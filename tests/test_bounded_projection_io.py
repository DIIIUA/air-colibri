"""Small checkpoints with tiny buffers exercise bounded reads and ring reuse."""
import weakref
import pytest
import torch
from safetensors.torch import save_file
from aircolibri_cli.model_source import PreadSafetensorsReader, open_model_source
from src.quantized_projection import pack_weight, pack_weight_rows


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.float8_e4m3fn])
def test_matrix_rectangles_bound_reads_and_scratch(tmp_path, dtype):
    matrix = (torch.arange(23*64).reshape(23, 64) % 17).to(dtype)
    path = tmp_path / "matrix.safetensors"
    save_file({"w": matrix}, str(path))
    reader = PreadSafetensorsReader(max_read_bytes=256)
    try:
        for bounds in [(0,23,0,64), (2,19,7,41), (1,20,11,12), (3,3,0,64), (1,5,9,9)]:
            r0,r1,c0,c1 = bounds
            got = reader.read_matrix_slice(path, "w", *bounds)
            assert torch.equal(got.float(), matrix[r0:r1,c0:c1].float())
        stats = reader.matrix_io_stats()
        assert 0 < stats["max_read_bytes"] <= 256
        assert stats["max_scratch_bytes"] <= 256
        assert stats["requested_bytes"] <= stats["read_bytes"] <= 8 * stats["requested_bytes"]
        with pytest.raises(ValueError, match="outside"):
            reader.read_matrix_slice(path, "w", 0, 24)
        with pytest.raises(ValueError, match="outside"):
            reader.read_matrix_slice(path, "w", 0, 1, -1, 4)
    finally:
        reader.close()


@pytest.mark.parametrize("bits", [4,6])
def test_row_packer_releases_tiles_and_recovers_from_reader_failure(bits):
    torch.manual_seed(671)
    weight = torch.randn(145, 128)
    expected = pack_weight(weight, bits)
    weak_tiles = []
    bounds = []
    def rows(a, b):
        assert all(ref() is None for ref in weak_tiles)
        tile = weight[a:b].clone()
        weak_tiles.append(weakref.ref(tile))
        bounds.append((a,b))
        return tile
    actual = pack_weight_rows(weight.shape, rows, bits, tile_elements=8*128, row_alignment=8)
    assert len(bounds) == 19 and bounds[-1] == (144,145)
    assert all(b-a <= 8 for a,b in bounds)
    assert all(torch.equal(actual[k], v) for k,v in expected.items())
    assert all(ref() is None for ref in weak_tiles)
    def interrupted(a,b):
        if a >= 16:
            raise RuntimeError("source interrupted")
        return rows(a,b)
    with pytest.raises(RuntimeError, match="source interrupted"):
        pack_weight_rows(weight.shape, interrupted, bits, tile_elements=8*128)
    recovered = pack_weight_rows(weight.shape, rows, bits, tile_elements=8*128)
    assert all(torch.equal(recovered[k], v) for k,v in expected.items())
    with pytest.raises(ValueError, match="aligned group"):
        pack_weight_rows(weight.shape, rows, bits, tile_elements=127)
    with pytest.raises(ValueError, match="Weight reader returned"):
        pack_weight_rows(weight.shape, lambda a,b: weight[:1], bits, tile_elements=8*128)


def write_checkpoint(folder, method):
    from transformers import AutoModelForCausalLM, Qwen2Config
    from test_hf_quantized_source import packed_fixture
    torch.manual_seed(479)
    config = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                         tie_word_embeddings=False)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    # Every projection must reach the 4096-element packing threshold; tiny raw
    # projections intentionally keep their existing full-read fallback.
    state = dict(model.state_dict())
    for name, weight in list(state.items()):
        if ".layers." not in name or weight.ndim != 2 or not name.endswith(".weight"):
            continue
        prefix = name.removesuffix("weight")
        if method == "fp8":
            state[name] = (weight*16).to(torch.float8_e4m3fn)
            state[prefix+"weight_scale_inv"] = torch.full((weight.shape[0]//16,weight.shape[1]//16),1/16,dtype=torch.bfloat16)
        elif method in {"awq","gptq"}:
            packed, _ = packed_fixture(*weight.shape, method)
            state.pop(name)
            state.update({prefix+k:v for k,v in packed.items()})
    if method != "plain":
        config.quantization_config = ({"quant_method":"fp8","fmt":"e4m3","weight_block_size":[16,16]}
            if method == "fp8" else {"quant_method":method,"bits":4,"group_size":16,"version":"gemm"})
    config.save_pretrained(folder)
    save_file({k:v.contiguous() for k,v in state.items()}, str(folder/"model.safetensors"))


@pytest.mark.parametrize("method", ["plain","fp8","awq","gptq"])
@pytest.mark.parametrize("bits", [4,6])
@torch.no_grad()
def test_tiny_buffer_matches_full_source_through_ring(tmp_path, monkeypatch, method, bits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from src.aircolibri_integrated import AirColibriModel
    from src import quantized_projection as quant
    write_checkpoint(tmp_path, method)
    reference_source = open_model_source(tmp_path)
    reference_source.read_weight_rows = None  # previous full-matrix path
    expected = []
    tokens = ([3,5,11], [7], [13])
    with AirColibriModel(str(tmp_path), weight_format=f"q{bits}", universal_quantization=True,
                         vram_resident_layers=4, model_source=reference_source) as resident:
        past = None
        for ids in tokens:
            out, past = resident._forward_step(torch.tensor([ids]), past_key_values=past, prefetch_next=True)
            expected.append(out.cpu())
    del past, out
    source = open_model_source(tmp_path)
    reader = source._pread_reader
    reader.max_read_bytes = 256
    read_full = reader.read_tensor
    def forbid_full_matrix(path, key):
        if len(reader._header(str(path))[0][key]["shape"]) == 2:
            raise AssertionError(f"full matrix read: {key}")
        return read_full(path, key)
    monkeypatch.setattr(reader, "read_tensor", forbid_full_matrix)
    monkeypatch.setattr(quant, "TILE_ELEMENTS", 1024)
    with AirColibriModel(str(tmp_path), weight_format=f"q{bits}", universal_quantization=True,
                         vram_resident_layers=1, num_slots=3, model_source=source,
                         weight_reader=reader) as ring:
        past = None
        for ids, ref in zip(tokens, expected):
            actual, past = ring._forward_step(torch.tensor([ids]), past_key_values=past, prefetch_next=True)
            assert torch.equal(actual.cpu(), ref)
        stats = source.projection_io_stats()
        assert stats["slices"] > 100
        assert 0 < stats["max_slice_bytes"] <= 1024 * 4
        assert stats["max_read_bytes"] <= 256 and stats["max_scratch_bytes"] <= 256
        if method != "plain":
            assert stats["decoded_tiles"] > 100 and stats["max_decoded_elements"] <= 1024


def test_transposed_hf_rows_preserve_conv1d_orientation(tmp_path):
    from transformers import GPT2Config, AutoModelForCausalLM
    torch.manual_seed(111)
    cfg = GPT2Config(vocab_size=64, n_embd=64, n_layer=1, n_head=4)
    model = AutoModelForCausalLM.from_config(cfg)
    cfg.save_pretrained(tmp_path)
    state = {k:v.clone().contiguous() for k,v in model.state_dict().items()}
    save_file(state,str(tmp_path/"model.safetensors"))
    source = open_model_source(tmp_path)
    try:
        entry = source.layer_entries(0)["model.layers.0.attn.c_attn.weight"]
        for a,b in [(0,11),(11,49),(183,192)]:
            got=source.read_weight_rows(entry,a,b,transpose=True)
            assert torch.equal(got,state[entry[1]].T[a:b])
    finally:
        source.close()


@pytest.mark.parametrize("method", ["fp8", "awq", "gptq"])
def test_unaligned_rows_cross_source_blocks_and_reject_bad_metadata(tmp_path, method):
    from safetensors.torch import load_file
    from aircolibri_cli.model_source import UnsupportedModelError
    write_checkpoint(tmp_path, method)
    source = open_model_source(tmp_path)
    try:
        entry = source.layer_entries(0)["model.layers.0.self_attn.q_proj.weight"]
        expected = source.read_decoded(entry)
        for a,b in [(0,1), (7,19), (15,33), (59,64), (5,5)]:
            assert torch.equal(source.read_weight_rows(entry,a,b), expected[a:b])
        with pytest.raises(ValueError, match="outside"):
            source.read_weight_rows(entry, 0, 65)
    finally:
        source.close()
    path = tmp_path / "model.safetensors"
    state = load_file(str(path))
    suffix = "weight_scale_inv" if method == "fp8" else "qweight"
    key = "model.layers.0.self_attn.q_proj." + suffix
    state[key] = torch.cat((state[key], state[key][:,:1]), dim=1)
    save_file(state, str(path))
    with pytest.raises(UnsupportedModelError, match="scale grid|INT4 geometry"):
        open_model_source(tmp_path)
