"""Stored quantization is decoded before the common CPU/GPU ring path."""
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, AutoModelForCausalLM
from aircolibri_cli.hf_quantized_source import decode_fp8, decode_int4
from aircolibri_cli.model_source import open_model_source


def packed_fixture(n, k, method, group=16):
    # Build source integers first; the expected float matrix never uses decoding.
    q = torch.randint(0, 16, (k, n), dtype=torch.int32)
    zero = torch.randint(1, 15, (k//group, n), dtype=torch.int32)
    scale = (torch.randint(1, 17, zero.shape).float() / 1024).half()
    groups = torch.arange(k, dtype=torch.int32) // group
    if method == "gptq":
        groups = groups[torch.randperm(k)]  # exercise activation-order groups
    expected = ((q.float() - zero[groups].float()) * scale[groups].float()).T.contiguous()
    if method == "awq":
        words = torch.zeros(k, n//8, dtype=torch.int32)
        zeros = torch.zeros(k//group, n//8, dtype=torch.int32)
        for slot, col in enumerate([0, 2, 4, 6, 1, 3, 5, 7]):
            words |= q[:, col::8] << (4*slot)
            zeros |= zero[:, col::8] << (4*slot)
    else:
        words = torch.zeros(k//8, n, dtype=torch.int32)
        zeros = torch.zeros(k//group, n//8, dtype=torch.int32)
        for slot in range(8):
            words |= q[slot::8] << (4*slot)
            zeros |= (zero[:, slot::8]-1) << (4*slot)
    return {"qweight": words, "qzeros": zeros, "scales": scale,
            **({"g_idx": groups} if method == "gptq" else {})}, expected


@pytest.mark.parametrize("method", ["awq", "gptq"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_int4_matches_original_integer_matrix(method, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from aircolibri_cli import hf_quantized_source as codec
    monkeypatch.setattr(codec, "TILE_ELEMENTS", 512)  # multiple output tiles
    torch.manual_seed(81)
    packed, expected = packed_fixture(40, 64, method)
    actual = decode_int4(**packed, method=method, group_size=16, device=device)
    assert torch.equal(actual.cpu(), expected)
    if method == "gptq":
        packed["g_idx"][0] = 99
        with pytest.raises(ValueError, match="group index"):
            decode_int4(**packed, method=method, group_size=16, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_matches_transformers_reference_including_partial_blocks(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from aircolibri_cli import hf_quantized_source as codec
    from transformers.integrations.finegrained_fp8 import _dequantize_fp8_weight
    monkeypatch.setattr(codec, "TILE_ELEMENTS", 256)
    torch.manual_seed(19)
    w = torch.randn(35, 71).to(torch.float8_e4m3fn)
    scales = torch.rand(3, 5).to(torch.bfloat16)
    expected = _dequantize_fp8_weight(w, scales, [16, 16], torch.float32)
    assert torch.equal(decode_fp8(w, scales, [16,16], device).cpu(), expected)
    with pytest.raises(ValueError, match="scale grid"):
        decode_fp8(w, scales[:2], [16,16], device)


@pytest.mark.parametrize("method", ["awq", "gptq", "fp8"])
@torch.no_grad()
def test_encoded_checkpoint_matches_native_model_through_ring(tmp_path, method):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from src.aircolibri_integrated import AirColibriModel
    torch.manual_seed(934)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    # Construct in BF16 rather than casting all buffers afterwards: RoPE keeps
    # its frequencies in FP32 in the real from_pretrained(dtype=...) path.
    native = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    state = dict(native.state_dict())
    expected_state = dict(state)
    for name, weight in list(state.items()):
        if ".layers." not in name or not name.endswith(".weight") or weight.ndim != 2:
            continue
        prefix = name.removesuffix("weight")
        if method == "fp8":
            value = (weight * 16).to(torch.float8_e4m3fn)
            scales = torch.full((weight.shape[0]//16, weight.shape[1]//16), 1/16, dtype=torch.bfloat16)
            state[name] = value
            state[prefix+"weight_scale_inv"] = scales
            expected_state[name] = value.float() / 16
        else:
            packed, expected = packed_fixture(*weight.shape, method)
            state.pop(name)
            state.update({prefix+k: v for k,v in packed.items()})
            expected_state[name] = expected
    native.load_state_dict(expected_state)
    native.to(device="cuda")
    cfg.quantization_config = ({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [16,16]}
                              if method == "fp8" else {"quant_method": method, "bits": 4,
                                                       "group_size": 16, "version": "gemm"})
    cfg.save_pretrained(tmp_path)
    save_file({k: v.contiguous() for k,v in state.items()}, str(tmp_path / "model.safetensors"))
    source = open_model_source(tmp_path)
    for entries in source._layers.values():
        for name, entry in entries.items():
            if entry in source._quantized:
                assert torch.equal(source.read_decoded(entry), expected_state[name])
    with AirColibriModel(str(tmp_path), weight_format="bf16", vram_resident_layers=1,
                         num_slots=3, model_source=source, device=torch.device("cuda")) as runtime:
        native_past = ring_past = None
        for tokens in ([3, 7, 11], [8], [14]):
            ids = torch.tensor([tokens], device="cuda")
            out = native(ids, past_key_values=native_past, use_cache=True, logits_to_keep=1)
            got, ring_past = runtime._forward_step(ids, past_key_values=ring_past, prefetch_next=True)
            torch.testing.assert_close(got, out.logits[:, -1], rtol=0, atol=0)
            native_past = out.past_key_values
