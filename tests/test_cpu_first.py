import copy
import threading
from types import SimpleNamespace

import pytest
import torch

from src.progressive_execution import move_tensors


def test_cache_transfer_preserves_metadata_aliases_and_all_state_kinds():
    tensor = torch.arange(8.)
    layer = SimpleNamespace(keys=tensor, values=tensor, conv_states=[tensor],
                            recurrent_states={"r": tensor}, device=torch.device("cpu"), length=17)
    result = move_tensors(layer, torch.device("cpu"))
    assert result is layer and layer.length == 17
    assert layer.keys is layer.values is layer.conv_states[0] is layer.recurrent_states["r"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optional_recurrent_anchor_can_keep_cpu_original_after_gpu_adoption():
    from aircolibri_cli.thought_buffer import blend_recurrent_anchor
    original = torch.ones(2, 3)
    snapshot = SimpleNamespace(rec_states={0: original})
    live = torch.zeros(1, 2, 3, device="cuda")
    cache = SimpleNamespace(layers=[SimpleNamespace(recurrent_states=live)])
    blend_recurrent_anchor(cache, snapshot, .25)
    torch.testing.assert_close(live, torch.full_like(live, .25))
    assert torch.equal(original, torch.ones_like(original))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 6])
def test_cpu_expert_payload_moves_to_gpu_without_second_read_or_quantization(bits):
    from src.quantized_experts import QuantizedExpertCache
    class Source:
        uniform_shapes = has_gate = True
        inter_dim = hidden_dim = 128
        _spans = {(0, i): None for i in range(8)}
        def __init__(self, device):
            self.device = torch.device(device)
            self.reads = 0
        def load_batch(self, layer, ids, device, dtype, evict):
            self.reads += 1
            w = weights[ids].to(device=device, dtype=dtype)
            return w[:, 0].clone(), w[:, 1].clone(), w[:, 2].clone()
        def close(self):
            pass
    torch.manual_seed(82)
    weights = torch.randn(8, 3, 128, 128, dtype=torch.bfloat16)*.04
    module = torch.nn.Module()
    module.act_fn = torch.nn.SiLU()
    cpu = QuantizedExpertCache(Source("cpu"), bits, 1024**2)
    cpu.configure_layer(0, module)
    before = cpu.load_batch(0, [2, 5], "cpu", torch.bfloat16, True)
    assert 0 < cpu.cpu_packed_bytes <= cpu.cpu_packed_budget
    packet = cpu.detach_cpu_packed()
    assert cpu.cpu_packed_bytes == 0
    cpu.close()
    source = Source("cuda:0")
    gpu = QuantizedExpertCache(source, bits, 40*1024**2)
    gpu.configure_layer(0, module)
    try:
        gpu.adopt_cpu_packed(packet)
        after = gpu.load_batch(0, [2, 5], "cuda:0", torch.bfloat16, True)
        assert source.reads == 0 and gpu.conversions == 0 and gpu.cpu_startup_reuses == 2
        assert all(torch.equal(a, b.cpu()) for a, b in zip(before, after))
        assert gpu.cpu_packed_bytes == 0
    finally:
        gpu.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 6])
@torch.no_grad()
def test_common_start_and_readiness_handoff_preserve_state(tmp_path, bits):
    from transformers import LlamaConfig, LlamaForCausalLM
    from test_model_source import _write_tiny_hf_tokenizer
    from aircolibri_cli.model_source import open_model_source, PreadSafetensorsReader
    from aircolibri_cli.unified_runtime import plan_for_checkpoint
    from src.aircolibri_integrated import AirColibriModel

    torch.set_num_threads(6)
    torch.manual_seed(451)
    cfg = LlamaConfig(vocab_size=259, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    gate = threading.Event()
    prepared = threading.Event()
    gpu_entered = threading.Event()
    observed = []

    def plan(initializing_engine):
        observed.append(initializing_engine.progressive.cpu_layers)
        gpu_entered.set()
        assert gate.wait(30)
        free, total = torch.cuda.mem_get_info()
        p = plan_for_checkpoint(tmp_path, mode="normal", total_bytes=total, free_bytes=free,
                                weight_format=f"q{bits}", enable_mtp=False,
                                max_context_tokens=128, prefill_chunk_tokens=4,
                                projection_metadata=initializing_engine.weight_manager.quantized_projections)[1]
        prepared.set()
        return p

    source = open_model_source(tmp_path)
    reader = PreadSafetensorsReader()
    engine = AirColibriModel(str(tmp_path), model_source=source, weight_reader=reader,
                             cpu_first_plan=plan, weight_format=f"q{bits}",
                             universal_quantization=True, vram_resident_layers=0,
                             max_ram_layers=2, max_ram_bytes=8*1024**2)
    try:
        assert engine.device.type == "cpu" and engine.progressive.worker is not None
        assert gpu_entered.wait(10) and observed == [0]
        past = None
        first, past = engine._forward_step(torch.tensor([[3, 7, 4]]))
        assert first.device.type == "cpu" and past.get_seq_length() == 3
        assert engine.progressive.cpu_layers == 5
        assert engine.progressive.cpu_steps == 1
        gate.set()
        assert prepared.wait(10), str(engine.progressive.error)
        assert engine.progressive.backend_ready.wait(30), str(engine.progressive.error)
        snapshots = [first.float().cpu()]
        teacher = []
        for pos in range(3, 28):
            logits, past = engine._forward_step(torch.tensor([[pos % 17 + 3]]), past,
                                                cache_position=torch.tensor([pos]))
            assert past.get_seq_length() == pos + 1
            assert torch.isfinite(logits).all()
            snapshots.append(logits.float().cpu())
            teacher.append(pos % 17 + 3)
            if engine.progressive.complete:
                break
        assert observed and engine.progressive.complete, engine.progressive.stats()
        assert engine.device.type == "cuda"
        assert engine.progressive.gpu_layers > 0
        assert engine.progressive.kernels_ready
        assert all(layer.keys.device.type == "cuda" for layer in past.layers)
        # Continue from a copy on the same graph: migration must not replay or
        # truncate the prefix, including the first real decode after adoption.
        independent = copy.deepcopy(past)
        a, past = engine._forward_step(torch.tensor([[19]]), past)
        b, independent = engine._forward_step(torch.tensor([[19]]), independent)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Independent GPU-only trajectory from the same checkpoint and forced
        # tokens, rather than only comparing two copies after the handoff.
        gpu_source = open_model_source(tmp_path)
        gpu_reader = PreadSafetensorsReader()
        try:
            with AirColibriModel(str(tmp_path), model_source=gpu_source, weight_reader=gpu_reader,
                                  weight_format=f"q{bits}", universal_quantization=True,
                                  vram_resident_layers=5) as reference:
                expected, ref_past = reference._forward_step(torch.tensor([[3, 7, 4]]))
                torch.testing.assert_close(snapshots[0], expected.float().cpu(), rtol=.03, atol=.02)
                for token, actual in zip(teacher, snapshots[1:]):
                    expected, ref_past = reference._forward_step(torch.tensor([[token]]), ref_past)
                    torch.testing.assert_close(actual, expected.float().cpu(), rtol=.03, atol=.02)
        finally:
            gpu_source.close()
            gpu_reader.close()
    finally:
        gate.set()
        engine.close()
        source.close()
        reader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_public_common_start_does_not_order_devices_or_wait_for_gpu(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
    from test_model_source import _write_tiny_hf_tokenizer
    import aircolibri_cli.unified_runtime as runtime
    LlamaForCausalLM(LlamaConfig(vocab_size=259, hidden_size=64, intermediate_size=128,
                                num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=2)).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    gpu_entered, release_gpu = threading.Event(), threading.Event()
    original_plan = runtime.plan_for_checkpoint
    def blocked_gpu_plan(*args, **kwargs):
        gpu_entered.set()
        assert release_gpu.wait(30)
        return original_plan(*args, **kwargs)
    monkeypatch.setattr(runtime, "plan_for_checkpoint", blocked_gpu_plan)
    engine = runtime.AirColibriUnifiedEngine(tmp_path, enable_mtp=False, verbose=False)
    try:
        assert gpu_entered.wait(10)
        assert engine.engine.progressive.cpu_layers == 0
        assert engine.engine.progressive.start_event.is_set()
        with torch.no_grad():
            logits, past = engine.engine._forward_step(torch.tensor([[3, 7]]))
        assert logits.device.type == "cpu" and past.get_seq_length() == 2
        assert not release_gpu.is_set()  # CPU ran while GPU preparation was held.
        def forbidden(*args, **kwargs):
            raise AssertionError("GPU planner must reuse startup geometry")
        with monkeypatch.context() as guard:
            guard.setattr(AutoModelForCausalLM, "from_config", forbidden)
            release_gpu.set()
            assert engine.engine.progressive.backend_ready.wait(30), engine.engine.progressive.stats()
            assert engine.engine.progressive.bundle_ready.wait(30), engine.engine.progressive.stats()
            with torch.no_grad():
                _, past = engine.engine._forward_step(torch.tensor([[3, 7]]), past)
            assert engine.engine.progressive.complete and past.get_seq_length() == 4
    finally:
        release_gpu.set()
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 6])
@torch.no_grad()
def test_cpu_prefill_map_survives_ring_and_thought_anchor_handoff(tmp_path, monkeypatch, bits):
    from dataclasses import replace
    from test_probability_warmup import make_model
    import aircolibri_cli.unified_runtime as runtime
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    make_model(tmp_path, "granitemoe")
    original = runtime.plan_for_checkpoint
    def streaming(*args, **kwargs):
        profile, plan, mtp = original(*args, **kwargs)
        return profile, replace(plan, resident_layers=2, ring_slots=3,
                                full_residency=False), mtp
    monkeypatch.setattr(runtime, "plan_for_checkpoint", streaming)
    with runtime.AirColibriUnifiedEngine(tmp_path, weight_format=f"q{bits}",
                                         enable_mtp=False, verbose=False) as u:
        pending = u.enable_unified_memory(region_experts=2, host_cache_bytes=2**20)
        anchor = InitialProbabilityAnchor()
        logits, past = u.engine._forward_step(torch.tensor([[3, 7, 4]]), prefill=True)
        anchor.on_prefill_end(u, past, logits)
        fingerprint = pending.region_map.stats()["initial_sha256"]
        phi0 = anchor.phi0.clone()
        initial_probabilities = anchor.initial_probability_fingerprint()
        assert pending.region_map.initial
        assert u.engine.progressive.backend_ready.wait(30), u.engine.progressive.stats()
        for position in range(3, 45):
            logits, past = u.engine._forward_step(torch.tensor([[position % 19 + 2]]), past,
                                                  cache_position=torch.tensor([position]), prefill=False)
            adjusted = anchor.transform_logits(logits, past, position)
            anchor.on_step_committed(past, 1)
            assert torch.isfinite(adjusted).all()
            if u.engine.progressive.complete:
                break
        assert u.engine.progressive.complete, u.engine.progressive.stats()
        assert pending.active is u.engine.probability_warmup
        assert pending.active.region_map.stats()["initial_sha256"] == fingerprint
        assert torch.equal(anchor.phi0.cpu(), phi0.cpu())
        assert anchor.initial_probability_fingerprint() == initial_probabilities
        for position in range(45, 49):
            logits, past = u.engine._forward_step(torch.tensor([[5]]), past, prefill=False)
        stats = pending.active.stats()["shared_memory"]
        assert stats["roles"]["C"]["used_bytes"] == 0
        assert all(n > 0 for n in u.engine.model_source._expert_loader.shared_memory.area_reads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("family", ["mamba", "mamba2", "lfm2", "gemma2"])
@torch.no_grad()
def test_recurrent_and_sliding_state_survives_progressive_start(tmp_path, family):
    from transformers import AutoModelForCausalLM
    from test_session_state import tiny_config
    from test_model_source import _write_tiny_hf_tokenizer
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, _cache_seq_len
    config = tiny_config(family)
    config.vocab_size = 259
    AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    actual = []
    with AirColibriUnifiedEngine(tmp_path, weight_format="bf16", max_context_tokens=128,
                                 enable_mtp=False, verbose=False) as u:
        past = None
        sequences = [[3, 7, 4, 9], *[[i+3] for i in range(20)]]
        for iteration, tokens in enumerate(sequences):
            logits, past = u.engine._forward_step(torch.tensor([tokens]), past)
            actual.append(logits.float().cpu())
            if iteration == 0:
                assert u.engine.progressive.backend_ready.wait(30), u.engine.progressive.stats()
        assert u.engine.progressive.complete, u.engine.progressive.stats()
        length = _cache_seq_len(past)
    with AirColibriUnifiedEngine(tmp_path, weight_format="bf16", cpu_first=False,
                                 max_context_tokens=128, enable_mtp=False, verbose=False) as ref:
        past = None
        for tokens, observed in zip(sequences, actual):
            logits, past = ref.engine._forward_step(torch.tensor([tokens]), past)
            torch.testing.assert_close(observed, logits.float().cpu(), rtol=.04, atol=.025)
        assert _cache_seq_len(past) == length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cancel_drains_preparation_and_next_request_can_take_over(tmp_path):
    import json
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, GenerationCancelled
    LlamaForCausalLM(LlamaConfig(vocab_size=259, hidden_size=128, intermediate_size=256,
                                num_hidden_layers=5, num_attention_heads=4,
                                num_key_value_heads=2)).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    # A live text request needs actual byte tokens. The shape-only tokenizer
    # helper otherwise appends <eos> at id 259, outside this checkpoint's table.
    token_path = tmp_path / "tokenizer.json"
    tokenizer = json.loads(token_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"] = {"<eos>": 0, **{c: i+1 for i, c in enumerate(bytes_to_unicode().values())}}
    token_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    with AirColibriUnifiedEngine(tmp_path, enable_mtp=False, verbose=False,
                                 prefill_chunk_tokens=4, max_context_tokens=128) as u:
        with pytest.raises(GenerationCancelled):
            list(u.stream_chat([{"role": "user", "content": "Hi"}], max_new_tokens=8,
                                cancel_check=lambda: u.engine.progressive.cpu_layers >= 2))
        assert u.engine.progressive.worker is None
        assert not u.engine.progressive.pending and u.engine.progressive.jobs.empty()
        assert u.generations_cancelled == 1 and not u.engine._failed
        result = list(u.stream_chat([{"role": "user", "content": "Hi"}], max_new_tokens=12,
                                    stop_on_eos=False))[-1]
        assert result["type"] == "done"
        assert u.engine.progressive.complete
