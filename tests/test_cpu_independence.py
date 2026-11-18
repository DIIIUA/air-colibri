"""The disk/CPU executor must survive the absence or rejection of its GPU helper."""
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def tiny_checkpoint(path):
    import json
    from transformers import LlamaConfig, AutoModelForCausalLM
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from test_model_source import _write_tiny_hf_tokenizer
    torch.set_num_threads(6)
    torch.manual_seed(853)
    model = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=259, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2),
        dtype=torch.bfloat16).eval()
    model.save_pretrained(path)
    _write_tiny_hf_tokenizer(path, vocab_size=259)
    tokenizer_path = path / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"] = {"<eos>": 0, **{c: i+1 for i, c in enumerate(bytes_to_unicode().values())}}
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    return model


def reference_weights(model, runtime):
    """HF arithmetic over decoded checkpoint weights, independent of streaming hooks."""
    from src.quantized_projection import pack_weight, unpack_weight
    bits = runtime.engine.quant_bits
    if bits:
        for i, specs in runtime.engine.weight_manager.quantized_projections.items():
            for name, spec in specs.items():
                weight = model.model.layers[i].get_parameter(name)
                source = weight.T if spec["transpose"] else weight
                decoded = unpack_weight(pack_weight(source, bits, device="cpu"))
                weight.data.copy_(decoded.T if spec["transpose"] else decoded)
    return model


def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU execution touched a CUDA runtime operation")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("device", "Stream", "mem_get_info", "synchronize",
                 "get_device_name", "memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize("fmt", ["bf16", "q4", "q6"])
@torch.no_grad()
def test_no_gpu_streams_every_layer_and_preserves_logits_cache_and_session(tmp_path, monkeypatch, fmt):
    import aircolibri_cli.unified_runtime as runtime
    reference = tiny_checkpoint(tmp_path)
    forbid_cuda(monkeypatch)
    monkeypatch.setattr(runtime, "_streaming_ram_budget_bytes", lambda: 48*1024)
    with runtime.AirColibriUnifiedEngine(tmp_path, weight_format=fmt, verbose=False,
                                         max_context_tokens=128, prefill_chunk_tokens=4) as u:
        reference_weights(reference, u)
        p = u.engine.progressive
        assert p.worker is None
        assert u.engine.weight_manager.pinned_pool.arena is None
        assert u.engine.weight_manager.pinned_pool.total_pinned_bytes == 0
        actual_cache = expected_cache = None
        for tokens in ([[3, 7, 4]], [[9]], [[11]], [[13]]):
            ids = torch.tensor(tokens)
            actual, actual_cache = u.engine._forward_step(ids, actual_cache)
            expected = reference(ids, past_key_values=expected_cache, use_cache=True, logits_to_keep=1)
            expected_cache = expected.past_key_values
            torch.testing.assert_close(actual, expected.logits[:, -1], rtol=0, atol=0)
            for a, b in zip(actual_cache.layers, expected_cache.layers):
                torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
                torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
        assert p.cpu_layers == 16 and p.gpu_layers == 0
        assert p.jobs.empty() and not p.pending
        assert u.engine.weight_manager.ram_cache.peak_bytes <= 48*1024
        messages = []
        for index, question in enumerate(("Hi", "Hi again")):
            messages.append({"role": "user", "content": question})
            # With one output token, the first parked prefix is exactly the
            # prompt; random byte tokens cannot invalidate it on re-tokenizing.
            events = list(u.stream_chat(messages, max_new_tokens=1 if index == 0 else 3,
                                       stop_on_eos=False, use_session_cache=True))
            assert events[-1]["type"] == "done"
            if index:
                assert events[-1]["metrics"]["cached_prefix_tokens"] > 0
            messages.append({"role": "assistant", "content": events[-1]["answer_text"]})
        snap = u.status_snapshot()
        assert snap["progressive_execution"]["gpu_state"] == "unavailable"
        assert snap["allocated_vram_mib"] == 0
        assert u.engine.weight_manager.pinned_pool.arena is None


def test_large_staging_geometry_needs_no_memory_until_a_transfer():
    from src.bounded_memory_manager import BoundedPinnedBufferPool, SlotArenaLayout
    # Headers describing >8 GB per slot must not allocate three such buffers.
    layout = SlotArenaLayout.from_templates({"w": ((64000, 64000), torch.bfloat16)})
    with BoundedPinnedBufferPool(arena_layout=layout, num_slots=3,
                                 device=torch.device("cpu"), allocate=False) as pool:
        assert pool.layout.payload_bytes > 8_000_000_000
        assert pool.arena is None and pool.total_pinned_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.copy_into_slot(0, {})


def test_deferred_staging_allocates_only_on_actual_copy_and_stays_closed():
    from src.bounded_memory_manager import BoundedPinnedBufferPool
    value = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    pool = BoundedPinnedBufferPool({"w": value}, num_slots=2,
                                  device=torch.device("cpu"), allocate=False)
    assert pool.arena is None
    torch.testing.assert_close(pool.copy_into_slot(1, {"w": value})["w"], value)
    assert pool.total_pinned_bytes > 0
    pool.close()
    assert pool.total_pinned_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        pool.copy_into_slot(0, {"w": value})


@torch.no_grad()
def test_gpu_planning_failure_keeps_cpu_state_and_does_not_retry(tmp_path, monkeypatch):
    import aircolibri_cli.unified_runtime as runtime
    reference = tiny_checkpoint(tmp_path)
    calls = []
    def rejected(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("GPU working set does not fit")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1024, 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(uuid=None))
    monkeypatch.setattr(runtime, "plan_for_checkpoint", rejected)
    with runtime.AirColibriUnifiedEngine(tmp_path, weight_format="q6", verbose=False) as u:
        reference_weights(reference, u)
        u.engine.progressive.worker.join(10)
        cache = ref_cache = None
        for tokens in ([[3, 7]], [[11]], [[13]]):
            ids = torch.tensor(tokens)
            actual, cache = u.engine._forward_step(ids, cache)
            expected = reference(ids, past_key_values=ref_cache, use_cache=True, logits_to_keep=1)
            ref_cache = expected.past_key_values
            torch.testing.assert_close(actual, expected.logits[:, -1], rtol=0, atol=0)
        state = u.status_snapshot()["progressive_execution"]
        assert state["gpu_state"] == "failed" and "does not fit" in state["preparation_error"]
        assert state["queued_payloads"] == 0 and u.engine.progressive.worker is None
        assert len(calls) == 1


def test_explicit_cpu_choice_never_starts_gpu_and_doctor_accepts_it(tmp_path, monkeypatch):
    import json
    from aircolibri_cli.cli import build_parser, cmd_doctor
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    tiny_checkpoint(tmp_path)
    forbid_cuda(monkeypatch)
    # A CPU choice must stay independent even on a machine that has CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def no_pin(*args, **kwargs):
        raise AssertionError("CPU session tried to allocate CUDA-pinned memory")
    monkeypatch.setattr(torch.Tensor, "pin_memory", no_pin)
    with AirColibriUnifiedEngine(tmp_path, device="cpu", progressive_start=False, verbose=False) as u:
        assert u.engine.progressive.worker is None
        assert u.status_snapshot()["progressive_execution"]["gpu_state"] == "disabled"
        assert list(u.stream_chat([{"role": "user", "content": "Hi"}],
                                   max_new_tokens=2, stop_on_eos=False,
                                   use_session_cache=True))[-1]["type"] == "done"
    output = tmp_path / "doctor.json"
    args = build_parser().parse_args(["doctor", "--model", str(tmp_path),
                                      "--device", "cpu", "--json-out", str(output)])
    assert cmd_doctor(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["execution_device"] == "cpu"
    assert "vram_plan_error" not in report


def test_doctor_reports_gpu_plan_rejection_without_rejecting_cpu(tmp_path, monkeypatch):
    import json
    import aircolibri_cli.unified_runtime as runtime
    from aircolibri_cli.cli import build_parser, cmd_doctor
    tiny_checkpoint(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "test GPU")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1024, 1024))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(uuid=None))
    def rejected(*args, **kwargs):
        raise RuntimeError("GPU working set does not fit")
    monkeypatch.setattr(runtime, "plan_for_checkpoint", rejected)
    output = tmp_path / "doctor-fallback.json"
    args = build_parser().parse_args(["doctor", "--model", str(tmp_path), "--json-out", str(output)])
    assert cmd_doctor(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["execution_device"] == "cpu" and "does not fit" in report["vram_plan_error"]


@pytest.mark.parametrize("bits", [4, 6])
@torch.no_grad()
def test_cpu_prefill_map_and_original_probabilities_survive_decode(tmp_path, monkeypatch, bits):
    from test_probability_warmup import make_model
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    make_model(tmp_path, "granitemoe")
    forbid_cuda(monkeypatch)
    with AirColibriUnifiedEngine(tmp_path, weight_format=f"q{bits}", device="cpu", verbose=False) as u:
        pending = u.enable_unified_memory(region_experts=2, host_cache_bytes=2**20)
        anchor = InitialProbabilityAnchor()
        logits, past = u.engine._forward_step(torch.tensor([[3, 7, 4]]), prefill=True)
        anchor.on_prefill_end(u, past, logits)
        fingerprint = pending.region_map.stats()["initial_sha256"]
        original = anchor.initial_probability_fingerprint()
        assert pending.region_map.initial
        for position in range(3, 8):
            logits, past = u.engine._forward_step(torch.tensor([[5]]), past, prefill=False)
            assert torch.isfinite(anchor.transform_logits(logits, past, position)).all()
            anchor.on_step_committed(past, 1)
        assert pending.region_map.stats()["initial_sha256"] == fingerprint
        assert anchor.initial_probability_fingerprint() == original
        assert pending.active is None  # No GPU arena is fabricated for CPU execution.
        # The optional anchor can perform extra probes on the same CPU graph.
        assert u.engine.progressive.cpu_layers >= 30
        assert u.engine.progressive.gpu_layers == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("verification", [False, True])
@torch.no_grad()
def test_failed_helper_returns_adopted_layer_and_live_cache_to_cpu(tmp_path, monkeypatch, verification):
    import aircolibri_cli.unified_runtime as runtime
    from src.progressive_execution import ProgressiveExecution
    reference = tiny_checkpoint(tmp_path)
    second_copy, reject = threading.Event(), threading.Event()
    original = ProgressiveExecution._copy_payload
    copies = []
    def interrupted(self, tensors):
        copies.append(1)
        # Globals, buffers, first layer succeed; the second layer fails later.
        if len(copies) == 4:
            second_copy.set()
            assert reject.wait(30)
            raise RuntimeError("injected GPU transfer failure")
        return original(self, tensors)
    monkeypatch.setattr(ProgressiveExecution, "_copy_payload", interrupted)
    with runtime.AirColibriUnifiedEngine(tmp_path, weight_format="q6", verbose=False) as u:
        reference_weights(reference, u)
        p = u.engine.progressive
        ids = torch.tensor([[3, 7]])
        try:
            _, past = u.engine._forward_step(ids)
            expected = reference(ids, use_cache=True, logits_to_keep=1)
            ref_past = expected.past_key_values
            assert second_copy.wait(30), p.stats()
            # Consume the prepared first layer while the next transfer is held.
            _, past = u.engine._forward_step(torch.tensor([[9]]), past)
            ref_past = reference(torch.tensor([[9]]), past_key_values=ref_past,
                                 use_cache=True, logits_to_keep=1).past_key_values
            assert p.adopted == {0} and past.layers[0].keys.is_cuda
            reject.set()
            p.worker.join(10)
            assert not p.worker.is_alive()
            if verification:
                # A helper failure cannot move a live verification round after
                # its rollback snapshots were taken. Retire at the next boundary.
                with p.verification_forwards():
                    _, past = u.engine._forward_step(torch.tensor([[10]]), past)
                    assert p.gpu_enabled and past.layers[0].keys.is_cuda
                ref_past = reference(torch.tensor([[10]]), past_key_values=ref_past,
                                     use_cache=True, logits_to_keep=1).past_key_values
            actual, past = u.engine._forward_step(torch.tensor([[11]]), past)
            expected = reference(torch.tensor([[11]]), past_key_values=ref_past,
                                 use_cache=True, logits_to_keep=1)
            torch.testing.assert_close(actual, expected.logits[:, -1], rtol=.03, atol=.02)
            assert past.get_seq_length() == 4+int(verification)
            assert all(layer.keys.device.type == "cpu" for layer in past.layers)
            assert not any(t.is_cuda for t in u.engine.model.parameters())
            assert not any(t.is_cuda for t in u.engine.model.buffers())
            assert p.gpu_state == "failed" and not p.ready and not p.adopted
            assert p.global_payload is None and p.jobs.empty() and p.worker is None
        finally:
            reject.set()
