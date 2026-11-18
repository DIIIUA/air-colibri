"""Causal-cache, ring-wrap and cancellation checks for the three-stage wave."""
import copy
import json
import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def llama_folder(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetensors.torch import save_file
    from test_model_source import _write_tiny_hf_tokenizer
    torch.manual_seed(124)
    cfg = LlamaConfig(vocab_size=259, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=2,
                      tie_word_embeddings=False, eos_token_id=0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    cfg.save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_data = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    tokenizer_data["model"]["vocab"] = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(bytes_to_unicode().values())}}
    tokenizer_file.write_text(json.dumps(tokenizer_data), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    return tmp_path


@pytest.mark.parametrize("fmt,resident", [("bf16", 0), ("q4", 2), ("q6", 0), ("q6", 7)])
@torch.no_grad()
def test_wave_matches_chunked_reference_cache_and_decode(llama_folder, fmt, resident):
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source
    from src.gpu_ring_pipeline import SlotState
    with AirColibriModel(str(llama_folder), weight_format=fmt, universal_quantization=True,
                         vram_resident_layers=resident, num_slots=3, max_ram_layers=7,
                         model_source=open_model_source(llama_folder)) as engine:
        tokens = torch.arange(1, 30).unsqueeze(0)
        # Prefix reuse, uneven last chunk, multiple ring wraps, then ordinary decode.
        _, prefix = engine._forward_step(tokens[:, :5], prefetch_next=False)
        expected_cache = copy.deepcopy(prefix)
        for part in tokens[:, 5:].split(7, dim=1):
            expected, expected_cache = engine._forward_step(part, expected_cache, prefetch_next=False)
        expected = expected.clone()
        wave, cache = engine.prefill_wavefront(tokens[:, 5:], 7, prefix, collect_timing=True)
        torch.testing.assert_close(wave, expected, rtol=.02, atol=.008)
        for a, b in zip(cache.layers, expected_cache.layers):
            torch.testing.assert_close(a.keys, b.keys, rtol=.02, atol=.008)
            torch.testing.assert_close(a.values, b.values, rtol=.02, atol=.008)
        assert [l.get_seq_length() for l in cache.layers] == [29] * 7
        assert engine.last_wavefront_stats["weight_loads"] == 7 - resident
        assert all(s is None for s in engine.active_slot_per_layer)
        assert all(s.state == SlotState.FREE for s in getattr(engine.pipeline, "slots", []))
        for token in (30, 31, 32):
            ref, expected_cache = engine._forward_step(torch.tensor([[token]]), expected_cache, prefetch_next=False)
            got, cache = engine._forward_step(torch.tensor([[token]]), cache, prefetch_next=False)
            torch.testing.assert_close(got, ref, rtol=.02, atol=.008)


@torch.no_grad()
def test_cancel_mid_wave_recovers_without_busy_slots(llama_folder):
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source
    from aircolibri_cli.unified_runtime import GenerationCancelled
    with AirColibriModel(str(llama_folder), vram_resident_layers=0, num_slots=3,
                         model_source=open_model_source(llama_folder)) as engine:
        calls = [0]
        def cancel():
            calls[0] += 1
            if calls[0] == 9:
                raise GenerationCancelled("prefill")
        engine._step_cancel_check = cancel
        with pytest.raises(GenerationCancelled):
            engine.prefill_wavefront(torch.arange(1, 29).unsqueeze(0), 7)
        assert not engine._failed and not engine._wavefront_active
        assert all(s is None for s in engine.active_slot_per_layer)
        engine._step_cancel_check = None
        serial, _ = engine.prefill_wavefront(torch.arange(1, 29).unsqueeze(0), 7, concurrent=False)
        concurrent, _ = engine.prefill_wavefront(torch.arange(1, 29).unsqueeze(0), 7)
        torch.testing.assert_close(concurrent, serial, rtol=0, atol=0)


@torch.no_grad()
def test_unified_wave_session_cancel_and_thought_anchor(llama_folder, monkeypatch):
    from dataclasses import replace
    from aircolibri_cli import unified_runtime as runtime
    from aircolibri_cli.thought_buffer import ColibriThoughtBuffer
    original_plan = runtime.plan_for_checkpoint
    def force_ring(*args, **kwargs):
        profile, plan, mtp = original_plan(*args, **kwargs)
        return profile, replace(plan, full_residency=False, resident_layers=1, ring_slots=3), mtp
    monkeypatch.setattr(runtime, "plan_for_checkpoint", force_ring)
    with runtime.AirColibriUnifiedEngine(llama_folder, cpu_first=False, weight_format="q6", verbose=False,
                                        prefill_chunk_tokens=8, prefill_wavefront=True,
                                        enable_mtp=False, max_context_tokens=512) as engine:
        messages = [{"role": "user", "content": "The next operation uses the result of the previous operation. " * 4}]
        done = [e for e in engine.stream_chat(messages, max_new_tokens=3, stop_on_eos=False,
                                              enable_thinking=False, use_session_cache=True) if e["type"] == "done"][0]
        assert done["metrics"]["prefill_wavefront"]["windows"]
        messages += [{"role": "assistant", "content": done["answer_text"]}, {"role": "user", "content": "Continue with another operation. " * 4}]
        follow = [e for e in engine.stream_chat(messages, max_new_tokens=3, stop_on_eos=False,
                                                enable_thinking=False, use_session_cache=True) if e["type"] == "done"][0]
        assert follow["metrics"]["cached_prefix_tokens"] > 0
        count = [0]
        def cancel():
            count[0] += 1
            return count[0] > 12
        with pytest.raises(runtime.GenerationCancelled):
            list(engine.stream_chat(messages, max_new_tokens=3, cancel_check=cancel, use_session_cache=False))
        anchored = ColibriThoughtBuffer(engine).ask_anchored(messages[:1], max_new_tokens=3)
        assert anchored["anchor_telemetry"]["total_rows_scored"] > 0
        assert not engine.engine._wavefront_active
