"""The shared verifier must preserve the model trajectory with streamed weights."""
import copy

import pytest
import torch

from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, ColibriSpecGate


def collect(engine, prompt, count=9):
    events = list(engine._stream_token_ids(prompt, count, False, 0, None))
    tokens = [t for event in events if event["kind"] == "tokens" for t in event["new_ids"]]
    return tokens, events[-1]


def fixed_proposals(speculator, expected, prompt_len, rejection):
    def propose(history, count, native_ready):
        start = len(history)-prompt_len
        tokens = list(expected[start:start+count])
        if tokens and rejection == "first":
            tokens[0] = (tokens[0]+1) % 64
        elif len(tokens) > 1 and rejection == "second":
            tokens[1] = (tokens[1]+1) % 64
        return tokens, [.95]*len(tokens), ColibriSpecGate.SRC_LOOKUP
    speculator.propose = propose


@pytest.mark.parametrize("fmt", ["bf16", "q4", "q6"])
@pytest.mark.parametrize("rejection", ["none", "first", "second"])
@torch.no_grad()
def test_streamed_dense_accept_and_reject_preserve_tokens_and_cache(tmp_path, monkeypatch, fmt, rejection):
    from test_cpu_independence import tiny_checkpoint
    import aircolibri_cli.unified_runtime as runtime
    tiny_checkpoint(tmp_path)
    monkeypatch.setattr(runtime, "_streaming_ram_budget_bytes", lambda: 48*1024)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    prompt = [3, 7, 4, 9]
    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format=fmt, speculation="off", verbose=False) as base:
        expected, state = collect(base, prompt)
        reference = copy.deepcopy(state["past"])
    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format=fmt, speculation="lookup", verbose=False) as u:
        fixed_proposals(u.speculator, expected, len(prompt), rejection)
        actual, result = collect(u, prompt)
        assert actual == expected
        assert result["consumed_tokens"] == len(prompt)+len(actual)-1
        for a, b in zip(result["past"].layers, reference.layers):
            torch.testing.assert_close(a.keys, b.keys, rtol=.03, atol=.02)
            torch.testing.assert_close(a.values, b.values, rtol=.03, atol=.02)
        assert result["spec_gate"]["verified_batches"] > 0
        assert u.engine.weight_manager.ram_cache.peak_bytes <= 48*1024
        assert u.engine.weight_manager.pinned_pool.total_pinned_bytes == 0
        if rejection == "none":
            assert result["backbone_passes"] < len(actual)-1


@pytest.mark.parametrize("family", ["mamba", "mamba2", "lfm2", "gemma2"])
@torch.no_grad()
def test_rejected_rows_restore_recurrent_and_sliding_state(tmp_path, monkeypatch, family):
    from transformers import AutoModelForCausalLM
    from test_session_state import tiny_config
    from test_model_source import _write_tiny_hf_tokenizer
    torch.set_num_threads(6)
    torch.manual_seed(992)
    config = tiny_config(family)
    AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=64)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    prompt = [3, 7, 4]
    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format="bf16", speculation="off", verbose=False) as base:
        expected, state = collect(base, prompt)
        before = copy.deepcopy(state["past"])
    with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format="bf16", speculation="lookup", verbose=False) as u:
        fixed_proposals(u.speculator, expected, len(prompt), "second")
        actual, result = collect(u, prompt)
        assert actual == expected
        assert result["spec_gate"].get("replay_forwards", 0)+result["spec_gate"].get("row_state_restores", 0) > 0
        ids = torch.tensor([[17]])
        a, _ = u.engine._forward_step(ids, result["past"])
        b, _ = u.engine._forward_step(ids, before)
        torch.testing.assert_close(a, b, rtol=.03, atol=.02)


def test_native_head_budget_is_checked_before_tensor_reads(tmp_path, monkeypatch):
    from src.native_mtp import NativeMTPHead
    from types import SimpleNamespace
    from transformers import Qwen3_5TextConfig
    from safetensors.torch import save_file
    from aircolibri_cli.model_source import PreadSafetensorsReader
    save_file({"mtp.fc.weight": torch.zeros(128, 256, dtype=torch.bfloat16)}, str(tmp_path/"head.safetensors"))
    cfg = Qwen3_5TextConfig(hidden_size=128)
    engine = SimpleNamespace(config=cfg, dtype=torch.bfloat16, model_source=None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Over-budget head read tensor data")
    monkeypatch.setattr(PreadSafetensorsReader, "read_tensor", forbidden)
    with pytest.raises(ValueError, match="budget"):
        NativeMTPHead(engine, tmp_path/"head.safetensors", budget_bytes=1)


@torch.no_grad()
def test_native_adapter_cpu_cache_continuation_matches_plain(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
    from safetensors.torch import save_file
    from test_model_source import _write_tiny_hf_tokenizer
    import aircolibri_cli.unified_runtime as runtime
    torch.set_num_threads(6)
    torch.manual_seed(843)
    cfg = Qwen3_5TextConfig(vocab_size=259, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"]*2,
        rope_parameters={"rope_type":"default", "partial_rotary_factor":1., "mrope_section":[4,6,6]})
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    weights = {"mtp.layers.0."+k: v.contiguous() for k, v in model.model.layers[1].state_dict().items()}
    weights.update({"mtp.fc.weight": torch.randn(128, 256, dtype=torch.bfloat16)*.02})
    for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        weights[f"mtp.{name}.weight"] = torch.zeros(128, dtype=torch.bfloat16)
    head = tmp_path/"head.safetensors"
    save_file(weights, str(head))
    del model, weights
    monkeypatch.setattr(runtime, "_streaming_ram_budget_bytes", lambda: 48*1024)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    reference = None
    for mode in ("off", "native"):
        with AirColibriUnifiedEngine(tmp_path, device="cpu", weight_format="q6", speculation=mode,
                                      mtp_head_path=head if mode == "native" else None, verbose=False) as u:
            first, state = collect(u, [3, 7, 4])
            events = list(u._stream_token_ids(state["all_ids"]+[11], 4, False,
                                              state["consumed_tokens"], state["past"]))
            second = [t for e in events if e["kind"] == "tokens" for t in e["new_ids"]]
            if reference is None:
                reference = (first, second)
            else:
                assert (first, second) == reference
                assert events[-1]["spec_gate"]["source"] == "native_mtp"
                assert events[-1]["spec_gate"]["native_head_rows"] == 5
                assert u.speculator.stats()["native_batches"] > 0
                assert u.speculator.native.committed == events[-1]["consumed_tokens"]
                assert u.engine.progressive.worker is None
                assert u.engine.weight_manager.pinned_pool.total_pinned_bytes == 0


@torch.no_grad()
def test_unvalidated_decoder_uses_plain_stream_without_excluding_model(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel
    from test_model_source import _write_tiny_hf_tokenizer
    GPT2LMHeadModel(GPT2Config(vocab_size=259, n_embd=64, n_head=4, n_layer=2,
                              n_positions=64)).save_pretrained(tmp_path)
    _write_tiny_hf_tokenizer(tmp_path, vocab_size=259)
    reference = None
    for mode in ("off", "lookup"):
        with AirColibriUnifiedEngine(tmp_path, device="cpu", speculation=mode, verbose=False) as u:
            actual, state = collect(u, [3,7,3,7,3,7])
            if reference is None:
                reference = actual
            else:
                assert actual == reference
                assert not u.speculator.stats()["batching_supported"]
                assert state["spec_gate"]["decoder_contract_plain_steps"] > 0
