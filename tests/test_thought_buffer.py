"""CPU-only unit tests for `ColibriThoughtBuffer` and `InitialProbabilityAnchor`."""
from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from aircolibri_cli.thought_buffer import (
    ColibriThoughtBuffer,
    InitialProbabilityAnchor,
    blend_recurrent_anchor,
    capture_cache_snapshot,
    restore_cache_snapshot,
)


class _FakeKVLayer:
    def __init__(self, seq_len: int):
        self.keys = torch.randn(1, 2, seq_len, 4)
        self.values = torch.randn(1, 2, seq_len, 4)

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            new_len = self.keys.shape[-2] + max_length
        else:
            new_len = max_length
        self.keys = self.keys[..., :new_len, :]
        self.values = self.values[..., :new_len, :]


class _FakeDeltaNetLayer:
    def __init__(self):
        self.keys = None
        self.is_conv_states_initialized = {0: True}
        self.conv_states = [torch.randn(1, 8, 4)]
        self.recurrent_states = [torch.randn(1, 2, 8, 8)]


class _FakeHybridCache:
    def __init__(self, seq_len: int = 10):
        self.layers = [_FakeKVLayer(seq_len), _FakeDeltaNetLayer()]


def test_snapshot_and_restore_hybrid_cache():
    cache = _FakeHybridCache(seq_len=12)
    orig_keys = cache.layers[0].keys.clone()
    orig_conv = cache.layers[1].conv_states[0].clone()
    orig_rec = cache.layers[1].recurrent_states[0].clone()

    snap = capture_cache_snapshot(cache)
    assert snap.kv_lengths == {0: 12}
    assert snap.vram_bytes > 0

    # Mutate KV (append 5 draft/wave tokens) and overwrite recurrent state
    cache.layers[0].keys = torch.cat([cache.layers[0].keys, torch.randn(1, 2, 5, 4)], dim=-2)
    cache.layers[0].values = torch.cat([cache.layers[0].values, torch.randn(1, 2, 5, 4)], dim=-2)
    cache.layers[1].conv_states[0].zero_()
    cache.layers[1].recurrent_states[0].fill_(99.0)

    restore_cache_snapshot(cache, snap)
    assert cache.layers[0].keys.shape[-2] == 12
    assert torch.equal(cache.layers[0].keys, orig_keys)
    assert torch.equal(cache.layers[1].conv_states[0], orig_conv)
    assert torch.equal(cache.layers[1].recurrent_states[0], orig_rec)


def test_unknown_state_rejected_before_probe():
    import pytest
    with pytest.raises(NotImplementedError, match="state format"):
        capture_cache_snapshot([torch.ones(2)])


def test_conv_only_snapshot_and_no_false_reanchor_counter():
    cache = _FakeHybridCache()
    cache.layers[1].recurrent_states = [None]
    snap = capture_cache_snapshot(cache)
    assert not snap.rec_states
    cache.layers[1].conv_states[0].zero_()
    restore_cache_snapshot(cache, snap)
    assert torch.equal(cache.layers[1].conv_states[0], snap.conv_states[1])
    anchor = InitialProbabilityAnchor(rec_reanchor_every=1)
    anchor.snap0 = snap
    anchor.on_step_committed(cache, 10)
    assert anchor.telemetry.recurrent_reanchors == 0


def test_blend_recurrent_anchor_preserves_s0_channel():
    cache = _FakeHybridCache(seq_len=8)
    s0 = cache.layers[1].recurrent_states[0].clone()
    snap = capture_cache_snapshot(cache)

    # Simulate total erasure of S_0 in recurrent state
    cache.layers[1].recurrent_states[0].zero_()
    blend_recurrent_anchor(cache, snap, lam=0.125)
    assert torch.allclose(cache.layers[1].recurrent_states[0], 0.125 * s0, atol=1e-6)


def test_syntactic_support_mask_prevents_ungrammatical_intrusion():
    anchor = InitialProbabilityAnchor(
        alpha=0.5,
        syntax_margin=4.0,
        entropy_low=0.1,
        entropy_high=1.0,
    )
    vocab = 16
    phi = torch.zeros(vocab, dtype=torch.bfloat16)
    # Token 7 ("Во-первых") has maximal initial probability boost, but is syntactically invalid here
    phi[7] = 4.0
    # Token 3 is syntactically valid and supported by initial task constraints
    phi[3] = 3.0
    anchor.phi0 = phi.clone()
    anchor.phi_buf = phi.clone()

    logits = torch.full((1, 1, vocab), -15.0, dtype=torch.float32)
    logits[0, 0, 2] = 5.10  # slight local favorite
    logits[0, 0, 3] = 5.00  # valid syntactic alternative (within syntax_margin=4.0)
    logits[0, 0, 4] = 4.60  # raises step entropy into semantic fork regime
    logits[0, 0, 7] = -12.0  # outside syntax_margin -> MUST NOT receive boost

    out = anchor.transform_logits(logits, past_key_values=None, step_index=5)
    assert out.shape == (1, 1, vocab)
    assert float(out[0, 0, 7].item()) == -12.0
    assert int(out[0, 0].argmax().item()) == 3
    assert anchor.telemetry.argmax_corrections == 1
    assert anchor.telemetry.fork_rows_gated == 1


def test_entropy_gate_zero_on_deterministic_steps():
    anchor = InitialProbabilityAnchor(alpha=0.6, entropy_low=0.25, entropy_high=1.8)
    anchor.phi0 = torch.full((12,), 4.0, dtype=torch.bfloat16)
    anchor.phi_buf = anchor.phi0.clone()

    # Deterministic step (entropy ~ 0): anchor must not interfere at all
    logits = torch.zeros((1, 1, 12), dtype=torch.float32)
    logits[0, 0, 1] = 30.0
    out = anchor.transform_logits(logits, past_key_values=None, step_index=2)
    assert anchor.telemetry.fork_rows_gated == 0
    assert anchor.telemetry.argmax_corrections == 0
    assert torch.allclose(out, logits)


def test_mtp_speculative_rows_and_committed_reanchor():
    cache = _FakeHybridCache(seq_len=6)
    anchor = InitialProbabilityAnchor(alpha=0.4, rec_anchor_lambda=0.1, rec_reanchor_every=4)
    anchor.snap0 = capture_cache_snapshot(cache)
    anchor.phi0 = torch.linspace(-1.0, 2.0, steps=20, dtype=torch.bfloat16)
    anchor.phi_buf = anchor.phi0.clone()

    # 3 MTP speculative verification rows in bfloat16
    logits_mtp = torch.randn(1, 3, 20, dtype=torch.bfloat16)
    out = anchor.transform_logits(logits_mtp, past_key_values=cache, step_index=10)
    assert out.shape == (1, 3, 20)
    assert out.dtype == torch.bfloat16
    assert anchor.telemetry.total_rows_scored == 3

    cache.layers[1].recurrent_states[0].zero_()
    anchor.on_step_committed(cache, committed_tokens=2)
    assert anchor.telemetry.recurrent_reanchors == 0
    anchor.on_step_committed(cache, committed_tokens=2)
    assert anchor.telemetry.recurrent_reanchors == 1
    assert cache.layers[1].recurrent_states[0].abs().sum().item() > 0.0


def test_layer_conclusion_never_erases_positive_phi0():
    anchor = InitialProbabilityAnchor()
    anchor.phi0 = torch.tensor([3.2, 0.1, -0.5, 1.6], dtype=torch.bfloat16)
    anchor.phi_buf = anchor.phi0.clone()

    # Intermediate wave strongly dislikes token 0, likes token 1
    step_logits = torch.tensor([[[ -5.0, 6.0, 0.0, -4.0 ]]], dtype=torch.float32)
    lp_neutral = torch.log_softmax(torch.zeros(4, dtype=torch.float32), dim=-1)
    anchor.layer_conclusion_into_buffer(step_logits, lp_neutral, weight=0.5)

    buf = anchor.phi_buf.float()
    # Token 0 had phi0 = 3.2 > 0.5 -> must retain at least 0.75 * 3.2 = 2.4!
    assert buf[0].item() >= 0.74 * 3.2
    # Token 1 gained positive support from the verified wave conclusion
    assert buf[1].item() > 0.5


def test_ask_anchored_parses_stream_chat_events():
    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3] if text.strip() else []

    class _FakeUnifiedEngine:
        tokenizer = _FakeTokenizer()
        _active_thought_anchor = None

        def stream_chat(self, messages, **kwargs):
            anchor = self._active_thought_anchor
            assert anchor is not None
            anchor.phi0 = torch.tensor([0.0, 1.5, 2.0, 2.5, 0.5], dtype=torch.bfloat16)
            yield {"type": "delta", "phase": "thinking", "text": "step 1 "}
            yield {"type": "delta", "phase": "answer", "text": "set_wifi(ON)"}
            yield {
                "type": "done",
                "thinking_text": "step 1 ",
                "answer_text": "set_wifi(ON)",
                "metrics": {"decode_tok_per_s": 42.5, "generated_tokens": 12},
            }

    eng = _FakeUnifiedEngine()
    tb = ColibriThoughtBuffer(eng)
    res = tb.ask_anchored([{"role": "user", "content": "plan"}], enable_thinking=True)
    assert res["answer"] == "set_wifi(ON)"
    assert res["thinking"].strip() == "step 1"
    assert res["metrics"]["decode_tok_per_s"] == 42.5
    assert res["phi0_alignment"] > 1.5
    assert eng._active_thought_anchor is None


def test_original_channel_participates_even_when_working_field_is_empty():
    anchor = InitialProbabilityAnchor(alpha=0.5, initial_weight=0.75)
    anchor.phi0 = torch.tensor([0.0, 3.0, -1.0, 0.2])
    original = anchor.phi0.clone()
    anchor.phi_buf = torch.zeros(4)
    raw = torch.zeros(1, 4)
    for step in (1, 40, 1000):
        actual = anchor.transform_logits(raw, None, step)
        assert actual.argmax(-1).item() == 1
        assert not torch.equal(actual.softmax(-1), raw.softmax(-1))
        assert torch.equal(anchor.phi0, original)
    assert anchor.telemetry.initial_rows_active == 3
    assert anchor.telemetry.initial_logit_l1 > 0
    with pytest.raises(ValueError):
        InitialProbabilityAnchor(initial_weight=0)


def test_failed_mid_probe_restores_borrowed_cache_and_capture_metadata():
    cache = _FakeHybridCache(seq_len=12)
    original = capture_cache_snapshot(cache)
    keys = cache.layers[0].keys.clone()
    sentinel = torch.ones(2)
    engine = SimpleNamespace(_prefetch_next_step=False)
    u = SimpleNamespace(engine=engine, fast_qwen=False, captured={'post_norm':sentinel},
                        _last_prompt_opened_think=True, _last_prompt_chatml_think=True)
    u._neutral_logprob_cache = (torch.zeros(4),torch.zeros(4),[1,2])
    def fail(*args, **kwargs):
        cache.layers[0].keys = torch.cat([keys,keys[...,:2,:]],-2)
        cache.layers[0].values = torch.cat([cache.layers[0].values,keys[...,:2,:]],-2)
        cache.layers[1].recurrent_states[0].zero_()
        engine._prefetch_next_step = True
        u.captured['post_norm'] = torch.zeros(2)
        u._last_prompt_opened_think = False
        raise RuntimeError('mid-probe failed after mutation')
    engine._forward_step = fail
    anchor = InitialProbabilityAnchor()
    with pytest.raises(RuntimeError,match='mid-probe failed'):
        anchor.on_prefill_end(u,cache,torch.zeros(1,4))
    assert torch.equal(cache.layers[0].keys,keys)
    assert cache.layers[0].values.shape[-2] == 12
    assert torch.equal(cache.layers[1].recurrent_states[0],original.rec_states[1])
    assert u.captured['post_norm'] is sentinel
    assert u._last_prompt_opened_think and u._last_prompt_chatml_think
    assert engine._prefetch_next_step is False
    assert anchor.snap0 is None and anchor.phi0 is None
