"""CPU checks for the VRAM planner (`normal` / `turbo`, no fixed 4gb/8gb profiles)."""
import pytest

from aircolibri_cli.unified_runtime import GiB, MiB, plan_vram, q4_bytes_for_shape

# Qwen3.5-9B-like numbers: 32 layers of ~122 MiB in Q4, LM head 572 MiB, MTP ~130 MiB.
LAYERS = [122 * MiB] * 32
COMMON = dict(
    layer_bytes=LAYERS,
    lm_head_bytes=q4_bytes_for_shape((248320, 4096)),
    mtp_bytes=130 * MiB,
    kv_bytes_per_token=8 * 2 * 4 * 256 * 2,
    recurrent_state_bytes=24 * 32 * 128 * 128 * 4,
    hidden_size=4096,
    intermediate_size=12288,
)


def test_lm_head_q4_size():
    assert q4_bytes_for_shape((248320, 4096)) == 248320 * 4096 // 2 + 2 * 248320 * 64 * 2


def test_8gb_card_keeps_everything_resident_with_a_reserve():
    p = plan_vram("normal", total_bytes=8 * GiB, free_bytes=int(6.9 * GiB), **COMMON)
    assert p.full_residency and p.enable_mtp and p.ring_slots == 0
    assert p.reserve_bytes > 1 * GiB  # OS/WDDM guard + KV for 8k context + prefill workspace
    assert p.budget_bytes + p.reserve_bytes == int(6.9 * GiB)
    assert p.prefill_chunk_tokens == 512


def test_turbo_uses_more_vram_than_normal():
    n = plan_vram("normal", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    t = plan_vram("turbo", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    assert t.reserve_bytes < n.reserve_bytes
    assert t.budget_bytes > n.budget_bytes
    assert t.resident_layers >= n.resident_layers


def test_4gb_card_streams_through_equal_ring_slots():
    p = plan_vram("normal", total_bytes=4 * GiB, free_bytes=int(3.8 * GiB), **COMMON)
    assert not p.full_residency
    assert p.ring_slots >= 3 and p.slot_bytes == max(LAYERS)
    assert 0 < p.resident_layers < 32
    assert p.weights_bytes <= p.budget_bytes
    # normal mode leaves roughly 0.3-0.8 GiB free on a 4 GB card instead of taking everything
    assert 0.3 * GiB < p.reserve_bytes < 1.0 * GiB
    assert p.prefill_chunk_tokens == 256 and p.max_context_tokens == 4096


def test_fixed_profiles_are_rejected():
    for bad in ("4gb", "8gb", "auto"):
        with pytest.raises(ValueError):
            plan_vram(bad, total_bytes=8 * GiB, free_bytes=7 * GiB, **COMMON)


def test_too_little_vram_is_a_clear_error():
    with pytest.raises(RuntimeError, match="Not enough free VRAM"):
        plan_vram("turbo", total_bytes=2 * GiB, free_bytes=int(1.0 * GiB), **COMMON)
