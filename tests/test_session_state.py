"""Parking must preserve continuation, including recurrent and sliding caches."""
import pytest
import torch
from transformers import AutoModelForCausalLM, Gemma2Config, Lfm2Config, MambaConfig, Mamba2Config
from aircolibri_cli.unified_runtime import ColibriKVPrefixSession

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tiny_config(family):
    common = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2)
    if family == "mamba":
        return MambaConfig(**common, state_size=8, expand=2, time_step_rank=4)
    if family == "mamba2":
        return Mamba2Config(**common, state_size=8, expand=2, num_heads=4,
                            head_dim=16, n_groups=2, chunk_size=4)
    if family == "lfm2":
        return Lfm2Config(**common, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, full_attn_idxs=[1], block_multiple_of=16)
    return Gemma2Config(**common, intermediate_size=64, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, sliding_window=4)


@pytest.mark.parametrize("family", ["mamba", "mamba2", "lfm2", "gemma2"])
@torch.no_grad()
def test_parked_continuation_matches_uninterrupted(family):
    torch.manual_seed(192)
    config = tiny_config(family)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).to("cuda").eval()
    ids = torch.tensor([[3, 7, 4, 9, 12, 14, 8]], device="cuda")
    cache_name = "cache_params" if family.startswith("mamba") else "past_key_values"
    past = getattr(model(input_ids=ids, use_cache=True), cache_name)
    session = ColibriKVPrefixSession()
    session.record_turn_end(config, ids[0].tolist() + [11], past, None, None, None, {},
                            consumed_tokens=ids.shape[1])
    assert session.fed == ids[0].tolist() and session.unfed_tail == [11]
    assert session.parked_bytes > 0 and not session.tainted
    assert session.kv_prefix_reuse(ids[0].tolist() + [11, 15]) == ids.shape[1]
    # Real cache stays on GPU for the independent uninterrupted branch.
    restored, _ = session.unpark_to_gpu(torch.device("cuda"), None, 11)
    for position, token in enumerate([11, 15, 5], start=ids.shape[1]):
        next_id = torch.tensor([[token]], device="cuda")
        uninterrupted = model(input_ids=next_id, **{cache_name: past}, use_cache=True,
                              cache_position=torch.tensor([position], device="cuda"))
        resumed = model(input_ids=next_id, **{cache_name: restored}, use_cache=True,
                        cache_position=torch.tensor([position], device="cuda"))
        torch.testing.assert_close(resumed.logits, uninterrupted.logits, atol=0, rtol=0)
        past, restored = getattr(uninterrupted, cache_name), getattr(resumed, cache_name)
    session.reset()
    assert session.parked_backbone is None and session.parked_bytes == 0


@pytest.mark.parametrize("family", ["mamba", "mamba2", "lfm2", "gemma2"])
@torch.no_grad()
def test_probe_rollback_preserves_real_family_continuation(family):
    import copy
    from aircolibri_cli.thought_buffer import capture_cache_snapshot, restore_cache_snapshot

    torch.manual_seed(193)
    config = tiny_config(family)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).to("cuda").eval()
    ids = torch.tensor([[3, 7, 4, 9, 12, 14, 8]], device="cuda")
    name = "cache_params" if family.startswith("mamba") else "past_key_values"
    past = getattr(model(input_ids=ids, use_cache=True), name)
    independent = copy.deepcopy(past)
    snapshot = capture_cache_snapshot(past)
    # Past the sliding window; repeat restoration to expose snapshot aliasing.
    for _ in range(2):
        for pos, token in enumerate([13, 16, 6], start=7):
            model(input_ids=torch.tensor([[token]], device="cuda"), **{name: past},
                  use_cache=True, cache_position=torch.tensor([pos], device="cuda"))
        restore_cache_snapshot(past, snapshot)
    for pos, token in enumerate([11, 15, 5], start=7):
        kwargs = dict(input_ids=torch.tensor([[token]], device="cuda"), use_cache=True,
                      cache_position=torch.tensor([pos], device="cuda"))
        actual = model(**kwargs, **{name: past})
        expected = model(**kwargs, **{name: independent})
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        past, independent = getattr(actual, name), getattr(expected, name)
