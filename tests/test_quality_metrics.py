import pytest
import torch

from src.quality_metrics import compare_logits, summarize


def confident_rows(count):
    return torch.tensor([[8., 3., 2., 1., 0., -1.]]).repeat(count, 1)


def test_identical_distribution_requires_300_confident_positions():
    logits = confident_rows(300)
    rows = compare_logits(logits, logits)
    assert summarize(rows[:299])["gate"] == "INSUFFICIENT_EVIDENCE"
    result = summarize(rows)
    assert result["gate"] == "PASS"
    assert result["confident_flips"] == 0 and result["kl_forward_mean"] == 0


def test_confident_flip_fails_even_with_high_average_agreement():
    ref = confident_rows(300)
    candidate = ref.clone()
    candidate[0, :2] = candidate[0, :2].flip(0)
    result = summarize(compare_logits(ref, candidate))
    assert result["gate"] == "FAIL" and result["confident_flips"] == 1


def test_same_argmax_does_not_hide_distribution_damage():
    ref = confident_rows(300)
    result = summarize(compare_logits(ref, ref / 10))
    assert result["top1_pct"] == 100
    assert result["gate"] == "FAIL" and not result["checks"]["kl_forward"]


def test_unconfident_positions_do_not_count_toward_gate():
    ref = torch.tensor([[1., .9, .8, .7, .6, .5]]).repeat(500, 1)
    result = summarize(compare_logits(ref, ref))
    assert result["confident_count"] == 0 and result["gate"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_logits_rejected(bad):
    ref = confident_rows(1)
    candidate = ref.clone()
    candidate[0, 2] = bad
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_logits(ref, candidate)


def test_shape_mismatch_and_empty_results_rejected():
    with pytest.raises(ValueError):
        compare_logits(confident_rows(2), confident_rows(1))
    with pytest.raises(ValueError):
        summarize([])


def test_equal_logits_follow_greedy_argmax_not_topk_tie_order(monkeypatch):
    original_topk = torch.Tensor.topk
    def different_valid_tie_order(tensor, k, *args, **kwargs):
        values, indices = original_topk(tensor, k, *args, **kwargs)
        if bool((values == values[0]).all()):
            return values.flip(-1), indices.flip(-1)
        return values, indices
    monkeypatch.setattr(torch.Tensor, "topk", different_valid_tie_order)
    ref = torch.tensor([[8., 1., 0., -1., -2., -3., -4., -5.]])
    candidate = torch.full_like(ref, 8.)
    row = compare_logits(ref, candidate)[0]
    assert row["candidate_id"] == candidate.argmax().item() == 0
    assert row["top1_match"] and not row["ordered_topk_top1_match"]
    assert row["candidate_top1_ties"] == 8


def test_top3_includes_equal_probability_at_cutoff():
    ref = torch.tensor([[8., 6., 5., 5., 0., -1.]])
    candidate = torch.tensor([[0., 1., 2., 8., 0., -1.]])
    assert compare_logits(ref, candidate)[0]["in_top3"]
