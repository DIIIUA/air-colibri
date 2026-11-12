"""Strict historical Q4/BF16 acceptance metrics; no model or CUDA dependency."""
import math
import statistics

import torch
import torch.nn.functional as F


THRESHOLDS = dict(min_confident=300, confidence_margin=1.0,
                  max_confident_flips=0, min_top3_pct=100.0,
                  min_top5_mean=4.7, min_top5_median=5.0,
                  max_kl_forward_mean=0.03, min_cosine_mean=0.99)


def compare_logits(reference, candidate):
    """One record per batch row. Compare every vocabulary logit in float64."""
    if reference.ndim != 2 or reference.shape != candidate.shape or reference.shape[1] < 5:
        raise ValueError("Expected equal [batch, vocabulary>=5] logits")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite quality logits")
    records = []
    for bf, q4 in zip(reference.double(), candidate.double()):
        bv, bi = bf.topk(5)
        _, qi = q4.topk(5)
        margin = (bv[0] - bv[1]).item()
        span = (bf.max() - bf.min()).item()
        logp, logq = bf.log_softmax(-1), q4.log_softmax(-1)
        records.append(dict(
            bf16_id=bi[0].item(), candidate_id=qi[0].item(),
            top1_match=bi[0].item() == qi[0].item(),
            in_top3=qi[0].item() in bi[:3].tolist(),
            top5_overlap=len(set(bi.tolist()) & set(qi.tolist())),
            rank=int((q4 > q4[bi[0]]).sum()) + 1,
            margin=margin, norm_margin=margin / span if span else 0.0,
            kl_forward=max(0.0, (logp.exp() * (logp - logq)).sum().item()),
            kl_reverse=max(0.0, (logq.exp() * (logq - logp)).sum().item()),
            cosine=F.cosine_similarity(bf, q4, dim=0).item(),
            centered_cosine=F.cosine_similarity(bf-bf.mean(), q4-q4.mean(), dim=0).item(),
        ))
    return records


def summarize(records):
    if not records:
        raise ValueError("No quality positions evaluated")
    n = len(records)
    confident = [r for r in records if r["margin"] >= THRESHOLDS["confidence_margin"]]
    flips = [r for r in confident if not r["top1_match"]]
    mean = lambda key: statistics.mean(r[key] for r in records)
    quantile = lambda key, q: torch.quantile(torch.tensor([r[key] for r in records], dtype=torch.float64), q).item()
    result = dict(positions=n, confident_count=len(confident), confident_flips=len(flips),
        top1_pct=100 * sum(r["top1_match"] for r in records) / n,
        top3_pct=100 * sum(r["in_top3"] for r in records) / n,
        top5_mean=mean("top5_overlap"),
        top5_median=statistics.median(r["top5_overlap"] for r in records),
        kl_forward_mean=mean("kl_forward"), kl_forward_p99=quantile("kl_forward", .99),
        kl_forward_max=max(r["kl_forward"] for r in records),
        kl_reverse_mean=mean("kl_reverse"), cosine_mean=mean("cosine"),
        cosine_min=min(r["cosine"] for r in records),
        centered_cosine_mean=mean("centered_cosine"),
        mrr=statistics.mean(1/r["rank"] for r in records),
        mrr_confident=statistics.mean(1/r["rank"] for r in confident) if confident else None)
    checks = dict(
        enough_confident_positions=len(confident) >= THRESHOLDS["min_confident"],
        zero_confident_flips=len(flips) == THRESHOLDS["max_confident_flips"],
        top3=math.isclose(result["top3_pct"], THRESHOLDS["min_top3_pct"], abs_tol=1e-9),
        top5=result["top5_mean"] >= THRESHOLDS["min_top5_mean"] or result["top5_median"] >= THRESHOLDS["min_top5_median"],
        kl_forward=result["kl_forward_mean"] <= THRESHOLDS["max_kl_forward_mean"],
        cosine=result["cosine_mean"] >= THRESHOLDS["min_cosine_mean"],
    )
    quality_checks = [v for k, v in checks.items() if k != "enough_confident_positions"]
    gate = "FAIL" if not all(quality_checks) else "PASS" if all(checks.values()) else "INSUFFICIENT_EVIDENCE"
    return dict(**result, thresholds=THRESHOLDS, checks=checks, gate=gate)
