"""Эталонный benchmark (540 позиций / >=300 уверенных) согласия top-1 и логитов W4A16 (g64_asym) vs BF16.

Сравнивает:
1. BF16 baseline (teacher-forced greedy траектории на 15 промптах одинаковой длины x 36 шагов = 540 позиций)
2. Q4 (lm_head=bf16) — конфигурация 0c по умолчанию
3. Q4 (lm_head=Q4)   — альтернативная квантованная голова

Метрики по D18 / D23 / Z7:
- flips при BF16 margin >= 1.0 со знаменателем (число уверенных позиций) и по нормированному margin >= 0.05
- median и max (margin и margin/span) на несовпадениях
- rank_1 (top-1 %), MRR (общий и на уверенных), top-3 %, top-5 (mean и median, порог >= 4.7 / med 5.0)
- Оба направления KL: KL(BF16||Q4) и KL(Q4||BF16) (mean, p99, max)
- Cosine similarity и centered cosine (mean, p01, min) + разбор худших позиций
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import tempfile
import time
from typing import Any, Dict, List

import psutil
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.aircolibri_integrated import AirColibriModel  # noqa: E402

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"

# Все 15 разнодоменных промптов имеют ровно 10 токенов без padding'а
PROMPTS: List[str] = [
    "In Western Europe, the capital city of France is",
    "In our solar system, the largest planet is named",
    "Explain how photosynthesis works inside green plants:",
    "According to physics, Newton's second law states that",
    "def square_sum(a, b): return a +",
    "In SQL queries, INNER JOIN differs from LEFT JOIN",
    "Compute the arithmetic result: twenty seven times fourteen =",
    "If all mammals are warm blooded creatures then whales",
    "Официальной столицей современной Японии давно уже считается",
    "Быстрая сортировка массива работает за время",
    "The historic Apollo space mission safely landed on the lunar",
    "William Shakespeare wrote many famous tragedies including Hamlet and",
    "Return a valid JSON object containing name and age:",
    "Question: How can anyone reverse a Python string?",
    "In CUDA programming, pinned host memory speeds up transfers",
]


def _free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(torch.quantile(torch.tensor(vals, dtype=torch.float64), q).item())


def _release_engine(engine: AirColibriModel | None) -> None:
    if engine is not None:
        engine.close()
        del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check_paris_sanity(engine: AirColibriModel, label: str) -> None:
    out_text, tokens, _ = engine.generate(
        prompt="The capital of France is",
        max_new_tokens=2,
        temperature=0.0,
    )
    if tokens[-2] != 11751 or tokens[-1] != 13:
        raise AssertionError(
            f"[{label}] Paris sanity check failed: tokens[-2:]={tokens[-2:]}, out='{out_text}'"
        )
    print(f"[{label}] Paris sanity PASS (B=1): tokens[-2:]={tokens[-2:]} ({out_text!r})", flush=True)


def _encode_uniform_batch(tokenizer: Any, prompts: List[str]) -> torch.Tensor:
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    lens = [len(ids) for ids in encoded]
    if len(set(lens)) != 1:
        raise ValueError(f"Expected uniform prompt token lengths without padding, got {lens}")
    return torch.tensor(encoded, dtype=torch.long)


def run_bf16_reference(
    prompts: List[str],
    steps_per_prompt: int,
    bf16_ram_layers: int,
    ref_path: str,
) -> Dict[str, Any]:
    torch.manual_seed(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    free_before = _free_ram_gb()
    t_start = time.perf_counter()
    print(
        f"\n=== [1/3] BF16 Baseline | ram_layers={bf16_ram_layers} | free_ram={free_before:.2f} GiB ===",
        flush=True,
    )

    engine = AirColibriModel(
        model_dir=MODEL_DIR,
        vram_resident_layers=4,
        max_ram_layers=bf16_ram_layers,
        dtype=torch.bfloat16,
    )
    try:
        _check_paris_sanity(engine, "BF16")
        batch_prompt_ids = _encode_uniform_batch(engine.tokenizer, prompts)
        gen_steps: List[torch.Tensor] = []
        ref_logits_steps: List[torch.Tensor] = []
        past_kv = None

        with torch.no_grad():
            for step in range(steps_per_prompt):
                t_s = time.perf_counter()
                step_tokens = batch_prompt_ids if step == 0 else gen_steps[-1].unsqueeze(1)
                logits_gpu, past_kv = engine._forward_step(step_tokens, past_key_values=past_kv)
                if torch.isnan(logits_gpu).any():
                    raise ValueError(f"NaN logits in BF16 at step {step}")
                logits_cpu = logits_gpu.detach().to(torch.bfloat16).cpu()
                next_toks = torch.argmax(logits_cpu.float(), dim=-1)
                gen_steps.append(next_toks)
                ref_logits_steps.append(logits_cpu)
                print(
                    f"  BF16 step {step + 1:02d}/{steps_per_prompt} (batch={len(prompts)}) "
                    f"in {time.perf_counter() - t_s:.2f}s | free_ram={_free_ram_gb():.2f} GiB",
                    flush=True,
                )

        del past_kv
        gen_tensor = torch.stack(gen_steps, dim=1)
        ref_tensor = torch.stack(ref_logits_steps, dim=1)
        torch.save(
            {
                "batch_prompt_ids": batch_prompt_ids,
                "gen_tokens": gen_tensor,
                "ref_logits": ref_tensor,
            },
            ref_path,
        )
        del gen_tensor, ref_tensor, gen_steps, ref_logits_steps

        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        rss_mb = _rss_mb()
        elapsed_s = time.perf_counter() - t_start
    finally:
        _release_engine(engine)

    return {
        "name": "BF16",
        "peak_vram_mb": peak_vram_mb,
        "rss_mb": rss_mb,
        "free_ram_before_gb": free_before,
        "free_ram_after_gb": _free_ram_gb(),
        "elapsed_s": elapsed_s,
    }


def evaluate_q4_mode(
    label: str,
    quantize_lm_head: bool,
    prompts: List[str],
    steps_per_prompt: int,
    ref_path: str,
) -> Dict[str, Any]:
    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
    batch_prompt_ids: torch.Tensor = ref_data["batch_prompt_ids"]
    bf16_gen_tokens: torch.Tensor = ref_data["gen_tokens"]
    ref_logits_all: torch.Tensor = ref_data["ref_logits"]

    torch.manual_seed(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    free_before = _free_ram_gb()
    t_start = time.perf_counter()
    print(f"\n=== [{label}] free_ram={free_before:.2f} GiB | quantize_lm_head={quantize_lm_head} ===", flush=True)

    engine = AirColibriModel(
        model_dir=MODEL_DIR,
        vram_resident_layers=4,
        max_ram_layers=28,
        dtype=torch.bfloat16,
        weight_format="q4_g64_asym",
        quantize_lm_head=quantize_lm_head,
    )
    try:
        _check_paris_sanity(engine, label)

        total_pos = 0
        confident_count = 0
        norm_confident_count = 0
        top1_matches = 0
        top3_hits = 0
        top5_overlaps: List[int] = []
        mrr_all: List[float] = []
        mrr_conf: List[float] = []
        cos_list: List[float] = []
        centered_cos_list: List[float] = []
        kl_bf_q4_list: List[float] = []
        kl_q4_bf_list: List[float] = []
        mismatches: List[Dict[str, Any]] = []
        all_records: List[Dict[str, Any]] = []
        past_kv = None

        with torch.no_grad():
            for step in range(steps_per_prompt):
                t_s = time.perf_counter()
                step_tokens = batch_prompt_ids if step == 0 else bf16_gen_tokens[:, step - 1 : step]
                q4_logits_gpu, past_kv = engine._forward_step(step_tokens, past_key_values=past_kv)
                if torch.isnan(q4_logits_gpu).any():
                    raise ValueError(f"NaN logits in {label} at step {step}")

                q4_step_cpu = q4_logits_gpu.detach().cpu().to(torch.float64)
                bf_step_cpu = ref_logits_all[:, step, :].to(torch.float64)

                for p_idx in range(len(prompts)):
                    q4_f64 = q4_step_cpu[p_idx]
                    bf_f64 = bf_step_cpu[p_idx]

                    bf_vals, bf_idx = torch.topk(bf_f64, k=5, dim=-1)
                    bf_top1 = int(bf_idx[0].item())
                    margin = float((bf_vals[0] - bf_vals[1]).item())
                    span = float((bf_f64.max() - bf_f64.min()).item())
                    norm_margin = margin / span if span > 0.0 else 0.0

                    _, q4_idx = torch.topk(q4_f64, k=5, dim=-1)
                    q4_top1 = int(q4_idx[0].item())

                    rank = int((q4_f64 > q4_f64[bf_top1]).sum().item()) + 1
                    rr = 1.0 / rank

                    cos = float(F.cosine_similarity(bf_f64.unsqueeze(0), q4_f64.unsqueeze(0), dim=-1).item())
                    bf_c = bf_f64 - bf_f64.mean()
                    q4_c = q4_f64 - q4_f64.mean()
                    cos_centered = float(F.cosine_similarity(bf_c.unsqueeze(0), q4_c.unsqueeze(0), dim=-1).item())

                    log_p = F.log_softmax(bf_f64, dim=-1)
                    p = log_p.exp()
                    log_q = F.log_softmax(q4_f64, dim=-1)
                    q = log_q.exp()
                    kl_bf_q4 = float(torch.sum(p * (log_p - log_q)).clamp(min=0.0).item())
                    kl_q4_bf = float(torch.sum(q * (log_q - log_p)).clamp(min=0.0).item())

                    total_pos += 1
                    if margin >= 1.0:
                        confident_count += 1
                        mrr_conf.append(rr)
                    if norm_margin >= 0.05:
                        norm_confident_count += 1

                    mrr_all.append(rr)
                    cos_list.append(cos)
                    centered_cos_list.append(cos_centered)
                    kl_bf_q4_list.append(kl_bf_q4)
                    kl_q4_bf_list.append(kl_q4_bf)

                    in_top3 = q4_top1 in bf_idx[:3].tolist()
                    if in_top3:
                        top3_hits += 1
                    top5_overlaps.append(len(set(bf_idx.tolist()) & set(q4_idx.tolist())))

                    rec = {
                        "prompt_idx": p_idx,
                        "step": step,
                        "margin": margin,
                        "norm_margin": norm_margin,
                        "span": span,
                        "rank": rank,
                        "cos": cos,
                        "centered_cos": cos_centered,
                        "kl_bf_q4": kl_bf_q4,
                        "kl_q4_bf": kl_q4_bf,
                        "in_top3": in_top3,
                        "bf16_id": bf_top1,
                        "q4_id": q4_top1,
                        "bf16_tok": engine.tokenizer.decode([bf_top1]),
                        "q4_tok": engine.tokenizer.decode([q4_top1]),
                    }
                    all_records.append(rec)

                    if q4_top1 == bf_top1:
                        top1_matches += 1
                    else:
                        mismatches.append(rec)

                print(
                    f"  {label} step {step + 1:02d}/{steps_per_prompt} in {time.perf_counter() - t_s:.2f}s "
                    f"| cum_top1={top1_matches}/{total_pos} ({100.0 * top1_matches / total_pos:.2f}%)",
                    flush=True,
                )

        del past_kv
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        rss_mb = _rss_mb()
        elapsed_s = time.perf_counter() - t_start
    finally:
        _release_engine(engine)
        del ref_logits_all, ref_data
        gc.collect()

    mismatch_margins = [m["margin"] for m in mismatches]
    mismatch_norm_margins = [m["norm_margin"] for m in mismatches]
    med_margin = statistics.median(mismatch_margins) if mismatch_margins else 0.0
    max_margin = max(mismatch_margins) if mismatch_margins else 0.0
    med_norm_margin = statistics.median(mismatch_norm_margins) if mismatch_norm_margins else 0.0
    max_norm_margin = max(mismatch_norm_margins) if mismatch_norm_margins else 0.0

    flips_ge_1 = sum(1 for m in mismatches if m["margin"] >= 1.0)
    flips_norm = sum(1 for m in mismatches if m["norm_margin"] >= 0.05)

    top1_pct = 100.0 * top1_matches / total_pos
    top3_pct = 100.0 * top3_hits / total_pos
    top5_mean = statistics.mean(top5_overlaps)
    top5_median = statistics.median(top5_overlaps)
    mrr_mean = statistics.mean(mrr_all)
    mrr_conf_mean = statistics.mean(mrr_conf) if mrr_conf else 0.0

    kl_fwd_mean = statistics.mean(kl_bf_q4_list)
    kl_fwd_p99 = _quantile(kl_bf_q4_list, 0.99)
    kl_fwd_max = max(kl_bf_q4_list)

    kl_rev_mean = statistics.mean(kl_q4_bf_list)
    kl_rev_p99 = _quantile(kl_q4_bf_list, 0.99)
    kl_rev_max = max(kl_q4_bf_list)

    cos_mean = statistics.mean(cos_list)
    cos_p01 = _quantile(cos_list, 0.01)
    cos_min = min(cos_list)
    ccos_mean = statistics.mean(centered_cos_list)
    ccos_p01 = _quantile(centered_cos_list, 0.01)
    ccos_min = min(centered_cos_list)

    worst_cos = sorted(all_records, key=lambda r: r["cos"])[:5]
    top3_misses = [r for r in all_records if not r["in_top3"]]

    # Пороги D18 + D23 + Z7
    gate_pass = (
        confident_count >= 300
        and flips_ge_1 == 0
        and math.isclose(top3_pct, 100.0, abs_tol=1e-9)
        and (top5_mean >= 4.7 or top5_median >= 5.0)
        and kl_fwd_mean <= 0.03
        and cos_mean >= 0.99
    )

    return {
        "name": label,
        "quantize_lm_head": quantize_lm_head,
        "positions": total_pos,
        "confident_count": confident_count,
        "norm_confident_count": norm_confident_count,
        "top1_matches": top1_matches,
        "mismatches": mismatches,
        "top1_pct": top1_pct,
        "mrr_mean": mrr_mean,
        "mrr_conf_mean": mrr_conf_mean,
        "mismatch_median_margin": med_margin,
        "mismatch_max_margin": max_margin,
        "mismatch_median_norm_margin": med_norm_margin,
        "mismatch_max_norm_margin": max_norm_margin,
        "flips_margin_ge_1": flips_ge_1,
        "flips_norm_ge_005": flips_norm,
        "top3_pct": top3_pct,
        "top3_misses": top3_misses,
        "top5_overlap_mean": top5_mean,
        "top5_overlap_median": top5_median,
        "kl_fwd_mean": kl_fwd_mean,
        "kl_fwd_p99": kl_fwd_p99,
        "kl_fwd_max": kl_fwd_max,
        "kl_rev_mean": kl_rev_mean,
        "kl_rev_p99": kl_rev_p99,
        "kl_rev_max": kl_rev_max,
        "cos_mean": cos_mean,
        "cos_p01": cos_p01,
        "cos_min": cos_min,
        "ccos_mean": ccos_mean,
        "ccos_p01": ccos_p01,
        "ccos_min": ccos_min,
        "worst_cos": worst_cos,
        "gate_pass": gate_pass,
        "peak_vram_mb": peak_vram_mb,
        "rss_mb": rss_mb,
        "free_ram_before_gb": free_before,
        "free_ram_after_gb": _free_ram_gb(),
        "elapsed_s": elapsed_s,
    }


def _print_report(bf16_res: Dict[str, Any], q4_runs: List[Dict[str, Any]], num_prompts: int, steps: int) -> None:
    print("\n" + "=" * 100)
    print(f"Q4 vs BF16 TOP-1 & LOGIT AGREEMENT REPORT ({num_prompts} prompts x {steps} steps = {num_prompts * steps} positions)")
    print("=" * 100)
    if bf16_res is not None:
        print(
            f"BF16 Baseline      | PeakVRAM={bf16_res['peak_vram_mb']:.1f} MB | RSS={bf16_res['rss_mb']:.1f} MB | "
            f"free_ram: {bf16_res['free_ram_before_gb']:.2f} -> {bf16_res['free_ram_after_gb']:.2f} GiB | "
            f"time={bf16_res['elapsed_s']:.1f}s"
        )
    for res in q4_runs:
        n_mis = len(res["mismatches"])
        print("-" * 100)
        print(
            f"{res['name']:18s} | rank_1={res['top1_matches']}/{res['positions']} ({res['top1_pct']:.2f}%) "
            f"| MRR={res['mrr_mean']:.4f} (conf={res['mrr_conf_mean']:.4f})"
        )
        print(
            f"{'':18s} | flips(margin>=1.0)={res['flips_margin_ge_1']}/{res['confident_count']} "
            f"| flips(norm>=0.05)={res['flips_norm_ge_005']}/{res['norm_confident_count']} "
            f"| mismatches={n_mis}"
        )
        print(
            f"{'':18s} | margin@mis med={res['mismatch_median_margin']:.4f} max={res['mismatch_max_margin']:.4f} "
            f"| norm@mis med={res['mismatch_median_norm_margin']:.4f} max={res['mismatch_max_norm_margin']:.4f}"
        )
        print(
            f"{'':18s} | top-3={res['top3_pct']:.2f}% "
            f"| top-5 mean={res['top5_overlap_mean']:.3f}/5 med={res['top5_overlap_median']:.1f}/5"
        )
        print(
            f"{'':18s} | KL(BF||Q4) mean={res['kl_fwd_mean']:.5f} p99={res['kl_fwd_p99']:.5f} max={res['kl_fwd_max']:.5f} "
            f"| KL(Q4||BF) mean={res['kl_rev_mean']:.5f} p99={res['kl_rev_p99']:.5f} max={res['kl_rev_max']:.5f}"
        )
        print(
            f"{'':18s} | cos mean={res['cos_mean']:.6f} p01={res['cos_p01']:.6f} min={res['cos_min']:.6f} "
            f"| centered mean={res['ccos_mean']:.6f} p01={res['ccos_p01']:.6f} min={res['ccos_min']:.6f}"
        )
        print(
            f"{'':18s} | Gate(D23/Z7)={'PASS' if res['gate_pass'] else 'FAIL'} "
            f"| PeakVRAM={res['peak_vram_mb']:.1f} MB | RSS={res['rss_mb']:.1f} MB "
            f"| free_ram: {res['free_ram_before_gb']:.2f}->{res['free_ram_after_gb']:.2f} GiB | time={res['elapsed_s']:.1f}s"
        )
        if n_mis > 0:
            top_mis = sorted(res["mismatches"], key=lambda m: m["margin"], reverse=True)[:8]
            print("  Top mismatches by BF16 top1-top2 margin:")
            for m in top_mis:
                print(
                    f"    p={m['prompt_idx']:02d} step={m['step']:02d} margin={m['margin']:.4f} "
                    f"(norm={m['norm_margin']:.4f}) rank={m['rank']} "
                    f"bf16={m['bf16_id']}({ascii(m['bf16_tok'])}) vs q4={m['q4_id']}({ascii(m['q4_tok'])})"
                )
        if res["top3_misses"]:
            print(f"  top-3 misses ({len(res['top3_misses'])}):")
            for r in res["top3_misses"][:10]:
                print(
                    f"    p={r['prompt_idx']:02d} step={r['step']:02d} rank={r['rank']} margin={r['margin']:.4f} "
                    f"cos={r['cos']:.4f} kl_fwd={r['kl_bf_q4']:.4f} "
                    f"bf16={r['bf16_id']}({ascii(r['bf16_tok'])}) vs q4={r['q4_id']}({ascii(r['q4_tok'])})"
                )
        print("  Worst 5 positions by cosine:")
        for r in res["worst_cos"]:
            print(
                f"    p={r['prompt_idx']:02d} step={r['step']:02d} cos={r['cos']:.5f} ccos={r['centered_cos']:.5f} "
                f"span={r['span']:.2f} kl_fwd={r['kl_bf_q4']:.5f} kl_rev={r['kl_q4_bf']:.5f} rank={r['rank']}"
            )
    print("=" * 100)
    q4_lm = next((r for r in q4_runs if r["quantize_lm_head"]), None)
    q4_bf16 = next((r for r in q4_runs if not r["quantize_lm_head"]), None)
    if q4_bf16 and q4_lm:
        if q4_lm["gate_pass"] and q4_bf16["gate_pass"]:
            print("VERDICT: Both Q4 (lm_head=bf16) and Q4 (lm_head=Q4) PASS.")
        else:
            print(
                f"VERDICT: Keep lm_head=bf16 default (gate={'PASS' if q4_bf16['gate_pass'] else 'FAIL'}, "
                f"flips>=1.0={q4_bf16['flips_margin_ge_1']}/{q4_bf16['confident_count']}, "
                f"max_margin={q4_bf16['mismatch_max_margin']:.4f}); "
                f"lm_head=Q4 gate={'PASS' if q4_lm['gate_pass'] else 'FAIL'} "
                f"(flips>=1.0={q4_lm['flips_margin_ge_1']}/{q4_lm['confident_count']}, "
                f"max_margin={q4_lm['mismatch_max_margin']:.4f})."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark W4A16 (g64_asym) vs BF16 over >= 540 greedy positions.")
    parser.add_argument("--prompts", type=int, default=15, help="Number of fixed prompts (default: 15)")
    parser.add_argument("--steps", type=int, default=36, help="Greedy steps per prompt (default: 36 -> 540 total, >=300 confident)")
    parser.add_argument("--bf16-ram-layers", type=int, default=4, help="RAM cache layer limit for BF16 baseline")
    parser.add_argument("--save-ref", type=str, default="", help="Optional path to save BF16 reference logits")
    parser.add_argument("--load-ref", type=str, default="", help="Optional path to reuse saved BF16 reference logits")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for bench_q4_top1.py")

    selected_prompts = PROMPTS[: args.prompts]
    if len(selected_prompts) < args.prompts:
        raise ValueError(f"Only {len(PROMPTS)} built-in prompts available, requested {args.prompts}")

    if args.load_ref:
        bf16_res = None
        ref_path = args.load_ref
        res_bf16_head = evaluate_q4_mode("Q4(lm_head=bf16)", False, selected_prompts, args.steps, ref_path)
        res_q4_head = evaluate_q4_mode("Q4(lm_head=Q4)", True, selected_prompts, args.steps, ref_path)
    elif args.save_ref:
        ref_path = args.save_ref
        bf16_res = run_bf16_reference(selected_prompts, args.steps, args.bf16_ram_layers, ref_path)
        res_bf16_head = evaluate_q4_mode("Q4(lm_head=bf16)", False, selected_prompts, args.steps, ref_path)
        res_q4_head = evaluate_q4_mode("Q4(lm_head=Q4)", True, selected_prompts, args.steps, ref_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "bf16_ref.pt")
            bf16_res = run_bf16_reference(selected_prompts, args.steps, args.bf16_ram_layers, ref_path)
            res_bf16_head = evaluate_q4_mode("Q4(lm_head=bf16)", False, selected_prompts, args.steps, ref_path)
            res_q4_head = evaluate_q4_mode("Q4(lm_head=Q4)", True, selected_prompts, args.steps, ref_path)

    _print_report(bf16_res, [res_bf16_head, res_q4_head], len(selected_prompts), args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
