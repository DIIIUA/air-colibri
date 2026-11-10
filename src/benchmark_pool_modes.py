"""Benchmark Suite comparing AirLLM baseline vs AirColibri Reusable Slot Pools.

Evaluates on RTX 3060 Ti:
1. Baseline AirLLM: Dynamic allocation + torch.cuda.empty_cache() per layer
2. Single Slot Pool: 1 persistent pre-allocated slot (zero malloc/free)
3. Double Buffer Pool: 2 persistent slots + dual CUDA streams & events
4. Ring Pool with Colibri LFRU Cache: Dense ring + MoE resident expert cache
"""

import json
import os
import sys
import time

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import psutil
from src.hybrid_layer_runner import HybridLayerRunner


def run_comprehensive_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("AirColibri Memory Architecture Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"Total VRAM: {vram_total_mb:.1f} MB")
    print(f"Total System RAM: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.1f} GB")
    print("=" * 70)

    # Configuration for realistic layer workload
    NUM_LAYERS = 16
    HIDDEN_DIM = 2048
    INTER_DIM = 4096
    NUM_EXPERTS = 16
    TOP_K = 2
    NUM_TOKENS = 10
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    runner = HybridLayerRunner(
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        dtype=DTYPE,
        device=device
    )

    results = []

    # 1. Baseline Naive AirLLM
    print("\n[1/4] Running Baseline AirLLM (Dynamic malloc + empty_cache)...")
    res_baseline = runner.run_naive_airllm(batch_size=1, seq_len=1, num_tokens=NUM_TOKENS)
    print(f"  -> Avg Step: {res_baseline['avg_step_ms']:.2f} ms | Tok/s: {res_baseline['tok_per_sec']:.2f} | Peak VRAM: {res_baseline['peak_vram_mb']:.1f} MB")
    results.append(res_baseline)

    # 2. Single Slot Pool
    print("\n[2/4] Running Single Slot Pool (1 Persistent Slot, Zero Malloc)...")
    res_single = runner.run_single_slot(batch_size=1, seq_len=1, num_tokens=NUM_TOKENS)
    print(f"  -> Avg Step: {res_single['avg_step_ms']:.2f} ms | Tok/s: {res_single['tok_per_sec']:.2f} | Peak VRAM: {res_single['peak_vram_mb']:.1f} MB")
    results.append(res_single)

    # 3. Double Buffer Pool
    print("\n[3/4] Running Double Buffer Pool (2 Slots + Dual CUDA Streams/Events)...")
    res_double = runner.run_double_buffer(batch_size=1, seq_len=1, num_tokens=NUM_TOKENS)
    print(f"  -> Avg Step: {res_double['avg_step_ms']:.2f} ms | Tok/s: {res_double['tok_per_sec']:.2f} | Peak VRAM: {res_double['peak_vram_mb']:.1f} MB")
    results.append(res_double)

    # 4. Ring Pool with Colibri LFRU Cache
    print("\n[4/4] Running Ring Pool + Colibri LFRU Cache (Dense Ring + MoE Cache)...")
    res_ring = runner.run_ring_pool_with_cache(batch_size=1, seq_len=1, num_tokens=NUM_TOKENS, hot_expert_bias=0.8)
    print(f"  -> Avg Step: {res_ring['avg_step_ms']:.2f} ms | Tok/s: {res_ring['tok_per_sec']:.2f} | Hit Rate: {res_ring['hit_rate_pct']:.1f}% | Transferred: {res_ring['mb_transferred']:.1f} MB")
    results.append(res_ring)

    # Calculate speedups relative to baseline
    baseline_lat = res_baseline["avg_step_ms"]
    for r in results:
        r["speedup_vs_baseline"] = baseline_lat / r["avg_step_ms"]

    # Save to data directory
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw benchmark data to {output_path}")

    # Output formatted markdown table
    print("\n" + "=" * 95)
    print(f"{'Mode':<42} | {'Step (ms)':<10} | {'Tok/s':<8} | {'Peak VRAM':<10} | {'Speedup':<8} | {'Transfer (MB)':<12}")
    print("-" * 95)
    for r in results:
        print(f"{r['mode']:<42} | {r['avg_step_ms']:<10.2f} | {r['tok_per_sec']:<8.2f} | {r['peak_vram_mb']:<10.1f} | {r['speedup_vs_baseline']:<8.2f}x | {r['mb_transferred']:<12.1f}")
    print("=" * 95)

    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()
