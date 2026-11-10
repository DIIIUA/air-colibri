"""Comprehensive Comparative Benchmark Battery for AirColibri on DeepSeek-V4-9B (Dense).

Runs 4 configurations side-by-side on the exact same prompt:
1. Baseline Sequential (1 slot, pageable memory, synchronous)
2. Bounded Pinned Double Buffer (2 slots, pinned memory, async overlap)
3. Triple Ring Pipeline (3 slots, pinned memory, 3-phase ring)
4. Triple Ring + Tail-to-Head Prefetch (3 slots, pinned memory, tail-to-head prefetch)

Measures:
- Cold TTFT (Step 0: prompt prefill, ms)
- Warm TTFT / Step 1 (first generated token, ms)
- Decode Steps 2..10 (ms/step)
- Average Decode Latency (ms/step) & Throughput (tok/s)
- Peak VRAM Allocated & Reserved (MB)
- Process RAM Working Set (MB)
- Output Text & Token Verification
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.bounded_memory_manager import BoundedPinnedBufferPool, BoundedRAMCache, StableWeightManager
from src.gpu_ring_pipeline import GPURingPipeline, SlotState

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"
PROMPT = "The capital of France is"
NUM_NEW_TOKENS = 10


def run_single_benchmark(
    config_name: str,
    num_slots: int,
    use_pinned: bool,
    tail_prefetch: bool,
    max_ram_layers: int = 8
) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {config_name}")
    print(f"Slots: {num_slots} | Pinned DMA: {use_pinned} | Tail-to-Head Prefetch: {tail_prefetch}")
    print(f"{'='*80}")

    # Force garbage collection and CUDA cache flush before run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(MODEL_DIR, config_file="config.hf.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 1. Weight Manager
    manager = StableWeightManager(
        model_dir=MODEL_DIR,
        max_ram_layers=max_ram_layers,
        num_pinned_slots=num_slots,
        dtype=dtype
    )

    # 2. Model on meta device
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Permanent globals: norm and lm_head on GPU, embed on CPU
    norm_w = manager.globals["model.norm.weight"].to(device=device, dtype=dtype)
    set_module_tensor_to_device(model, "model.norm.weight", device, value=norm_w, clear_cache=False)

    lm_head_w = manager.globals["lm_head.weight"].to(device=device, dtype=dtype)
    set_module_tensor_to_device(model, "lm_head.weight", device, value=lm_head_w, clear_cache=False)

    embed_w_cpu = manager.globals["model.embed_tokens.weight"]

    # 3. Pipeline Setup
    template = {k: v for k, v in manager.pinned_pool.slots[0].items()}
    pipeline = GPURingPipeline(
        template_tensors=template,
        num_slots=num_slots,
        device=device,
        dtype=dtype,
        tail_to_head_prefetch=tail_prefetch
    )

    num_layers = config.num_hidden_layers
    pipeline.tail_prefetched = False

    layer_param_names = [[k for k, _ in m.named_parameters()] for m in model.model.layers]
    active_slot_per_layer = [None] * num_layers

    # 4. Install Hooks
    for idx, layer_mod in enumerate(model.model.layers):
        layer_mod._air_idx = idx

        def _pre(m, a):
            i = m._air_idx
            if num_slots == 1 and not use_pinned:
                # Baseline sequential mode: synchronous load on demand
                cached = manager.ram_cache.get(i)
                if cached is None:
                    layer_file = os.path.join(MODEL_DIR, f"model-{i:05d}.safetensors")
                    import safetensors
                    cached = {}
                    prefix = f"model.layers.{i}."
                    with safetensors.safe_open(layer_file, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            short_k = k.replace(prefix, "")
                            cached[short_k] = f.get_slice(k)[:].to(dtype=dtype)
                    manager.ram_cache.put(i, cached)
                slot = pipeline.slots[0]
                for k, t in cached.items():
                    slot.tensors[k].copy_(t, non_blocking=False)
                for k in layer_param_names[i]:
                    if k in slot.tensors:
                        set_module_tensor_to_device(m, k, device, value=slot.tensors[k], clear_cache=False)
                active_slot_per_layer[i] = slot
            else:
                # Ring pipeline mode: acquire slot
                slot = pipeline.acquire_slot_for_compute(i)
                active_slot_per_layer[i] = slot
                for k in layer_param_names[i]:
                    if k in slot.tensors:
                        set_module_tensor_to_device(m, k, device, value=slot.tensors[k], clear_cache=False)

        def _post(m, a, o):
            i = m._air_idx
            slot = active_slot_per_layer[i]
            if num_slots == 1 and not use_pinned:
                pass
            else:
                pipeline.release_slot_after_compute(slot)
                next_layer = i + num_slots
                if next_layer < num_layers:
                    pinned_t = manager.load_layer_to_pinned_slot(next_layer, target_slot_idx=next_layer % num_slots)
                    next_slot = pipeline.get_slot_for_loading(next_layer)
                    pipeline.load_tensors_async(next_slot, pinned_t)
                elif next_layer == num_layers and pipeline.tail_to_head_prefetch:
                    pinned_t = manager.load_layer_to_pinned_slot(0, target_slot_idx=0)
                    pipeline.prefetch_tail_to_head(pinned_t)
                    pipeline.tail_prefetched = True
            return o

        layer_mod.register_forward_pre_hook(_pre)
        layer_mod.register_forward_hook(_post)

    # 5. Run Generation
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = input_ids[0].tolist()
    past_key_values = None
    step_latencies = []

    process = psutil.Process()
    ram_start = process.memory_info().rss / (1024**2)

    with torch.no_grad():
        for step in range(NUM_NEW_TOKENS):
            t0 = time.perf_counter()

            if past_key_values is None:
                step_tokens = torch.tensor([generated_ids], dtype=torch.long)
            else:
                step_tokens = torch.tensor([[generated_ids[-1]]], dtype=torch.long)

            inputs_embeds = torch.nn.functional.embedding(step_tokens, embed_w_cpu).to(device=device, dtype=dtype)

            if num_slots > 1 or use_pinned:
                for s in range(min(num_slots, num_layers)):
                    if s == 0 and pipeline.tail_prefetched:
                        pipeline.tail_prefetched = False
                        continue
                    pinned_t = manager.load_layer_to_pinned_slot(s, target_slot_idx=s)
                    slot = pipeline.get_slot_for_loading(s)
                    pipeline.load_tensors_async(slot, pinned_t)

            outputs = model(
                inputs_embeds=inputs_embeds,
                input_ids=None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if torch.isnan(logits).any():
                raise ValueError(f"NaN in logits at step {step}!")

            next_tok = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_tok)

            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(step_ms)

            tok_str = tokenizer.decode([next_tok])
            print(f"  Step {step + 1:2d}/{NUM_NEW_TOKENS}: ID={next_tok:6d} ('{tok_str}') | Latency: {step_ms:7.1f} ms", flush=True)

    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    vram_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
    vram_res = torch.cuda.max_memory_reserved(device) / (1024**2)
    ram_end = process.memory_info().rss / (1024**2)

    cold_ttft = step_latencies[0]
    decode_latencies = step_latencies[1:]
    avg_decode = sum(decode_latencies) / len(decode_latencies) if decode_latencies else cold_ttft
    effective_tok_s = 1000.0 / avg_decode if avg_decode > 0 else 0.0

    print(f"\nResult: '{result_text}'")
    print(f"Cold TTFT: {cold_ttft:.1f} ms | Avg Decode: {avg_decode:.1f} ms ({effective_tok_s:.2f} tok/s)")
    print(f"Peak VRAM: {vram_alloc:.1f} MB (Alloc) / {vram_res:.1f} MB (Res) | Host RAM: {ram_end:.1f} MB")

    return {
        "config_name": config_name,
        "num_slots": num_slots,
        "use_pinned": use_pinned,
        "tail_prefetch": tail_prefetch,
        "cold_ttft_ms": cold_ttft,
        "avg_decode_ms": avg_decode,
        "effective_tok_s": effective_tok_s,
        "step_latencies": step_latencies,
        "peak_vram_alloc_mb": vram_alloc,
        "peak_vram_res_mb": vram_res,
        "host_ram_mb": ram_end,
        "result_text": result_text
    }


def main():
    print("=" * 80)
    print("AIRCOLIBRI COMPARATIVE BENCHMARK BATTERY (STAGE 5)")
    print(f"Model: DeepSeek-V4-9B (32 layers) | Device: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: '{PROMPT}' | Target: {NUM_NEW_TOKENS} new tokens")
    print("=" * 80)

    configs = [
        ("1. Baseline Sequential (1 Slot, Pageable)", 1, False, False),
        ("2. Bounded Pinned Double Buffer (2 Slots)", 2, True, False),
        ("3. Triple Ring Pipeline (3 Slots)", 3, True, False),
        ("4. Triple Ring + Tail-to-Head Prefetch (3 Slots)", 3, True, True),
    ]

    all_results = []
    for name, slots, pinned, prefetch in configs:
        res = run_single_benchmark(name, slots, pinned, prefetch)
        all_results.append(res)

    print("\n\n" + "=" * 95)
    print("FINAL COMPARATIVE BENCHMARK SUMMARY TABLE")
    print("=" * 95)
    header = f"{'Configuration':<45} | {'Cold TTFT':<11} | {'Avg Decode':<11} | {'Speed':<9} | {'Peak VRAM':<10}"
    print(header)
    print("-" * 95)

    for r in all_results:
        row = f"{r['config_name']:<45} | {r['cold_ttft_ms']:>8.1f} ms | {r['avg_decode_ms']:>8.1f} ms | {r['effective_tok_s']:>6.2f} t/s | {r['peak_vram_alloc_mb']:>7.1f} MB"
        print(row)
    print("=" * 95)

    # Save results to json
    out_json = "scripts/benchmark_battery_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed benchmark results saved to {out_json}")


if __name__ == "__main__":
    main()
