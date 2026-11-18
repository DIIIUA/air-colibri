"""Short real-model comparisons with a forced three-slot, nonresident backbone.

Caches are bounded; OS caches are not flushed. This is execution validation,
not a large-model capability or statistically controlled speedup certificate.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("qwen", "granite"), required=True)
    p.add_argument("--device", choices=("cpu", "auto"), default="auto")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--expert-cache-mib", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import psutil
    from aircolibri_cli import unified_runtime as runtime
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    torch.set_num_threads(6)
    original = runtime.plan_for_checkpoint
    original_ram_budget = runtime._streaming_ram_budget_bytes
    def streamed(*a, **kw):
        kw["force_universal"] = True
        profile, plan, mtp = original(*a, **kw)
        return profile, replace(plan, resident_layers=0, full_residency=False, ring_slots=3,
            budget_bytes=plan.weights_bytes+(256+args.expert_cache_mib)*2**20), mtp
    runtime.plan_for_checkpoint = streamed
    runtime._streaming_ram_budget_bytes = lambda: 128*2**20
    name = ("12_gguf_Qwen3.5-0.8B-Q4_K_M" if args.case == "qwen"
            else "28_hf_granite-3.1-1b-a400m-MoE")
    folder = ROOT/"models"/name
    head = ROOT/"data"/"native-mtp"/"qwen3.5-0.8b.safetensors"
    stamps = {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in folder.rglob("*")
              if f.suffix in (".gguf", ".safetensors")}
    sources = ["src/general_speculation.py", "src/native_mtp.py", "src/rowwise_verification.py",
               "src/route_exclusion.py", "src/progressive_execution.py", "src/probability_warmup.py",
               "aircolibri_cli/unified_runtime.py", "benchmarks/general_speculation.py"]
    report = dict(model=name, device_choice=args.device, checkpoint_bytes=sum(s[0] for s in stamps.values()),
                  protocol=__doc__, cpu_threads=6,
                  sources={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}, results=[])
    expected = None
    for mode in ("off", "native" if args.case == "qwen" else "lookup"):
        result = dict(mode=mode)
        u = None
        started = time.perf_counter()
        print(f"START {name} {mode}", flush=True)
        try:
            u = runtime.AirColibriUnifiedEngine(folder, device=args.device,
                progressive_start=args.device == "cpu", weight_format="q6", speculation=mode,
                mtp_head_path=head if mode == "native" else None, verbose=False,
                max_context_tokens=256, prefill_chunk_tokens=16)
            result["constructor_s"] = time.perf_counter()-started
            warm = None
            if args.case == "granite":
                warm = u.engine.enable_probability_warmup(region_experts=8, gpu_budget_bytes=32*2**20,
                    host_cache_bytes=16*2**20, unified_memory=True, dynamic_regions=True)
                u._active_thought_anchor = InitialProbabilityAnchor()
            prompt, _, _, _ = u._prepare_prompt_and_cache(
                [{"role":"user", "content":"Continue: 1, 2, 3, 4, 5,"}], False, False, max_new_tokens=args.tokens)
            result["requests"] = []
            past, prefix = None, 0
            for turn, count in enumerate((args.tokens, 4)):
                tokens, final = [], None
                for event in u._stream_token_ids(prompt, count, False, prefix, past):
                    if event["kind"] == "tokens":
                        tokens.extend(event["new_ids"])
                        print(f"{mode} turn={turn} tokens={len(tokens)}", flush=True)
                    else:
                        final = event
                past = final["past"]
                prefix = final["consumed_tokens"]
                result["requests"].append(dict(tokens=tokens, text=u.tokenizer.decode(tokens),
                    prefix=prefix, prefill_ms=final["prefill_ms"], decode_s=final["decode_s"],
                    backbone_passes=final["backbone_passes"], speculation=final.get("spec_gate", {})))
                # Continue from the actual committed cache, feeding the unfed
                # output and new input without recomputing the old backbone.
                prompt = final["all_ids"]+u.tokenizer.encode("\nContinue:", add_special_tokens=False)
            result["status"] = u.status_snapshot()
            if warm is not None:
                warm = getattr(warm, "active", None) or warm
                if hasattr(warm, "prefetch"):
                    warm.prefetch.drain()
                    result["warmup"] = warm.stats()
                    total_experts = sum(u.engine.model_source._expert_loader.num_experts_by_layer.values())
                    result["expert_slots_below_whole_model"] = result["warmup"]["shared_memory"]["num_slots"] < total_experts
                    result["total_model_experts"] = total_experts
                result["initial_probability_sha256"] = u._active_thought_anchor.initial_probability_fingerprint()
            cache = u.engine.weight_manager.ram_cache
            result["ram_cache"] = dict(cap=cache.max_bytes, peak=cache.peak_bytes,
                layers=cache.max_cached_layers, evictions=cache.evictions)
            result["staging_bytes"] = u.engine.weight_manager.pinned_pool.total_pinned_bytes
            actual = [r["tokens"] for r in result["requests"]]
            if expected is None:
                expected = actual
            result["tokens_match_plain"] = actual == expected
            result["passed"] = (actual == expected and cache.peak_bytes <= cache.max_bytes
                                and result.get("expert_slots_below_whole_model", True))
        except Exception as exc:
            result.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
            print(result["traceback"], flush=True)
        finally:
            if u is not None:
                u.close()
            past = final = None
            gc.collect()
            result["total_s"] = time.perf_counter()-started
            result["rss_mib_at_end"] = psutil.Process().memory_info().rss/2**20
            result["weights_size_mtime_unchanged"] = all(
                (Path(f).stat().st_size, Path(f).stat().st_mtime_ns) == stamp for f, stamp in stamps.items())
            report["results"].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: result.get(k) for k in ("mode", "passed", "total_s", "error")}), flush=True)
    runtime.plan_for_checkpoint = original
    runtime._streaming_ram_budget_bytes = original_ram_budget
    return 0 if all(r["passed"] for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
