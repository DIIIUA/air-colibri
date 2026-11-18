"""Repeatable offline GPU benchmark; run one process per configuration.

Auto is the public placement policy. The two granite-* modes deliberately force
six streamed layers and the same configurable expert budget to exercise memory rotation
on a small model. Those diagnostic placements are not default-product speeds.
No per-layer observers, logit copies, or KV hashes run inside timed requests.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--weight-format", choices=["q4", "q6", "bf16"], default="q6")
    p.add_argument("--placement", choices=["auto", "granite-ring", "granite-dynamic"], default="auto")
    p.add_argument("--anchor", action="store_true")
    p.add_argument("--expert-slots", type=int, default=700,
                   help="Starting expert slots for controlled Granite placements; ignored in auto mode")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.trials < 2 or args.tokens < 2:
        p.error("At least two trials and tokens are required")
    import psutil
    import torch
    from aircolibri_cli import unified_runtime as runtime

    model = Path(args.model).resolve()
    source_hash = hashlib.sha256()
    for folder in ("aircolibri_cli", "src"):
        for file in sorted((ROOT/folder).rglob("*")):
            if file.suffix in (".py", ".cu", ".h", ".c"):
                source_hash.update(file.relative_to(ROOT).as_posix().encode())
                source_hash.update(file.read_bytes())
    report = dict(model=model.name, weight_format=args.weight_format, placement=args.placement,
                  controlled_expert_slots=args.expert_slots if args.placement != "auto" else None,
                  source_tree_sha256=source_hash.hexdigest(),
                  anchor=args.anchor, status="starting", measured_trials=[],
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), cpu_threads=torch.get_num_threads(),
                  available_ram_mib=round(psutil.virtual_memory().available / 2**20, 1),
                  model_bytes=sum(f.stat().st_size for f in model.rglob("*") if f.is_file()),
                  cache_note="OS file cache is not flushed; constructor time is not cold-disk time.")
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    if args.placement != "auto":
        if model.name != "28_hf_granite-3.1-1b-a400m-MoE" or args.weight_format not in ("q4", "q6"):
            raise ValueError("Controlled streaming placement requires the documented Granite 1B Q4/Q6 fixture")
        bits = int(args.weight_format[1:])
        expert_bytes = int(3 * 512 * 1024 * (bits/8 + 4/64))
        reserve = 8 * 6 * 512 * 1024 + 32 * 2**20
        original = runtime.plan_for_checkpoint
        def forced_plan(*a, **kw):
            profile, plan, mtp = original(*a, **kw)
            return profile, replace(plan, full_residency=False, resident_layers=18, ring_slots=3,
                budget_bytes=plan.weights_bytes + 256*2**20 + reserve + args.expert_slots*expert_bytes), mtp
        runtime.plan_for_checkpoint = forced_plan
        runtime._streaming_ram_budget_bytes = lambda: 256*2**20

    engine = None
    process = psutil.Process()
    def cache_counters():
        stats = engine.expert_cache_stats() or {}
        result = {k:v for k,v in stats.items() if isinstance(v,(int,float,bool))}
        for key in ('source','shared_memory'):
            section = stats.get(key) or {}
            result[key] = {k:v for k,v in section.items()
                           if isinstance(v,(int,float,bool)) and k != 'allocation_ptr'}
        return result
    def generate(messages, tokens, *, measured=False):
        if args.anchor:
            from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
            engine._active_thought_anchor = InitialProbabilityAnchor()
        engine.reset_session()
        torch.cuda.synchronize()
        before_cache = cache_counters()
        torch.cuda.reset_peak_memory_stats()
        io0 = process.io_counters()
        t0 = time.perf_counter()
        first = None
        done = None
        for event in engine.stream_chat(messages, max_new_tokens=tokens, enable_thinking=False,
                                        stop_on_eos=not measured, use_session_cache=False):
            if event["type"] == "delta" and event.get("text") and first is None:
                first = time.perf_counter() - t0
            if event["type"] == "done":
                done = event
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        io1 = process.io_counters()
        if done is None:
            raise RuntimeError("Generation returned no completion event")
        result = dict(wall_s=round(wall, 6), first_visible_delta_s=first,
                      metrics=done["metrics"], answer=done["answer_text"],
                      output_sha256=hashlib.sha256((done["thinking_text"]+done["answer_text"]).encode()).hexdigest(),
                      process_read_bytes=io1.read_bytes-io0.read_bytes,
                      process_read_count=io1.read_count-io0.read_count)
        result.update(cache_before=before_cache, cache_after=cache_counters())
        engine._active_thought_anchor = None
        return result
    try:
        t0 = time.perf_counter()
        engine = runtime.AirColibriUnifiedEngine(model_dir=model, weight_format=args.weight_format,
            enable_mtp=False, verbose=False, max_context_tokens=2048, prefill_chunk_tokens=64)
        if args.placement == "granite-dynamic":
            engine.enable_unified_memory(region_experts=12, gpu_budget_bytes=0, host_cache_bytes=64*2**20)
        report.update(constructor_s=round(time.perf_counter()-t0, 6), plan=asdict(engine.plan), status="running")
        save()
        messages = [{"role": "user", "content": "Explain why the sky appears blue during the day. Use three short sentences."}]
        report["warmup"] = generate(messages, args.tokens, measured=True)
        save()
        for trial in range(args.trials):
            result = generate(messages, args.tokens, measured=True)
            report["measured_trials"].append(result)
            save()
            print(f"trial {trial+1}: {result['metrics']['decode_tok_per_s']} tok/s", flush=True)
        report["quality_probes"] = [generate([{"role":"user", "content": text}], 16)
            for text in ("What is 17 + 25? Answer with only the number.", "Name the capital of Japan. Answer with one word.")]
        rates = [x["metrics"]["decode_tok_per_s"] for x in report["measured_trials"]]
        report["summary"] = dict(median_tok_s=statistics.median(rates), min_tok_s=min(rates), max_tok_s=max(rates),
            repeated_text_equal=len({x["output_sha256"] for x in report["measured_trials"]}) == 1,
            median_prefill_ms=statistics.median(x["metrics"]["prefill_ms"] for x in report["measured_trials"]))
        warm = getattr(engine.engine, "probability_warmup", None)
        if warm is not None:
            report["warmup_state"] = warm.stats()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if engine is not None:
            engine.close()
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        report["allocated_after_close_mib"] = round(torch.cuda.memory_allocated()/2**20, 3)
        save()


if __name__ == "__main__":
    main()
