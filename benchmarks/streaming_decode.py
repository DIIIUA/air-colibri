"""Paired decode measurements: resident / half streamed / all streamed.

One inference process, equal Q6 arithmetic and bounded host cache. Placement order
is A B C C B A, with an unscored warmup before every block. A separate idle process
can hold VRAM before planning. No token/hidden observers inside timed requests.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIB = 2**20


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def hardware(pids):
    def query(args):
        result = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    processes = []
    for line in query(["--query-compute-apps=pid,used_gpu_memory"]).splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit() and int(first) in pids:
            processes.append(line)
    return dict(gpu=query(["--query-gpu=name,memory.total,memory.used,memory.free,power.limit,driver_version"]),
                own_process_memory_mib=processes)


def hold(args):
    import torch
    try:
        payload = torch.empty(args.hold_vram*MIB, dtype=torch.uint8, device="cuda").fill_(1)
        torch.cuda.synchronize()
        write(args.ready, dict(pid=os.getpid(), allocated_bytes=torch.cuda.memory_allocated(),
                               reserved_bytes=torch.cuda.memory_reserved(), requested_mib=args.hold_vram))
        while not args.stop.exists():
            time.sleep(.2)
        del payload
        torch.cuda.empty_cache()
    except Exception:
        write(args.ready, dict(error=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["01_gguf_Llama-3.2-1B-Instruct-Q4_K_M",
                                                    "12_gguf_Qwen3.5-0.8B-Q4_K_M"])
    p.add_argument("--pressures-mib", nargs="+", type=int, default=[0, 2048])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--trials-per-block", type=int, default=2)
    p.add_argument("--ram-cache-mib", type=int, default=896)
    p.add_argument("--output", type=Path)
    p.add_argument("--hold-vram", type=int)
    p.add_argument("--ready", type=Path)
    p.add_argument("--stop", type=Path)
    args = p.parse_args()
    if args.hold_vram:
        return hold(args)
    if args.output is None or args.tokens < 32 or args.trials_per_block < 2:
        p.error("Output, >=32 tokens and >=2 trials/block required")
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import psutil
    import torch
    from aircolibri_cli import unified_runtime as runtime
    torch.set_num_threads(6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_plan = runtime.plan_for_checkpoint
    original_budget = runtime._streaming_ram_budget_bytes
    source = hashlib.sha256()
    for directory in ("src", "aircolibri_cli"):
        for f in sorted((ROOT/directory).rglob("*")):
            if f.suffix in (".py", ".cu", ".h", ".c"):
                source.update(f.relative_to(ROOT).as_posix().encode())
                source.update(f.read_bytes().replace(b"\r\n", b"\n"))
    report = dict(protocol=__doc__, tokens=args.tokens, trials_per_block=args.trials_per_block,
        order=["resident", "half", "stream", "stream", "half", "resident"],
        ram_cache_cap_bytes=args.ram_cache_mib*MIB, cpu_threads=6, format="q6", mtp=False, anchor=False,
        max_context=2048, prefill_chunk=64, source_lf_sha256=source.hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda,
        note="OS caches not flushed. Host-layer count raised to all layers within the fixed byte cap, to distinguish RAM-to-GPU streaming from disk cache misses. Controlled placements are not default startup speeds. VRAM pressure exists before planning; no claim of live replanning after arbitrary later allocations.",
        models=[], status="running")
    process = psutil.Process()
    try:
        for name in args.models:
            model = ROOT/"models"/name
            model_result = dict(model=name, cases=[])
            report["models"].append(model_result)
            stamps = {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in model.rglob("*")
                      if f.suffix in (".gguf", ".safetensors")}
            model_result["checkpoint_bytes"] = sum(x[0] for x in stamps.values())
            for pressure in args.pressures_mib:
                holder = log = None
                stop = args.output.parent/f"pressure_{model.name}_{pressure}.stop"
                ready = args.output.parent/f"pressure_{model.name}_{pressure}.ready.json"
                if pressure and (stop.exists() or ready.exists()):
                    raise FileExistsError("Use a fresh output directory for memory-pressure control files")
                try:
                    pressure_info = None
                    if pressure:
                        log = (args.output.parent/f"pressure_{model.name}_{pressure}.log").open("w")
                        holder = subprocess.Popen([sys.executable, str(Path(__file__)), "--hold-vram", str(pressure),
                            "--ready", str(ready), "--stop", str(stop)], stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        deadline = time.monotonic()+180
                        while not ready.exists():
                            if holder.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError("VRAM holder failed to become ready; inspect its log")
                            time.sleep(.2)
                        pressure_info = json.loads(ready.read_text("utf-8"))
                        if "error" in pressure_info:
                            raise RuntimeError(pressure_info["error"])
                    for block, placement in enumerate(report["order"]):
                        print(f"START {name} pressure={pressure} {placement} block={block}", flush=True)
                        case = dict(placement=placement, pressure_mib=pressure, pressure_process=pressure_info,
                                    block=block, trials=[], status="starting")
                        model_result["cases"].append(case)
                        write(args.output, report)
                        engine = None
                        def forced(*a, **kw):
                            kw["force_universal"] = True
                            profile, plan, mtp = original_plan(*a, **kw)
                            case["automatic_plan_before_override"] = asdict(plan)
                            resident = {"resident":plan.layers_total, "half":plan.layers_total//2, "stream":0}[placement]
                            if placement == "resident" and not plan.full_residency:
                                raise RuntimeError("Full-resident reference does not fit the measured free VRAM")
                            return profile, replace(plan, resident_layers=resident,
                                full_residency=placement == "resident", ring_slots=0 if placement == "resident" else 3), mtp
                        try:
                            runtime.plan_for_checkpoint = forced
                            runtime._streaming_ram_budget_bytes = lambda: args.ram_cache_mib*MIB
                            case["hardware_before"] = hardware([os.getpid()]+([pressure_info["pid"]] if holder else []))
                            case["available_ram_before_bytes"] = psutil.virtual_memory().available
                            started = time.perf_counter()
                            engine = runtime.AirColibriUnifiedEngine(model_dir=model, progressive_start=False,
                                weight_format="q6", speculation="off", prune_regions=False, enable_mtp=False,
                                max_context_tokens=2048, prefill_chunk_tokens=64, verbose=False)
                            case["constructor_s"] = time.perf_counter()-started
                            case["startup_memory_probe"] = engine.status_snapshot().get("startup_memory_probe")
                            if case["startup_memory_probe"] is not None:
                                assert case["automatic_plan_before_override"]["free_bytes"] == case["startup_memory_probe"]["free_bytes"]
                            cache = engine.engine.weight_manager.ram_cache
                            cache.max_cached_layers = engine.plan.layers_total
                            strides = getattr(engine.engine.pipeline, "slot_strides", ())
                            if strides:
                                engine.plan = replace(engine.plan, slot_bytes=max(strides))
                            case["controlled_plan"] = asdict(engine.plan)
                            def generate(count):
                                engine.reset_session()
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                io0 = process.io_counters()
                                before = dict(hits=cache.hits, misses=cache.misses, evictions=cache.evictions)
                                done = None
                                for event in engine.stream_chat([{"role":"user", "content":
                                    "Explain why the sky appears blue during the day. Use three short sentences."}],
                                    max_new_tokens=count, enable_thinking=False, use_session_cache=False, stop_on_eos=False):
                                    if event["type"] == "done":
                                        done = event
                                torch.cuda.synchronize()
                                io1 = process.io_counters()
                                if done is None or done["metrics"]["generated_tokens"] != count:
                                    raise RuntimeError("Incomplete measured generation")
                                return dict(metrics=done["metrics"], text=done["answer_text"],
                                    text_sha256=hashlib.sha256((done["thinking_text"]+done["answer_text"]).encode()).hexdigest(),
                                    read_bytes=io1.read_bytes-io0.read_bytes,
                                    ram_hits=cache.hits-before["hits"], ram_misses=cache.misses-before["misses"],
                                    ram_evictions=cache.evictions-before["evictions"])
                            case["warmup"] = generate(8)
                            case["status"] = "running"
                            write(args.output, report)
                            for trial in range(args.trials_per_block):
                                result = generate(args.tokens)
                                case["trials"].append(result)
                                write(args.output, report)
                                print(f"TRIAL {name} pressure={pressure} {placement}: {result['metrics']['decode_tok_per_s']} tok/s", flush=True)
                            case["ram_cache"] = dict(peak_bytes=cache.peak_bytes, cap_bytes=cache.max_bytes,
                                                     layers=cache.current_layers_count)
                            case["hardware_after"] = hardware([os.getpid()]+([pressure_info["pid"]] if holder else []))
                            case["status"] = "passed"
                        finally:
                            runtime.plan_for_checkpoint = original_plan
                            runtime._streaming_ram_budget_bytes = original_budget
                            if engine is not None:
                                engine.close()
                            engine = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            write(args.output, report)
                finally:
                    if holder is not None:
                        stop.touch()
                        try:
                            holder.wait(20)
                        except subprocess.TimeoutExpired:
                            holder.kill(); holder.wait()
                    if log is not None:
                        log.close()
            model_result["weights_size_mtime_unchanged"] = all(
                (Path(f).stat().st_size, Path(f).stat().st_mtime_ns) == stamp for f, stamp in stamps.items())
            model_result["summary"] = []
            for pressure in args.pressures_mib:
                for placement in ("resident", "half", "stream"):
                    trials = [t for c in model_result["cases"] if c["pressure_mib"] == pressure and c["placement"] == placement for t in c["trials"]]
                    rates = [t["metrics"]["decode_tok_per_s"] for t in trials]
                    model_result["summary"].append(dict(pressure_mib=pressure, placement=placement, samples=len(rates),
                        median_tok_s=statistics.median(rates), min_tok_s=min(rates), max_tok_s=max(rates),
                        median_prefill_ms=statistics.median(t["metrics"]["prefill_ms"] for t in trials),
                        output_hashes=sorted({t["text_sha256"] for t in trials}),
                        median_read_bytes=statistics.median(t["read_bytes"] for t in trials)))
            write(args.output, report)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write(args.output, report)


if __name__ == "__main__":
    main()
