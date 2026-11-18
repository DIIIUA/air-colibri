"""Bounded common-start / GPU-first comparison, without downloads or cache flushing."""
import argparse
import gc
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine


def run(folder, cpu_first, bits, tokens):
    from src.colibri_deltanet_nvrtc import KERNEL_CALL_STATS
    kernel_before = dict(KERNEL_CALL_STATS)
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in folder.rglob("*") if p.suffix in {".gguf", ".safetensors"}}
    start = time.perf_counter()
    result = {"model": folder.name, "progressive_start": cpu_first,
              "mode": "simultaneous_readiness" if cpu_first else "gpu_first_control", "format": f"q{bits}"}
    engine = None
    try:
        engine = AirColibriUnifiedEngine(folder, cpu_first=cpu_first, weight_format=f"q{bits}",
                                         enable_mtp=False, max_context_tokens=512,
                                         prefill_chunk_tokens=8, verbose=False)
        result["constructor_s"] = time.perf_counter()-start
        result["device_after_constructor"] = str(engine.engine.device)
        first = None
        for event in engine.stream_chat([{"role": "user", "content": "What is 6 times 7? Answer briefly."}],
                                         max_new_tokens=tokens, enable_thinking=False,
                                         stop_on_eos=False, use_session_cache=True):
            if event["type"] == "delta" and first is None:
                first = time.perf_counter()-start
            if event["type"] == "done":
                result.update(answer=event["answer_text"], metrics=event["metrics"])
        result["first_text_from_constructor_start_s"] = first
        result["total_s"] = time.perf_counter()-start
        result["progressive_state"] = engine.engine.progressive.stats() if cpu_first else None
        result["final_device"] = str(engine.engine.device)
        # A second turn exercises parking and continuation after the takeover.
        messages = [{"role": "user", "content": "What is 6 times 7? Answer briefly."},
                    {"role": "assistant", "content": result["answer"]},
                    {"role": "user", "content": "Name the capital of Japan."}]
        for event in engine.stream_chat(messages,
                                         max_new_tokens=12, enable_thinking=False,
                                         use_session_cache=True, stop_on_eos=False):
            if event["type"] == "done":
                result["second_answer"] = event["answer_text"]
                result["second_metrics"] = event["metrics"]
        result["passed"] = bool(result["final_device"].startswith("cuda") and
                                  (not cpu_first or result["progressive_state"]["complete"]))
    except Exception as exc:
        import traceback
        result.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
    finally:
        result["native_kernel_calls"] = {k: v-kernel_before.get(k, 0) for k, v in KERNEL_CALL_STATS.items()}
        if engine is not None:
            engine.close()
        gc.collect()
        torch.cuda.empty_cache()
        result["weight_files_unchanged"] = all(p.exists() and
            (p.stat().st_size, p.stat().st_mtime_ns) == stamp for name, stamp in before.items()
            for p in [Path(name)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--bits", type=int, choices=[4, 6], default=6)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progressive-only", "--cpu-first-only", action="store_true", dest="progressive_only")
    args = parser.parse_args()
    torch.set_num_threads(6)
    report = {"protocol": "one request per fresh engine, OS caches not flushed; not a cold-disk or causal speedup proof",
              "threads": 6, "results": [],
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT/'src/progressive_execution.py', ROOT/'aircolibri_cli/unified_runtime.py',
                                          ROOT/'src/quantized_experts.py', ROOT/'src/aircolibri_integrated.py',
                                          ROOT/'src/colibri_deltanet_nvrtc.py', ROOT/'src/colibri_nvrtc.py',
                                          ROOT/'src/native/colibri_fused_quant.cu', ROOT/'aircolibri_cli/model_source.py']}}
    for model in args.models:
        for cpu_first in ((True,) if args.progressive_only else (True, False)):
            print(f"RUN {model} progressive_start={cpu_first}", flush=True)
            result = run(ROOT / "models" / model, cpu_first, args.bits, args.tokens)
            report["results"].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: result.get(k) for k in ("model", "progressive_start", "passed", "total_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
