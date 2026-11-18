"""Small real checkpoints through the disk/CPU executor; no quality or speedup claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--formats", nargs="+", choices=("bf16", "q4", "q6"), default=["bf16", "q6"])
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--hide-cuda", action="store_true")
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hide_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    import psutil
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    torch.set_num_threads(6)
    paths = ["src/progressive_execution.py", "src/aircolibri_integrated.py",
             "aircolibri_cli/unified_runtime.py", "aircolibri_cli/cli.py", __file__]
    report = {"protocol": "Two short requests per checkpoint/format; OS caches not flushed. Technical execution only; no answer-quality or causal speedup certification.",
              "device_choice": args.device, "cuda_available": torch.cuda.is_available(), "cpu_threads": 6,
              "source_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
              "results": []}
    for name in args.models:
        folder = ROOT / "models" / name
        stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in folder.rglob("*") if p.suffix in (".gguf", ".safetensors")}
        for fmt in args.formats:
            print(f"RUN {name} {fmt} {args.device}", flush=True)
            result = {"model": name, "format": fmt, "requests": [],
                      "checkpoint_bytes": sum(size for size, _ in stamps.values())}
            engine = None
            started = time.perf_counter()
            try:
                engine = AirColibriUnifiedEngine(folder, device=args.device, weight_format=fmt,
                    enable_mtp=False, max_context_tokens=256, prefill_chunk_tokens=8, verbose=False)
                result["constructor_s"] = time.perf_counter() - started
                messages = []
                for prompt, count in (("What is 6 times 7? Answer briefly.", args.tokens), ("Name the capital of Japan.", min(6, args.tokens))):
                    messages.append({"role": "user", "content": prompt})
                    first = None
                    request_start = time.perf_counter()
                    for event in engine.stream_chat(messages, max_new_tokens=count,
                            stop_on_eos=False, use_session_cache=True, enable_thinking=False):
                        if event["type"] == "delta" and first is None:
                            first = time.perf_counter() - request_start
                        if event["type"] == "done":
                            result["requests"].append({"answer": event["answer_text"], "metrics": event["metrics"],
                                                       "first_text_s": first})
                            messages.append({"role": "assistant", "content": event["answer_text"]})
                result["status"] = engine.status_snapshot()
                cache = engine.engine.weight_manager.ram_cache
                result["ram_layer_cache"] = {"peak_bytes": cache.peak_bytes, "max_bytes": cache.max_bytes,
                                             "max_layers": cache.max_cached_layers, "evictions": cache.evictions}
                result["host_staging_bytes"] = engine.engine.weight_manager.pinned_pool.total_pinned_bytes
                result["passed"] = (engine.engine.device.type == "cpu" and len(result["requests"]) == 2
                                    and engine.engine.progressive.worker is None)
            except Exception as exc:
                import traceback
                result.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
            finally:
                if engine is not None:
                    engine.close()
                gc.collect()
                result["total_s"] = time.perf_counter() - started
                result["rss_mib_at_end"] = psutil.Process().memory_info().rss / 2**20
                result["weight_size_mtime_unchanged"] = all(Path(p).exists() and
                    (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == stamp for p, stamp in stamps.items())
            report["results"].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: result.get(k) for k in ("model", "format", "passed", "total_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
