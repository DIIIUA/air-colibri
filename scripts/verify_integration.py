"""Real-checkpoint integration probe. Each case runs in a fresh process.

Results distinguish logits/state parity, lease coverage and physical actions.
The 2.8 GiB RAM gate is unchanged from the historical real-weight test.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
import torch
from src.aircolibri_integrated import AirColibriModel
from src.upstream_bridge import NativeTierPolicy


def state_hashes(obj, prefix="state", seen=None):
    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {prefix: dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(raw).hexdigest())}
    if id(obj) in seen:
        return {}
    seen.add(id(obj))
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("transformers"):
        items = vars(obj).items()
    else:
        return {}
    result = {}
    for key, value in items:
        result.update(state_hashes(value, f"{prefix}.{key}", seen))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=["off", "observe", "enforce"], default="enforce")
    p.add_argument("--pipeline", choices=["sequential", "double_buffer", "triple_ring_prefetch"], default="sequential")
    p.add_argument("--native", action="store_true")
    p.add_argument("--weight-format", choices=["q4", "bf16"], default="q4")
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--model", default=str(ROOT / "models/deepseek-v4-9b-colibri"))
    p.add_argument("--cache", default=str(ROOT / "data/q4-g64-cache"))
    p.add_argument("--manifest", default=str(ROOT / "data/integration-20261112/trusted-checkpoint-manifest.json"))
    p.add_argument("--airllm", default=r"D:\Antigravity Lair\10_ML_AND_TENSORS\air-colibri\upstream_airllm")
    p.add_argument("--output", default=str(ROOT / "data/integration-20261112"))
    a = p.parse_args()
    if a.steps < 1 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in a.case):
        p.error("Invalid steps or case name")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    result = dict(case=a.case, status="STARTED", control=a.mode, pipeline=a.pipeline,
                  native=a.native, prompt=a.prompt, steps_requested=a.steps,
                  ram_available_before=psutil.virtual_memory().available,
                  torch_version=torch.__version__, is_synthetic=False)
    result["weight_format"] = a.weight_format
    result["runtime_sources"] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in sorted((ROOT / "src").glob("*.py"))}
    dest = out / (a.case + ".json")
    save = lambda: dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["ram_available_before"] < 2.8 * 2**30 or not torch.cuda.is_available():
        result.update(status="SAFE_SKIP", reason="Requires CUDA and >=2.8 GiB available RAM")
        save(); print(json.dumps(result)); return
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    engine = None
    try:
        policy = NativeTierPolicy(ROOT / "data/native/colibri_tier.dll", 32, 2) if a.native else None
        started = time.perf_counter()
        engine = AirColibriModel(a.model, pipeline_mode=a.pipeline, vram_resident_layers=0,
            max_ram_layers=2, max_ram_bytes=a.cache_mib * 2**20, cache_policy="lru", weight_format=a.weight_format,
            q4_cache_dir=a.cache if a.weight_format=="q4" else None,
            checkpoint_manifest=a.manifest if a.weight_format=="q4" else None, control_mode=a.mode,
            airllm_source=a.airllm if a.native else None, native_policy=policy)
        result["init_seconds"] = time.perf_counter() - started
        ids = engine.tokenizer(a.prompt, return_tensors="pt")["input_ids"]
        result["input_ids"] = ids.tolist()
        result["model_type"] = engine.config.model_type
        result["layers"] = engine.config.num_hidden_layers
        result["runs"] = []
        logits_all, state_all = [], []
        cache = None
        for step in range(a.steps):
            print(f"[{a.case}] step {step+1}/{a.steps}", flush=True)
            start = time.perf_counter()
            with torch.no_grad():
                logits, cache = engine._forward_step(ids, cache, prefetch_next=step + 1 < a.steps)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite logits")
            logits_all.append(logits.detach().cpu())
            state_all.append(state_hashes(cache))
            if not state_all[-1]:
                raise RuntimeError("No KV/recurrent state inspected")
            token = int(logits.argmax(-1).item())
            result["runs"].append(dict(step=step, token=token, seconds=elapsed,
                state_tensors=len(state_all[-1]), ram_available=psutil.virtual_memory().available,
                rss=psutil.Process().memory_info().rss))
            ids = torch.tensor([[token]])
        result.update(status="PASS", cache=engine.weight_manager.prepacked_cache.stats if engine.weight_manager.prepacked_cache else None,
            requantizations=engine.weight_manager.quantization_calls,
            loads=engine.pipeline.total_loads, computes=engine.pipeline.total_computes,
            tail_prefetches=engine.pipeline.tail_prefetches,
            ram_cache_bytes=engine.weight_manager.ram_cache.current_bytes,
            ram_hits=engine.weight_manager.ram_cache.hits,
            ram_admission_rejected=engine.weight_manager.ram_cache.rejected_admissions,
            mmap_handles=len(engine.weight_manager.mmap_handles),
            peak_vram=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        if engine.ack_adapter:
            snapshot = engine.ack_adapter.snapshot()
            result["lease"] = {k:v for k,v in snapshot.items() if k != "events"}
        if engine.trace_adapter:
            result["trace"] = engine.trace_adapter.contract.snapshot()
            result["linear_hooks"] = engine.trace_adapter.hooked_linears
        if engine.weight_reader:
            result["airllm_reader_calls"] = engine.weight_reader.calls
        if policy:
            result["native_tier"] = policy.stats
        if result["requantizations"] or result["loads"] != a.steps * result["layers"]:
            raise RuntimeError("Unexpected conversion or transfer count")
        torch.save(logits_all, out / (a.case + "-logits.pt"))
        (out / (a.case + "-state.json")).write_text(json.dumps(state_all, indent=2), encoding="utf-8")
    except BaseException as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
    finally:
        cache = None
        logits = None
        if engine is not None:
            engine.close()
        gc.collect()
        torch.cuda.empty_cache()
        result["allocated_after_close"] = torch.cuda.memory_allocated()
        save()
    print(json.dumps({k:v for k,v in result.items() if k not in ("trace", "lease")}, ensure_ascii=False), flush=True)
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
