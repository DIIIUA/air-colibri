"""Separate-process oracle/resident/shared-ring runs with fixed input tokens."""
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
import transformers


def state_hashes(obj, prefix="state", seen=None):
    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        return {prefix:dict(shape=list(t.shape), dtype=str(t.dtype), sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())}
    if id(obj) in seen:
        return {}
    seen.add(id(obj))
    if isinstance(obj, dict):
        pairs = obj.items()
    elif isinstance(obj, (list, tuple)):
        pairs = enumerate(obj)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("transformers"):
        pairs = vars(obj).items()
    else:
        return {}
    result = {}
    for k, value in pairs:
        result.update(state_hashes(value, f"{prefix}.{k}", seen))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reference", "resident", "streamed"], required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--model", default=str(ROOT / "models/OLMoE-1B-7B-0125-Instruct"))
    p.add_argument("--airllm", help="Original AirLLM checkout; otherwise use the CLI local configuration")
    p.add_argument("--native", action="store_true")
    p.add_argument("--read-backend",choices=["mmap","pread"])
    p.add_argument("--vram-gib", type=float, default=3)
    p.add_argument("--cache-mib", type=int, default=64)
    p.add_argument("--output", default=str(ROOT / "data/shared-moe-20261113"))
    a = p.parse_args()
    if not a.case or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in a.case) or a.vram_gib <= 0:
        p.error("Invalid case/budget")
    dest = Path(a.output); dest.mkdir(parents=True, exist_ok=True)
    path = dest / (a.case + ".json")
    if path.exists():
        p.error("Case already exists; use another name to preserve evidence")
    report = dict(status="STARTED", mode=a.mode, native=a.native, vram_budget_gib=a.vram_gib,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        runtime_sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/"src").glob("*.py"))},
        oracle_source_sha256=hashlib.sha256((ROOT/"scripts/reference_olmoe.py").read_bytes()).hexdigest())
    folder = Path(a.model)
    report["checkpoint_metadata_sha256"] = {n:hashlib.sha256((folder/n).read_bytes()).hexdigest()
        for n in ("config.json","model.safetensors.index.json","tokenizer.json")}
    report["checkpoint_shards"] = {f.name:dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
        for f in sorted(folder.glob("*.safetensors"))}
    from transformers.models.olmoe import modeling_olmoe
    report["hf_olmoe_source_sha256"] = hashlib.sha256(Path(modeling_olmoe.__file__).read_bytes()).hexdigest()
    save = lambda: path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    runtime = state = logits = None
    try:
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 2.8*2**30:
            report.update(status="SAFE_SKIP", reason="CUDA and >=2.8 GiB available RAM required")
            return
        from src.resource_plan import windows_commit_available
        commit = windows_commit_available()
        if commit is not None and commit < max(x["bytes"] for x in report["checkpoint_shards"].values())+2*2**30:
            report.update(status="SAFE_SKIP",reason="Insufficient commit headroom for mapped source and runtime")
            return
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        torch.cuda.set_per_process_memory_fraction(min(1, a.vram_gib*2**30/torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if a.mode == "reference":
            from scripts.reference_olmoe import ReferenceOLMoE
            runtime = ReferenceOLMoE(a.model)
        else:
            from src.olmoe_integrated import AirColibriOLMoE
            from src.cli import default_airllm_source
            extra = {"stream_dense":True} if a.mode == "streamed" else {}
            runtime = AirColibriOLMoE(a.model, a.airllm or default_airllm_source(), ram_bytes=a.cache_mib*2**20,
                native_dll=ROOT/"data/native/colibri_tier.dll" if a.native else None,
                io_backend=a.read_backend, **extra)
            report["airllm_io_backend"]=runtime.provider.reader.io_backend
        report["initialization_seconds"] = time.perf_counter()-started
        rows, values = [], []
        # Fixed teacher-forced inputs prevent a token divergence changing later work.
        for prompt in ("Hello", "The answer is 42. Код: x = 1"):
            state = None
            inputs = [runtime.tokenizer(prompt, return_tensors="pt")["input_ids"], torch.tensor([[13]]), torch.tensor([[42]])]
            for ids in inputs:
                if psutil.virtual_memory().available < 512*2**20:
                    raise MemoryError("RAM pressure before forward")
                print(f"{a.case}: forward {len(rows)+1}/6 ({ids.shape[1]} tokens)", flush=True)
                t = time.perf_counter()
                logits, state = runtime.forward(ids, state)
                torch.cuda.synchronize()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite logits")
                values.append(logits.detach().cpu())
                rows.append(dict(input_ids=ids.tolist(), state=state_hashes(state), seconds=time.perf_counter()-t,
                    available_ram=psutil.virtual_memory().available, rss=psutil.Process().memory_info().rss))
                report["runs"] = rows
                report["completed_forwards"] = len(rows)
                # Native faults cannot execute finally: preserve each completed
                # observation without marking a partially completed run PASS.
                torch.save(values, dest/(a.case+"-logits.pt"))
                save()
        report.update(status="PASS", runs=rows, router_choices=runtime.router_choices,
                      peak_vram=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        if a.mode != "reference":
            report.update(trace=runtime.trace.snapshot(), expert_reads=runtime.store.reads,
                          h2d_expert_bytes=runtime.store.logical_h2d_bytes)
            if getattr(runtime, "resources", None):
                report["resources"] = runtime.resources.snapshot()
            if runtime.store.policy:
                report["native_policy"] = runtime.store.policy.stats
        else:
            report.update(reference_layers=runtime.loaded_layers, reference_logical_read_bytes=runtime.read_bytes)
        torch.save(values, dest/(a.case+"-logits.pt"))
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
    finally:
        state = logits = None
        if runtime:
            runtime.close()
        gc.collect()
        report["allocated_after_close"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        save()
        print(json.dumps({k:v for k,v in report.items() if k not in ("runs","router_choices","trace","runtime_sources","resources")}, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
