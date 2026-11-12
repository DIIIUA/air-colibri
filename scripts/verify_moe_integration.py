"""Run the full real OLMoE causal graph; compare sparse and all-expert backends."""
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
from src.olmoe_integrated import AirColibriOLMoE
from scripts.verify_integration import state_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--full-experts", action="store_true")
    p.add_argument("--native", action="store_true")
    p.add_argument("--shadow", action="store_true")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--prompt", default="Hello")
    a = p.parse_args()
    if a.steps < 1 or not a.case or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in a.case):
        p.error("Invalid case name or steps")
    out = ROOT / "data/integration-20261112"
    result = dict(case=a.case, full_expert_baseline=a.full_experts, native=a.native,
                  status="STARTED", model="OLMoE-1B-7B-0125-Instruct", prompt=a.prompt,
                  ram_available_before=psutil.virtual_memory().available)
    result["runtime_sources"] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in sorted((ROOT / "src").glob("*.py"))}
    result["native_mode"] = "shadow" if a.shadow else "apply" if a.native else "off"
    (out / (a.case + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    runtime = state = logits = None
    try:
        if result["ram_available_before"] < 2.8*2**30 or not torch.cuda.is_available():
            result.update(status="SAFE_SKIP", reason="Need CUDA and >=2.8 GiB available RAM")
            return
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        runtime = AirColibriOLMoE(ROOT / "models/OLMoE-1B-7B-0125-Instruct",
            r"D:\Antigravity Lair\10_ML_AND_TENSORS\air-colibri\upstream_airllm",
            native_dll=ROOT / "data/native/colibri_tier.dll" if a.native or a.shadow else None,
            full_expert_baseline=a.full_experts, native_mode="shadow" if a.shadow else "apply")
        ids = runtime.tokenizer(a.prompt, return_tensors="pt")["input_ids"]
        result["input_ids"] = ids.tolist()
        values, states, rows = [], [], []
        for step in range(a.steps):
            print(f"{a.case}: full forward {step+1}/{a.steps}", flush=True)
            start = time.perf_counter()
            logits, state = runtime.forward(ids, state)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits")
            values.append(logits.cpu())
            states.append(state_hashes(state))
            token = int(logits.argmax(-1).item())
            rows.append(dict(step=step, token=token, seconds=elapsed, state_tensors=len(states[-1]),
                available_ram=psutil.virtual_memory().available, rss=psutil.Process().memory_info().rss))
            ids = torch.tensor([[token]])
        result.update(status="PASS", runs=rows, trace=runtime.trace.snapshot(),
            router_choices=runtime.router_choices, expert_calls=runtime.store.expert_calls,
            expert_reads=runtime.store.reads, logical_h2d_expert_bytes=runtime.store.logical_h2d_bytes,
            airllm_calls=runtime.provider.reader.calls, ram_cache_bytes=runtime.store.cache.current_bytes,
            peak_vram=torch.cuda.max_memory_allocated(),
            placement_counts=dict(runtime.store.cache.placement_counts),
            placement_peak_bytes=runtime.store.cache.peak_bytes)
        if runtime.store.policy:
            result["native_tier"] = runtime.store.policy.stats
        torch.save(values, out / (a.case + "-logits.pt"))
        (out / (a.case + "-state.json")).write_text(json.dumps(states, indent=2))
    except BaseException as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
    finally:
        state = logits = None
        if runtime:
            runtime.close()
        gc.collect()
        result["allocated_after_close"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        (out / (a.case + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k not in ("router_choices", "trace")}))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
