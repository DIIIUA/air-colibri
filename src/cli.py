"""Unified, explicit entrypoint for the verified AirColibri backends."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
def default_airllm_source():
    """Explicit environment/local configuration, then a checkout-relative path."""
    if os.environ.get("AIRCOLIBRI_AIRLLM_SOURCE"):
        return os.environ["AIRCOLIBRI_AIRLLM_SOURCE"]
    local = ROOT / "aircolibri.local.json"
    if local.is_file():
        configured = json.loads(local.read_text(encoding="utf-8")).get("airllm_source")
        if configured:
            return configured
    return str(ROOT / "upstream_airllm")


def main():
    p = argparse.ArgumentParser(description="AirLLM readers + original Transformers forward + Colibri native tier")
    p.add_argument("action", choices=["plan", "run", "doctor"])
    p.add_argument("--model", choices=["qwen9b", "olmoe"], default="qwen9b")
    p.add_argument("--model-dir")
    p.add_argument("--weight-format", choices=["bf16", "q4"], default="bf16")
    p.add_argument("--group-size", type=int, choices=[16,32,64], default=64)
    p.add_argument("--pipeline", choices=["sequential","double_buffer","triple_ring_prefetch"], default="sequential")
    p.add_argument("--policy", choices=["off","shadow","apply"], default="off")
    p.add_argument("--control", choices=["off","observe","enforce"], default="enforce")
    p.add_argument("--ram-gib", type=float, default=4)
    p.add_argument("--vram-gib", type=float, default=5)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--airllm-source", default=default_airllm_source())
    p.add_argument("--native-library", default=str(ROOT / "data/native/colibri_tier.dll"))
    p.add_argument("--q4-cache", default=str(ROOT / "data/q4-g64-cache"))
    p.add_argument("--checkpoint-manifest", default=str(ROOT / "data/integration-20261112/trusted-checkpoint-manifest.json"))
    p.add_argument("--report", help="Optional JSON result path; defaults to data/latest-run.json")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    if args.model == "olmoe" and (args.pipeline != "sequential" or args.control != "enforce"):
        p.error("OLMoE currently supports --pipeline sequential and --control enforce only")
    model_dir = Path(args.model_dir or ROOT / "models" / (
        "deepseek-v4-9b-colibri" if args.model == "qwen9b" else "OLMoE-1B-7B-0125-Instruct")).resolve()
    from src.resource_plan import make_plan, windows_commit_available, GIB
    import psutil
    import torch
    import transformers
    if args.action == "doctor":
        print(json.dumps(dict(root=str(ROOT), python=sys.version, torch=torch.__version__,
            transformers=transformers.__version__, cuda=torch.cuda.is_available(),
            available_ram=psutil.virtual_memory().available, commit_available=windows_commit_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            airllm_present=(Path(args.airllm_source)/"air_llm/airllm/utils.py").is_file(),
            native_tier_present=Path(args.native_library).is_file(),
            native_cuda_probe=str(ROOT/"data/integration-20261112/native-cuda-probe.json")),indent=2))
        return
    config = json.loads((model_dir/"config.json").read_text(encoding="utf-8"))
    expected = "qwen3_5_text" if args.model == "qwen9b" else "olmoe"
    if config["model_type"] != expected:
        p.error("--model and checkpoint config disagree")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                        **({"fix_mistral_regex":False} if expected == "qwen3_5_text" else {}))
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    slots = {"sequential":1, "double_buffer":2, "triple_ring_prefetch":3}[args.pipeline]
    plan = make_plan(model_dir, args.weight_format, args.group_size, slots,
        args.ram_gib,args.vram_gib,args.cache_mib,args.policy,ids.shape[1]+args.max_new_tokens)
    if args.action == "plan":
        print(json.dumps(plan.as_dict(),indent=2)); return
    report = dict(status="STARTED", plan=plan.as_dict(), prompt=args.prompt, input_ids=ids.tolist(),
                  weight_format=args.weight_format, policy=args.policy, control=args.control,
                  capabilities=dict(native_c_ram_admission=args.policy!="off", native_colibri_model_math=False,
                    original_airllm_subset_reader=True, dynamic_vram_placement=False, asynchronous_prefetch_cancel=False),
                  q4_quality="EXPERIMENTAL; historical strict BF16 quality gate not passed" if args.weight_format=="q4" else "unquantized BF16 compute; Qwen checkpoint F16, OLMoE checkpoint BF16")
    report["provenance"] = dict(
        model_dir=str(model_dir),
        checkpoint_metadata_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (model_dir/"config.json",model_dir/"tokenizer.json",model_dir/"model.safetensors.index.json") if f.is_file()},
        runtime_sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((ROOT/"src").glob("*.py"))},
        airllm_source=args.airllm_source,native_library=args.native_library if args.policy!="off" else None)
    dest = Path(args.report or ROOT / "data/latest-run.json").resolve()
    dest.parent.mkdir(parents=True,exist_ok=True)
    def save():
        dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    save()
    runtime = state = logits = None
    guards = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("These full-model runtime adapters require CUDA")
        plan.validate(available_vram=torch.cuda.mem_get_info()[0])
        torch.cuda.set_per_process_memory_fraction(min(1,args.vram_gib*GIB/torch.cuda.get_device_properties(0).total_memory))
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        if args.model == "qwen9b":
            from src.aircolibri_integrated import AirColibriModel
            from src.upstream_bridge import NativeTierPolicy
            policy = NativeTierPolicy(args.native_library,config["num_hidden_layers"],2,mode=args.policy) if args.policy!="off" else None
            runtime=AirColibriModel(str(model_dir),pipeline_mode=args.pipeline,weight_format=args.weight_format,
                q4_group_size=args.group_size,max_ram_layers=2,max_ram_bytes=plan.ram_cache_bytes,cache_policy="lru",
                q4_cache_dir=args.q4_cache if args.weight_format=="q4" else None,
                checkpoint_manifest=args.checkpoint_manifest if args.weight_format=="q4" else None,
                airllm_source=args.airllm_source,native_policy=policy,control_mode=args.control)
            cache=runtime.weight_manager.ram_cache
        else:
            from src.olmoe_integrated import AirColibriOLMoE
            runtime=AirColibriOLMoE(model_dir,args.airllm_source,
                native_dll=args.native_library if args.policy!="off" else None,
                native_mode=args.policy if args.policy!="off" else "apply",ram_bytes=plan.ram_cache_bytes)
            cache=runtime.store.cache
        report["initialization_seconds"]=time.perf_counter()-start
        active_policy = runtime.native_policy if args.model=="qwen9b" else runtime.store.policy
        if active_policy:
            report["provenance"]["native_manifest"] = active_policy.provenance
        process=psutil.Process()
        def guard(_module,_args):
            if psutil.virtual_memory().available < 512*2**20:
                raise MemoryError("RAM pressure: stopped at layer boundary")
            if process.memory_info().rss > plan.ram_budget_bytes:
                raise MemoryError("Process RSS exceeded declared RAM budget")
        for layer in runtime.model.model.layers:
            guards.append(layer.register_forward_pre_hook(guard,prepend=True))
        generated=ids[0].tolist()
        rows=[]
        for step in range(args.max_new_tokens):
            started=time.perf_counter()
            with torch.no_grad():
                if args.model=="qwen9b":
                    logits,state=runtime._forward_step(ids,state,prefetch_next=step+1<args.max_new_tokens)
                else:
                    logits,state=runtime.forward(ids,state)
            torch.cuda.synchronize()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite logits")
            token=int(logits.argmax(-1).item())
            generated.append(token)
            rows.append(dict(step=step+1,token=token,seconds=time.perf_counter()-started))
            print(runtime.tokenizer.decode([token]),end="",flush=True)
            ids=torch.tensor([[token]])
        print()
        report.update(status="PASS", generated_ids=generated, text=runtime.tokenizer.decode(generated),
            steps=rows,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            ram_cache_peak_bytes=cache.peak_bytes,placement_counts=dict(cache.placement_counts))
        if args.model=="qwen9b":
            report.update(loads=runtime.pipeline.total_loads,computes=runtime.pipeline.total_computes,
                          tail_prefetches=runtime.pipeline.tail_prefetches)
            if runtime.ack_adapter:
                report["lease_event_counts"]=dict(runtime.ack_adapter.event_counts)
            if runtime.native_policy:
                report["native_policy"]=runtime.native_policy.stats
        elif runtime.store.policy:
            report["native_policy"]=runtime.store.policy.stats
    except KeyboardInterrupt:
        report.update(status="CANCELLED",reason="User interrupted at current operation")
    except Exception as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
    finally:
        for handle in guards:
            handle.remove()
        state=logits=None
        if runtime:
            runtime.close()
        gc.collect()
        report["cuda_allocated_after_close"]=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        save()
    print(f"{report['status']}: {dest}")
    if report["status"]!="PASS":
        raise SystemExit(130 if report["status"]=="CANCELLED" else 1)
