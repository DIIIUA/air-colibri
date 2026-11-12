"""Inject real load/cancellation failures, close twice and reopen in one process."""
import gc
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import psutil
import torch
from src.aircolibri_integrated import AirColibriModel
from src.olmoe_integrated import AirColibriOLMoE
from scripts.verify_integration import state_hashes

OUT=ROOT/"data/integration-20261112"
AIRLLM=r"D:\Antigravity Lair\10_ML_AND_TENSORS\air-colibri\upstream_airllm"


def dense():
    return AirColibriModel(str(ROOT/"models/deepseek-v4-9b-colibri"),pipeline_mode="triple_ring_prefetch",
        weight_format="q4",max_ram_layers=2,max_ram_bytes=256*2**20,cache_policy="lru",control_mode="enforce",
        q4_cache_dir=str(ROOT/"data/q4-g64-cache"),checkpoint_manifest=str(OUT/"trusted-checkpoint-manifest.json"),airllm_source=AIRLLM)


def moe():
    return AirColibriOLMoE(ROOT/"models/OLMoE-1B-7B-0125-Instruct",AIRLLM,native_dll=ROOT/"data/native/colibri_tier.dll")


def step(runtime,ids,state=None,final=False):
    with torch.no_grad():
        if isinstance(runtime,AirColibriModel):
            return runtime._forward_step(ids,state,prefetch_next=not final)
        return runtime.forward(ids,state)


def main():
    if psutil.virtual_memory().available<2.8*2**30:
        raise MemoryError("Keep 2.8 GiB startup gate")
    torch.set_num_threads(4)
    rows=[]
    for kind,factory,reference in [("dense",dense,"off-sequential-v2"),("moe",moe,"moe-sparse-off")]:
        runtime=factory()
        try:
            ids=runtime.tokenizer("Hello",return_tensors="pt")["input_ids"]
            if kind=="dense":
                original=runtime.weight_manager.load_layer_to_pinned_slot
                def broken(layer_idx,*args,**kwargs):
                    if layer_idx==1:
                        raise OSError("Injected initial prefetch read failure")
                    return original(layer_idx,*args,**kwargs)
                runtime.weight_manager.load_layer_to_pinned_slot=broken
                error=OSError
            else:
                def interrupt(module,args):
                    raise KeyboardInterrupt("Injected cancellation after earlier CUDA consumers")
                handle=runtime.model.model.layers[3].register_forward_pre_hook(interrupt)
                error=KeyboardInterrupt
            try:
                step(runtime,ids)
            except error:
                pass
            else:
                raise AssertionError("Failure injection did not fire")
            try:
                step(runtime,ids)
            except RuntimeError as exc:
                assert "failed" in str(exc)
            else:
                raise AssertionError("Failed runtime accepted reuse")
        finally:
            runtime.close()
            runtime.close()
        runtime=None
        if kind=="dense":
            original=None
        gc.collect()
        after_failure=torch.cuda.memory_allocated()
        assert after_failure<=16*2**20, after_failure
        runtime=factory()
        state=logits=None
        try:
            ids=runtime.tokenizer("Hello",return_tensors="pt")["input_ids"]
            expected=torch.load(OUT/(reference+"-logits.pt"),map_location="cpu",weights_only=True)
            expected_states=json.loads((OUT/(reference+"-state.json")).read_text())
            for i in range(2):
                logits,state=step(runtime,ids,state,final=i==1)
                torch.cuda.synchronize()
                assert torch.equal(logits.cpu(),expected[i])
                assert state_hashes(state)==expected_states[i]
                ids=torch.tensor([[int(logits.argmax(-1).item())]])
        finally:
            state=logits=None
            runtime.close()
        runtime=None
        gc.collect()
        after_reopen=torch.cuda.memory_allocated()
        assert after_reopen<=16*2**20, after_reopen
        rows.append(dict(model=kind,status="PASS",injected="read failure" if kind=="dense" else "KeyboardInterrupt",
                         reuse_rejected=True,close_twice=True,reopened_exact_logits_and_state=True,
                         allocated_after_failure=after_failure,allocated_after_reopen=after_reopen))
        (OUT/"lifecycle-validation.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps(rows[-1]),flush=True)


if __name__=="__main__": main()
