"""Compare completed separate-process runs; exact logits, routes and KV state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def compare(folder, reference, candidates):
    def read(case):
        result=json.loads((folder/(case+".json")).read_text(encoding="utf-8"))
        if result["status"] != "PASS":
            raise ValueError(f"{case} is not a completed PASS run")
        values=torch.load(folder/(case+"-logits.pt"),map_location="cpu",weights_only=True)
        if len(values)!=len(result["runs"]):
            raise ValueError("Logit/run count mismatch")
        return result,values
    base,expected=read(reference)
    if base["mode"]!="reference":
        raise ValueError("Reference must use the independent harness")
    results=[]
    for case in candidates:
        candidate,actual=read(case)
        if len(actual)!=len(expected):
            raise ValueError("Different workload lengths")
        same_inputs=[r["input_ids"] for r in base["runs"]]==[r["input_ids"] for r in candidate["runs"]]
        logits_equal=all(a.dtype==b.dtype and a.shape==b.shape
                         and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
                         for a,b in zip(expected,actual))
        state_equal=[r["state"] for r in base["runs"]]==[r["state"] for r in candidate["runs"]]
        routers_equal=base["router_choices"]==candidate["router_choices"]
        oracle_source_matches=base["oracle_source_sha256"]==candidate["oracle_source_sha256"]
        identities={key:(base[key]==candidate[key] if key in base and key in candidate else None)
            for key in ("checkpoint_metadata_sha256","checkpoint_shards","hf_olmoe_source_sha256")}
        row=dict(case=case,status="PASS" if same_inputs and logits_equal and state_equal and routers_equal and oracle_source_matches
                 and all(value is not False for value in identities.values()) else "FAIL",
            identical_inputs=same_inputs,exact_logits=logits_equal,exact_kv_state_hashes=state_equal,
            bitwise_logits=logits_equal,source_identity_matches=identities,
            exact_router_choices=routers_equal,forward_count=len(actual),logit_values=sum(x.numel() for x in actual),
            max_abs_logit_error=max(float((a.float()-b.float()).abs().max()) for a,b in zip(expected,actual)),
            peak_vram=candidate["peak_vram"],peak_reserved=candidate["peak_reserved"],
            allocator_budget_gib=candidate["vram_budget_gib"],allocated_after_close=candidate["allocated_after_close"],
            oracle_source_matches=oracle_source_matches,
            native=candidate["native"],native_policy=candidate.get("native_policy"),
            airllm_io_backend=candidate.get("airllm_io_backend","upstream-default (historical mmap)"),
            runtime_sources=candidate["runtime_sources"])
        if "resources" in candidate:
            row["resources"]={k:v for k,v in candidate["resources"].items() if k not in ("events","leases")}
        results.append(row)
    return dict(status="PASS" if all(x["status"]=="PASS" for x in results) else "FAIL",reference=reference,
        reference_forward_count=len(expected),reference_peak_vram=base["peak_vram"],
        reference_source_sha256=base["oracle_source_sha256"],results=results,
        input_artifact_sha256={case:hashlib.sha256((folder/(case+".json")).read_bytes()).hexdigest() for case in [reference,*candidates]},
        workload=base.get("workload","historical-short"),
        limits="Same installed Transformers graph; independent loading/materialization only. Workload-scoped parity; no speed, long-context certification or physical-I/O claim.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--reference",default="reference-v1")
    p.add_argument("--candidates",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.folder,a.reference,a.candidates)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"cases":[{k:v for k,v in r.items() if k not in ("runtime_sources","resources","native_policy")} for r in result["results"]]}))
    raise SystemExit(0 if result["status"]=="PASS" else 1)
