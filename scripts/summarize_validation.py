"""Compare saved complete logits/state, separate correctness from timing."""
import hashlib
import json
from pathlib import Path
import statistics
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/integration-20261112"


def read(name):
    return json.loads((OUT / (name + ".json")).read_text(encoding="utf-8-sig"))


def compare(names):
    rows = [read(n) for n in names]
    assert all(r["status"] == "PASS" for r in rows), names
    logits = [torch.load(OUT / (n+"-logits.pt"), map_location="cpu", weights_only=True) for n in names]
    states = [read(n+"-state") for n in names]
    assert all(len(v) == len(logits[0]) for v in logits)
    exact = all(torch.equal(a,b) for candidate in logits[1:] for a,b in zip(logits[0],candidate))
    same_state = all(v==states[0] for v in states)
    routers = all(r.get("router_choices")==rows[0].get("router_choices") for r in rows)
    result = dict(cases=names,steps_per_case=len(logits[0]),logits_bitwise_equal=exact,
                  state_hashes_equal=same_state,router_choices_equal=routers,
                  max_abs_logit_difference=max(float((a-b).abs().max()) for candidate in logits[1:] for a,b in zip(logits[0],candidate)))
    assert exact and same_state and routers, result
    return result


def main():
    result = {}
    names = [f"moe-bench-{mode}-{i}" for i in range(1,6) for mode in ("off","native")]
    result["moe_benchmark_parity"] = compare(names)
    snapshots=[read(n)["runtime_sources"] for n in names]
    assert all(s==snapshots[0] for s in snapshots)
    result["moe_benchmark_sources_identical"] = True
    result["moe_benchmark"] = {}
    for mode in ("off","native"):
        rows=[read(f"moe-bench-{mode}-{i}") for i in range(1,6)]
        totals=[sum(x["seconds"] for x in r["runs"]) for r in rows]
        result["moe_benchmark"][mode] = dict(forward_total_seconds=totals,
            median_forward_seconds=statistics.median(totals),
            median_decode_seconds=statistics.median(statistics.mean(x["seconds"] for x in r["runs"][1:]) for r in rows),
            expert_reads=[r["expert_reads"] for r in rows],
            airllm_calls=[r["airllm_calls"] for r in rows],
            logical_expert_h2d_bytes=[r["logical_h2d_expert_bytes"] for r in rows],
            peak_vram_bytes=[r["peak_vram"] for r in rows])
    off,native=(result["moe_benchmark"][m] for m in ("off","native"))
    result["moe_benchmark"]["median_forward_time_reduction_percent"] = 100*(1-native["median_forward_seconds"]/off["median_forward_seconds"])
    result["moe_benchmark"]["expert_read_reduction_percent"] = 100*(1-native["expert_reads"][0]/off["expert_reads"][0])
    result["timing_limit"]="5 fresh-process pairs; same Hello prompt, 8 steps; OS cache not flushed or controlled; no physical SSD counters or general speed guarantee"
    groups = {
        "moe_full_expert_reference":["moe-all-experts","moe-sparse-off","moe-sparse-native-v2"],
        "dense_q4":["off-sequential-v2","observe-sequential","native-sequential-v2","native-ring"],
        "dense_bf16":["bf16-off","bf16-native"],
        "moe_multitoken":["moe-multitoken-off","moe-multitoken-native","moe-multitoken-shadow"],
    }
    for key,cases in groups.items():
        if all((OUT/(c+"-logits.pt")).exists() for c in cases):
            result[key] = compare(cases)
    (OUT/"validation-summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
