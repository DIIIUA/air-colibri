"""Publish compact, qualified Graphify evidence without copying upstream graphs.

Full AST graphs stay local in data/shared-moe-20261113/topology-graphify. The
crosswalk below is an analyst-assigned functional analogy, not an extracted
cross-repository call edge. Node existence is checked before publication.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/shared-moe-20261113"
BASE = DATA / "topology-graphify"
A = "air_llm_airllm_"
AB = A + "airllm_base_airllmbasemodel_"
ROWS = [
    ("F01", "resolve model", [A+"auto_model_automodel_from_pretrained"], ["c_family_registry_resolve_model", "c_coli_engine_for"]),
    ("F02", "address weights", [A+"utils_split_and_save_layers"], ["c_st_st_init"]),
    ("F03", "read requested payload", [A+"utils_load_layer_subset"], ["c_olmoe_load_expert_merged"]),
    ("F05", "materialize for compute", [AB+"move_layer_to_device"], ["c_olmoe_cuda_chain_occ_start", "c_vk_tier_vkt_note"]),
    ("F06", "readiness boundary", [AB+"pre_hook"], ["c_olmoe_expert_get"]),
    ("F07", "execute model forward", [AB+"forward"], ["c_olmoe_step", "c_olmoe_cuda_chain_occ_forward"]),
    ("F08", "expert branch boundary, not same router implementation", [AB+"setup_expert_streaming"], ["c_olmoe_moe_route_row"]),
    ("F09", "stateful forward boundary", [AB+"forward"], ["c_olmoe_step"]),
    ("F10", "end consumer lifetime", [AB+"post_hook"], ["c_olmoe_expert_put", "c_vk_tier_quiesce"]),
    ("F11", "future work", [AB+"next_streamed_idx", AB+"pre_hook"], ["c_olmoe_pilot_prefetch"]),
    ("F14", "token-loop entry", [AB+"generate"], ["c_olmoe_serve_one_turn"]),
    ("F16", "cleanup boundary, not equivalent shutdown", [A+"utils_clean_memory"], ["c_vk_tier_vkt_shutdown"]),
]


def build():
    graphs, stats = {}, {}
    for name in ("airllm", "colibri"):
        path = BASE / name / "graphify-out/graph.json"
        raw = path.read_bytes()
        g = json.loads(raw)
        graphs[name] = g
        diag = json.loads((DATA / f"graphify-{name}-diagnostic.json").read_text(encoding="utf-8-sig"))
        summary = {k: v for k, v in diag["summary"].items() if k != "input_path"}
        stats[name] = dict(nodes=len(g["nodes"]), edges=len(g["edges"]),
                           input_tokens=g["input_tokens"], output_tokens=g["output_tokens"],
                           graph_sha256=hashlib.sha256(raw).hexdigest(),
                           relations=dict(Counter(e.get("relation") for e in g["edges"])),
                           diagnostic=summary,
                           local_graph=str(path.relative_to(ROOT)).replace('\\', '/'))
    indices = {name: {n['id']: n for n in g['nodes']} for name, g in graphs.items()}
    crosswalk = []
    for ident, function, air, coli in ROWS:
        record = dict(id=ident, function=function, relationship="analyst_functional_analogy_not_call_edge")
        for name, ids in (("airllm", air), ("colibri", coli)):
            record[name] = [indices[name][i] for i in ids]
        crosswalk.append(record)
    false_edges = [e for e in graphs['colibri']['edges'] if
                   e['source'] == 'c_olmoe_expert_get' and
                   e['target'] == 'c_tests_test_qwen36_tier_hold_sleep_ms']
    if len(false_edges) != 1:
        raise ValueError("Expected reviewed Graphify false-edge example changed")
    selected = {}
    for name, sources in {
        'airllm': [AB+'pre_hook', AB+'load_streamed_layer'],
        'colibri': ['c_olmoe_step', 'c_olmoe_expert_get', 'c_olmoe_cuda_chain_occ_start'],
    }.items():
        selected[name] = [e for e in graphs[name]['edges'] if e['source'] in sources and e.get('relation') == 'calls']
    report = dict(kind="graphify-topology-crosscheck-v1", date="2026-11-13", tool="graphifyy 0.9.18",
                  mode="code-only, no-cluster, max-workers=2, separate outputs", graphs=stats,
                  function_crosswalk=crosswalk, selected_extracted_calls=selected,
                  excluded_false_edge=dict(raw=false_edges[0], reason="olmoe.c L87-91 defines platform macro sleep_ms as Sleep/usleep, not a call into test code"),
                  limitations=[
                      "Graph includes tests, tooling, configuration and all conditional compile branches, not just active inference.",
                      "AST labels EXTRACTED do not prove correct global symbol resolution.",
                      "Common C header/source names can collapse to a declaration node; verify the implementation separately.",
                      "Dynamic Python hooks and C macro/function-pointer dispatch need source review.",
                      "2 AirLLM and 93 Colibri detected source files produced zero nodes; logs preserve warnings.",
                      "Dangling edges can denote external dependencies; counts are not a completeness certificate.",
                      "No full graph merge or simple-graph reduction was used to infer functional equivalence.",
                  ])
    dest = ROOT / 'docs/evidence/2026-11-13/graphify-topology.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{'nodes':s['nodes'],'edges':s['edges']} for name,s in stats.items()}))


if __name__ == '__main__':
    build()
