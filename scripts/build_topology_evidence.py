"""Record pinned, read-only source anchors for the original-program topology audit.

This records identities/coverage, not a runtime test or a proof that every line
of each referenced file was reviewed. Upstream paths are supplied explicitly;
local absolute source paths are not embedded in the public evidence.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "airllm": "e496ed3c5c8f614df943b9401cf17e156ef7aca0",
    "colibri": "0a5504a1dc64f86fcd0fe7db298b0448c730e7e6",
}
AUTHOR_REPOS = {"airllm": "lyogavin/airllm", "colibri": "JustVugg/colibri"}

# The ranges identify the inspected mechanism; file hashes cover the full file.
ANCHORS = [
    ("A01", "airllm", "AutoModel dispatch", "air_llm/airllm/auto_model.py", [(17, 79)]),
    ("A02", "airllm", "Constructor, meta model and initialization", "air_llm/airllm/airllm_base.py", [(58, 215), (267, 379)]),
    ("A03", "airllm", "CPU loading, device materialization, resident tables", "air_llm/airllm/airllm_base.py", [(430, 458), (600, 632), (673, 747)]),
    ("A04", "airllm", "Layer/expert hooks, prefetch and delegation", "air_llm/airllm/airllm_base.py", [(749, 929)]),
    ("A05", "airllm", "Cleanup, device choice, mmap tables", "air_llm/airllm/utils.py", [(78, 102), (239, 343)]),
    ("A06", "airllm", "Checkpoint preparation and split lifecycle", "air_llm/airllm/utils.py", [(395, 422), (544, 886)]),
    ("A07", "airllm", "Whole-layer versus subset reader", "air_llm/airllm/utils.py", [(425, 460)]),
    ("A08", "airllm", "Platform persister selection", "air_llm/airllm/persist/model_persister.py", None),
    ("A08", "airllm", "Safetensors persistence and marker", "air_llm/airllm/persist/safetensor_model_persister.py", None),
    ("A08", "airllm", "MLX persistence and name mapping", "air_llm/airllm/persist/mlx_model_persister.py", None),
    ("A09", "airllm", "Kimi expert specialization", "air_llm/airllm/airllm_kimi_k3.py", None),
    ("A10", "airllm", "Qwen dense VL and resident vision", "air_llm/airllm/airllm_qwen3_5.py", None),
    ("A11", "airllm", "Qwen4 packed experts and PLE", "air_llm/airllm/airllm_qwen4_exp.py", None),
    ("A12", "airllm", "MLX executor lifecycle", "air_llm/airllm/airllm_llama_mlx.py", [(233, 461)]),
    ("A13", "airllm", "LoRA reload/recompute lifecycle", "air_llm/airllm/airllm_lora.py", [(1, 76), (192, 219), (329, 378), (381, 397)]),
    ("C01", "colibri", "Family dispatch and capability registry", "c/family_registry.py", [(28, 240), (1303, 1885), (1953, 2130)]),
    ("C02", "colibri", "CLI dispatch and serving entry", "c/coli", [(430, 450), (1758, 1795), (3431, 3491)]),
    ("C03", "colibri", "Resource plan outputs and environment handoff", "c/resource_plan.py", [(3700, 3892)]),
    ("C03", "colibri", "Setup discovery and model preparation", "c/setup_flow.py", [(1840, 1860), (2001, 2196), (2367, 2380)]),
    ("C04", "colibri", "Native scoring primitives", "c/tier.h", None),
    ("C05", "colibri", "Gateway engine ownership and shutdown", "c/openai_server.py", [(6770, 6915), (7639, 7680), (10220, 10270), (10360, 10397)]),
    ("C06", "colibri", "OLMoE initialization and backend selection", "c/olmoe.c", [(900, 1062)]),
    ("C06", "colibri", "Native shard reads and mapping boundaries", "c/st.h", [(1340, 1555)]),
    ("C07", "colibri", "OLMoE slot ownership, loading and KV-room shrink", "c/olmoe.c", [(280, 307), (1067, 1175), (1358, 1586)]),
    ("C08", "colibri", "OLMoE route, grouped expert compute and layer loop", "c/olmoe.c", [(1657, 1705), (1790, 1860), (1959, 1991), (2149, 2240)]),
    ("C09", "colibri", "OLMoE speculative worker and process lifetime", "c/olmoe.c", [(2291, 2460), (3342, 3420), (3470, 3559)]),
    ("C10", "colibri", "Common device tier: staging, quiesce, heat and shutdown", "c/vk_tier.c", [(35, 52), (549, 611), (730, 824), (2220, 2248), (2385, 2479), (3268, 3285), (3322, 3349)]),
    ("C10", "colibri", "OLMoE tier source ownership and RAM callback", "c/olmoe.c", [(1178, 1260)]),
    ("C11", "colibri", "OLMoE full-resident CUDA chain", "c/olmoe_cuda_chain.h", [(1, 62), (118, 143), (217, 342), (401, 448), (524, 606)]),
    ("C12", "colibri", "CUDA frame synchronization and shutdown", "c/cuda_chain.cu", [(1573, 1586), (1667, 1685), (2440, 2501)]),
    ("C13", "colibri", "CUDA tier interface and limitations", "c/cu_tier.h", [(1, 65)]),
    ("C13", "colibri", "Compile shared tier over CUDA ops", "c/cu_tier.c", None),
    ("C14", "colibri", "Qwen hybrid layers, dispatch, MTP boundary, Dense residency", "c/qwen36.c", [(5956, 5995), (6495, 6547), (6595, 6605), (6688, 6803), (9620, 9688)]),
    ("C15", "colibri", "GLM per-layer repin scope and emitted-token cadence", "c/colibri.c", [(10889, 10905), (10980, 10995), (11057, 11067), (14355, 14366)]),
    ("C16", "colibri", "Documented optional route substitution versus prefetch", "docs/CACHE_ROUTE.md", None),
    ("C17", "colibri", "OLMoE session/prefix, serving and embedded cleanup", "c/olmoe.c", [(2694, 2835), (2957, 3022), (3121, 3278), (3576, 3605), (3680, 3743)]),
    ("C18", "colibri", "Documented Metal topology and its numerical boundaries", "docs/metal.md", None),
    ("C18", "colibri", "Qwen Metal affine-store initialization", "c/qwen36.c", [(9620, 9637)]),
    ("C19", "colibri", "Documented expert-worker topology", "docs/cluster.md", [(1, 65)]),
    ("C20", "colibri", "Image encode/denoise/decode boundary", "c/qwenimage.c", [(1059, 1105)]),
    ("I01", "aircolibri", "Bridge, local events and decay cadence", "src/upstream_bridge.py", [(57, 130)]),
    ("I01", "aircolibri", "C ABI bridge to upstream tier.h", "src/native/tier_bridge.c", None),
    ("I02", "aircolibri", "Current OLMoE HF integration", "src/olmoe_integrated.py", None),
    ("I03", "aircolibri", "Shared arenas and GPU residency", "src/shared_resources.py", None),
    ("I04", "aircolibri", "Existing project Ring and fences", "src/gpu_ring_pipeline.py", None),
    ("I05", "aircolibri", "Current startup resource estimates", "src/resource_plan.py", None),
]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def build(airllm, colibri):
    roots = {"airllm": Path(airllm), "colibri": Path(colibri), "aircolibri": ROOT}
    sources = {}
    for key in REVISIONS:
        root = roots[key]
        revision = git(root, "rev-parse", "HEAD")
        if revision != REVISIONS[key]:
            raise ValueError(f"{key}: pinned revision differs: {revision}")
        clean = not git(root, "status", "--porcelain")
        if not clean:
            raise ValueError(f"{key}: source worktree is modified")
        sources[key] = dict(revision=revision, author_repo=AUTHOR_REPOS[key],
                            checkout_origin=git(root, "remote", "get-url", "origin"),
                            worktree_clean=clean,
                            author_commit_verified_via_github_api="2026-11-13",
                            author_commit_url=f"https://github.com/{AUTHOR_REPOS[key]}/commit/{revision}")
    records = []
    for ident, project, purpose, rel, ranges in ANCHORS:
        path = roots[project] / rel
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        ranges = ranges or [(1, len(lines))]
        for first, last in ranges:
            if not 1 <= first <= last <= len(lines):
                raise ValueError(f"Invalid anchor {ident} {rel}: {first}:{last}/{len(lines)}")
        record = dict(id=ident, project=project, purpose=purpose, path=rel,
                      sha256=hashlib.sha256(raw).hexdigest(), lines=len(lines),
                      ranges=ranges, coverage="selected mechanism; not exhaustive branch validation")
        if project != "aircolibri":
            record["revision"] = REVISIONS[project]
            record["url"] = f"https://github.com/{AUTHOR_REPOS[project]}/blob/{REVISIONS[project]}/{rel}"
        else:
            record["base_commit"] = git(ROOT, "rev-parse", "HEAD")
            record["source_state"] = "working tree after compact-slot validation"
        records.append(record)
    tree = ast.parse((roots["colibri"] / "c/family_registry.py").read_text(encoding="utf-8"))
    families = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FamilyDescriptor":
            item = dict(line=node.lineno, modality="text", has_cli_adapter=False, has_gateway_adapter=False)
            for kw in node.keywords:
                if kw.arg in ("id", "engine_artifact", "model_types", "modality", "planner_id", "has_cli_adapter", "has_gateway_adapter"):
                    item[kw.arg] = ast.literal_eval(kw.value)
            families.append(item)
    if len(families) != 13:
        raise ValueError("Family inventory changed")
    report = dict(kind="original-program-topology-source-audit-v1", date="2026-11-13",
                  status="STATIC_SOURCE_MAP", runtime_validation=False, sources=sources,
                  anchors=records, colibri_families=families,
                  coverage={
                      "airllm": "end-to-end core HF hooks, preparation, resident/mmap, Kimi experts; MLX and LoRA lifecycle",
                      "colibri": "all 13 family dispatch entries; detailed OLMoE CPU/CUDA and common GPU tier; Qwen state, GLM repin, gateway lifetime",
                      "boundary_only": ["Metal and cluster documentation/interface", "image/decision dispatch", "MTP", "other family kernel graphs", "segment/edge APIs"],
                  }, limitations=[
                      "Not a run of each original engine, model, backend or optional path.",
                      "Full-file hashes identify source; they do not imply full semantic review.",
                      "Local pinned revisions, not a claim of latest remote HEAD.",
                      "Current AirColibri wrapper semantics are separate from original program behavior.",
                      "Source-weight files were not loaded or changed by this audit.",
                  ])
    output = ROOT / "docs/evidence/2026-11-13/original-topology.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# Указатель исходников: две оригинальные топологии\n",
          "Статический source audit, 13 ноября 2026. [Начать с карты](ORIGINALS_TOPOLOGY_RU.md).\n",
          "SHA-256 относится ко всему файлу и подтверждает его идентичность, а не полное чтение/исполнение каждой ветки. Диапазоны указывают механизмы, на которые опирается карта.\n",
          "Локальные checkout origin ведут на форки DIIIUA. Наличие обоих закреплённых commits в авторских репозиториях проверено GitHub API; новые upstream версии не подменяют этот срез.\n"]
    for key, source in sources.items():
        md.append(f"- **{key}**: [{source['revision']}]({source['author_commit_url']}); worktree clean.\n")
    for record in records:
        md += [f"\n## {record['id']} — {record['purpose']}\n", f"`{record['project']}/{record['path']}`\n"]
        if "url" in record:
            links = [f"[L{a}–L{b}]({record['url']}#L{a}-L{b})" for a, b in record["ranges"]]
        else:
            links = [f"[{record['path']}]({ '../' + record['path']}) (working-tree snapshot)"]
        md += [" · ".join(links) + "\n", f"\nSHA-256: `{record['sha256']}`\n"]
    md += ["\n## Пределы покрытия\n", "Полная карта lifecycle не равна аудиту каждого kernel. Внешняя HF-математика, остальные family-specific native kernels, все formats/backends, MTP verification и cluster recovery требуют собственных проверок. Список всех 13 dispatch entries и границ — в [JSON](evidence/2026-11-13/original-topology.json).\n"]
    (ROOT / "docs/ORIGINALS_SOURCE_INDEX.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"anchors": len(records), "unique_files": len({(r['project'], r['path']) for r in records}),
                      "families": len(families), "status": report["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airllm", required=True)
    parser.add_argument("--colibri", required=True)
    args = parser.parse_args()
    build(args.airllm, args.colibri)
