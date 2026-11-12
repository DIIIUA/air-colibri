"""Use ORIGINAL CrossLayerPlanMap replan in a throw-away clone only.

The historical planner ranks *number of components*, not their exact byte
sizes; the wrapper additionally checks source byte budgets, but MUST NOT claim
VRAM residency or actual I/O was changed. It emits advisory PlacementIntents.
"""
from __future__ import annotations
import copy
from collections import Counter
from dataclasses import asdict

from src.catalog_bridge import ExistingCrossLayerCatalog
from src.control_contract import TraceContract


def shadow_replan(catalog: ExistingCrossLayerCatalog,
                  trace: TraceContract, vram_budget_bytes: int,
                  pinned_ram_budget_bytes: int,
                  slots_vram: int = 8,
                  slots_ram: int = 24):
    if vram_budget_bytes < 0 or pinned_ram_budget_bytes < 0:
        raise ValueError("Negative byte budgets")
    if trace.current_step is not None:
        raise ValueError("Do not replan inside an incomplete forward")
    original = {
        (i, k): v.current_tier
        for i, groups in catalog.map.matrix.items()
        for k, v in groups.items()
    }
    clone = copy.deepcopy(catalog.map)
    historical_result = clone.trigger_replan(
        trigger_reason="shadow:completed_real_forward",
        vram_slot_budget=slots_vram,
        ram_slot_budget=slots_ram)
    allocation = {"vram_stand": 0, "ram_pinned": 0}
    quota = {"vram_stand": vram_budget_bytes,
             "ram_pinned": pinned_ram_budget_bytes}
    target_counts = Counter()
    proposals = []
    rejected = []
    # Deterministic ranking from original CrossLayerPlanMap.
    by_score = sorted(
        (v for groups in clone.matrix.values() for v in groups.values()),
        key=lambda c: (-c.predicted_score, c.layer_idx, c.component_id))
    for comp in by_score:
        intended = comp.current_tier.value
        if intended not in quota:
            continue
        estimated_bytes = comp.nbytes  # actual checkpoint metadata, conservative
        if allocation[intended] + estimated_bytes > quota[intended]:
            rejected.append({
                "layer": comp.layer_idx, "component": comp.component_id,
                "target": intended, "source_bytes": estimated_bytes,
                "reason": "model-source-byte-budget-exceeded"})
            continue
        proposal = trace.suggest(comp.layer_idx, comp.component_id,
                                 intended, estimated_bytes,
                                 "original CrossLayerPlanMap shadow replan")
        allocation[intended] += estimated_bytes
        proposals.append(asdict(proposal))
        target_counts[intended] += 1
    current = {
        (i, k): v.current_tier
        for i, groups in catalog.map.matrix.items()
        for k, v in groups.items()
    }
    assert current == original, "Original map tiers MUST stay unchanged"
    assert all(x["advisory"] and not x["dispatched"] for x in proposals)
    return {
        "source_planner": "src.adaptive_streaming.CrossLayerPlanMap.trigger_replan",
        "shadow_result": historical_result,
        "real_model_components": sum(len(x) for x in catalog.entries.values()),
        "model_source_bytes_used_for_admission": True,
        "vram_proposal_bytes": allocation["vram_stand"],
        "pinned_ram_proposal_bytes": allocation["ram_pinned"],
        "vram_budget_bytes": vram_budget_bytes,
        "pinned_ram_budget_bytes": pinned_ram_budget_bytes,
        "accepted_intents": proposals,
        "rejected_over_byte_budget": rejected,
        "counts": dict(target_counts),
        "original_map_mutated": False,
        "actual_backend_schedule_changed": False,
        "actual_h2d_bytes_saved": None,
        "actual_ssd_bytes_saved": None,
        "physical_residency_assumed_from_metadata": False,
    }
