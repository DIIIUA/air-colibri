"""Reuse the existing CrossLayerPlanMap with REAL safetensors metadata.

No tensors are loaded, no new SSD reader or storage scheme is implemented.
Do not call CrossLayerPlanMap.trigger_replan as if it moved physical weights:
its current implementation only changes metadata.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re

from safetensors import safe_open
from src.adaptive_streaming import (ComponentType, CrossLayerPlanMap, ResidencyTier)

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "BOOL": 1,
}
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.")


def component_for_tensor(local: str) -> str:
    found = EXPERT_RE.match(local)
    if found:
        return f"expert:{found.group(1)}"
    if local.startswith("linear_attn."):
        return "linear_attn"
    if local.startswith("self_attn."):
        return "self_attn"
    if local.startswith("mlp."):
        return "mlp"
    if "norm" in local:
        return "norm"
    return "other"


def component_for_module(local: str) -> str:
    """Classify an actually executed module, NEVER imaginary expert IDs."""
    found = EXPERT_RE.match(local + ".")
    if found:
        return f"expert:{found.group(1)}"
    if local.startswith("linear_attn."):
        return "linear_attn"
    if local.startswith("self_attn."):
        return "self_attn"
    if local.startswith("mlp."):
        return "mlp"
    if "norm" in local:
        return "norm"
    return "other"


def component_type(component: str) -> ComponentType:
    if component.startswith("expert:"):
        return ComponentType.ROUTED_EXPERT
    return {
        "linear_attn": ComponentType.LINEAR_ATTN,
        "self_attn": ComponentType.ATTENTION,
        "mlp": ComponentType.SHARED_MLP,
        "norm": ComponentType.NORM,
        "other": ComponentType.SHARED_MLP,
    }[component]


class ExistingCrossLayerCatalog:
    """Only actual layer IDs, tensor keys/shapes/dtypes from local checkpoint."""
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir).resolve()
        self.config = json.loads((self.model_dir / "config.json").read_text())
        if self.config.get("model_type") != "qwen3_5_text":
            raise ValueError("This proof targets qwen3_5_text only")
        self.num_layers = int(self.config["num_hidden_layers"])
        self.layer_types = self.config["layer_types"]
        if len(self.layer_types) != self.num_layers:
            raise ValueError("layer_types/count mismatch")
        self.entries: dict[int, dict[str, dict]] = {}
        self.keys: dict[int, dict[str, list[str]]] = {}
        self.actual_tensor_count = 0
        self.logical_source_bytes = 0
        self.map = CrossLayerPlanMap(self.num_layers)
        self._scan_headers()

    def _scan_headers(self):
        for index in range(self.num_layers):
            path = self.model_dir / f"model-{index:05d}.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Missing real layer {index}: {path}")
            prefix = f"model.layers.{index}."
            grouped: dict[str, dict] = defaultdict(
                lambda: {"keys": [], "source_bytes": 0, "tensors": 0})
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for name in f.keys():
                    if not name.startswith(prefix):
                        raise ValueError(f"Unexpected key: {name}")
                    local = name[len(prefix):]
                    group = component_for_tensor(local)
                    slice_ = f.get_slice(name)
                    dtype = slice_.get_dtype()
                    if dtype not in DTYPE_BYTES:
                        raise ValueError(f"Unsupported dtype {dtype}: {name}")
                    size = DTYPE_BYTES[dtype]
                    for dim in slice_.get_shape():
                        size *= dim
                    grouped[group]["keys"].append(name)
                    grouped[group]["source_bytes"] += size
                    grouped[group]["tensors"] += 1
                    self.logical_source_bytes += size
                    self.actual_tensor_count += 1
            if not grouped:
                raise ValueError(f"No checkpoint tensors in layer {index}")
            expected = ("self_attn" if self.layer_types[index] == "full_attention"
                        else "linear_attn")
            if expected not in grouped or "mlp" not in grouped:
                raise ValueError(f"Layer {index} lacks real {expected}/mlp keys")
            if ("self_attn" if expected == "linear_attn" else "linear_attn") in grouped:
                raise ValueError(f"Both attention families present in layer {index}")
            self.entries[index] = dict(grouped)
            self.keys[index] = {k: sorted(v["keys"]) for k, v in grouped.items()}
            for group, stats in sorted(grouped.items()):
                self.map.register_component(
                    index, group, component_type(group), stats["keys"],
                    stats["source_bytes"], ResidencyTier.DISK_STREAM)

    def record_actual_access(self, layer, component, step):
        if component not in self.entries[layer]:
            raise ValueError(f"Unknown real component {layer}/{component}")
        self.map.record_access(layer, component, step)

    def summary(self):
        return {
            "source": str(self.model_dir),
            "model_type": self.config["model_type"],
            "layers": self.num_layers,
            "actual_tensor_count": self.actual_tensor_count,
            "logical_source_bytes": self.logical_source_bytes,
            "routed_expert_components": sum(
                k.startswith("expert:") for v in self.entries.values() for k in v),
            "per_layer": {
                str(i): {k: {
                    "source_bytes": v["source_bytes"],
                    "tensors": v["tensors"], "actual_keys": self.keys[i][k]}
                    for k, v in groups.items()}
                for i, groups in self.entries.items()},
        }

    def plan_snapshot(self):
        """Existing map state WITHOUT mutating actual residency or making I/O."""
        return {
            "layers": self.map.num_layers,
            "step_counter": self.map.step_counter,
            "replan_count": self.map.replan_count,
            "observed": [
                {"layer": layer, "component": component,
                 "access_count": meta.access_count,
                 "last_step_accessed": meta.last_step_accessed,
                 "metadata_tier": meta.current_tier.value,
                 "source_bytes": meta.nbytes}
                for layer, groups in self.map.matrix.items()
                for component, meta in groups.items()],
            "physical_residency_changed": False,
        }
