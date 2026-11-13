"""Byte-based startup plan shared by the dense and MoE entrypoints.

Reservations are conservative estimates, reported separately from measured
allocator peaks. Native tier admission can only use the reserved cache budget.
"""
from dataclasses import asdict, dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import struct

import psutil

MIB = 2**20
GIB = 2**30
DTYPE_BYTES = {"F16":2, "BF16":2, "F32":4, "F64":8, "I8":1, "U8":1,
               "I16":2, "U16":2, "I32":4, "U32":4, "I64":8, "U64":8, "BOOL":1}


def tensor_headers(path):
    path = Path(path)
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 64*MIB:
            raise ValueError("Unexpectedly large safetensors header")
        records = json.loads(f.read(length))
    data_size = path.stat().st_size - 8 - length
    result = {}
    for key, meta in records.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"Invalid tensor offsets: {key}")
        result[key] = meta
    return result


def windows_commit_available():
    if os.name != "nt":
        return None
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (n, ctypes.c_uint32) for n in ("HandleCount", "ProcessCount", "ThreadCount")]
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    fn = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    fn.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_uint32]
    fn.restype = ctypes.c_int
    if not fn(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return (info.CommitLimit-info.CommitTotal)*info.PageSize


@dataclass
class RuntimePlan:
    model_type: str
    layers: int
    source_weight_bytes: int
    largest_shard_bytes: int
    ram_budget_bytes: int
    vram_budget_bytes: int
    ram_cache_bytes: int
    pinned_bytes: int
    gpu_weight_bytes: int
    host_transient_bytes: int
    runtime_reserve_bytes: int
    gpu_state_scratch_reserve_bytes: int
    ram_estimate_bytes: int
    vram_estimate_bytes: int
    policy: str
    weight_placement: str = "dense-ring"

    def as_dict(self):
        return asdict(self)

    def validate(self, available_vram=None, available_ram=None, commit_available=None):
        if self.ram_estimate_bytes > self.ram_budget_bytes:
            raise MemoryError("Planned RAM reservations exceed --ram-gib; reduce cache/slots or increase budget")
        if self.vram_estimate_bytes > self.vram_budget_bytes:
            raise MemoryError("Planned VRAM reservations exceed --vram-gib")
        available_ram = psutil.virtual_memory().available if available_ram is None else available_ram
        if available_ram < max(int(2.8*GIB), self.ram_estimate_bytes-self.runtime_reserve_bytes):
            raise MemoryError("Insufficient available RAM for the plan (minimum 2.8 GiB)")
        if available_vram is not None and self.vram_estimate_bytes > available_vram:
            raise MemoryError("Available VRAM is smaller than the plan")
        commit_available = windows_commit_available() if commit_available is None else commit_available
        if commit_available is not None and commit_available < self.largest_shard_bytes+self.ram_estimate_bytes:
            raise MemoryError("Insufficient Windows commit headroom for the largest mapped shard and RAM reservations")


def make_plan(model_dir, weight_format="bf16", group_size=64, slots=1,
              ram_gib=4, vram_gib=5, cache_mib=256, policy="off", sequence_tokens=64,
              moe_placement="streamed"):
    folder = Path(model_dir).resolve()
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    kind = config["model_type"]
    if kind not in ("qwen3_5_text", "olmoe"):
        raise ValueError("Supported architectures: qwen3_5_text and olmoe")
    if min(ram_gib, vram_gib) <= 0 or cache_mib < 0 or not 1 <= slots <= 4 or sequence_tokens < 1:
        raise ValueError("Invalid resource budget")
    files = list(folder.glob("model-*.safetensors"))
    if not files:
        raise ValueError("No checkpoint shards")
    total = sum(p.stat().st_size for p in files)
    largest = max(p.stat().st_size for p in files)
    runtime = 1024*MIB
    # Include logits, attention workspace and persistent generation state.
    logits_bytes = config["vocab_size"]*sequence_tokens*2
    attention_bytes = config["num_attention_heads"]*sequence_tokens*sequence_tokens*4
    scratch = max(512*MIB, logits_bytes*2+attention_bytes+128*MIB)
    cache = int(cache_mib*MIB)
    if kind == "qwen3_5_text":
        if weight_format not in ("bf16", "q4") or group_size not in (16,32,64):
            raise ValueError("Unsupported Qwen weight format/group")
        from src.bounded_memory_manager import SlotArenaLayout
        from src.quantization_q4 import is_q4_quantizable, q4_tensor_specs
        import torch
        specs = {}
        max_payload = 0
        max_source_layer = 0
        for idx in range(config["num_hidden_layers"]):
            max_source_layer = max(max_source_layer, (folder / f"model-{idx:05d}.safetensors").stat().st_size)
            layer_specs = {}
            for name, meta in tensor_headers(folder / f"model-{idx:05d}.safetensors").items():
                prefix = f"model.layers.{idx}."
                if not name.startswith(prefix):
                    raise ValueError("Unexpected layer key")
                local, shape = name[len(prefix):], tuple(meta["shape"])
                entries = q4_tensor_specs(local, shape, group_size, torch.bfloat16) if weight_format == "q4" and is_q4_quantizable(local, shape, group_size) else {local:(shape,torch.bfloat16)}
                for key, spec in entries.items():
                    if key in specs and specs[key] != spec:
                        raise ValueError("Heterogeneous layer layout is not supported")
                    specs[key] = spec
                layer_specs.update(entries)
            payload = sum(math.prod(shape)*torch.empty((),dtype=dtype).element_size() for shape,dtype in layer_specs.values())
            max_payload = max(max_payload,payload)
        arena = SlotArenaLayout.from_templates(specs, dtype=torch.bfloat16)
        pinned = arena.slot_stride*slots
        globals_ = tensor_headers(folder / "model-globals.safetensors")
        permanent = sum(math.prod(v["shape"])*2 for k,v in globals_.items() if k in ("lm_head.weight","model.norm.weight"))
        gpu_weights = pinned+permanent
        host_transient = max_payload+64*MIB+(max_source_layer if weight_format == "q4" else 0)
        # Q4 cache payload + metadata and original header mappings are temporary;
        # globals has the largest commit mapping even though embedding is lazy.
    else:
        if weight_format != "bf16":
            raise ValueError("OLMoE adapter supports original BF16 weights")
        if moe_placement not in ("streamed", "resident"):
            raise ValueError("Unsupported OLMoE weight placement")
        pinned = 0
        dense_bytes = 0
        globals_bytes = 0
        layer_bytes = {}
        templates = {}
        for f in files:
            for name,meta in tensor_headers(f).items():
                if ".experts." not in name:
                    size = math.prod(meta["shape"])*2
                    dense_bytes += size
                    if name.startswith("model.layers."):
                        _, _, index, local = name.split(".", 3)
                        layer_bytes[index] = layer_bytes.get(index, 0)+size
                        key = "dense."+local
                        shape = tuple(meta["shape"])
                        if key in templates and templates[key] != shape:
                            raise ValueError("Heterogeneous OLMoE Dense layouts are unsupported")
                        templates[key] = shape
                    else:
                        globals_bytes += size
        expert_bytes = 3*config["hidden_size"]*config["intermediate_size"]*2
        if moe_placement == "streamed":
            from src.bounded_memory_manager import SlotArenaLayout
            import torch
            templates.update({"expert.gate_up_proj":(2*config["intermediate_size"], config["hidden_size"]),
                              "expert.down_proj":(config["hidden_size"], config["intermediate_size"])})
            pinned = 2*SlotArenaLayout.from_templates(templates, dtype=torch.bfloat16).slot_stride
            gpu_weights = globals_bytes+pinned
        else:
            gpu_weights = dense_bytes+expert_bytes*2
        host_transient = max(expert_bytes*2,config["vocab_size"]*config["hidden_size"]*2)
        host_transient = max(host_transient, max(layer_bytes.values(),default=0))
    ram_estimate = runtime+cache+pinned+host_transient
    return RuntimePlan(kind,config["num_hidden_layers"],total,largest,int(ram_gib*GIB),int(vram_gib*GIB),
        cache,pinned,gpu_weights,host_transient,runtime,scratch,ram_estimate,gpu_weights+scratch,policy,
        "shared-dense-expert-ring" if kind=="olmoe" and moe_placement=="streamed" else "resident-dense-expert-copies" if kind=="olmoe" else "dense-ring")
