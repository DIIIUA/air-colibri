"""Bounded Colibri affine Q4/Q6 storage shared by streaming and compute.

This is a runtime representation, not a GGUF block format. Source quantization
is decoded by the source adapter before conversion. Routers and parameters
consumed directly by architecture-specific kernels are deliberately preserved.
"""
from __future__ import annotations

import inspect
import math
import torch
from torch import nn
from torch.nn import functional as F

PACKED_FIELDS = ("packed_lo", "packed_hi", "qscale", "qzero")
TILE_ELEMENTS = 1024 * 1024


def packed_specs(name, shape, bits, group_size=64):
    n, k = shape
    prefix = name.removesuffix("weight")
    result = {
        prefix + "packed_lo": ((n, k // 2), torch.uint8),
        prefix + "qscale": ((n, k // group_size), torch.bfloat16),
        prefix + "qzero": ((n, k // group_size), torch.bfloat16),
    }
    if bits == 6:
        result[prefix + "packed_hi"] = ((n, k // 4), torch.uint8)
    return result


@torch.no_grad()
def pack_weight(weight, bits, group_size=64, device=None):
    """Quantize in bounded row tiles; never allocate a full FP32 weight copy."""
    dst = torch.device(device) if device is not None else weight.device
    return pack_weight_rows(weight.shape, lambda start, end: weight[start:end],
                            bits, group_size, dst)


@torch.no_grad()
def pack_weight_rows(shape, read_rows, bits, group_size=64, device="cpu", tile_elements=None,
                     row_alignment=1):
    """Pack successive source tiles directly into the final Q4/Q6 allocation.

    read_rows(start, end) returns [end-start, in_features]. No full uncompressed
    source matrix is needed. The final packed matrix still must fit the caller's
    resident/streaming slot budget.
    """
    if bits not in (4, 6) or group_size not in (16, 32, 64):
        raise ValueError("Colibri supports Q4/Q6 with groups of 16, 32 or 64")
    n, k = shape
    if n < 0 or k <= 0 or k % group_size:
        raise ValueError("Input width must be divisible by group size")
    limit = TILE_ELEMENTS if tile_elements is None else int(tile_elements)
    if limit <= 0:
        raise ValueError("tile_elements must be positive")
    alignment = int(row_alignment)
    if alignment <= 0 or limit < min(n, alignment) * k:
        raise ValueError("tile_elements cannot hold one aligned group of weight rows")
    dst = torch.device(device)
    result = {key: torch.empty(shape, dtype=dt, device=dst)
              for key, (shape, dt) in packed_specs("weight", (n, k), bits, group_size).items()}
    rows = max(alignment, (limit // k // alignment) * alignment)
    for start in range(0, n, rows):
        end = min(n, start + rows)
        tile = read_rows(start, end)
        if tuple(tile.shape) != (end-start, k):
            raise ValueError(f"Weight reader returned {tuple(tile.shape)} for rows {start}:{end}, expected {(end-start, k)}")
        w = tile.to(device=dst, dtype=torch.float32).reshape(-1, k // group_size, group_size)
        zero = w.amin(-1).to(torch.bfloat16)
        # CPU and CUDA FP32 division can land on different sides of a BF16
        # midpoint. Keep rounding decisions in FP64 so placement does not alter
        # the packed weights. Conversion still operates on a bounded row tile.
        scale = ((w.amax(-1).double() - zero.double()) / ((1 << bits) - 1)).clamp_min_(1e-8).to(torch.bfloat16)
        q = ((w.double() - zero.double().unsqueeze(-1)) / scale.double().unsqueeze(-1)).round_().clamp_(0, (1 << bits) - 1).to(torch.uint8).reshape(-1, k)
        result["packed_lo"][start:end] = (q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)
        if bits == 6:
            hi = q >> 4
            result["packed_hi"][start:end] = hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)
            del hi
        result["qscale"][start:end] = scale
        result["qzero"][start:end] = zero
        del tile, w, zero, scale, q
    return result


def unpack_weight(packed, group_size=64, dtype=torch.bfloat16):
    """Decode a caller-bounded tile (also used as the numerical reference)."""
    lo = packed["packed_lo"]
    n, half = lo.shape
    q = torch.stack((lo & 15, lo >> 4), dim=-1).reshape(n, half * 2)
    hi = packed.get("packed_hi")
    if hi is not None:
        upper = torch.stack(tuple((hi >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1).reshape_as(q)
        q = q | (upper << 4)
    w = q.float().reshape(n, -1, group_size)
    return (w * packed["qscale"].float().unsqueeze(-1) + packed["qzero"].float().unsqueeze(-1)).reshape(n, -1).to(dtype)


class PackedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, bias=False, group_size=64):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        for name, (shape, dt) in packed_specs("weight", (out_features, in_features), bits, group_size).items():
            self.register_buffer(name, torch.empty(shape, dtype=dt, device="meta"))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device="meta"), requires_grad=False) if bias else None
        self.fused_calls = 0
        self.tiled_calls = 0

    def packed(self):
        return {name: getattr(self, name) for name in PACKED_FIELDS if hasattr(self, name)}

    def forward(self, x):
        from src.colibri_nvrtc import can_use_fused_dense_q4, fused_dense_linear_q4, fused_dense_linear_q6
        if x.dtype == torch.bfloat16 and can_use_fused_dense_q4(x, self.packed_lo, self.qscale, self.qzero, self.group_size):
            self.fused_calls += 1
            if self.bits == 6:
                return fused_dense_linear_q6(x, self.packed_lo, self.packed_hi, self.qscale, self.qzero, self.bias, self.group_size)
            return fused_dense_linear_q4(x, self.packed_lo, self.qscale, self.qzero, self.bias, self.group_size)
        self.tiled_calls += 1
        flat = x.reshape(-1, self.in_features)
        out = torch.empty((flat.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        rows = max(1, TILE_ELEMENTS // self.in_features)
        packed = self.packed()
        for start in range(0, self.out_features, rows):
            end = min(self.out_features, start + rows)
            w = unpack_weight({key: value[start:end] for key, value in packed.items()}, self.group_size, x.dtype)
            out[:, start:end] = F.linear(flat, w, None if self.bias is None else self.bias[start:end])
        return out.view(*x.shape[:-1], self.out_features)


def projection_map(model, layer_shapes, bits, group_size=64, replace=False):
    """Select actual module calls, never infer execution semantics from shapes alone."""
    from aircolibri_cli.model_source import resolve_decoder_layers
    from transformers.pytorch_utils import Conv1D
    result = {}
    for idx, layer in enumerate(resolve_decoder_layers(model)):
        specs = {}
        shapes = {key.removeprefix(f"model.layers.{idx}."): value for key, value in layer_shapes[idx].items()}
        modules = dict(layer.named_modules())
        for name, mod in list(modules.items()):
            if type(mod) not in (nn.Linear, Conv1D):
                continue
            key = name + ".weight"
            shape = shapes.get(key)
            if shape is None or len(shape) != 2 or tuple(mod.weight.shape) != tuple(shape):
                continue
            # Keep routing logits exact and bypass fused recurrent-weight consumers.
            if any(part in ("gate", "router", "router_proj") or part.endswith("_gate") for part in name.split(".")):
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = modules[parent_name]
            try:
                direct_weight_use = f".{leaf}.weight" in inspect.getsource(type(parent))
            except (OSError, TypeError):
                direct_weight_use = True
            if direct_weight_use:
                continue
            transpose = type(mod) is Conv1D
            canonical = tuple(reversed(shape)) if transpose else tuple(shape)
            n, k = canonical
            if k % group_size or math.prod(canonical) < 4096:
                continue
            specs[key] = {"shape": canonical, "transpose": transpose}
            if replace:
                setattr(parent, leaf, PackedLinear(k, n, bits, mod.bias is not None, group_size))
        result[idx] = specs
    return result


def projection_stats(model):
    mods = [m for m in model.modules() if isinstance(m, PackedLinear)]
    return {"modules": len(mods), "bits": sorted({m.bits for m in mods}),
            "fused_calls": sum(m.fused_calls for m in mods),
            "tiled_calls": sum(m.tiled_calls for m in mods)}
