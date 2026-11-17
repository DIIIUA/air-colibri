"""Decode stored HF quantization into the universal engine's weight contract.

Universal Q4/Q6 packing reads output-row tiles directly from the checkpoint.
Direct-weight consumers retain a full-projection decode fallback; the checkpoint
itself is never converted or rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import torch

from aircolibri_cli.model_source import (
    HFShardedSource, UnsupportedModelError, _ShapeOnlySlice, raw_text_config_dict,
)

TILE_ELEMENTS = 1024 * 1024


def decode_fp8(weight, scales, block_size, device=None):
    """E4M3/E5M2 values multiplied by the corresponding block scale."""
    if weight.ndim != 2 or len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("FP8 requires a matrix and two positive block dimensions")
    n, k = weight.shape
    br, bc = block_size
    if tuple(scales.shape) != (math.ceil(n/br), math.ceil(k/bc)):
        raise ValueError("FP8 scale grid does not cover the weight matrix")
    dst = device or weight.device
    out = torch.empty((n, k), dtype=torch.float32, device=dst)
    scale = scales.to(device=dst, dtype=torch.float32)
    cols = torch.arange(k, device=dst) // bc
    rows = max(1, TILE_ELEMENTS // k)
    for start in range(0, n, rows):
        end = min(n, start + rows)
        grid = scale[(torch.arange(start, end, device=dst)//br)[:, None], cols[None, :]]
        out[start:end] = weight[start:end].to(device=dst, dtype=torch.float32) * grid
    return out


def decode_int4(qweight, qzeros, scales, *, method, group_size, g_idx=None,
                zero_offset=1, device=None):
    """AWQ GEMM and GPTQ have different packing axes and zero encodings."""
    if method not in {"awq", "gptq"} or qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("Expected AWQ/GPTQ packed matrices")
    n = scales.shape[1]
    k = qweight.shape[0] * (8 if method == "gptq" else 1)
    expected = (k // 8, n) if method == "gptq" else (k, n // 8)
    group_size = k if group_size == -1 else group_size
    if (n % 8 or k % 8 or tuple(qweight.shape) != expected or group_size <= 0
            or tuple(qzeros.shape) != (scales.shape[0], n // 8)
            or scales.shape[0] != math.ceil(k/group_size)):
        raise ValueError("Inconsistent packed INT4 geometry")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("INT4 words must be int32")
    dst = device or qweight.device
    out = torch.empty((n, k), dtype=torch.float32, device=dst)
    groups = (torch.arange(k, device=dst) // group_size if g_idx is None
              else g_idx.to(device=dst, dtype=torch.long))
    if groups.shape != (k,) or bool(((groups < 0) | (groups >= scales.shape[0])).any()):
        raise ValueError("GPTQ g_idx contains an invalid group index")
    if method == "awq" and g_idx is not None:
        raise ValueError("AWQ GEMM does not use GPTQ activation-order indices")
    source_cols = torch.arange(k, device=dst)
    awq_positions = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dst)
    # Multiples of eight keep every output tile aligned to packed words.
    rows = max(8, (TILE_ELEMENTS // k // 8) * 8)
    for start in range(0, n, rows):
        end = min(n, start + rows)
        outputs = torch.arange(end-start, device=dst)
        s = scales[:, start:end].to(device=dst, dtype=torch.float32)
        z = qzeros[:, start//8:end//8].to(dst)
        shifts = (awq_positions[outputs % 8] if method == "awq" else outputs % 8) * 4
        zeros = (z[:, outputs//8] >> shifts[None, :]) & 15
        if method == "gptq":
            zeros = zeros + zero_offset
            words = qweight[:, start:end].to(dst)
            q = (words[source_cols//8, :] >> ((source_cols%8)*4)[:, None]) & 15
        else:
            words = qweight[:, start//8:end//8].to(dst)
            q = (words[:, outputs//8] >> shifts[None, :]) & 15
        out[start:end] = ((q.float() - zeros[groups].float()) * s[groups]).T
    return out


class _QuantizedLayerView:
    def __init__(self, source, layer_idx):
        self.source = source
        self.entries = source.layer_entries(layer_idx)

    def keys(self):
        return list(self.entries)

    def get_slice(self, key):
        return _ShapeOnlySlice(self.source.tensor_shape(self.entries[key]))

    def get_tensor(self, key, device=None):
        return self.source.read_decoded(self.entries[key], device)


class HFQuantizedSource(HFShardedSource):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.quant_config = dict(raw_text_config_dict(model_dir)["quantization_config"])
        method = self.quant_config["quant_method"]
        if method == "fp8":
            if self.quant_config.get("fmt", "e4m3") not in {"e4m3", "e5m2"}:
                raise UnsupportedModelError("Unsupported FP8 value encoding")
            block = self.quant_config.get("weight_block_size")
            if not isinstance(block, (tuple, list)) or len(block) != 2 or min(block) <= 0:
                raise UnsupportedModelError("FP8 source requires explicit weight_block_size")
        else:
            if int(self.quant_config.get("bits", 0)) != 4:
                raise UnsupportedModelError("HF packed source currently requires 4-bit AWQ/GPTQ")
            if method == "awq" and str(self.quant_config.get("version", "gemm")).lower() != "gemm":
                raise UnsupportedModelError("AWQ source requires GEMM packing")
            if method == "awq" and not self.quant_config.get("zero_point", True):
                raise UnsupportedModelError("AWQ source requires explicit zero points")
            if method == "gptq" and self.quant_config.get("checkpoint_format", "gptq") not in {"gptq", "gptq_v2"}:
                raise UnsupportedModelError("Unsupported GPTQ checkpoint packing")
        self._quantized = {}
        self._tile_lock = threading.Lock()
        self._tile_stats = dict(decoded_tiles=0, max_decoded_elements=0)
        for layer_idx, entries in self._layers.items():
            updated = dict(entries)
            for name, entry in entries.items():
                meta = self._tensor_meta(*entry)
                if method == "fp8" and meta["dtype"] in {"F8_E4M3", "F8_E5M2"}:
                    sibling = name.removesuffix(".weight") + ".weight_scale_inv"
                    if sibling not in entries:
                        raise UnsupportedModelError(f"FP8 weight {name} misses its scale grid")
                    shape = meta["shape"]
                    grid = self._tensor_meta(*entries[sibling])["shape"]
                    if len(shape) != 2 or tuple(grid) != tuple(math.ceil(d/b) for d,b in zip(shape, block)):
                        raise UnsupportedModelError(f"FP8 scale grid does not cover {name}")
                    self._quantized[entry] = {"scale": entries[sibling], "shape": tuple(meta["shape"])}
                    updated.pop(sibling)
                elif method in {"awq", "gptq"} and name.endswith(".qweight"):
                    prefix = name.removesuffix("qweight")
                    required = ("qzeros", "scales", "g_idx") if method == "gptq" else ("qzeros", "scales")
                    if any(prefix+k not in entries for k in required):
                        raise UnsupportedModelError(f"Packed weight {name} misses {required}")
                    parts = {k: entries[prefix+k] for k in required}
                    scale_shape = self._tensor_meta(*parts["scales"])["shape"]
                    if len(scale_shape) != 2 or len(meta["shape"]) != 2:
                        raise UnsupportedModelError(f"Packed INT4 requires matrices: {name}")
                    parts["shape"] = (scale_shape[1], meta["shape"][0] * (8 if method == "gptq" else 1))
                    n, k = parts["shape"]
                    group = int(self.quant_config["group_size"])
                    group = k if group == -1 else group
                    expected = (k//8, n) if method == "gptq" else (k, n//8)
                    zeros = self._tensor_meta(*parts["qzeros"])
                    if (n % 8 or k % 8 or group <= 0 or tuple(meta["shape"]) != expected
                            or tuple(zeros["shape"]) != (scale_shape[0], n//8)
                            or scale_shape[0] != math.ceil(k/group)
                            or meta["dtype"] != "I32" or zeros["dtype"] != "I32"):
                        raise UnsupportedModelError(f"Inconsistent packed INT4 geometry: {name}")
                    self._quantized[entry] = parts
                    for k in ("qweight", *required):
                        updated.pop(prefix+k)
                    updated[prefix+"weight"] = entry
            self._layers[layer_idx] = updated
        # Globals use the existing bounded embedding/head path. Do not silently
        # cast a packed global matrix without its scales.
        if any(self._tensor_meta(*e)["dtype"].startswith("F8_") or e[1].endswith(".qweight")
               for e in self._globals.values()):
            raise UnsupportedModelError("Quantized global embedding/head requires a sliced global adapter")

    def tensor_shape(self, entry):
        spec = self._quantized.get(entry)
        return spec["shape"] if spec else super().tensor_shape(entry)

    def layer_identity(self, layer_idx):
        identity = super().layer_identity(layer_idx)
        # Include the codec/config and every companion's file range and stamp.
        parts = []
        for entry in self.layer_entries(layer_idx).values():
            for name, companion in self._quantized.get(entry, {}).items():
                if name == "shape":
                    continue
                from pathlib import Path
                st = Path(companion[0]).stat()
                parts.append((companion, self._tensor_meta(*companion), st.st_size, st.st_mtime_ns))
        identity["sha256"] = hashlib.sha256(json.dumps([identity["sha256"], self.quant_config, parts], sort_keys=True).encode()).hexdigest()
        for name, entry in self.layer_entries(layer_idx).items():
            identity["tensors"][name]["shape"] = list(self.tensor_shape(entry))
        return identity

    def open_layer(self, layer_idx, dtype=None):
        return _QuantizedLayerView(self, layer_idx)

    def weight_row_alignment(self, entry):
        return 8 if entry in self._quantized and self.quant_config["quant_method"] in {"awq", "gptq"} else 1

    def projection_io_stats(self):
        result = super().projection_io_stats()
        with self._tile_lock:
            result.update(self._tile_stats)
        return result

    def read_weight_rows(self, entry, r0, r1, *, transpose=False, device=None):
        spec = self._quantized.get(entry)
        if spec is None:
            return super().read_weight_rows(entry, r0, r1, transpose=transpose, device=device)
        if transpose:
            raise UnsupportedModelError("Packed HF Conv1D requires a separate orientation adapter")
        n, k = spec["shape"]
        if not 0 <= r0 <= r1 <= n:
            raise ValueError("Decoded weight rows are outside the matrix")
        if r0 == r1:
            return torch.empty((0, k), dtype=torch.float32, device=device or "cpu")
        reader = self._pread_reader
        method = self.quant_config["quant_method"]
        if method == "fp8":
            br, bc = self.quant_config["weight_block_size"]
            weight = reader.read_matrix_slice(*entry, r0, r1)
            s0, s1 = r0//br, math.ceil(r1/br)
            scale = reader.read_matrix_slice(*spec["scale"], s0, s1).to(device=device, dtype=torch.float32)
            rows = torch.arange(r0, r1, device=scale.device)//br - s0
            cols = torch.arange(k, device=scale.device)//bc
            tile = weight.to(device=scale.device, dtype=torch.float32) * scale[rows[:, None], cols[None, :]]
        else:
            # Cover whole words even for an unaligned caller. The packer asks
            # for aligned rows, so its temporary allocation stays within budget.
            a, b = (r0//8)*8, min(n, math.ceil(r1/8)*8)
            stored_rows = self._tensor_meta(*entry)["shape"][0]
            weight = reader.read_matrix_slice(*entry, 0, stored_rows,
                                              a//8 if method == "awq" else a,
                                              b//8 if method == "awq" else b)
            group_count = self._tensor_meta(*spec["scales"])["shape"][0]
            scale = reader.read_matrix_slice(*spec["scales"], 0, group_count, a, b)
            zeros = reader.read_matrix_slice(*spec["qzeros"], 0, group_count, a//8, b//8)
            groups = reader.read_tensor(*spec["g_idx"]) if "g_idx" in spec else None
            tile = decode_int4(weight, zeros, scale, method=method,
                               group_size=int(self.quant_config["group_size"]), g_idx=groups,
                               zero_offset=0 if self.quant_config.get("checkpoint_format") == "gptq_v2" else 1,
                               device=device)
        with self._tile_lock:
            self._tile_stats["decoded_tiles"] += 1
            self._tile_stats["max_decoded_elements"] = max(self._tile_stats["max_decoded_elements"], tile.numel())
        return tile if method == "fp8" else tile[r0-a:r1-a].contiguous()

    def read_decoded(self, entry, device=None):
        weight = self._pread_reader.read_tensor(*entry)
        spec = self._quantized.get(entry)
        if spec is None:
            return weight.to(device) if device is not None else weight
        method = self.quant_config["quant_method"]
        if method == "fp8":
            scale = self._pread_reader.read_tensor(*spec["scale"])
            return decode_fp8(weight, scale, self.quant_config["weight_block_size"], device)
        parts = {key: self._pread_reader.read_tensor(*spec[key]) for key in ("qzeros", "scales")}
        groups = self._pread_reader.read_tensor(*spec["g_idx"]) if "g_idx" in spec else None
        return decode_int4(weight, **parts, method=method, group_size=int(self.quant_config["group_size"]),
                           g_idx=groups, zero_offset=0 if self.quant_config.get("checkpoint_format") == "gptq_v2" else 1,
                           device=device)
