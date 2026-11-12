"""Квантизация W4A16 (Group-wise Asymmetric uint4, g=64) поверх неразрывных 2 MiB-арен.

Хранение линейных матриц (0 B per-group LUT):
  - qweight: [N, K // 2]  torch.uint8    (2 ниббла на байт, lo = [0..15], hi = [0..15])
  - scales:  [N, K // 64] torch.bfloat16
  - zeros:   [N, K // 64] torch.bfloat16 (w = q * scales + zeros)

Распаковка на GPU выполняется побайтовым bitwise in-place в единый
тайловый буфер scratch <= 12 MiB (0 B int64, 0 B gather/LUT), чтобы полный
пик (включая сам scratch) оставался минимальным (~14 MiB).
"""

from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

Q4_GROUP_SIZE = 64
Q4_WEIGHT_SUFFIX = ".weight"
Q4_SCRATCH_ELEMS = 6 * 1024 * 1024  # 12.0 MiB в bfloat16 (один общий scratch на весь конвейер)
_Q4_SMALL_TILE_ROWS = 1024          # 8.0 MiB тайл при K = 4096


def is_q4_quantizable(
    short_name: str,
    shape: Tuple[int, ...],
    group_size: int = Q4_GROUP_SIZE,
    include_lm_head: bool = False,
) -> bool:
    """Определяет, переводится ли тензор в packed Q4."""
    if len(shape) != 2 or shape[1] <= 0 or shape[1] % group_size != 0 or not short_name.endswith(Q4_WEIGHT_SUFFIX):
        return False
    if short_name == "lm_head.weight":
        return include_lm_head
    return (
        short_name.startswith("mlp.")
        or short_name.startswith("linear_attn.")
        or short_name.startswith("self_attn.")
    )


def q4_subkey_names(short_name: str) -> Tuple[str, str, str]:
    """Возвращает тройку имён (qweight, scales, zeros) для матрицы *.weight."""
    prefix = short_name[: -len(Q4_WEIGHT_SUFFIX)] if short_name.endswith(Q4_WEIGHT_SUFFIX) else short_name
    return f"{prefix}.qweight", f"{prefix}.scales", f"{prefix}.zeros"


def q4_tensor_specs(
    short_name: str,
    shape: Tuple[int, ...],
    group_size: int = Q4_GROUP_SIZE,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    """Возвращает формы и dtype субядер Q4 для размещения в SlotArenaLayout."""
    out_features, in_features = shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} must be divisible by group_size={group_size} for {short_name}")
    groups = in_features // group_size
    q_k, s_k, z_k = q4_subkey_names(short_name)
    return {
        q_k: ((out_features, in_features // 2), torch.uint8),
        s_k: ((out_features, groups), compute_dtype),
        z_k: ((out_features, groups), compute_dtype),
    }


def _quantize_chunk_q4(
    w_chunk: torch.Tensor,
    group_size: int,
    compute_dtype: torch.dtype,
    work_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = w_chunk.shape
    groups = in_features // group_size
    w_work = w_chunk.to(device=work_device, dtype=torch.float32) if w_chunk.device != work_device or w_chunk.dtype != torch.float32 else w_chunk
    wg = w_work.view(out_features, groups, group_size)

    w_min = wg.amin(dim=-1, keepdim=True)
    w_max = wg.amax(dim=-1, keepdim=True)
    zeros_bf16 = w_min.to(compute_dtype)
    scales_bf16 = ((w_max - zeros_bf16.float()) / 15.0).clamp(min=1e-7).to(compute_dtype)

    q = torch.round((wg - zeros_bf16.float()) / scales_bf16.float().clamp(min=1e-7)).clamp(0, 15).to(torch.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous().view(out_features, in_features // 2)
    scales = scales_bf16.squeeze(-1).contiguous()
    zeros = zeros_bf16.squeeze(-1).contiguous()
    return packed, scales, zeros


def quantize_tensor_q4(
    weight: torch.Tensor,
    group_size: int = Q4_GROUP_SIZE,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_device: Optional[torch.device] = None,
    target_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Квантует 2D-матрицу (N, K) в packed uint8 + bf16 scales + bf16 zeros (w = q * s + z)."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")

    dst_device = target_device or weight.device
    work_device = quant_device or (torch.device("cuda") if torch.cuda.is_available() else dst_device)
    groups = in_features // group_size

    # Чанкуем по 2048 строк даже для 12288x4096 и lm_head, чтобы float32-буферы не раздували PeakVRAM (~96 MiB вместо ~580 MiB)
    chunk_rows = 2048
    if out_features <= chunk_rows:
        packed, scales, zeros = _quantize_chunk_q4(weight, group_size, compute_dtype, work_device)
        if work_device != dst_device:
            return packed.to(dst_device), scales.to(dst_device), zeros.to(dst_device)
        return packed, scales, zeros

    packed_out = torch.empty((out_features, in_features // 2), dtype=torch.uint8, device=dst_device)
    scales_out = torch.empty((out_features, groups), dtype=compute_dtype, device=dst_device)
    zeros_out = torch.empty((out_features, groups), dtype=compute_dtype, device=dst_device)

    for r0 in range(0, out_features, chunk_rows):
        r1 = min(r0 + chunk_rows, out_features)
        p_c, s_c, z_c = _quantize_chunk_q4(weight[r0:r1], group_size, compute_dtype, work_device)
        packed_out[r0:r1].copy_(p_c.to(dst_device, non_blocking=False) if p_c.device != dst_device else p_c)
        scales_out[r0:r1].copy_(s_c.to(dst_device, non_blocking=False) if s_c.device != dst_device else s_c)
        zeros_out[r0:r1].copy_(z_c.to(dst_device, non_blocking=False) if z_c.device != dst_device else z_c)
        del p_c, s_c, z_c

    return packed_out, scales_out, zeros_out


def _compute_tile_rows(out_features: int, in_features: int, scratch_elems: int) -> int:
    if in_features <= 4096:
        max_rows = min(_Q4_SMALL_TILE_ROWS, scratch_elems // in_features)
    else:
        max_rows = scratch_elems // in_features
    tile = max(64, min(out_features, max_rows))
    return tile


def _unpack_q4_tile_into(
    qweight_slice: torch.Tensor,
    scales_slice: torch.Tensor,
    zeros_slice: torch.Tensor,
    dst_weight: torch.Tensor,
    group_size: int = Q4_GROUP_SIZE,
) -> None:
    """Распаковывает тайл uint8 нибблов прямо in-place в dst_weight без int64 и gather/LUT."""
    rows, packed_cols = qweight_slice.shape
    in_features = packed_cols * 2
    groups = in_features // group_size
    wpair = dst_weight.view(rows, packed_cols, 2)
    wpair[..., 0].copy_(qweight_slice & 0x0F)
    wpair[..., 1].copy_(qweight_slice >> 4)
    dst_weight.view(rows, groups, group_size).mul_(scales_slice.unsqueeze(-1)).add_(zeros_slice.unsqueeze(-1))


def dequantize_q4_tensor(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = Q4_GROUP_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Деквантует (qweight, scales, zeros) в (N, K) dtype через побайтовый bitwise in-place."""
    out_features, packed_cols = qweight.shape
    in_features = packed_cols * 2

    if out is None:
        out = torch.empty((out_features, in_features), dtype=dtype, device=qweight.device)
    else:
        out = out[: out_features * in_features].view(out_features, in_features)

    tile = _compute_tile_rows(out_features, in_features, Q4_SCRATCH_ELEMS)
    for r0 in range(0, out_features, tile):
        r1 = min(r0 + tile, out_features)
        _unpack_q4_tile_into(
            qweight[r0:r1],
            scales[r0:r1],
            zeros[r0:r1],
            out[r0:r1],
            group_size=group_size,
        )
    return out


def q4_tiled_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = Q4_GROUP_SIZE,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Выполняет F.linear в W4A16 тайлами через переиспользуемый scratch <= 12 MiB (без int64/gather)."""
    out_features, packed_cols = qweight.shape
    in_features = packed_cols * 2
    dtype = scales.dtype

    scratch_elems = scratch.numel() if scratch is not None else Q4_SCRATCH_ELEMS
    tile = _compute_tile_rows(out_features, in_features, scratch_elems)
    work_scratch = scratch if scratch is not None else torch.empty(
        min(out_features, tile) * in_features,
        dtype=dtype,
        device=qweight.device,
    )

    x_compute = x if x.dtype == dtype else x.to(dtype)

    if out_features <= tile:
        w_view = work_scratch[: out_features * in_features].view(out_features, in_features)
        _unpack_q4_tile_into(qweight, scales, zeros, w_view, group_size=group_size)
        return F.linear(x_compute, w_view, bias)

    chunks = []
    for r0 in range(0, out_features, tile):
        r1 = min(r0 + tile, out_features)
        rows = r1 - r0
        w_view = work_scratch[: rows * in_features].view(rows, in_features)
        _unpack_q4_tile_into(
            qweight[r0:r1],
            scales[r0:r1],
            zeros[r0:r1],
            w_view,
            group_size=group_size,
        )
        b_slice = bias[r0:r1] if bias is not None else None
        chunks.append(F.linear(x_compute, w_view, b_slice))
    return torch.cat(chunks, dim=-1)


class ColibriQ4Linear(nn.Module):
    """Линейный слой W4A16 (g=64 asym) с zero-copy привязкой буферов слота и общим scratch."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        has_bias: bool = False,
        group_size: int = Q4_GROUP_SIZE,
        compute_dtype: torch.dtype = torch.bfloat16,
        scratch_provider: Optional[Callable[[torch.device], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self._scratch_provider = scratch_provider
        groups = in_features // group_size
        buf_dev = device or torch.device("meta")

        self.register_buffer("qweight", torch.empty((out_features, in_features // 2), dtype=torch.uint8, device=buf_dev))
        self.register_buffer("scales", torch.empty((out_features, groups), dtype=compute_dtype, device=buf_dev))
        self.register_buffer("zeros", torch.empty((out_features, groups), dtype=compute_dtype, device=buf_dev))
        if has_bias:
            self.register_buffer("bias", torch.empty((out_features,), dtype=compute_dtype, device=buf_dev))
        else:
            self.register_buffer("bias", None)

    def bind_q4_tensors(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Zero-copy привязка квантованных тензоров из VRAM-слота или резидентного хранилища."""
        self._buffers["qweight"] = qweight
        self._buffers["scales"] = scales
        self._buffers["zeros"] = zeros
        if bias is not None:
            self._buffers["bias"] = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qweight = self._buffers["qweight"]
        scales = self._buffers["scales"]
        zeros = self._buffers["zeros"]
        bias = self._buffers["bias"]
        scratch = self._scratch_provider(qweight.device) if self._scratch_provider is not None else None
        return q4_tiled_linear(
            x,
            qweight,
            scales,
            zeros,
            bias=bias,
            group_size=self.group_size,
            scratch=scratch,
        )


def replace_linear_with_q4(
    model: nn.Module,
    quantize_lm_head: bool = False,
    scratch_provider: Optional[Callable[[torch.device], torch.Tensor]] = None,
    group_size: int = Q4_GROUP_SIZE,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Заменяет квантуемые nn.Linear декодера (и опционально lm_head) на ColibriQ4Linear."""
    replaced = 0

    def _replace_in_layer(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and child.in_features % group_size == 0:
                if (
                    full.startswith("mlp.")
                    or full.startswith("linear_attn.")
                    or full.startswith("self_attn.")
                ):
                    q_mod = ColibriQ4Linear(
                        out_features=child.out_features,
                        in_features=child.in_features,
                        has_bias=(child.bias is not None),
                        group_size=group_size,
                        compute_dtype=compute_dtype,
                        scratch_provider=scratch_provider,
                        device=child.weight.device,
                    )
                    setattr(parent, child_name, q_mod)
                    replaced += 1
                    continue
            _replace_in_layer(child, full)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_mod in model.model.layers:
            _replace_in_layer(layer_mod)

    if quantize_lm_head and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        old_head = model.lm_head
        if old_head.in_features % group_size == 0:
            model.lm_head = ColibriQ4Linear(
                out_features=old_head.out_features,
                in_features=old_head.in_features,
                has_bias=(old_head.bias is not None),
                group_size=group_size,
                compute_dtype=compute_dtype,
                scratch_provider=scratch_provider,
                device=old_head.weight.device,
            )
            replaced += 1

    return replaced
