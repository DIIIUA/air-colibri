"""Self-contained NVRTC + CUDA Driver API runtime for Colibri fused W4A16 / W6A16 kernels.

Compiles `src/native/colibri_fused_quant.cu` once via PyTorch's bundled NVRTC
(`nvrtc64_*.dll` on Windows / `libnvrtc.so*` on Linux), caches PTX keyed by
source SHA-256 and target SM capability under `data/native/ptx-cache/`, and
dispatches kernels directly onto `torch.cuda.current_stream(device)` via
`nvcuda.dll` (`cuLaunchKernel`) with zero unpack-scratch allocations.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import threading
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
CU_SOURCE_PATH = ROOT / "src" / "native" / "colibri_fused_quant.cu"
PTX_CACHE_DIR = ROOT / "data" / "native" / "ptx-cache"

_KERNEL_NAMES = (
    b"routed_gate_up_swiglu_q4",
    b"routed_down_q4",
    b"routed_gate_up_swiglu_q6",
    b"routed_down_q6",
    b"reduce_k_to_bf16",
    b"f32_to_bf16_vec",
    b"dense_gemv_q4_bf16",
)

_LOCK = threading.Lock()
_NVRTC = None
_NVCUDA = None
_DEVICE_MODULES: dict[int, dict[bytes, ctypes.c_void_p]] = {}


def _load_driver_and_nvrtc():
    global _NVRTC, _NVCUDA
    if _NVCUDA is not None and _NVRTC is not None:
        return _NVCUDA, _NVRTC

    torch_lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt":
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
        nvcuda = ctypes.WinDLL("nvcuda.dll")
        candidates = sorted(torch_lib.glob("nvrtc64_*.dll"))
        if not candidates:
            raise RuntimeError(f"Bundled NVRTC DLL not found in {torch_lib}")
        nvrtc = ctypes.WinDLL(str(candidates[0]))
    else:
        nvcuda = ctypes.CDLL("libcuda.so.1")
        candidates = sorted(torch_lib.glob("libnvrtc.so*"))
        nvrtc = ctypes.CDLL(str(candidates[0])) if candidates else ctypes.CDLL("libnvrtc.so")

    # Explicit 64-bit pointer signatures (critical on 64-bit Windows)
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetPTX.restype = ctypes.c_int
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

    nvcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    nvcuda.cuModuleLoadData.restype = ctypes.c_int
    nvcuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    nvcuda.cuModuleGetFunction.restype = ctypes.c_int
    nvcuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,  # f
        ctypes.c_uint,    # gridDimX
        ctypes.c_uint,    # gridDimY
        ctypes.c_uint,    # gridDimZ
        ctypes.c_uint,    # blockDimX
        ctypes.c_uint,    # blockDimY
        ctypes.c_uint,    # blockDimZ
        ctypes.c_uint,    # sharedMemBytes
        ctypes.c_void_p,  # hStream
        ctypes.POINTER(ctypes.c_void_p),  # kernelParams
        ctypes.POINTER(ctypes.c_void_p),  # extra
    ]
    nvcuda.cuLaunchKernel.restype = ctypes.c_int

    _NVCUDA = nvcuda
    _NVRTC = nvrtc
    return _NVCUDA, _NVRTC


def _compile_or_load_ptx(major: int, minor: int, nvrtc) -> bytes:
    src_bytes = CU_SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"colibri_fused_quant_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        data = cache_file.read_bytes()
        if data.endswith(b"\x00"):
            return data
        return data + b"\x00"

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"colibri_fused_quant.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed with error {rc}")
    try:
        arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        opts = (ctypes.c_char_p * 2)(arch_flag, b"--use_fast_math")
        rc = nvrtc.nvrtcCompileProgram(prog, 2, opts)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"NVRTC compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}")
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        rc = nvrtc.nvrtcGetPTX(prog, ptx_buf)
        if rc != 0:
            raise RuntimeError(f"nvrtcGetPTX failed with error {rc}")
        raw_ptx = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(raw_ptx)
    os.replace(tmp, cache_file)
    return raw_ptx


def get_kernels(device: torch.device) -> tuple[object, dict[bytes, ctypes.c_void_p]]:
    dev_idx = 0 if device.index is None else int(device.index)
    nvcuda, nvrtc = _load_driver_and_nvrtc()
    cached = _DEVICE_MODULES.get(dev_idx)
    if cached is not None:
        return nvcuda, cached

    with _LOCK:
        cached = _DEVICE_MODULES.get(dev_idx)
        if cached is not None:
            return nvcuda, cached
        with torch.cuda.device(dev_idx):
            # Ensure PyTorch's primary context on dev_idx is initialized and current
            _ = torch.empty(1, device=f"cuda:{dev_idx}")
            major, minor = torch.cuda.get_device_capability(dev_idx)
            ptx = _compile_or_load_ptx(major, minor, nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData failed with CUresult={rc}")
            funcs: dict[bytes, ctypes.c_void_p] = {"_module": mod}
            for name in _KERNEL_NAMES:
                fn = ctypes.c_void_p()
                rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name)
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({name.decode()}) failed with CUresult={rc}")
                funcs[name] = fn
            _DEVICE_MODULES[dev_idx] = funcs
            return nvcuda, funcs


def _check_launch(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel({name}) failed with CUresult={rc}")


def fused_routed_moe_q4(
    x_bf16: torch.Tensor,          # (1, H) bf16 contiguous
    topk_idx: torch.Tensor,        # (K_SUB,) int64 contiguous on CUDA
    topk_weights: torch.Tensor,    # (K_SUB,) bf16 contiguous on CUDA
    gu_q: torch.Tensor,            # (E, 2*I, H//2) uint8
    gu_s: torch.Tensor,            # (E, 2*I, H//GS) bf16
    gu_z: torch.Tensor,            # (E, 2*I, H//GS) bf16
    dn_q: torch.Tensor,            # (E, H, I//2) uint8
    dn_s: torch.Tensor,            # (E, H, I//GS) bf16
    dn_z: torch.Tensor,            # (E, H, I//GS) bf16
    inter_buf: torch.Tensor,       # (>=K_SUB, I) float32
    part_buf: torch.Tensor,        # (>=K_SUB, H) float32
    group_size: int,
    reduce: bool = True,
) -> torch.Tensor:
    """Execute single-token routed W4A16 MoE directly in registers with zero unpack buffer."""
    dev = x_bf16.device
    nvcuda, fns = get_kernels(dev)
    H = int(x_bf16.shape[-1])
    I = int(dn_q.shape[2] * 2)
    K_SUB = int(topk_idx.numel())
    GS = int(group_size)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
    idx_c = topk_idx if (topk_idx.dtype == torch.int64 and topk_idx.is_contiguous()) else topk_idx.to(torch.int64).contiguous()
    w_c = topk_weights if (topk_weights.dtype == torch.bfloat16 and topk_weights.is_contiguous()) else topk_weights.to(torch.bfloat16).contiguous()

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_idx = ctypes.c_void_p(idx_c.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_guq = ctypes.c_void_p(gu_q.data_ptr())
    p_gus = ctypes.c_void_p(gu_s.data_ptr())
    p_guz = ctypes.c_void_p(gu_z.data_ptr())
    p_dnq = ctypes.c_void_p(dn_q.data_ptr())
    p_dns = ctypes.c_void_p(dn_s.data_ptr())
    p_dnz = ctypes.c_void_p(dn_z.data_ptr())
    p_inter = ctypes.c_void_p(inter_buf.data_ptr())
    p_part = ctypes.c_void_p(part_buf.data_ptr())

    c_H = ctypes.c_int(H)
    c_I = ctypes.c_int(I)
    c_GS = ctypes.c_int(GS)

    args_gu = (ctypes.c_void_p * 9)(
        ctypes.addressof(p_x), ctypes.addressof(p_idx),
        ctypes.addressof(p_guq), ctypes.addressof(p_gus), ctypes.addressof(p_guz),
        ctypes.addressof(p_inter), ctypes.addressof(c_H), ctypes.addressof(c_I), ctypes.addressof(c_GS),
    )
    smem_gu = (H // GS) * 4 + H * 2
    grid_gu_x = (I + 7) // 8
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"routed_gate_up_swiglu_q4"], grid_gu_x, K_SUB, 1, 256, 1, 1, smem_gu, stream, args_gu, None),
        "routed_gate_up_swiglu_q4",
    )

    args_dn = (ctypes.c_void_p * 10)(
        ctypes.addressof(p_inter), ctypes.addressof(p_idx), ctypes.addressof(p_w),
        ctypes.addressof(p_dnq), ctypes.addressof(p_dns), ctypes.addressof(p_dnz),
        ctypes.addressof(p_part), ctypes.addressof(c_H), ctypes.addressof(c_I), ctypes.addressof(c_GS),
    )
    smem_dn = (I + I // GS) * 4
    grid_dn_x = (H + 7) // 8
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"routed_down_q4"], grid_dn_x, K_SUB, 1, 256, 1, 1, smem_dn, stream, args_dn, None),
        "routed_down_q4",
    )

    if reduce:
        out = torch.empty((1, H), dtype=torch.bfloat16, device=dev)
        p_out = ctypes.c_void_p(out.data_ptr())
        c_K = ctypes.c_int(K_SUB)
        args_red = (ctypes.c_void_p * 4)(
            ctypes.addressof(p_part), ctypes.addressof(p_out), ctypes.addressof(c_K), ctypes.addressof(c_H),
        )
        _check_launch(
            nvcuda.cuLaunchKernel(fns[b"reduce_k_to_bf16"], (H + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args_red, None),
            "reduce_k_to_bf16",
        )
        return out

    out_parts = torch.empty((K_SUB, H), dtype=torch.bfloat16, device=dev)
    p_outp = ctypes.c_void_p(out_parts.data_ptr())
    total_elems = K_SUB * H
    c_tot = ctypes.c_int(total_elems)
    args_conv = (ctypes.c_void_p * 3)(ctypes.addressof(p_part), ctypes.addressof(p_outp), ctypes.addressof(c_tot))
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"f32_to_bf16_vec"], (total_elems + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args_conv, None),
        "f32_to_bf16_vec",
    )
    return out_parts


def fused_routed_moe_q6(
    x_bf16: torch.Tensor,          # (1, H) bf16
    topk_idx: torch.Tensor,        # (K_SUB,) int64
    topk_weights: torch.Tensor,    # (K_SUB,) bf16
    gu_qlo: torch.Tensor,          # (E, 2*I, H//2) uint8
    gu_qhi: torch.Tensor,          # (E, 2*I, H//4) uint8
    gu_s: torch.Tensor,            # (E, 2*I, H//GS) bf16
    gu_z: torch.Tensor,            # (E, 2*I, H//GS) bf16
    dn_qlo: torch.Tensor,          # (E, H, I//2) uint8
    dn_qhi: torch.Tensor,          # (E, H, I//4) uint8
    dn_s: torch.Tensor,            # (E, H, I//GS) bf16
    dn_z: torch.Tensor,            # (E, H, I//GS) bf16
    inter_buf: torch.Tensor,       # (>=K_SUB, I) float32
    part_buf: torch.Tensor,        # (>=K_SUB, H) float32
    group_size: int,
    reduce: bool = True,
) -> torch.Tensor:
    """Execute single-token routed W6A16 (Q6) MoE directly in registers with zero unpack buffer."""
    dev = x_bf16.device
    nvcuda, fns = get_kernels(dev)
    H = int(x_bf16.shape[-1])
    I = int(dn_qlo.shape[2] * 2)
    K_SUB = int(topk_idx.numel())
    GS = int(group_size)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
    idx_c = topk_idx if (topk_idx.dtype == torch.int64 and topk_idx.is_contiguous()) else topk_idx.to(torch.int64).contiguous()
    w_c = topk_weights if (topk_weights.dtype == torch.bfloat16 and topk_weights.is_contiguous()) else topk_weights.to(torch.bfloat16).contiguous()

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_idx = ctypes.c_void_p(idx_c.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_gulo = ctypes.c_void_p(gu_qlo.data_ptr())
    p_guhi = ctypes.c_void_p(gu_qhi.data_ptr())
    p_gus = ctypes.c_void_p(gu_s.data_ptr())
    p_guz = ctypes.c_void_p(gu_z.data_ptr())
    p_dnlo = ctypes.c_void_p(dn_qlo.data_ptr())
    p_dnhi = ctypes.c_void_p(dn_qhi.data_ptr())
    p_dns = ctypes.c_void_p(dn_s.data_ptr())
    p_dnz = ctypes.c_void_p(dn_z.data_ptr())
    p_inter = ctypes.c_void_p(inter_buf.data_ptr())
    p_part = ctypes.c_void_p(part_buf.data_ptr())

    c_H = ctypes.c_int(H)
    c_I = ctypes.c_int(I)
    c_GS = ctypes.c_int(GS)

    args_gu = (ctypes.c_void_p * 10)(
        ctypes.addressof(p_x), ctypes.addressof(p_idx),
        ctypes.addressof(p_gulo), ctypes.addressof(p_guhi), ctypes.addressof(p_gus), ctypes.addressof(p_guz),
        ctypes.addressof(p_inter), ctypes.addressof(c_H), ctypes.addressof(c_I), ctypes.addressof(c_GS),
    )
    smem_gu = (H // GS) * 4 + H * 2
    grid_gu_x = (I + 7) // 8
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"routed_gate_up_swiglu_q6"], grid_gu_x, K_SUB, 1, 256, 1, 1, smem_gu, stream, args_gu, None),
        "routed_gate_up_swiglu_q6",
    )

    args_dn = (ctypes.c_void_p * 11)(
        ctypes.addressof(p_inter), ctypes.addressof(p_idx), ctypes.addressof(p_w),
        ctypes.addressof(p_dnlo), ctypes.addressof(p_dnhi), ctypes.addressof(p_dns), ctypes.addressof(p_dnz),
        ctypes.addressof(p_part), ctypes.addressof(c_H), ctypes.addressof(c_I), ctypes.addressof(c_GS),
    )
    smem_dn = (I + I // GS) * 4
    grid_dn_x = (H + 7) // 8
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"routed_down_q6"], grid_dn_x, K_SUB, 1, 256, 1, 1, smem_dn, stream, args_dn, None),
        "routed_down_q6",
    )

    if reduce:
        out = torch.empty((1, H), dtype=torch.bfloat16, device=dev)
        p_out = ctypes.c_void_p(out.data_ptr())
        c_K = ctypes.c_int(K_SUB)
        args_red = (ctypes.c_void_p * 4)(
            ctypes.addressof(p_part), ctypes.addressof(p_out), ctypes.addressof(c_K), ctypes.addressof(c_H),
        )
        _check_launch(
            nvcuda.cuLaunchKernel(fns[b"reduce_k_to_bf16"], (H + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args_red, None),
            "reduce_k_to_bf16",
        )
        return out

    out_parts = torch.empty((K_SUB, H), dtype=torch.bfloat16, device=dev)
    p_outp = ctypes.c_void_p(out_parts.data_ptr())
    total_elems = K_SUB * H
    c_tot = ctypes.c_int(total_elems)
    args_conv = (ctypes.c_void_p * 3)(ctypes.addressof(p_part), ctypes.addressof(p_outp), ctypes.addressof(c_tot))
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"f32_to_bf16_vec"], (total_elems + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args_conv, None),
        "f32_to_bf16_vec",
    )
    return out_parts


def can_use_fused_dense_q4(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> bool:
    if not (x.is_cuda and qweight.is_cuda and scales.dtype == torch.bfloat16):
        return False
    if group_size not in (16, 32, 64):
        return False
    K = int(qweight.shape[1] * 2)
    M = int(x.numel() // K) if K > 0 else 0
    if M < 1 or M > 8:
        return False
    smem_bytes = (K // group_size) * 4 + K * 2
    return smem_bytes <= 49152


def fused_dense_linear_q4(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    """Zero-scratch W4A16 GEMV / small-M linear (`y = x @ dequant(W)^T + bias`)."""
    dev = qweight.device
    nvcuda, fns = get_kernels(dev)
    N = int(qweight.shape[0])
    K = int(qweight.shape[1] * 2)
    orig_shape = x.shape[:-1]
    M = int(x.numel() // K)

    x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    x_2d = x_bf16.reshape(M, K)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    qw = qweight if qweight.is_contiguous() else qweight.contiguous()
    sc = scales if scales.is_contiguous() else scales.contiguous()
    zr = zeros if zeros.is_contiguous() else zeros.contiguous()
    bi = None
    if bias is not None:
        bi_t = bias if bias.dtype == torch.bfloat16 else bias.to(torch.bfloat16)
        bi = bi_t if bi_t.is_contiguous() else bi_t.contiguous()

    y_2d = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    p_x = ctypes.c_void_p(x_2d.data_ptr())
    p_q = ctypes.c_void_p(qw.data_ptr())
    p_s = ctypes.c_void_p(sc.data_ptr())
    p_z = ctypes.c_void_p(zr.data_ptr())
    p_b = ctypes.c_void_p(bi.data_ptr() if bi is not None else 0)
    p_y = ctypes.c_void_p(y_2d.data_ptr())

    c_M = ctypes.c_int(M)
    c_N = ctypes.c_int(N)
    c_K = ctypes.c_int(K)
    c_GS = ctypes.c_int(int(group_size))

    args = (ctypes.c_void_p * 10)(
        ctypes.addressof(p_x), ctypes.addressof(p_q), ctypes.addressof(p_s), ctypes.addressof(p_z),
        ctypes.addressof(p_b), ctypes.addressof(p_y),
        ctypes.addressof(c_M), ctypes.addressof(c_N), ctypes.addressof(c_K), ctypes.addressof(c_GS),
    )
    smem = (K // group_size) * 4 + K * 2
    grid_x = (N + 7) // 8
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"dense_gemv_q4_bf16"], grid_x, M, 1, 256, 1, 1, smem, stream, args, None),
        "dense_gemv_q4_bf16",
    )
    return y_2d.view(*orig_shape, N)
