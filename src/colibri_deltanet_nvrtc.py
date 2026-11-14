"""Self-contained NVRTC CUDA kernels for GatedDeltaNet & CausalConv1d in AirColibri.

Provides zero-dependency native CUDA kernels compiled at runtime via PyTorch's bundled
NVRTC (`nvrtc64_*.dll` / `libnvrtc.so*`) and cached as PTX under `data/native/ptx-cache/`:
- `causal_conv1d_update` (single-step / short-step depthwise causal conv1d + state update + SiLU)
- `causal_conv1d_fn` (full-sequence prefill depthwise causal conv1d + SiLU, arbitrary strides)
- `fused_recurrent_gated_delta_rule` (register-resident FP32 recurrent GatedDeltaNet scan)
- `chunk_gated_delta_rule` (register-resident FP32 GatedDeltaNet scan without `solve_triangular`)
- `fused_rmsnorm_gated_silu` & `fused_rmsnorm_zero_centered`

Eliminates `transformers` reference PyTorch fallbacks on Windows and Linux without requiring
`causal-conv1d`, `flash-linear-attention`, Triton, or a host C++/CUDA compiler.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import inspect
import math
import os
import sys
import threading
import types
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.colibri_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

DELTANET_CUDA_SRC = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_rne(float val) {
    unsigned int bits = __float_as_uint(val);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return 0x7fc0u;
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}

__device__ __forceinline__ float fp16_to_f32(unsigned short h) {
    float res;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(res) : "h"(h));
    return res;
}

__device__ __forceinline__ unsigned short f32_to_fp16_rn(float val) {
    unsigned short res;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(res) : "f"(val));
    return res;
}

// dt codes: 0 = float32, 1 = bfloat16, 2 = float16
__device__ __forceinline__ float ld_elem(const void* ptr, long long idx, int dt) {
    if (dt == 1) return bf16_to_f32(reinterpret_cast<const unsigned short*>(ptr)[idx]);
    if (dt == 2) return fp16_to_f32(reinterpret_cast<const unsigned short*>(ptr)[idx]);
    return reinterpret_cast<const float*>(ptr)[idx];
}

__device__ __forceinline__ void st_elem(void* ptr, long long idx, float val, int dt) {
    if (dt == 1) reinterpret_cast<unsigned short*>(ptr)[idx] = f32_to_bf16_rne(val);
    else if (dt == 2) reinterpret_cast<unsigned short*>(ptr)[idx] = f32_to_fp16_rn(val);
    else reinterpret_cast<float*>(ptr)[idx] = val;
}

__device__ __forceinline__ float cast_round(float val, int dt) {
    if (dt == 1) return bf16_to_f32(f32_to_bf16_rne(val));
    if (dt == 2) return fp16_to_f32(f32_to_fp16_rn(val));
    return val;
}

extern "C" __global__ void causal_conv1d_update_kernel(
    const void* __restrict__ x_ptr,
    void* __restrict__ cs_ptr,
    const void* __restrict__ w_ptr,
    const void* __restrict__ b_ptr,
    void* __restrict__ out_ptr,
    int B, int C, int L, int S, int K,
    long long xs_b, long long xs_c, long long xs_l,
    long long cs_b, long long cs_c, long long cs_s,
    int x_dt, int cs_dt, int w_dt, int b_dt,
    int has_bias, int act_mode
) {
    __shared__ float s_hist[32 * 128];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 128 + tid;
    if (idx >= B * C) return;
    int b = idx / C;
    int c = idx % C;

    long long cs_base = (long long)b * cs_b + (long long)c * cs_c;
    long long x_base = (long long)b * xs_b + (long long)c * xs_c;

    for (int s = 0; s < S; ++s) {
        s_hist[s * 128 + tid] = cast_round(ld_elem(cs_ptr, cs_base + (long long)s * cs_s, cs_dt), w_dt);
    }
    for (int t = 0; t < L; ++t) {
        s_hist[(S + t) * 128 + tid] = cast_round(ld_elem(x_ptr, x_base + (long long)t * xs_l, x_dt), w_dt);
    }

    float bias_v = has_bias ? ld_elem(b_ptr, c, b_dt) : 0.0f;
    long long w_base = (long long)c * K;
    long long out_base = (long long)idx * L;
    int total_len = S + L;

    for (int t = 0; t < L; ++t) {
        int start_p = S - K + 1 + t;
        float acc = bias_v;
        for (int k = 0; k < K; ++k) {
            int p = start_p + k;
            float xv = (p >= 0 && p < total_len) ? s_hist[p * 128 + tid] : 0.0f;
            acc = fmaf(xv, ld_elem(w_ptr, w_base + k, w_dt), acc);
        }
        acc = cast_round(acc, w_dt);
        if (act_mode == 1) {
            acc = cast_round(acc / (1.0f + expf(-acc)), w_dt);
        }
        st_elem(out_ptr, out_base + t, acc, x_dt);
    }

    for (int s = 0; s < S; ++s) {
        int p = L + s;
        float nv = s_hist[p * 128 + tid];
        st_elem(cs_ptr, cs_base + (long long)s * cs_s, nv, cs_dt);
    }
}

extern "C" __global__ void causal_conv1d_fn_kernel(
    const void* __restrict__ x_ptr,
    const void* __restrict__ w_ptr,
    const void* __restrict__ b_ptr,
    void* __restrict__ out_ptr,
    int B, int C, int L, int K,
    long long xs_b, long long xs_c, long long xs_l,
    int x_dt, int w_dt, int b_dt,
    int has_bias, int act_mode
) {
    __shared__ float s_w[16];
    __shared__ float s_x[256 + 16];

    int bc = blockIdx.x;
    if (bc >= B * C) return;
    int b = bc / C;
    int c = bc % C;
    int tid = threadIdx.x;
    int pad = K - 1;
    int t_start = blockIdx.y * 256;

    if (tid < K) {
        s_w[tid] = ld_elem(w_ptr, (long long)c * K + tid, w_dt);
    }

    long long x_base = (long long)b * xs_b + (long long)c * xs_c;
    int total_sh = 256 + pad;
    for (int i = tid; i < total_sh; i += 256) {
        int src_t = t_start - pad + i;
        float xv = 0.0f;
        if (src_t >= 0 && src_t < L) {
            xv = cast_round(ld_elem(x_ptr, x_base + (long long)src_t * xs_l, x_dt), w_dt);
        }
        s_x[i] = xv;
    }
    __syncthreads();

    int t = t_start + tid;
    if (t < L) {
        float acc = has_bias ? ld_elem(b_ptr, c, b_dt) : 0.0f;
        for (int k = 0; k < K; ++k) {
            acc = fmaf(s_x[tid + k], s_w[k], acc);
        }
        acc = cast_round(acc, w_dt);
        if (act_mode == 1) {
            acc = cast_round(acc / (1.0f + expf(-acc)), w_dt);
        }
        st_elem(out_ptr, (long long)bc * L + t, acc, x_dt);
    }
}

extern "C" __global__ void gated_delta_rule_d128_kernel(
    const void* __restrict__ q_ptr,
    const void* __restrict__ k_ptr,
    const void* __restrict__ v_ptr,
    const void* __restrict__ g_ptr,
    const void* __restrict__ beta_ptr,
    const void* __restrict__ init_state_ptr,
    void* __restrict__ out_ptr,
    float* __restrict__ final_state_ptr,
    int B, int L, int H_q, int H_k, int H_v,
    int q_dt, int k_dt, int v_dt, int g_dt, int beta_dt, int init_dt,
    int has_init, int write_final, int use_l2norm
) {
    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_part[4][128];
    __shared__ float s_red_q[4];
    __shared__ float s_red_k[4];
    __shared__ float s_scalars[2];

    int bh = blockIdx.x;
    if (bh >= B * H_v) return;
    int b = bh / H_v;
    int vh = bh % H_v;
    int qh = vh / (H_v / H_q);
    int kh = vh / (H_v / H_k);

    int j = threadIdx.x;     // 0..127 (column in D_v)
    int part = threadIdx.y;  // 0..3   (32-row slice in D_k)
    int k_off = part * 32;

    float s_reg[32];
    if (has_init) {
        long long st_base = (long long)bh * (128 * 128) + (long long)k_off * 128 + j;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            s_reg[r] = ld_elem(init_state_ptr, st_base + r * 128, init_dt);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            s_reg[r] = 0.0f;
        }
    }

    for (int t = 0; t < L; ++t) {
        float qv = 0.0f;
        float kv = 0.0f;
        if (part == 0) {
            long long q_idx = (((long long)b * L + t) * H_q + qh) * 128 + j;
            long long k_idx = (((long long)b * L + t) * H_k + kh) * 128 + j;
            qv = ld_elem(q_ptr, q_idx, q_dt);
            kv = ld_elem(k_ptr, k_idx, k_dt);
            if (use_l2norm) {
                float sq_q = qv * qv;
                float sq_k = kv * kv;
                int warp = j >> 5;
                int lane = j & 31;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    sq_q += __shfl_down_sync(0xffffffffu, sq_q, off);
                    sq_k += __shfl_down_sync(0xffffffffu, sq_k, off);
                }
                if (lane == 0) {
                    s_red_q[warp] = sq_q;
                    s_red_k[warp] = sq_k;
                }
            }
            if (j == 0) {
                long long gb_idx = ((long long)b * L + t) * H_v + vh;
                s_scalars[0] = expf(ld_elem(g_ptr, gb_idx, g_dt));
                s_scalars[1] = ld_elem(beta_ptr, gb_idx, beta_dt);
            }
        }
        __syncthreads();

        if (part == 0) {
            float inv_q = 0.08838834764831845f;
            float inv_k = 1.0f;
            if (use_l2norm) {
                inv_q *= rsqrtf((s_red_q[0] + s_red_q[1]) + (s_red_q[2] + s_red_q[3]) + 1e-6f);
                inv_k = rsqrtf((s_red_k[0] + s_red_k[1]) + (s_red_k[2] + s_red_k[3]) + 1e-6f);
            }
            s_q[j] = qv * inv_q;
            s_k[j] = kv * inv_k;
        }
        __syncthreads();

        float decay_t = s_scalars[0];
        float beta_t = s_scalars[1];
        float kv_p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float sv = s_reg[r] * decay_t;
            s_reg[r] = sv;
            kv_p = fmaf(sv, s_k[k_off + r], kv_p);
        }
        s_part[part][j] = kv_p;
        __syncthreads();

        long long v_idx = (((long long)b * L + t) * H_v + vh) * 128 + j;
        float v_val = ld_elem(v_ptr, v_idx, v_dt);
        float kv_mem = (s_part[0][j] + s_part[1][j]) + (s_part[2][j] + s_part[3][j]);
        float delta_j = (v_val - kv_mem) * beta_t;
        float out_p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float sv = fmaf(s_k[k_off + r], delta_j, s_reg[r]);
            s_reg[r] = sv;
            out_p = fmaf(sv, s_q[k_off + r], out_p);
        }
        s_part[part][j] = out_p;
        __syncthreads();

        if (part == 0) {
            float out_j = (s_part[0][j] + s_part[1][j]) + (s_part[2][j] + s_part[3][j]);
            st_elem(out_ptr, v_idx, out_j, q_dt);
        }
    }

    if (write_final) {
        long long st_base = (long long)bh * (128 * 128) + (long long)k_off * 128 + j;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            final_state_ptr[st_base + r * 128] = s_reg[r];
        }
    }
}

extern "C" __global__ void gated_delta_rule_generic_kernel(
    const void* __restrict__ q_ptr,
    const void* __restrict__ k_ptr,
    const void* __restrict__ v_ptr,
    const void* __restrict__ g_ptr,
    const void* __restrict__ beta_ptr,
    const void* __restrict__ init_state_ptr,
    void* __restrict__ out_ptr,
    float* __restrict__ state_ws_ptr,
    int B, int L, int H_q, int H_k, int H_v, int D_k, int D_v,
    int q_dt, int k_dt, int v_dt, int g_dt, int beta_dt, int init_dt,
    int has_init, int use_l2norm
) {
    __shared__ float s_q[256];
    __shared__ float s_k[256];
    __shared__ float s_red_q[8];
    __shared__ float s_red_k[8];
    __shared__ float s_scalars[2];

    int bh = blockIdx.x;
    if (bh >= B * H_v) return;
    int b = bh / H_v;
    int vh = bh % H_v;
    int qh = vh / (H_v / H_q);
    int kh = vh / (H_v / H_k);

    int tid = threadIdx.x; // 0..255
    int warp = tid >> 5;
    int lane = tid & 31;

    float* st_head = state_ws_ptr + (long long)bh * D_k * D_v;
    int total_st = D_k * D_v;
    if (has_init) {
        long long st_base = (long long)bh * total_st;
        for (int idx = tid; idx < total_st; idx += 256) {
            st_head[idx] = ld_elem(init_state_ptr, st_base + idx, init_dt);
        }
    } else {
        for (int idx = tid; idx < total_st; idx += 256) {
            st_head[idx] = 0.0f;
        }
    }
    __syncthreads();

    float inv_sqrt_dk = rsqrtf((float)D_k);

    for (int t = 0; t < L; ++t) {
        float qv = 0.0f;
        float kv = 0.0f;
        if (tid < D_k) {
            qv = ld_elem(q_ptr, (((long long)b * L + t) * H_q + qh) * D_k + tid, q_dt);
            kv = ld_elem(k_ptr, (((long long)b * L + t) * H_k + kh) * D_k + tid, k_dt);
        }
        if (use_l2norm) {
            float sq_q = qv * qv;
            float sq_k = kv * kv;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sq_q += __shfl_down_sync(0xffffffffu, sq_q, off);
                sq_k += __shfl_down_sync(0xffffffffu, sq_k, off);
            }
            if (lane == 0) {
                s_red_q[warp] = sq_q;
                s_red_k[warp] = sq_k;
            }
        }
        if (tid == 0) {
            long long gb_idx = ((long long)b * L + t) * H_v + vh;
            s_scalars[0] = expf(ld_elem(g_ptr, gb_idx, g_dt));
            s_scalars[1] = ld_elem(beta_ptr, gb_idx, beta_dt);
        }
        __syncthreads();

        if (tid < D_k) {
            float inv_q = inv_sqrt_dk;
            float inv_k = 1.0f;
            if (use_l2norm) {
                float sum_q = 0.0f, sum_k = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    sum_q += s_red_q[w];
                    sum_k += s_red_k[w];
                }
                inv_q *= rsqrtf(sum_q + 1e-6f);
                inv_k = rsqrtf(sum_k + 1e-6f);
            }
            s_q[tid] = qv * inv_q;
            s_k[tid] = kv * inv_k;
        }
        __syncthreads();

        if (tid < D_v) {
            float decay_t = s_scalars[0];
            float beta_t = s_scalars[1];
            float kv_mem = 0.0f;
            for (int i = 0; i < D_k; ++i) {
                float s_ij = st_head[i * D_v + tid] * decay_t;
                st_head[i * D_v + tid] = s_ij;
                kv_mem = fmaf(s_ij, s_k[i], kv_mem);
            }
            long long v_idx = (((long long)b * L + t) * H_v + vh) * D_v + tid;
            float v_val = ld_elem(v_ptr, v_idx, v_dt);
            float delta_j = (v_val - kv_mem) * beta_t;
            float out_j = 0.0f;
            for (int i = 0; i < D_k; ++i) {
                float s_ij = fmaf(s_k[i], delta_j, st_head[i * D_v + tid]);
                st_head[i * D_v + tid] = s_ij;
                out_j = fmaf(s_ij, s_q[i], out_j);
            }
            st_elem(out_ptr, v_idx, out_j, q_dt);
        }
        __syncthreads();
    }
}

extern "C" __global__ void fused_rmsnorm_gated_silu_kernel(
    const void* __restrict__ x_ptr,
    const void* __restrict__ gate_ptr,
    const void* __restrict__ w_ptr,
    void* __restrict__ out_ptr,
    int rows, int D, float eps,
    int x_dt, int g_dt, int w_dt
) {
    __shared__ float s_red[8];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    long long base = (long long)row * D;

    float sq = 0.0f;
    for (int c = tid; c < D; c += 256) {
        float v = ld_elem(x_ptr, base + c, x_dt);
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();

    float tot = 0.0f;
    if (tid < 8) tot = s_red[tid];
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        tot += __shfl_down_sync(0xffffffffu, tot, off);
    }
    if (tid == 0) s_red[0] = rsqrtf(tot / (float)D + eps);
    __syncthreads();

    float inv_rms = s_red[0];
    for (int c = tid; c < D; c += 256) {
        float x_f32 = ld_elem(x_ptr, base + c, x_dt);
        float normed = cast_round(x_f32 * inv_rms, x_dt);
        float w_val = ld_elem(w_ptr, c, w_dt);
        float scaled = (w_dt == x_dt) ? cast_round(w_val * normed, x_dt) : (w_val * normed);
        float g_f32 = ld_elem(gate_ptr, base + c, g_dt);
        float silu_g = g_f32 / (1.0f + expf(-g_f32));
        st_elem(out_ptr, base + c, scaled * silu_g, x_dt);
    }
}

extern "C" __global__ void fused_rmsnorm_zero_centered_kernel(
    const void* __restrict__ x_ptr,
    const void* __restrict__ w_ptr,
    void* __restrict__ out_ptr,
    int rows, int D, float eps,
    int x_dt, int w_dt
) {
    __shared__ float s_red[8];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    long long base = (long long)row * D;

    float sq = 0.0f;
    for (int c = tid; c < D; c += 256) {
        float v = ld_elem(x_ptr, base + c, x_dt);
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();

    float tot = 0.0f;
    if (tid < 8) tot = s_red[tid];
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        tot += __shfl_down_sync(0xffffffffu, tot, off);
    }
    if (tid == 0) s_red[0] = rsqrtf(tot / (float)D + eps);
    __syncthreads();

    float inv_rms = s_red[0];
    for (int c = tid; c < D; c += 256) {
        float v = ld_elem(x_ptr, base + c, x_dt) * inv_rms * (1.0f + ld_elem(w_ptr, c, w_dt));
        st_elem(out_ptr, base + c, v, x_dt);
    }
}
"""

_KERNEL_NAMES = (
    b"causal_conv1d_update_kernel",
    b"causal_conv1d_fn_kernel",
    b"gated_delta_rule_d128_kernel",
    b"gated_delta_rule_generic_kernel",
    b"fused_rmsnorm_gated_silu_kernel",
    b"fused_rmsnorm_zero_centered_kernel",
)

_DTYPE_CODE = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
}

_LOCK = threading.Lock()
_DEVICE_MODULES: dict[int, tuple[Any, dict[bytes, ctypes.c_void_p]]] = {}
_PATCHED_MODULES: set[str] = set()

KERNEL_CALL_STATS: dict[str, int] = {
    "causal_conv1d_update": 0,
    "causal_conv1d_fn": 0,
    "fused_recurrent_gated_delta_rule": 0,
    "chunk_gated_delta_rule": 0,
    "rmsnorm_gated": 0,
    "rmsnorm_zero_centered": 0,
}

_TARGET_MODELING_MODULES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    "transformers.models.qwen3_next.modeling_qwen3_next",
    "transformers.models.olmo_hybrid.modeling_olmo_hybrid",
    "transformers.models.qwen4_exp.modeling_qwen4_exp",
)


def compile_deltanet_ptx(major: int, minor: int, nvrtc: Any | None = None) -> bytes:
    """Compiles ``DELTANET_CUDA_SRC`` to PTX via bundled NVRTC and caches on disk."""
    src_bytes = DELTANET_CUDA_SRC.encode("utf-8")
    digest = hashlib.sha256(src_bytes).hexdigest()[:16]
    PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PTX_CACHE_DIR / f"colibri_deltanet_sm{major}{minor}_{digest}.ptx"
    if cache_file.is_file():
        raw = cache_file.read_bytes()
        return raw if raw.endswith(b"\x00") else raw + b"\x00"

    if nvrtc is None:
        _, nvrtc = _load_driver_and_nvrtc()

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"colibri_deltanet.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram(colibri_deltanet.cu) failed with error {rc}")
    try:
        arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
        opts = (ctypes.c_char_p * 1)(arch_flag)
        rc = nvrtc.nvrtcCompileProgram(prog, 1, opts)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(
                f"NVRTC compile failed ({rc}):\n{log_buf.value.decode('utf-8', errors='replace')}"
            )
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
    return raw_ptx if raw_ptx.endswith(b"\x00") else raw_ptx + b"\x00"


def get_deltanet_kernels(device: torch.device) -> tuple[Any, dict[bytes, ctypes.c_void_p]]:
    dev_idx = 0 if device.index is None else int(device.index)
    cached = _DEVICE_MODULES.get(dev_idx)
    if cached is not None:
        return cached

    with _LOCK:
        cached = _DEVICE_MODULES.get(dev_idx)
        if cached is not None:
            return cached
        nvcuda, nvrtc = _load_driver_and_nvrtc()
        with torch.cuda.device(dev_idx):
            _ = torch.empty(1, device=f"cuda:{dev_idx}")
            major, minor = torch.cuda.get_device_capability(dev_idx)
            ptx = compile_deltanet_ptx(major, minor, nvrtc=nvrtc)
            ptx_buf = ctypes.create_string_buffer(ptx)
            mod = ctypes.c_void_p()
            rc = nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p))
            if rc != 0:
                raise RuntimeError(f"cuModuleLoadData(colibri_deltanet) failed with CUresult={rc}")
            funcs: dict[bytes, ctypes.c_void_p] = {"_module": mod}
            for name in _KERNEL_NAMES:
                fn = ctypes.c_void_p()
                rc = nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, name)
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({name.decode()}) failed with CUresult={rc}")
                funcs[name] = fn
            _DEVICE_MODULES[dev_idx] = (nvcuda, funcs)
            return nvcuda, funcs


def _check_launch(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel({name}) failed with CUresult={rc}")


def _act_code(activation: str | None) -> int | None:
    if activation is None:
        return 0
    if activation in ("silu", "swish"):
        return 1
    return None


def nvrtc_causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    _fallback: Callable | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    act_mode = _act_code(activation)
    w2d = weight.squeeze(1) if weight.ndim == 3 else weight
    if (
        not hidden_states.is_cuda
        or hidden_states.ndim != 3
        or w2d.ndim != 2
        or w2d.shape[-1] > 16
        or act_mode is None
        or hidden_states.dtype not in _DTYPE_CODE
        or w2d.dtype not in _DTYPE_CODE
        or (bias is not None and bias.dtype not in _DTYPE_CODE)
        or kwargs.get("seq_idx") is not None
        or kwargs.get("cu_seqlens") is not None
    ):
        if _fallback is not None:
            return _fallback(hidden_states, weight, bias=bias, activation=activation, **kwargs)
        padding = w2d.shape[-1] - 1
        out = F.conv1d(
            hidden_states.to(w2d.dtype),
            weight=w2d.unsqueeze(1),
            bias=bias,
            padding=padding,
            groups=hidden_states.shape[1],
        )[:, :, : hidden_states.shape[-1]]
        if activation in ("silu", "swish"):
            out = F.silu(out)
        return out.to(hidden_states.dtype)

    B, C, L = hidden_states.shape
    K = int(w2d.shape[1])
    if B == 0 or C == 0 or L == 0:
        return hidden_states.contiguous()

    dev = hidden_states.device
    nvcuda, fns = get_deltanet_kernels(dev)
    w_c = w2d if w2d.is_contiguous() else w2d.contiguous()
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else None
    out = torch.empty((B, C, L), dtype=hidden_states.dtype, device=dev)

    xs_b, xs_c, xs_l = hidden_states.stride()
    p_x = ctypes.c_void_p(hidden_states.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_b = ctypes.c_void_p(b_c.data_ptr() if b_c is not None else 0)
    p_o = ctypes.c_void_p(out.data_ptr())

    c_B, c_C, c_L, c_K = ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(L), ctypes.c_int(K)
    c_xsb, c_xsc, c_xsl = ctypes.c_longlong(xs_b), ctypes.c_longlong(xs_c), ctypes.c_longlong(xs_l)
    c_xdt = ctypes.c_int(_DTYPE_CODE[hidden_states.dtype])
    c_wdt = ctypes.c_int(_DTYPE_CODE[w_c.dtype])
    c_bdt = ctypes.c_int(_DTYPE_CODE[b_c.dtype] if b_c is not None else 0)
    c_hb = ctypes.c_int(1 if b_c is not None else 0)
    c_act = ctypes.c_int(act_mode)

    args = (ctypes.c_void_p * 16)(
        ctypes.addressof(p_x), ctypes.addressof(p_w), ctypes.addressof(p_b), ctypes.addressof(p_o),
        ctypes.addressof(c_B), ctypes.addressof(c_C), ctypes.addressof(c_L), ctypes.addressof(c_K),
        ctypes.addressof(c_xsb), ctypes.addressof(c_xsc), ctypes.addressof(c_xsl),
        ctypes.addressof(c_xdt), ctypes.addressof(c_wdt), ctypes.addressof(c_bdt),
        ctypes.addressof(c_hb), ctypes.addressof(c_act),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    # Process L in 65535*256-safe tiles along gridDim.y
    grid_y = (L + 255) // 256
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"causal_conv1d_fn_kernel"], B * C, grid_y, 1, 256, 1, 1, 0, stream, args, None),
        "causal_conv1d_fn_kernel",
    )
    KERNEL_CALL_STATS["causal_conv1d_fn"] += 1
    return out


def nvrtc_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    _fallback: Callable | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    squeeze_seq = hidden_states.ndim == 2
    x3d = hidden_states.unsqueeze(-1) if squeeze_seq else hidden_states
    w2d = weight.squeeze(1) if weight.ndim == 3 else weight
    act_mode = _act_code(activation)

    if (
        not x3d.is_cuda
        or x3d.ndim != 3
        or conv_state.ndim != 3
        or w2d.ndim != 2
        or act_mode is None
        or x3d.dtype not in _DTYPE_CODE
        or conv_state.dtype not in _DTYPE_CODE
        or w2d.dtype not in _DTYPE_CODE
        or (bias is not None and bias.dtype not in _DTYPE_CODE)
        or kwargs.get("cache_seqlens") is not None
        or kwargs.get("conv_state_indices") is not None
    ):
        if _fallback is not None:
            return _fallback(hidden_states, conv_state, weight, bias=bias, activation=activation)
        _, hidden_size, seq_len = x3d.shape
        state_len = conv_state.shape[-1]
        hs_new = torch.cat([conv_state, x3d], dim=-1).to(w2d.dtype)
        conv_state.copy_(hs_new[:, :, -state_len:])
        out = F.conv1d(hs_new, w2d.unsqueeze(1), bias, padding=0, groups=hidden_size)[:, :, -seq_len:]
        if activation in ("silu", "swish"):
            out = F.silu(out)
        out = out.to(hidden_states.dtype)
        return out.squeeze(-1) if squeeze_seq else out

    B, C, L = x3d.shape
    S = int(conv_state.shape[-1])
    K = int(w2d.shape[-1])
    if L > 16 or S > 16 or K > 16:
        hs_new = torch.cat([conv_state.to(w2d.dtype), x3d.to(w2d.dtype)], dim=-1)
        conv_state.copy_(hs_new[:, :, -S:])
        conv_full = nvrtc_causal_conv1d_fn(hs_new, w2d, bias=bias, activation=activation)
        out = conv_full[:, :, -L:].to(hidden_states.dtype)
        return out.squeeze(-1) if squeeze_seq else out

    dev = x3d.device
    nvcuda, fns = get_deltanet_kernels(dev)
    w_c = w2d if w2d.is_contiguous() else w2d.contiguous()
    b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else None
    out = torch.empty((B, C, L), dtype=hidden_states.dtype, device=dev)

    xs_b, xs_c, xs_l = x3d.stride()
    cs_b, cs_c, cs_s = conv_state.stride()
    p_x = ctypes.c_void_p(x3d.data_ptr())
    p_cs = ctypes.c_void_p(conv_state.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_b = ctypes.c_void_p(b_c.data_ptr() if b_c is not None else 0)
    p_o = ctypes.c_void_p(out.data_ptr())

    c_B, c_C, c_L, c_S, c_K = (
        ctypes.c_int(B), ctypes.c_int(C), ctypes.c_int(L), ctypes.c_int(S), ctypes.c_int(K),
    )
    c_xsb, c_xsc, c_xsl = ctypes.c_longlong(xs_b), ctypes.c_longlong(xs_c), ctypes.c_longlong(xs_l)
    c_csb, c_csc, c_css = ctypes.c_longlong(cs_b), ctypes.c_longlong(cs_c), ctypes.c_longlong(cs_s)
    c_xdt = ctypes.c_int(_DTYPE_CODE[x3d.dtype])
    c_csdt = ctypes.c_int(_DTYPE_CODE[conv_state.dtype])
    c_wdt = ctypes.c_int(_DTYPE_CODE[w_c.dtype])
    c_bdt = ctypes.c_int(_DTYPE_CODE[b_c.dtype] if b_c is not None else 0)
    c_hb = ctypes.c_int(1 if b_c is not None else 0)
    c_act = ctypes.c_int(act_mode)

    args = (ctypes.c_void_p * 22)(
        ctypes.addressof(p_x), ctypes.addressof(p_cs), ctypes.addressof(p_w),
        ctypes.addressof(p_b), ctypes.addressof(p_o),
        ctypes.addressof(c_B), ctypes.addressof(c_C), ctypes.addressof(c_L),
        ctypes.addressof(c_S), ctypes.addressof(c_K),
        ctypes.addressof(c_xsb), ctypes.addressof(c_xsc), ctypes.addressof(c_xsl),
        ctypes.addressof(c_csb), ctypes.addressof(c_csc), ctypes.addressof(c_css),
        ctypes.addressof(c_xdt), ctypes.addressof(c_csdt), ctypes.addressof(c_wdt), ctypes.addressof(c_bdt),
        ctypes.addressof(c_hb), ctypes.addressof(c_act),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    grid_x = (B * C + 127) // 128
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"causal_conv1d_update_kernel"], grid_x, 1, 1, 128, 1, 1, 0, stream, args, None),
        "causal_conv1d_update_kernel",
    )
    KERNEL_CALL_STATS["causal_conv1d_update"] += 1
    return out.squeeze(-1) if squeeze_seq else out


def _launch_gated_delta_rule_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dev = query.device
    B, L, H_q, D_k = query.shape
    _, _, H_k, _ = key.shape
    _, _, H_v, D_v = value.shape

    if B == 0 or L == 0:
        out = torch.empty((B, L, H_v, D_v), dtype=query.dtype, device=dev)
        if not output_final_state:
            return out, None
        if initial_state is not None:
            return out, initial_state.to(dtype=torch.float32, copy=True)
        return out, torch.zeros((B, H_v, D_k, D_v), dtype=torch.float32, device=dev)

    nvcuda, fns = get_deltanet_kernels(dev)
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    q_c = query if query.is_contiguous() else query.contiguous()
    k_c = key if key.is_contiguous() else key.contiguous()
    v_c = value if value.is_contiguous() else value.contiguous()
    g_c = g if g.is_contiguous() else g.contiguous()
    b_c = beta if beta.is_contiguous() else beta.contiguous()
    init_c = (initial_state if initial_state.is_contiguous() else initial_state.contiguous()) if initial_state is not None else None

    out = torch.empty((B, L, H_v, D_v), dtype=query.dtype, device=dev)
    use_d128 = D_k == 128 and D_v == 128

    # Slice very long sequences (>2048 tokens) so each launch completes in <2 ms on any GPU.
    max_slice = 2048
    cur_init = init_c
    final_state: torch.Tensor | None = None
    if output_final_state or not use_d128 or L > max_slice:
        final_state = torch.empty((B, H_v, D_k, D_v), dtype=torch.float32, device=dev)

    for s0 in range(0, L, max_slice):
        s1 = min(L, s0 + max_slice)
        sub_len = s1 - s0
        q_s = q_c[:, s0:s1] if s0 > 0 or s1 < L else q_c
        k_s = k_c[:, s0:s1] if s0 > 0 or s1 < L else k_c
        v_s = v_c[:, s0:s1] if s0 > 0 or s1 < L else v_c
        g_s = g_c[:, s0:s1] if s0 > 0 or s1 < L else g_c
        b_s = b_c[:, s0:s1] if s0 > 0 or s1 < L else b_c
        out_s = out[:, s0:s1] if s0 > 0 or s1 < L else out
        if not q_s.is_contiguous():
            q_s, k_s, v_s = q_s.contiguous(), k_s.contiguous(), v_s.contiguous()
            g_s, b_s = g_s.contiguous(), b_s.contiguous()
        out_target = out_s if out_s.is_contiguous() else torch.empty_like(v_s, dtype=query.dtype)

        need_write_state = output_final_state or (s1 < L) or (not use_d128)
        p_q = ctypes.c_void_p(q_s.data_ptr())
        p_k = ctypes.c_void_p(k_s.data_ptr())
        p_v = ctypes.c_void_p(v_s.data_ptr())
        p_g = ctypes.c_void_p(g_s.data_ptr())
        p_b = ctypes.c_void_p(b_s.data_ptr())
        p_init = ctypes.c_void_p(cur_init.data_ptr() if cur_init is not None else 0)
        p_out = ctypes.c_void_p(out_target.data_ptr())
        p_fin = ctypes.c_void_p(final_state.data_ptr() if (final_state is not None and need_write_state) else 0)

        c_B = ctypes.c_int(B)
        c_L = ctypes.c_int(sub_len)
        c_Hq = ctypes.c_int(H_q)
        c_Hk = ctypes.c_int(H_k)
        c_Hv = ctypes.c_int(H_v)
        c_qdt = ctypes.c_int(_DTYPE_CODE[q_s.dtype])
        c_kdt = ctypes.c_int(_DTYPE_CODE[k_s.dtype])
        c_vdt = ctypes.c_int(_DTYPE_CODE[v_s.dtype])
        c_gdt = ctypes.c_int(_DTYPE_CODE[g_s.dtype])
        c_bdt = ctypes.c_int(_DTYPE_CODE[b_s.dtype])
        c_idt = ctypes.c_int(_DTYPE_CODE[cur_init.dtype] if cur_init is not None else 0)
        c_hi = ctypes.c_int(1 if cur_init is not None else 0)
        c_l2 = ctypes.c_int(1 if use_qk_l2norm_in_kernel else 0)

        if use_d128:
            c_wf = ctypes.c_int(1 if need_write_state else 0)
            args = (ctypes.c_void_p * 22)(
                ctypes.addressof(p_q), ctypes.addressof(p_k), ctypes.addressof(p_v),
                ctypes.addressof(p_g), ctypes.addressof(p_b), ctypes.addressof(p_init),
                ctypes.addressof(p_out), ctypes.addressof(p_fin),
                ctypes.addressof(c_B), ctypes.addressof(c_L),
                ctypes.addressof(c_Hq), ctypes.addressof(c_Hk), ctypes.addressof(c_Hv),
                ctypes.addressof(c_qdt), ctypes.addressof(c_kdt), ctypes.addressof(c_vdt),
                ctypes.addressof(c_gdt), ctypes.addressof(c_bdt), ctypes.addressof(c_idt),
                ctypes.addressof(c_hi), ctypes.addressof(c_wf), ctypes.addressof(c_l2),
            )
            _check_launch(
                nvcuda.cuLaunchKernel(
                    fns[b"gated_delta_rule_d128_kernel"], B * H_v, 1, 1, 128, 4, 1, 0, stream, args, None
                ),
                "gated_delta_rule_d128_kernel",
            )
        else:
            c_Dk = ctypes.c_int(D_k)
            c_Dv = ctypes.c_int(D_v)
            args = (ctypes.c_void_p * 23)(
                ctypes.addressof(p_q), ctypes.addressof(p_k), ctypes.addressof(p_v),
                ctypes.addressof(p_g), ctypes.addressof(p_b), ctypes.addressof(p_init),
                ctypes.addressof(p_out), ctypes.addressof(p_fin),
                ctypes.addressof(c_B), ctypes.addressof(c_L),
                ctypes.addressof(c_Hq), ctypes.addressof(c_Hk), ctypes.addressof(c_Hv),
                ctypes.addressof(c_Dk), ctypes.addressof(c_Dv),
                ctypes.addressof(c_qdt), ctypes.addressof(c_kdt), ctypes.addressof(c_vdt),
                ctypes.addressof(c_gdt), ctypes.addressof(c_bdt), ctypes.addressof(c_idt),
                ctypes.addressof(c_hi), ctypes.addressof(c_l2),
            )
            _check_launch(
                nvcuda.cuLaunchKernel(
                    fns[b"gated_delta_rule_generic_kernel"], B * H_v, 1, 1, 256, 1, 1, 0, stream, args, None
                ),
                "gated_delta_rule_generic_kernel",
            )

        if out_target.data_ptr() != out_s.data_ptr():
            out_s.copy_(out_target)
        cur_init = final_state

    return out, (final_state if output_final_state else None)


def _can_run_deltanet_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> bool:
    if not query.is_cuda or kwargs.get("cu_seqlens") is not None:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return False
    D_k = int(query.shape[-1])
    D_v = int(value.shape[-1])
    H_q, H_k, H_v = int(query.shape[2]), int(key.shape[2]), int(value.shape[2])
    if D_k < 1 or D_k > 256 or D_v < 1 or D_v > 256 or int(key.shape[-1]) != D_k:
        return False
    if H_q < 1 or H_k < 1 or H_v < 1 or (H_v % H_q != 0) or (H_v % H_k != 0):
        return False
    for t in (query, key, value, g, beta):
        if t.dtype not in _DTYPE_CODE:
            return False
    if initial_state is not None and initial_state.dtype not in _DTYPE_CODE:
        return False
    return True


def nvrtc_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    _fallback: Callable | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _can_run_deltanet_cuda(query, key, value, g, beta, initial_state, kwargs):
        out = _launch_gated_delta_rule_cuda(
            query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
        )
        KERNEL_CALL_STATS["fused_recurrent_gated_delta_rule"] += 1
        return out
    if _fallback is not None:
        return _fallback(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )
    raise RuntimeError("No fallback available for recurrent_gated_delta_rule")


def nvrtc_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    _fallback: Callable | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _can_run_deltanet_cuda(query, key, value, g, beta, initial_state, kwargs):
        out = _launch_gated_delta_rule_cuda(
            query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
        )
        KERNEL_CALL_STATS["chunk_gated_delta_rule"] += 1
        return out
    if _fallback is not None:
        return _fallback(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )
    raise RuntimeError("No fallback available for chunk_gated_delta_rule")


def nvrtc_rmsnorm_gated_silu(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dev = hidden_states.device
    nvcuda, fns = get_deltanet_kernels(dev)
    D = int(hidden_states.shape[-1])
    rows = int(hidden_states.numel() // D)
    if rows == 0:
        return hidden_states.contiguous()

    x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    g_c = gate if gate.is_contiguous() else gate.contiguous()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    out = torch.empty_like(x_c)

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_g = ctypes.c_void_p(g_c.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_o = ctypes.c_void_p(out.data_ptr())
    c_rows = ctypes.c_int(rows)
    c_D = ctypes.c_int(D)
    c_eps = ctypes.c_float(float(eps))
    c_xdt = ctypes.c_int(_DTYPE_CODE[x_c.dtype])
    c_gdt = ctypes.c_int(_DTYPE_CODE[g_c.dtype])
    c_wdt = ctypes.c_int(_DTYPE_CODE[w_c.dtype])

    args = (ctypes.c_void_p * 10)(
        ctypes.addressof(p_x), ctypes.addressof(p_g), ctypes.addressof(p_w), ctypes.addressof(p_o),
        ctypes.addressof(c_rows), ctypes.addressof(c_D), ctypes.addressof(c_eps),
        ctypes.addressof(c_xdt), ctypes.addressof(c_gdt), ctypes.addressof(c_wdt),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"fused_rmsnorm_gated_silu_kernel"], rows, 1, 1, 256, 1, 1, 0, stream, args, None),
        "fused_rmsnorm_gated_silu_kernel",
    )
    KERNEL_CALL_STATS["rmsnorm_gated"] += 1
    return out


def nvrtc_rmsnorm_zero_centered(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dev = hidden_states.device
    nvcuda, fns = get_deltanet_kernels(dev)
    D = int(hidden_states.shape[-1])
    rows = int(hidden_states.numel() // D)
    if rows == 0:
        return hidden_states.contiguous()

    x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    out = torch.empty_like(x_c)

    p_x = ctypes.c_void_p(x_c.data_ptr())
    p_w = ctypes.c_void_p(w_c.data_ptr())
    p_o = ctypes.c_void_p(out.data_ptr())
    c_rows = ctypes.c_int(rows)
    c_D = ctypes.c_int(D)
    c_eps = ctypes.c_float(float(eps))
    c_xdt = ctypes.c_int(_DTYPE_CODE[x_c.dtype])
    c_wdt = ctypes.c_int(_DTYPE_CODE[w_c.dtype])

    args = (ctypes.c_void_p * 8)(
        ctypes.addressof(p_x), ctypes.addressof(p_w), ctypes.addressof(p_o),
        ctypes.addressof(c_rows), ctypes.addressof(c_D), ctypes.addressof(c_eps),
        ctypes.addressof(c_xdt), ctypes.addressof(c_wdt),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    _check_launch(
        nvcuda.cuLaunchKernel(fns[b"fused_rmsnorm_zero_centered_kernel"], rows, 1, 1, 256, 1, 1, 0, stream, args, None),
        "fused_rmsnorm_zero_centered_kernel",
    )
    KERNEL_CALL_STATS["rmsnorm_zero_centered"] += 1
    return out


def patch_modeling_module(mod: Any) -> bool:
    """Replaces PyTorch fallback DeltaNet / CausalConv1d symbols on a `transformers` modeling module."""
    if mod is None:
        return False
    mod_name = getattr(mod, "__name__", "")
    if mod_name in _PATCHED_MODULES:
        return True

    patched = False
    if hasattr(mod, "causal_conv1d_fn"):
        raw_fn = inspect.unwrap(getattr(mod, "causal_conv1d_fn"))

        def _conv_fn(hidden_states, weight, bias=None, activation=None, _fb=raw_fn, **kwargs):
            return nvrtc_causal_conv1d_fn(hidden_states, weight, bias=bias, activation=activation, _fallback=_fb, **kwargs)

        mod.causal_conv1d_fn = _conv_fn
        patched = True

    if hasattr(mod, "causal_conv1d_update"):
        raw_upd = inspect.unwrap(getattr(mod, "causal_conv1d_update"))

        def _conv_upd(hidden_states, conv_state, weight, bias=None, activation=None, _fb=raw_upd, **kwargs):
            return nvrtc_causal_conv1d_update(
                hidden_states, conv_state, weight, bias=bias, activation=activation, _fallback=_fb, **kwargs
            )

        mod.causal_conv1d_update = _conv_upd
        patched = True

    if hasattr(mod, "torch_recurrent_gated_delta_rule"):
        raw_rec = inspect.unwrap(getattr(mod, "torch_recurrent_gated_delta_rule"))

        def _rec_fn(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            _fb=raw_rec,
            **kwargs,
        ):
            return nvrtc_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                _fallback=_fb,
                **kwargs,
            )

        mod.torch_recurrent_gated_delta_rule = _rec_fn
        patched = True

    if hasattr(mod, "torch_chunk_gated_delta_rule"):
        raw_chk = inspect.unwrap(getattr(mod, "torch_chunk_gated_delta_rule"))

        def _chk_fn(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            _fb=raw_chk,
            **kwargs,
        ):
            return nvrtc_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                _fallback=_fb,
                **kwargs,
            )

        mod.torch_chunk_gated_delta_rule = _chk_fn
        patched = True

    if patched and mod_name:
        _PATCHED_MODULES.add(mod_name)
    return patched


def install_deltanet_nvrtc_kernels(
    model: Any = None,
    patch_norms: bool = True,
    patch_zero_centered_norm: bool | None = None,
) -> dict[str, int]:
    """Installs NVRTC GatedDeltaNet, CausalConv1d, and RMSNorm kernels on `model` and its modeling module."""
    counts = {"patched_modules": 0, "rmsnorm_gated": 0, "rmsnorm_zero_centered": 0}
    if patch_zero_centered_norm is None:
        patch_zero_centered_norm = patch_norms

    for mod_name in _TARGET_MODELING_MODULES:
        loaded = sys.modules.get(mod_name)
        if loaded is not None and patch_modeling_module(loaded):
            counts["patched_modules"] += 1

    if model is not None:
        model_mod = inspect.getmodule(type(model))
        if model_mod is not None and patch_modeling_module(model_mod):
            counts["patched_modules"] += 1
        inner = getattr(model, "model", None)
        if inner is not None:
            inner_mod = inspect.getmodule(type(inner))
            if inner_mod is not None and patch_modeling_module(inner_mod):
                counts["patched_modules"] += 1

        if (patch_norms or patch_zero_centered_norm) and hasattr(model, "modules"):
            for submod in model.modules():
                cls_name = type(submod).__name__
                if patch_norms and cls_name.endswith("RMSNormGated") and getattr(submod, "activation", None) == "silu":
                    if not getattr(submod, "_colibri_nvrtc_norm", False):
                        orig_f = submod.forward
                        eps_v = float(getattr(submod, "variance_epsilon", getattr(submod, "eps", 1e-6)))

                        def _make_gated_fwd(orig_fn, eps_val):
                            def _fwd(self_m, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                                if (
                                    hidden_states.is_cuda
                                    and gate.is_cuda
                                    and self_m.weight.is_cuda
                                    and hidden_states.shape == gate.shape
                                    and hidden_states.dtype in _DTYPE_CODE
                                    and gate.dtype in _DTYPE_CODE
                                    and self_m.weight.dtype in _DTYPE_CODE
                                ):
                                    return nvrtc_rmsnorm_gated_silu(hidden_states, gate, self_m.weight, eps_val)
                                return orig_fn(hidden_states, gate)

                            return _fwd

                        submod.forward = types.MethodType(_make_gated_fwd(orig_f, eps_v), submod)
                        submod._colibri_nvrtc_norm = True
                        counts["rmsnorm_gated"] += 1
                elif patch_zero_centered_norm and cls_name in ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm", "Qwen3NextRMSNorm"):
                    if not getattr(submod, "_colibri_nvrtc_norm", False):
                        orig_f = submod.forward
                        eps_v = float(getattr(submod, "eps", getattr(submod, "variance_epsilon", 1e-6)))

                        def _make_zc_fwd(orig_fn, eps_val):
                            def _fwd(self_m, x: torch.Tensor) -> torch.Tensor:
                                if (
                                    x.is_cuda
                                    and self_m.weight.is_cuda
                                    and x.dtype in _DTYPE_CODE
                                    and self_m.weight.dtype in _DTYPE_CODE
                                ):
                                    return nvrtc_rmsnorm_zero_centered(x, self_m.weight, eps_val)
                                return orig_fn(x)

                            return _fwd

                        submod.forward = types.MethodType(_make_zc_fwd(orig_f, eps_v), submod)
                        submod._colibri_nvrtc_norm = True
                        counts["rmsnorm_zero_centered"] += 1

    return counts
