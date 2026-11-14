"""Standalone NVRTC CUDA Kernel Suite for AirColibri Unified Public Runtime.

Includes:
1. `GrandUnifiedKernels`:
   - `fused_zero_centered_rmsnorm_bf16`
   - `fused_dense_gate_up_swiglu_q4_bf16`
   - `fused_qkvzba_q4_bf16`
   - `fused_causal_conv1d_silu_batched`
   - `fused_gated_delta_recurrent_batched`
   - `indexed_vocab_gemv_q4_f32`
   - `persistent_dense_gemv_q4_bf16`
2. `SingleReadMultiRowKernels` (`template<int M>` for `M in {1, 2, 3, 4}`):
   - Reads every 4-bit weight byte ONCE across `M` speculative candidate tokens,
     using dynamic shared memory `M * 8448 B` for 83-100% SM occupancy and
     100% identical FP32 reduction tree across `M=1` and `M=2..4`.
3. `ZeroProbSubspaceVocabHead`:
   - Optional Gamma-weighted oblique SVD vocabulary dead-zone pruner (`-98.35%` dead rows).
"""
from __future__ import annotations

import ctypes
import torch

from src.colibri_nvrtc import _load_driver_and_nvrtc


UNIFIED_CUDA_SRC = r"""
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

extern "C" __global__ void fused_zero_centered_rmsnorm_bf16(
    const unsigned short* __restrict__ x_bf16,
    const unsigned short* __restrict__ w_bf16,
    unsigned short* __restrict__ y_bf16,
    int D,
    float eps
) {
    __shared__ float s_red[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    const unsigned short* xr = x_bf16 + (long long)row * D;
    unsigned short* yr = y_bf16 + (long long)row * D;

    float sq = 0.0f;
    for (int i = tid; i < D; i += 256) {
        float v = bf16_to_f32(xr[i]);
        sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) s_red[warp] = sq;
    __syncthreads();

    float total_sq = 0.0f;
    if (tid < 8) total_sq = s_red[tid];
    for (int off = 4; off > 0; off >>= 1) {
        total_sq += __shfl_down_sync(0xffffffffu, total_sq, off);
    }
    if (tid == 0) s_red[0] = rsqrtf(total_sq / (float)D + eps);
    __syncthreads();

    float inv_rms = s_red[0];
    for (int i = tid; i < D; i += 256) {
        float v = bf16_to_f32(xr[i]) * inv_rms * (1.0f + bf16_to_f32(w_bf16[i]));
        yr[i] = f32_to_bf16_rne(v);
    }
}

extern "C" __global__ void fused_dense_gate_up_swiglu_q4_bf16(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ g_q,
    const unsigned short* __restrict__ g_s,
    const unsigned short* __restrict__ g_z,
    const unsigned char* __restrict__ u_q,
    const unsigned short* __restrict__ u_s,
    const unsigned short* __restrict__ u_z,
    unsigned short* __restrict__ y_bf16,
    int M, int I, int K
) {
    extern __shared__ char smem_raw[];
    int num_groups = K >> 6;
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int m_idx = blockIdx.y;
    if (m_idx >= M) return;

    const unsigned short* xr = x_bf16 + (long long)m_idx * K;
    for (int c = tid; c < K; c += 256) {
        s_x[c] = xr[c];
    }
    __syncthreads();

    for (int g = tid; g < num_groups; g += 256) {
        float sum = 0.0f;
        int base = g << 6;
        #pragma unroll 8
        for (int j = 0; j < 64; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int row = (blockIdx.x << 3) + warp_id;
    if (row >= I) return;

    int packed_K = K >> 1;
    const uint4* gq_vec = reinterpret_cast<const uint4*>(g_q + (long long)row * packed_K);
    const uint4* uq_vec = reinterpret_cast<const uint4*>(u_q + (long long)row * packed_K);
    const unsigned short* gs_row = g_s + (long long)row * num_groups;
    const unsigned short* gz_row = g_z + (long long)row * num_groups;
    const unsigned short* us_row = u_s + (long long)row * num_groups;
    const unsigned short* uz_row = u_z + (long long)row * num_groups;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    int vec_count = packed_K >> 4;
    for (int v = lane; v < vec_count; v += 32) {
        uint4 qg = gq_vec[v];
        uint4 qu = uq_vec[v];
        int col_base = v << 5;
        int g = col_base >> 6;
        float svg = bf16_to_f32(gs_row[g]);
        float svu = bf16_to_f32(us_row[g]);
        unsigned int wg[4] = {qg.x, qg.y, qg.z, qg.w};
        unsigned int wu[4] = {qu.x, qu.y, qu.z, qu.w};
        float dot_g = 0.0f;
        float dot_u = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int wwg = wg[w];
            unsigned int wwu = wu[w];
            int c0 = col_base + (w << 3);
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                float xv = bf16_to_f32(s_x[c0 + nib]);
                dot_g += (float)((wwg >> (nib * 4)) & 0xFu) * xv;
                dot_u += (float)((wwu >> (nib * 4)) & 0xFu) * xv;
            }
        }
        acc_g += svg * dot_g;
        acc_u += svu * dot_u;
    }

    for (int g = lane; g < num_groups; g += 32) {
        float xs = s_xsum[g];
        acc_g += bf16_to_f32(gz_row[g]) * xs;
        acc_u += bf16_to_f32(uz_row[g]) * xs;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, off);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, off);
    }

    if (lane == 0) {
        float g_bf = bf16_to_f32(f32_to_bf16_rne(acc_g));
        float u_bf = bf16_to_f32(f32_to_bf16_rne(acc_u));
        float silu_g = bf16_to_f32(f32_to_bf16_rne(g_bf / (1.0f + expf(-g_bf))));
        y_bf16[(long long)m_idx * I + row] = f32_to_bf16_rne(silu_g * u_bf);
    }
}

extern "C" __global__ void fused_qkvzba_q4_bf16(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ qkv_q, const unsigned short* __restrict__ qkv_s, const unsigned short* __restrict__ qkv_z,
    const unsigned char* __restrict__ z_q,   const unsigned short* __restrict__ z_s,   const unsigned short* __restrict__ z_z,
    const unsigned char* __restrict__ b_q,   const unsigned short* __restrict__ b_s,   const unsigned short* __restrict__ b_z,
    const unsigned char* __restrict__ a_q,   const unsigned short* __restrict__ a_s,   const unsigned short* __restrict__ a_z,
    unsigned short* __restrict__ out_qkv,
    unsigned short* __restrict__ out_z,
    unsigned short* __restrict__ out_b,
    unsigned short* __restrict__ out_a,
    int M
) {
    __shared__ float s_xsum[64];
    __shared__ unsigned short s_x[4096];

    int tid = threadIdx.x;
    int m_idx = blockIdx.y;
    if (m_idx >= M) return;

    const unsigned short* xr = x_bf16 + (long long)m_idx * 4096;
    for (int c = tid; c < 4096; c += 256) s_x[c] = xr[c];
    __syncthreads();

    if (tid < 64) {
        float sum = 0.0f;
        int base = tid << 6;
        #pragma unroll 8
        for (int j = 0; j < 64; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[tid] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int global_row = (blockIdx.x << 3) + warp_id;
    if (global_row >= 12352) return;

    const unsigned char* q_ptr;
    const unsigned short* s_row;
    const unsigned short* z_row;
    unsigned short* dst_ptr;

    if (global_row < 8192) {
        int r = global_row;
        q_ptr = qkv_q + (long long)r * 2048;
        s_row = qkv_s + (long long)r * 64;
        z_row = qkv_z + (long long)r * 64;
        dst_ptr = out_qkv + (long long)m_idx * 8192 + r;
    } else if (global_row < 12288) {
        int r = global_row - 8192;
        q_ptr = z_q + (long long)r * 2048;
        s_row = z_s + (long long)r * 64;
        z_row = z_z + (long long)r * 64;
        dst_ptr = out_z + (long long)m_idx * 4096 + r;
    } else if (global_row < 12320) {
        int r = global_row - 12288;
        q_ptr = b_q + (long long)r * 2048;
        s_row = b_s + (long long)r * 64;
        z_row = b_z + (long long)r * 64;
        dst_ptr = out_b + (long long)m_idx * 32 + r;
    } else {
        int r = global_row - 12320;
        q_ptr = a_q + (long long)r * 2048;
        s_row = a_s + (long long)r * 64;
        z_row = a_z + (long long)r * 64;
        dst_ptr = out_a + (long long)m_idx * 32 + r;
    }

    const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
    float acc = 0.0f;
    for (int v = lane; v < 128; v += 32) {
        uint4 qv = q_vec[v];
        int col_base = v << 5;
        float sv = bf16_to_f32(s_row[col_base >> 6]);
        unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
        float qdot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int ww = words[w];
            int c0 = col_base + (w << 3);
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                qdot += (float)((ww >> (nib * 4)) & 0xFu) * bf16_to_f32(s_x[c0 + nib]);
            }
        }
        acc += sv * qdot;
    }
    for (int g = lane; g < 64; g += 32) {
        acc += bf16_to_f32(z_row[g]) * s_xsum[g];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        *dst_ptr = f32_to_bf16_rne(acc);
    }
}

extern "C" __global__ void fused_causal_conv1d_silu_batched(
    const unsigned short* __restrict__ x_bf16,
    unsigned short* __restrict__ conv_state_bf16,
    const unsigned short* __restrict__ weight_bf16,
    unsigned short* __restrict__ out_bf16,
    int state_len
) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int b = blockIdx.y;
    if (c >= 8192) return;

    unsigned short* st_row = conv_state_bf16 + ((long long)b * 8192 + c) * state_len;
    unsigned short x_new = x_bf16[(long long)b * 8192 + c];

    unsigned short s_m3 = (state_len >= 3) ? st_row[state_len - 3] : 0u;
    unsigned short s_m2 = (state_len >= 2) ? st_row[state_len - 2] : 0u;
    unsigned short s_m1 = (state_len >= 1) ? st_row[state_len - 1] : 0u;

    for (int t = 0; t < state_len - 1; ++t) st_row[t] = st_row[t + 1];
    if (state_len >= 1) st_row[state_len - 1] = x_new;

    const unsigned short* w_row = weight_bf16 + c * 4;
    float acc = bf16_to_f32(s_m3) * bf16_to_f32(w_row[0])
              + bf16_to_f32(s_m2) * bf16_to_f32(w_row[1])
              + bf16_to_f32(s_m1) * bf16_to_f32(w_row[2])
              + bf16_to_f32(x_new) * bf16_to_f32(w_row[3]);
    float conv_bf = bf16_to_f32(f32_to_bf16_rne(acc));
    out_bf16[(long long)b * 8192 + c] = f32_to_bf16_rne(conv_bf / (1.0f + expf(-conv_bf)));
}

extern "C" __global__ void fused_gated_delta_recurrent_batched(
    const unsigned short* __restrict__ qkv_bf16,
    const unsigned short* __restrict__ z_bf16,
    const unsigned short* __restrict__ b_bf16,
    const unsigned short* __restrict__ a_bf16,
    const void* __restrict__ A_log_ptr,
    const void* __restrict__ dt_bias_ptr,
    const unsigned short* __restrict__ norm_w_bf16,
    float* __restrict__ state_f32,
    unsigned short* __restrict__ out_bf16,
    int a_log_is_bf16,
    int dt_bias_is_bf16
) {
    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float s_red0[4];
    __shared__ float s_red1[4];

    int vh = blockIdx.x;
    int bi = blockIdx.y;
    int kh = vh >> 1;
    int j = threadIdx.x;
    int warp = j >> 5;
    int lane = j & 31;

    const unsigned short* qkv_b = qkv_bf16 + (long long)bi * 8192;
    float q_raw = bf16_to_f32(qkv_b[kh * 128 + j]);
    float k_raw = bf16_to_f32(qkv_b[2048 + kh * 128 + j]);
    float v_val = bf16_to_f32(qkv_b[4096 + vh * 128 + j]);

    float wq = q_raw * q_raw;
    float wk = k_raw * k_raw;
    for (int off = 16; off > 0; off >>= 1) {
        wq += __shfl_down_sync(0xffffffffu, wq, off);
        wk += __shfl_down_sync(0xffffffffu, wk, off);
    }
    if (lane == 0) { s_red0[warp] = wq; s_red1[warp] = wk; }
    __syncthreads();

    float inv_q = rsqrtf(s_red0[0] + s_red0[1] + s_red0[2] + s_red0[3] + 1e-6f) * 0.08838834764831845f;
    float inv_k = rsqrtf(s_red1[0] + s_red1[1] + s_red1[2] + s_red1[3] + 1e-6f);
    s_q[j] = q_raw * inv_q;
    s_k[j] = k_raw * inv_k;
    __syncthreads();

    float b_val = bf16_to_f32(b_bf16[bi * 32 + vh]);
    float beta_t = 1.0f / (1.0f + expf(-b_val));
    float a_log_v = a_log_is_bf16 ? bf16_to_f32(reinterpret_cast<const unsigned short*>(A_log_ptr)[vh])
                                  : reinterpret_cast<const float*>(A_log_ptr)[vh];
    float dt_b_v  = dt_bias_is_bf16 ? bf16_to_f32(reinterpret_cast<const unsigned short*>(dt_bias_ptr)[vh])
                                    : reinterpret_cast<const float*>(dt_bias_ptr)[vh];
    float sp_in = bf16_to_f32(a_bf16[bi * 32 + vh]) + dt_b_v;
    float softplus_v = (sp_in > 20.0f) ? sp_in : log1pf(expf(sp_in));
    float decay_t = expf(-expf(a_log_v) * softplus_v);

    float* st_head = state_f32 + ((long long)bi * 32 + vh) * (128 * 128);
    float kv_mem_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) {
        float s_ij = st_head[i * 128 + j] * decay_t;
        st_head[i * 128 + j] = s_ij;
        kv_mem_j += s_ij * s_k[i];
    }
    float delta_j = (v_val - kv_mem_j) * beta_t;

    float out_j = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < 128; ++i) {
        float s_ij = st_head[i * 128 + j] + s_k[i] * delta_j;
        st_head[i * 128 + j] = s_ij;
        out_j += s_ij * s_q[i];
    }

    float out_bf_f32 = bf16_to_f32(f32_to_bf16_rne(out_j));
    float sq = out_bf_f32 * out_bf_f32;
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
    if (lane == 0) s_red0[warp] = sq;
    __syncthreads();

    float mean_sq = (s_red0[0] + s_red0[1] + s_red0[2] + s_red0[3]) * (1.0f / 128.0f);
    float normed_bf = bf16_to_f32(f32_to_bf16_rne(out_bf_f32 * rsqrtf(mean_sq + 1e-6f)));
    float weighted_bf = bf16_to_f32(f32_to_bf16_rne(bf16_to_f32(norm_w_bf16[j]) * normed_bf));
    float z_f32 = bf16_to_f32(z_bf16[(long long)bi * 4096 + vh * 128 + j]);
    out_bf16[(long long)bi * 4096 + vh * 128 + j] = f32_to_bf16_rne(weighted_bf * (z_f32 / (1.0f + expf(-z_f32))));
}

extern "C" __global__ void indexed_vocab_gemv_q4_f32(
    const unsigned short* __restrict__ x_bf16,
    const int* __restrict__ cand_idx,
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    float* __restrict__ out_scores,
    int M, int C_surv, int K
) {
    extern __shared__ char smem_raw[];
    int num_groups = K >> 6;
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int m_idx = blockIdx.y;
    if (m_idx >= M) return;

    const unsigned short* xr = x_bf16 + (long long)m_idx * K;
    for (int c = tid; c < K; c += 256) s_x[c] = xr[c];
    __syncthreads();

    for (int g = tid; g < num_groups; g += 256) {
        float sum = 0.0f;
        int base = g << 6;
        #pragma unroll 8
        for (int j = 0; j < 64; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int slot = (blockIdx.x << 3) + warp_id;
    if (slot >= C_surv) return;

    int row = cand_idx[(long long)m_idx * C_surv + slot];
    int packed_K = K >> 1;
    const uint4* q_vec = reinterpret_cast<const uint4*>(qweight + (long long)row * packed_K);
    const unsigned short* s_row = scales + (long long)row * num_groups;
    const unsigned short* z_row = zeros  + (long long)row * num_groups;

    float acc = 0.0f;
    int vec_count = packed_K >> 4;
    for (int v = lane; v < vec_count; v += 32) {
        uint4 qv = q_vec[v];
        int col_base = v << 5;
        float sv = bf16_to_f32(s_row[col_base >> 6]);
        unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
        float qdot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int ww = words[w];
            int c0 = col_base + (w << 3);
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                qdot += (float)((ww >> (nib * 4)) & 0xFu) * bf16_to_f32(s_x[c0 + nib]);
            }
        }
        acc += sv * qdot;
    }
    for (int g = lane; g < num_groups; g += 32) {
        acc += bf16_to_f32(z_row[g]) * s_xsum[g];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        out_scores[(long long)m_idx * C_surv + slot] = acc;
    }
}

extern "C" __global__ void persistent_dense_gemv_q4_bf16(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    unsigned short* __restrict__ y_bf16,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    int num_groups = K >> 6;
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int m_idx = blockIdx.y;
    if (m_idx >= M) return;

    const unsigned short* xr = x_bf16 + (long long)m_idx * K;
    for (int c = tid; c < K; c += 256) s_x[c] = xr[c];
    __syncthreads();

    for (int g = tid; g < num_groups; g += 256) {
        float sum = 0.0f;
        int base = g << 6;
        #pragma unroll 8
        for (int j = 0; j < 64; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int packed_K = K >> 1;
    int vec_count = packed_K >> 4;
    int total_warps = gridDim.x << 3;
    int start_row = (blockIdx.x << 3) + warp_id;

    for (int row = start_row; row < N; row += total_warps) {
        const uint4* q_vec = reinterpret_cast<const uint4*>(qweight + (long long)row * packed_K);
        const unsigned short* s_row = scales + (long long)row * num_groups;
        const unsigned short* z_row = zeros  + (long long)row * num_groups;

        float acc = 0.0f;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 qv = q_vec[v];
            int col_base = v << 5;
            float sv = bf16_to_f32(s_row[col_base >> 6]);
            unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
            float qdot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                unsigned int ww = words[w];
                int c0 = col_base + (w << 3);
                #pragma unroll
                for (int nib = 0; nib < 8; ++nib) {
                    qdot += (float)((ww >> (nib * 4)) & 0xFu) * bf16_to_f32(s_x[c0 + nib]);
                }
            }
            acc += sv * qdot;
        }
        for (int g = lane; g < num_groups; g += 32) {
            acc += bf16_to_f32(z_row[g]) * s_xsum[g];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            y_bf16[(long long)m_idx * N + row] = f32_to_bf16_rne(acc);
        }
    }
}
"""


MULTIROW_SINGLE_READ_CU = r"""
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

template<int M>
__device__ __forceinline__ void multirow_dense_gemv_impl(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ zeros,
    const unsigned short* __restrict__ bias_bf16,
    unsigned short* __restrict__ y_bf16,
    int N, int K
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + M * 64);

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int packed_K = K >> 1;
    int num_groups = K >> 6;
    int total_warps = gridDim.x << 3;
    int start_row = (blockIdx.x << 3) + warp_id;

    if (K <= 4096) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const unsigned short* xr = x_bf16 + (long long)m * K;
            unsigned short* sxm = s_x + m * 4096;
            for (int c = tid; c < K; c += 256) sxm[c] = xr[c];
        }
        __syncthreads();
        if (tid < num_groups) {
            int base = tid << 6;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const unsigned short* sxm = s_x + m * 4096;
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < 64; ++j) sum += bf16_to_f32(sxm[base + j]);
                s_xsum[m * 64 + tid] = sum;
            }
        }
        __syncthreads();

        int vec_count = packed_K >> 4;
        for (int row = start_row; row < N; row += total_warps) {
            const uint4* q_vec = reinterpret_cast<const uint4*>(qweight + (long long)row * packed_K);
            const unsigned short* s_row = scales + (long long)row * num_groups;
            const unsigned short* z_row = zeros  + (long long)row * num_groups;
            float acc[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.0f;

            for (int v = lane; v < vec_count; v += 32) {
                uint4 qv = q_vec[v];
                int col_base = v << 5;
                float sv = bf16_to_f32(s_row[col_base >> 6]);
                unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
                float qdot[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) qdot[m] = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int ww = words[w];
                    int c0 = col_base + (w << 3);
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        float qf = (float)((ww >> (nib * 4)) & 0xFu);
                        int idx = c0 + nib;
                        #pragma unroll
                        for (int m = 0; m < M; ++m) {
                            qdot[m] += qf * bf16_to_f32(s_x[m * 4096 + idx]);
                        }
                    }
                }
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += sv * qdot[m];
            }
            for (int g = lane; g < num_groups; g += 32) {
                float zv = bf16_to_f32(z_row[g]);
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += zv * s_xsum[m * 64 + g];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
                }
            }
            if (lane == 0) {
                float bval = (bias_bf16 != nullptr) ? bf16_to_f32(bias_bf16[row]) : 0.0f;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    y_bf16[(long long)m * N + row] = f32_to_bf16_rne(acc[m] + bval);
                }
            }
        }
        return;
    }

    int row = start_row;
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int k_base = 0; k_base < K; k_base += 4096) {
        if (k_base > 0) __syncthreads();
        int chunk_K = (K - k_base < 4096) ? (K - k_base) : 4096;
        int chunk_groups = chunk_K >> 6;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const unsigned short* xr = x_bf16 + (long long)m * K + k_base;
            unsigned short* sxm = s_x + m * 4096;
            for (int c = tid; c < chunk_K; c += 256) sxm[c] = xr[c];
        }
        __syncthreads();
        if (tid < chunk_groups) {
            int base = tid << 6;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const unsigned short* sxm = s_x + m * 4096;
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < 64; ++j) sum += bf16_to_f32(sxm[base + j]);
                s_xsum[m * 64 + tid] = sum;
            }
        }
        __syncthreads();

        if (row < N) {
            const uint4* q_vec = reinterpret_cast<const uint4*>(qweight + (long long)row * packed_K + (k_base >> 1));
            const unsigned short* s_row = scales + (long long)row * num_groups + (k_base >> 6);
            const unsigned short* z_row = zeros  + (long long)row * num_groups + (k_base >> 6);
            int vec_count = chunk_K >> 5;
            for (int v = lane; v < vec_count; v += 32) {
                uint4 qv = q_vec[v];
                int col_base = v << 5;
                float sv = bf16_to_f32(s_row[col_base >> 6]);
                unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
                float qdot[M];
                #pragma unroll
                for (int m = 0; m < M; ++m) qdot[m] = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int ww = words[w];
                    int c0 = col_base + (w << 3);
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        float qf = (float)((ww >> (nib * 4)) & 0xFu);
                        int idx = c0 + nib;
                        #pragma unroll
                        for (int m = 0; m < M; ++m) {
                            qdot[m] += qf * bf16_to_f32(s_x[m * 4096 + idx]);
                        }
                    }
                }
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += sv * qdot[m];
            }
            for (int g = lane; g < chunk_groups; g += 32) {
                float zv = bf16_to_f32(z_row[g]);
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += zv * s_xsum[m * 64 + g];
            }
        }
    }
    if (row < N) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
            }
        }
        if (lane == 0) {
            float bval = (bias_bf16 != nullptr) ? bf16_to_f32(bias_bf16[row]) : 0.0f;
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                y_bf16[(long long)m * N + row] = f32_to_bf16_rne(acc[m] + bval);
            }
        }
    }
}

template<int M>
__device__ __forceinline__ void multirow_gate_up_swiglu_impl(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ g_q,
    const unsigned short* __restrict__ g_s,
    const unsigned short* __restrict__ g_z,
    const unsigned char* __restrict__ u_q,
    const unsigned short* __restrict__ u_s,
    const unsigned short* __restrict__ u_z,
    unsigned short* __restrict__ y_bf16,
    int I
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + M * 64);

    int tid = threadIdx.x;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const unsigned short* xr = x_bf16 + (long long)m * 4096;
        unsigned short* sxm = s_x + m * 4096;
        for (int c = tid; c < 4096; c += 256) sxm[c] = xr[c];
    }
    __syncthreads();
    if (tid < 64) {
        int base = tid << 6;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const unsigned short* sxm = s_x + m * 4096;
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < 64; ++j) sum += bf16_to_f32(sxm[base + j]);
            s_xsum[m * 64 + tid] = sum;
        }
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int row = (blockIdx.x << 3) + warp_id;
    if (row >= I) return;

    const uint4* gq_vec = reinterpret_cast<const uint4*>(g_q + (long long)row * 2048);
    const uint4* uq_vec = reinterpret_cast<const uint4*>(u_q + (long long)row * 2048);
    const unsigned short* gs_row = g_s + (long long)row * 64;
    const unsigned short* gz_row = g_z + (long long)row * 64;
    const unsigned short* us_row = u_s + (long long)row * 64;
    const unsigned short* uz_row = u_z + (long long)row * 64;

    float acc_g[M], acc_u[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (int v = lane; v < 128; v += 32) {
        uint4 qg = gq_vec[v];
        uint4 qu = uq_vec[v];
        int col_base = v << 5;
        int g = col_base >> 6;
        float svg = bf16_to_f32(gs_row[g]);
        float svu = bf16_to_f32(us_row[g]);
        unsigned int wg[4] = {qg.x, qg.y, qg.z, qg.w};
        unsigned int wu[4] = {qu.x, qu.y, qu.z, qu.w};
        float dg[M], du[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) { dg[m] = 0.0f; du[m] = 0.0f; }
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int wwg = wg[w];
            unsigned int wwu = wu[w];
            int c0 = col_base + (w << 3);
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                float qfg = (float)((wwg >> (nib * 4)) & 0xFu);
                float qfu = (float)((wwu >> (nib * 4)) & 0xFu);
                int idx = c0 + nib;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xm = bf16_to_f32(s_x[m * 4096 + idx]);
                    dg[m] += qfg * xm;
                    du[m] += qfu * xm;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            acc_g[m] += svg * dg[m];
            acc_u[m] += svu * du[m];
        }
    }
    for (int g = lane; g < 64; g += 32) {
        float gzv = bf16_to_f32(gz_row[g]);
        float uzv = bf16_to_f32(uz_row[g]);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float xsm = s_xsum[m * 64 + g];
            acc_g[m] += gzv * xsm;
            acc_u[m] += uzv * xsm;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            acc_g[m] += __shfl_down_sync(0xffffffffu, acc_g[m], off);
            acc_u[m] += __shfl_down_sync(0xffffffffu, acc_u[m], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float g_bf = bf16_to_f32(f32_to_bf16_rne(acc_g[m]));
            float u_bf = bf16_to_f32(f32_to_bf16_rne(acc_u[m]));
            float silu_g = bf16_to_f32(f32_to_bf16_rne(g_bf / (1.0f + expf(-g_bf))));
            y_bf16[(long long)m * I + row] = f32_to_bf16_rne(silu_g * u_bf);
        }
    }
}

template<int M>
__device__ __forceinline__ void multirow_qkvzba_impl(
    const unsigned short* __restrict__ x_bf16,
    const unsigned char* __restrict__ qkv_q, const unsigned short* __restrict__ qkv_s, const unsigned short* __restrict__ qkv_z,
    const unsigned char* __restrict__ z_q,   const unsigned short* __restrict__ z_s,   const unsigned short* __restrict__ z_z,
    const unsigned char* __restrict__ b_q,   const unsigned short* __restrict__ b_s,   const unsigned short* __restrict__ b_z,
    const unsigned char* __restrict__ a_q,   const unsigned short* __restrict__ a_s,   const unsigned short* __restrict__ a_z,
    unsigned short* __restrict__ out_qkv,
    unsigned short* __restrict__ out_z,
    unsigned short* __restrict__ out_b,
    unsigned short* __restrict__ out_a
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + M * 64);

    int tid = threadIdx.x;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const unsigned short* xr = x_bf16 + (long long)m * 4096;
        unsigned short* sxm = s_x + m * 4096;
        for (int c = tid; c < 4096; c += 256) sxm[c] = xr[c];
    }
    __syncthreads();
    if (tid < 64) {
        int base = tid << 6;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const unsigned short* sxm = s_x + m * 4096;
            float sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < 64; ++j) sum += bf16_to_f32(sxm[base + j]);
            s_xsum[m * 64 + tid] = sum;
        }
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int global_row = (blockIdx.x << 3) + warp_id;
    if (global_row >= 12352) return;

    const unsigned char* q_ptr;
    const unsigned short* s_row;
    const unsigned short* z_row;
    unsigned short* dst_base;
    int dst_stride;
    int r_local;

    if (global_row < 8192) {
        r_local = global_row;
        q_ptr = qkv_q + (long long)r_local * 2048;
        s_row = qkv_s + (long long)r_local * 64;
        z_row = qkv_z + (long long)r_local * 64;
        dst_base = out_qkv;
        dst_stride = 8192;
    } else if (global_row < 12288) {
        r_local = global_row - 8192;
        q_ptr = z_q + (long long)r_local * 2048;
        s_row = z_s + (long long)r_local * 64;
        z_row = z_z + (long long)r_local * 64;
        dst_base = out_z;
        dst_stride = 4096;
    } else if (global_row < 12320) {
        r_local = global_row - 12288;
        q_ptr = b_q + (long long)r_local * 2048;
        s_row = b_s + (long long)r_local * 64;
        z_row = b_z + (long long)r_local * 64;
        dst_base = out_b;
        dst_stride = 32;
    } else {
        r_local = global_row - 12320;
        q_ptr = a_q + (long long)r_local * 2048;
        s_row = a_s + (long long)r_local * 64;
        z_row = a_z + (long long)r_local * 64;
        dst_base = out_a;
        dst_stride = 32;
    }

    const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int v = lane; v < 128; v += 32) {
        uint4 qv = q_vec[v];
        int col_base = v << 5;
        float sv = bf16_to_f32(s_row[col_base >> 6]);
        unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
        float qdot[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) qdot[m] = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            unsigned int ww = words[w];
            int c0 = col_base + (w << 3);
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                float qf = (float)((ww >> (nib * 4)) & 0xFu);
                int idx = c0 + nib;
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    qdot[m] += qf * bf16_to_f32(s_x[m * 4096 + idx]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] += sv * qdot[m];
    }
    for (int g = lane; g < 64; g += 32) {
        float zv = bf16_to_f32(z_row[g]);
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] += zv * s_xsum[m * 64 + g];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            dst_base[(long long)m * dst_stride + r_local] = f32_to_bf16_rne(acc[m]);
        }
    }
}

#define EXPORT_MR_KERNELS(MVAL) \
extern "C" __global__ void multirow_dense_gemv_m##MVAL( \
    const unsigned short* __restrict__ x_bf16, \
    const unsigned char* __restrict__ qweight, \
    const unsigned short* __restrict__ scales, \
    const unsigned short* __restrict__ zeros, \
    const unsigned short* __restrict__ bias_bf16, \
    unsigned short* __restrict__ y_bf16, \
    int N, int K \
) { multirow_dense_gemv_impl<MVAL>(x_bf16, qweight, scales, zeros, bias_bf16, y_bf16, N, K); } \
extern "C" __global__ void multirow_gate_up_swiglu_m##MVAL( \
    const unsigned short* __restrict__ x_bf16, \
    const unsigned char* __restrict__ g_q, const unsigned short* __restrict__ g_s, const unsigned short* __restrict__ g_z, \
    const unsigned char* __restrict__ u_q, const unsigned short* __restrict__ u_s, const unsigned short* __restrict__ u_z, \
    unsigned short* __restrict__ y_bf16, int I \
) { multirow_gate_up_swiglu_impl<MVAL>(x_bf16, g_q, g_s, g_z, u_q, u_s, u_z, y_bf16, I); } \
extern "C" __global__ void multirow_qkvzba_m##MVAL( \
    const unsigned short* __restrict__ x_bf16, \
    const unsigned char* __restrict__ qkv_q, const unsigned short* __restrict__ qkv_s, const unsigned short* __restrict__ qkv_z, \
    const unsigned char* __restrict__ z_q,   const unsigned short* __restrict__ z_s,   const unsigned short* __restrict__ z_z, \
    const unsigned char* __restrict__ b_q,   const unsigned short* __restrict__ b_s,   const unsigned short* __restrict__ b_z, \
    const unsigned char* __restrict__ a_q,   const unsigned short* __restrict__ a_s,   const unsigned short* __restrict__ a_z, \
    unsigned short* __restrict__ out_qkv, unsigned short* __restrict__ out_z, \
    unsigned short* __restrict__ out_b,   unsigned short* __restrict__ out_a \
) { multirow_qkvzba_impl<MVAL>(x_bf16, qkv_q, qkv_s, qkv_z, z_q, z_s, z_z, b_q, b_s, b_z, a_q, a_s, a_z, out_qkv, out_z, out_b, out_a); }

EXPORT_MR_KERNELS(1)
EXPORT_MR_KERNELS(2)
EXPORT_MR_KERNELS(3)
EXPORT_MR_KERNELS(4)
"""


class GrandUnifiedKernels:
    def __init__(self, device: torch.device, max_batch: int = 8):
        self.device = device
        self.nvcuda, self.nvrtc = _load_driver_and_nvrtc()
        major, minor = torch.cuda.get_device_capability(device)

        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), UNIFIED_CUDA_SRC.encode("utf-8"), b"grand_unified.cu", 0, None, None
        )
        if rc != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")
        try:
            arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
            opts = (ctypes.c_char_p * 1)(arch_flag)
            rc = self.nvrtc.nvrtcCompileProgram(prog, 1, opts)
            if rc != 0:
                lsz = ctypes.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(lsz))
                lbuf = ctypes.create_string_buffer(lsz.value)
                self.nvrtc.nvrtcGetProgramLog(prog, lbuf)
                raise RuntimeError(f"NVRTC compile failed ({rc}):\n{lbuf.value.decode('utf-8', errors='replace')}")
            psz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(psz))
            pbuf = ctypes.create_string_buffer(psz.value)
            self.nvrtc.nvrtcGetPTX(prog, pbuf)
            ptx = pbuf.raw if pbuf.raw.endswith(b"\x00") else pbuf.raw + b"\x00"
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        torch.cuda.init()
        _ = torch.empty(1, device=device)
        self.mod = ctypes.c_void_p()
        rc = self.nvcuda.cuModuleLoadData(ctypes.byref(self.mod), ctypes.c_char_p(ptx))
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {rc}")

        self.fns = {}
        for nm in (
            b"fused_zero_centered_rmsnorm_bf16",
            b"fused_dense_gate_up_swiglu_q4_bf16",
            b"fused_qkvzba_q4_bf16",
            b"fused_causal_conv1d_silu_batched",
            b"fused_gated_delta_recurrent_batched",
            b"indexed_vocab_gemv_q4_f32",
            b"persistent_dense_gemv_q4_bf16",
        ):
            fn = ctypes.c_void_p()
            rc = self.nvcuda.cuModuleGetFunction(ctypes.byref(fn), self.mod, nm)
            if rc != 0:
                raise RuntimeError(f"cuModuleGetFunction({nm}) failed: {rc}")
            self.fns[nm] = fn

        self.ws_mlp_act = torch.empty((max_batch, 12288), dtype=torch.bfloat16, device=device)
        self.ws_qkv = torch.empty((max_batch, 8192), dtype=torch.bfloat16, device=device)
        self.ws_qkv_conv = torch.empty((max_batch, 8192), dtype=torch.bfloat16, device=device)
        self.ws_z = torch.empty((max_batch, 4096), dtype=torch.bfloat16, device=device)
        self.ws_b = torch.empty((max_batch, 32), dtype=torch.bfloat16, device=device)
        self.ws_a = torch.empty((max_batch, 32), dtype=torch.bfloat16, device=device)
        self.ws_core = torch.empty((max_batch, 4096), dtype=torch.bfloat16, device=device)

    def persistent_gemv_q4(self, x: torch.Tensor, q4_mod) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        N = int(q4_mod.out_features)
        x_2d = x if x.is_contiguous() else x.contiguous()
        y_2d = torch.empty((M, N), dtype=torch.bfloat16, device=self.device)
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        smem = (K // 64) * 4 + K * 2
        c_x = ctypes.c_uint64(x_2d.data_ptr())
        c_qw = ctypes.c_uint64(q4_mod.qweight.data_ptr())
        c_sc = ctypes.c_uint64(q4_mod.scales.data_ptr())
        c_zr = ctypes.c_uint64(q4_mod.zeros.data_ptr())
        c_y = ctypes.c_uint64(y_2d.data_ptr())
        c_M = ctypes.c_int(M)
        c_N = ctypes.c_int(N)
        c_K = ctypes.c_int(K)
        params = (ctypes.c_void_p * 8)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_sc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_zr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_N), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        )
        grid_x = min((N + 7) // 8, 304 if N >= 32768 else 512)
        rc = self.nvcuda.cuLaunchKernel(
            self.fns[b"persistent_dense_gemv_q4_bf16"], grid_x, M, 1, 256, 1, 1, smem, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"persistent_gemv launch failed: {rc}")
        return y_2d.view(*orig_shape[:-1], N)

    def rmsnorm_zero_centered(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        D = x.shape[-1]
        R = x.numel() // D
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        c_x = ctypes.c_uint64(x.data_ptr())
        c_w = ctypes.c_uint64(weight.data_ptr())
        c_y = ctypes.c_uint64(out.data_ptr())
        c_D = ctypes.c_int(D)
        c_eps = ctypes.c_float(eps)
        params = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_D), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_eps), ctypes.c_void_p),
        )
        rc = self.nvcuda.cuLaunchKernel(
            self.fns[b"fused_zero_centered_rmsnorm_bf16"], R, 1, 1, 256, 1, 1, 0, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"rmsnorm launch failed: {rc}")
        return out

    def fused_mlp_gate_up_swiglu(self, x: torch.Tensor, gate_mod, up_mod) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        I = int(gate_mod.out_features)
        x_2d = x if x.is_contiguous() else x.contiguous()
        y_view = (
            self.ws_mlp_act[:M]
            if (M <= self.ws_mlp_act.shape[0] and I == 12288)
            else torch.empty((M, I), dtype=torch.bfloat16, device=self.device)
        )
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        smem = (K // 64) * 4 + K * 2
        c_x = ctypes.c_uint64(x_2d.data_ptr())
        c_gq = ctypes.c_uint64(gate_mod.qweight.data_ptr())
        c_gs = ctypes.c_uint64(gate_mod.scales.data_ptr())
        c_gz = ctypes.c_uint64(gate_mod.zeros.data_ptr())
        c_uq = ctypes.c_uint64(up_mod.qweight.data_ptr())
        c_us = ctypes.c_uint64(up_mod.scales.data_ptr())
        c_uz = ctypes.c_uint64(up_mod.zeros.data_ptr())
        c_y = ctypes.c_uint64(y_view.data_ptr())
        c_M = ctypes.c_int(M)
        c_I = ctypes.c_int(I)
        c_K = ctypes.c_int(K)
        params = (ctypes.c_void_p * 11)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gq), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_uq), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_us), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_uz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_I), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        )
        grid_x = (I + 7) // 8
        rc = self.nvcuda.cuLaunchKernel(
            self.fns[b"fused_dense_gate_up_swiglu_q4_bf16"], grid_x, M, 1, 256, 1, 1, smem, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"fused_mlp launch failed: {rc}")
        return y_view.view(*orig_shape[:-1], I)

    def fused_deltanet_full_step(
        self, lat_mod, hidden_states: torch.Tensor, conv_state: torch.Tensor, recurrent_state: torch.Tensor
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

        c_x = ctypes.c_uint64(x_c.data_ptr())
        c_qkv_q = ctypes.c_uint64(lat_mod.in_proj_qkv.qweight.data_ptr())
        c_qkv_s = ctypes.c_uint64(lat_mod.in_proj_qkv.scales.data_ptr())
        c_qkv_z = ctypes.c_uint64(lat_mod.in_proj_qkv.zeros.data_ptr())
        c_z_q = ctypes.c_uint64(lat_mod.in_proj_z.qweight.data_ptr())
        c_z_s = ctypes.c_uint64(lat_mod.in_proj_z.scales.data_ptr())
        c_z_z = ctypes.c_uint64(lat_mod.in_proj_z.zeros.data_ptr())
        c_b_q = ctypes.c_uint64(lat_mod.in_proj_b.qweight.data_ptr())
        c_b_s = ctypes.c_uint64(lat_mod.in_proj_b.scales.data_ptr())
        c_b_z = ctypes.c_uint64(lat_mod.in_proj_b.zeros.data_ptr())
        c_a_q = ctypes.c_uint64(lat_mod.in_proj_a.qweight.data_ptr())
        c_a_s = ctypes.c_uint64(lat_mod.in_proj_a.scales.data_ptr())
        c_a_z = ctypes.c_uint64(lat_mod.in_proj_a.zeros.data_ptr())

        c_oqkv = ctypes.c_uint64(self.ws_qkv.data_ptr())
        c_oz = ctypes.c_uint64(self.ws_z.data_ptr())
        c_ob = ctypes.c_uint64(self.ws_b.data_ptr())
        c_oa = ctypes.c_uint64(self.ws_a.data_ptr())
        c_M = ctypes.c_int(B)

        p_proj = (ctypes.c_void_p * 18)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qkv_q), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qkv_s), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qkv_z), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_z_q), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_z_s), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_z_z), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_b_q), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_b_s), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_b_z), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_a_q), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_a_s), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_a_z), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_oqkv), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_oz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ob), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_oa), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
        )
        rc0 = self.nvcuda.cuLaunchKernel(self.fns[b"fused_qkvzba_q4_bf16"], 1544, B, 1, 256, 1, 1, 0, stream_ptr, p_proj, None)
        if rc0 != 0:
            raise RuntimeError(f"fused_qkvzba failed: {rc0}")

        w2d = lat_mod.conv1d.weight.squeeze(1)
        if not w2d.is_contiguous():
            w2d = w2d.contiguous()
        c_st = ctypes.c_uint64(conv_state.data_ptr())
        c_w = ctypes.c_uint64(w2d.data_ptr())
        c_qkv_c = ctypes.c_uint64(self.ws_qkv_conv.data_ptr())
        c_slen = ctypes.c_int(int(conv_state.shape[-1]))
        p_conv = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_oqkv), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_st), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_w), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qkv_c), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_slen), ctypes.c_void_p),
        )
        rc1 = self.nvcuda.cuLaunchKernel(
            self.fns[b"fused_causal_conv1d_silu_batched"], 32, B, 1, 256, 1, 1, 0, stream_ptr, p_conv, None
        )
        if rc1 != 0:
            raise RuntimeError(f"conv_batched failed: {rc1}")

        c_alog = ctypes.c_uint64(lat_mod.A_log.data_ptr())
        c_dtb = ctypes.c_uint64(lat_mod.dt_bias.data_ptr())
        c_nw = ctypes.c_uint64(lat_mod.norm.weight.data_ptr())
        c_rec = ctypes.c_uint64(recurrent_state.data_ptr())
        c_core = ctypes.c_uint64(self.ws_core.data_ptr())
        c_abf = ctypes.c_int(1 if lat_mod.A_log.dtype == torch.bfloat16 else 0)
        c_dbf = ctypes.c_int(1 if lat_mod.dt_bias.dtype == torch.bfloat16 else 0)
        p_rec = (ctypes.c_void_p * 11)(
            ctypes.cast(ctypes.byref(c_qkv_c), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_oz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ob), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_oa), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_alog), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dtb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_nw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_rec), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_core), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_abf), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_dbf), ctypes.c_void_p),
        )
        rc2 = self.nvcuda.cuLaunchKernel(
            self.fns[b"fused_gated_delta_recurrent_batched"], 32, B, 1, 128, 1, 1, 0, stream_ptr, p_rec, None
        )
        if rc2 != 0:
            raise RuntimeError(f"delta_batched failed: {rc2}")

        return lat_mod.out_proj(self.ws_core[:B].view(B, 1, 4096))

    def indexed_vocab_scores(self, x_bf16: torch.Tensor, cand_idx_i32: torch.Tensor, lm_head_mod) -> torch.Tensor:
        K = x_bf16.shape[-1]
        M = x_bf16.numel() // K
        C_surv = cand_idx_i32.shape[-1]
        x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
        idx_c = cand_idx_i32 if cand_idx_i32.is_contiguous() else cand_idx_i32.contiguous()
        out_f32 = torch.empty((M, C_surv), dtype=torch.float32, device=self.device)

        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        smem = (K // 64) * 4 + K * 2
        c_x = ctypes.c_uint64(x_c.data_ptr())
        c_idx = ctypes.c_uint64(idx_c.data_ptr())
        c_qw = ctypes.c_uint64(lm_head_mod.qweight.data_ptr())
        c_sc = ctypes.c_uint64(lm_head_mod.scales.data_ptr())
        c_zr = ctypes.c_uint64(lm_head_mod.zeros.data_ptr())
        c_out = ctypes.c_uint64(out_f32.data_ptr())
        c_M = ctypes.c_int(M)
        c_C = ctypes.c_int(C_surv)
        c_K = ctypes.c_int(K)
        params = (ctypes.c_void_p * 9)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_idx), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_sc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_zr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_C), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        )
        grid_x = (C_surv + 7) // 8
        rc = self.nvcuda.cuLaunchKernel(
            self.fns[b"indexed_vocab_gemv_q4_f32"], grid_x, M, 1, 256, 1, 1, smem, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"indexed_vocab launch failed: {rc}")
        return out_f32


class SingleReadMultiRowKernels:
    """Compiles template-specialized single-weight-read multi-row (`M in {1, 2, 3, 4}`) CUDA kernels via NVRTC."""

    def __init__(self, device: torch.device):
        self.device = device
        self.nvcuda, self.nvrtc = _load_driver_and_nvrtc()
        major, minor = torch.cuda.get_device_capability(device)
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), MULTIROW_SINGLE_READ_CU.encode("utf-8"), b"multirow_single_read.cu", 0, None, None
        )
        if rc != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")
        try:
            arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode("utf-8")
            opts = (ctypes.c_char_p * 1)(arch_flag)
            rc = self.nvrtc.nvrtcCompileProgram(prog, 1, opts)
            if rc != 0:
                lsz = ctypes.c_size_t()
                self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(lsz))
                lbuf = ctypes.create_string_buffer(lsz.value)
                self.nvrtc.nvrtcGetProgramLog(prog, lbuf)
                raise RuntimeError(f"NVRTC multirow compile failed ({rc}):\n{lbuf.value.decode('utf-8', errors='replace')}")
            psz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(psz))
            pbuf = ctypes.create_string_buffer(psz.value)
            self.nvrtc.nvrtcGetPTX(prog, pbuf)
            ptx = pbuf.raw if pbuf.raw.endswith(b"\x00") else pbuf.raw + b"\x00"
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        self.mod = ctypes.c_void_p()
        rc = self.nvcuda.cuModuleLoadData(ctypes.byref(self.mod), ctypes.c_char_p(ptx))
        if rc != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {rc}")
        self.dense_fns = {}
        self.mlp_fns = {}
        self.qkv_fns = {}
        for m in (1, 2, 3, 4):
            for dst, prefix in (
                (self.dense_fns, f"multirow_dense_gemv_m{m}"),
                (self.mlp_fns, f"multirow_gate_up_swiglu_m{m}"),
                (self.qkv_fns, f"multirow_qkvzba_m{m}"),
            ):
                fn = ctypes.c_void_p()
                rc = self.nvcuda.cuModuleGetFunction(ctypes.byref(fn), self.mod, prefix.encode("utf-8"))
                if rc != 0:
                    raise RuntimeError(f"cuModuleGetFunction({prefix}) failed: {rc}")
                dst[m] = fn
        self.ws_mlp = torch.empty((4, 12288), dtype=torch.bfloat16, device=device)

    def linear_q4(self, x: torch.Tensor, q4_mod) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        assert 1 <= M <= 4
        N = int(q4_mod.out_features)
        x_2d = x.reshape(M, K)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        y_2d = torch.empty((M, N), dtype=torch.bfloat16, device=self.device)
        bias = q4_mod._buffers.get("bias", None)
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        c_x = ctypes.c_uint64(x_2d.data_ptr())
        c_qw = ctypes.c_uint64(q4_mod.qweight.data_ptr())
        c_sc = ctypes.c_uint64(q4_mod.scales.data_ptr())
        c_zr = ctypes.c_uint64(q4_mod.zeros.data_ptr())
        c_bi = ctypes.c_uint64(bias.data_ptr() if bias is not None else 0)
        c_y = ctypes.c_uint64(y_2d.data_ptr())
        c_N = ctypes.c_int(N)
        c_K = ctypes.c_int(K)
        params = (ctypes.c_void_p * 8)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_qw), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_sc), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_zr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_bi), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_N), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_K), ctypes.c_void_p),
        )
        grid_x = 304 if (K <= 4096 and N >= 32768) else ((N + 7) // 8)
        smem = M * 8448
        rc = self.nvcuda.cuLaunchKernel(
            self.dense_fns[M], grid_x, 1, 1, 256, 1, 1, smem, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"multirow_dense_gemv_m{M} launch failed: {rc}")
        return y_2d.view(*orig_shape[:-1], N)

    def gate_up_swiglu(self, x: torch.Tensor, gate_mod, up_mod) -> torch.Tensor:
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        assert 2 <= M <= 4
        I = int(gate_mod.out_features)
        x_2d = x.reshape(M, K)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        y_view = self.ws_mlp[:M] if I == 12288 else torch.empty((M, I), dtype=torch.bfloat16, device=self.device)
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        c_x = ctypes.c_uint64(x_2d.data_ptr())
        c_gq = ctypes.c_uint64(gate_mod.qweight.data_ptr())
        c_gs = ctypes.c_uint64(gate_mod.scales.data_ptr())
        c_gz = ctypes.c_uint64(gate_mod.zeros.data_ptr())
        c_uq = ctypes.c_uint64(up_mod.qweight.data_ptr())
        c_us = ctypes.c_uint64(up_mod.scales.data_ptr())
        c_uz = ctypes.c_uint64(up_mod.zeros.data_ptr())
        c_y = ctypes.c_uint64(y_view.data_ptr())
        c_I = ctypes.c_int(I)
        params = (ctypes.c_void_p * 9)(
            ctypes.cast(ctypes.byref(c_x), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gq), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_gz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_uq), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_us), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_uz), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_y), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_I), ctypes.c_void_p),
        )
        smem = M * 8448
        rc = self.nvcuda.cuLaunchKernel(
            self.mlp_fns[M], (I + 7) // 8, 1, 1, 256, 1, 1, smem, stream_ptr, params, None
        )
        if rc != 0:
            raise RuntimeError(f"multirow_gate_up_m{M} launch failed: {rc}")
        return y_view.view(*orig_shape[:-1], I)


class ZeroProbSubspaceVocabHead(torch.nn.Module):
    """Gamma-Weighted Oblique Projector Vocabulary Dead-Zone Pruner (`-98.35%` dead rows)."""

    def __init__(
        self,
        q4_lm_head,
        gk: GrandUnifiedKernels,
        norm_weight: torch.Tensor,
        zero_centered_norm: bool = True,
        rank_r: int = 512,
        survivors: int = 4096,
        pinned_token_ids: list[int] | None = None,
    ):
        super().__init__()
        self.q4_head = q4_lm_head
        self.gk = gk
        self.rank_r = rank_r
        self.survivors = survivors
        self.vocab_size = int(q4_lm_head.out_features)
        self.hidden_size = int(q4_lm_head.in_features)
        valid_pins = sorted({int(t) for t in (pinned_token_ids or []) if 0 <= int(t) < self.vocab_size})
        self.pinned_ids = (
            torch.tensor(valid_pins, dtype=torch.int32, device=gk.device) if valid_pins else None
        )

        with torch.no_grad():
            gamma = (1.0 + norm_weight.detach().float()) if zero_centered_norm else norm_weight.detach().float()
            gamma_safe = torch.where(gamma.abs() < 1e-4, torch.full_like(gamma, 1e-4), gamma)

            step_v = max(1, self.vocab_size // 16384)
            sample_idx = torch.arange(0, self.vocab_size, step_v, device=gk.device, dtype=torch.long)
            qw_s = q4_lm_head.qweight[sample_idx]
            sc_s = q4_lm_head.scales[sample_idx].float()
            zr_s = q4_lm_head.zeros[sample_idx].float()
            lo = (qw_s & 0x0F).to(torch.float32)
            hi = (qw_s >> 4).to(torch.float32)
            w_unp = torch.stack([lo, hi], dim=-1).reshape(sample_idx.numel(), -1, 64)
            w_sample = (w_unp * sc_s.unsqueeze(-1) + zr_s.unsqueeze(-1)).reshape(sample_idx.numel(), self.hidden_size)
            w_eff = (w_sample - w_sample.mean(dim=0, keepdim=True)) * gamma_safe.unsqueeze(0)
            _, _, V = torch.pca_lowrank(w_eff, q=rank_r, center=False, niter=3)

            self.V_left = (V / gamma_safe.unsqueeze(1)).to(torch.bfloat16).contiguous()
            v_right_t = (V * gamma_safe.unsqueeze(1)).t().to(torch.bfloat16).contiguous()
            del w_eff, w_sample, w_unp, lo, hi, qw_s, sc_s, zr_s

            w_sub_t = torch.empty((rank_r, self.vocab_size), dtype=torch.bfloat16, device=gk.device)
            w_sub_t.copy_(q4_lm_head(v_right_t.unsqueeze(0)).squeeze(0))
            self.W_sub_T = w_sub_t.contiguous()
            del v_right_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and 1 <= (x.shape[0] * x.shape[1]) <= 4:
            B, S, D = x.shape
            M = B * S
            x2d = x.reshape(M, D)
            coarse = (x2d @ self.V_left) @ self.W_sub_T
            if self.pinned_ids is not None and self.pinned_ids.numel() < self.survivors:
                k_dyn = self.survivors - int(self.pinned_ids.numel())
                dyn_idx = torch.topk(coarse, k=k_dyn, dim=-1).indices.to(torch.int32)
                pins = self.pinned_ids.unsqueeze(0).expand(M, -1)
                top_idx = torch.cat([dyn_idx, pins], dim=-1).contiguous()
            else:
                top_idx = torch.topk(coarse, k=self.survivors, dim=-1).indices.to(torch.int32)
            exact_surv = self.gk.indexed_vocab_scores(x2d, top_idx, self.q4_head)
            full_out = torch.full((M, self.vocab_size), -1e4, dtype=x.dtype, device=x.device)
            full_out.scatter_(1, top_idx.long(), exact_surv.to(x.dtype))
            return full_out.view(B, S, self.vocab_size)
        return self.q4_head(x)


_MR_KERNELS: SingleReadMultiRowKernels | None = None


def get_mr_kernels(device: torch.device) -> SingleReadMultiRowKernels:
    global _MR_KERNELS
    if _MR_KERNELS is None:
        _MR_KERNELS = SingleReadMultiRowKernels(device)
    return _MR_KERNELS
