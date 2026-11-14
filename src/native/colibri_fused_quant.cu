/* Fused Colibri Affine Quantization CUDA Kernels (W4A16 & W6A16, BF16 I/O).
 *
 * Implements zero-scratch register-level dequantization using Colibri's
 * grouped affine identity (`affine_quant.h`):
 *   sum_{j=0..GS-1} (scale * q_j + zero) * x_j = scale * sum(q_j * x_j) + zero * sum(x_j)
 *
 * Shared memory holds input activations `s_x` (bf16, 2 B/elem) and pre-reduced
 * group sums `s_xsum` (f32, 4 B/group), keeping dynamic shared memory <= 27 KiB
 * even at K = 12,288 (Qwen3.5-9B MLP down_proj) without special opt-in limits.
 */

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

/* ============================================================================
 * 1. Routed MoE Gate+Up+SwiGLU — W4A16 (GS in {16, 32, 64})
 * ============================================================================ */
extern "C" __global__ void routed_gate_up_swiglu_q4(
    const unsigned short* __restrict__ x_bf16,      // [H]
    const long long* __restrict__ topk_idx,         // [K_SUB]
    const unsigned char* __restrict__ gu_q,         // [E, 2*I, H/2]
    const unsigned short* __restrict__ gu_s,        // [E, 2*I, H/GS]
    const unsigned short* __restrict__ gu_z,        // [E, 2*I, H/GS]
    float* __restrict__ inter_out,                  // [K_SUB, I]
    int H, int I, int GS
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    int num_groups = H / GS;
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    for (int c = tid; c < H; c += block_threads) {
        s_x[c] = x_bf16[c];
    }
    __syncthreads();

    for (int g = tid; g < num_groups; g += block_threads) {
        float sum = 0.0f;
        int base = g * GS;
        for (int j = 0; j < GS; ++j) {
            sum += bf16_to_f32(s_x[base + j]);
        }
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = block_threads >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    int k_slot = blockIdx.y;
    if (row >= I) return;

    int e = (int)topk_idx[k_slot];
    int two_I = 2 * I;
    int packed_H = H >> 1;

    long long exp_q_off = (long long)e * two_I * packed_H;
    long long exp_s_off = (long long)e * two_I * num_groups;

    const unsigned char* q_gate_ptr = gu_q + exp_q_off + (long long)row * packed_H;
    const unsigned char* q_up_ptr   = gu_q + exp_q_off + (long long)(row + I) * packed_H;
    const unsigned short* s_gate_row = gu_s + exp_s_off + (long long)row * num_groups;
    const unsigned short* z_gate_row = gu_z + exp_s_off + (long long)row * num_groups;
    const unsigned short* s_up_row   = gu_s + exp_s_off + (long long)(row + I) * num_groups;
    const unsigned short* z_up_row   = gu_z + exp_s_off + (long long)(row + I) * num_groups;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;

    if (H >= 64 && (H & 31) == 0 && (((unsigned long long)q_gate_ptr | (unsigned long long)q_up_ptr) & 15ull) == 0ull) {
        const uint4* q_gate_vec = reinterpret_cast<const uint4*>(q_gate_ptr);
        const uint4* q_up_vec   = reinterpret_cast<const uint4*>(q_up_ptr);
        int vec_count = packed_H >> 4;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 qg = q_gate_vec[v];
            uint4 qu = q_up_vec[v];
            int col_base = v * 32;
            unsigned int words_g[4] = {qg.x, qg.y, qg.z, qg.w};
            unsigned int words_u[4] = {qu.x, qu.y, qu.z, qu.w};

            if (GS >= 32) {
                int g = col_base / GS;
                float sg = bf16_to_f32(s_gate_row[g]);
                float su = bf16_to_f32(s_up_row[g]);
                float qdot_g = 0.0f;
                float qdot_u = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int wg = words_g[w];
                    unsigned int wu = words_u[w];
                    int c0 = col_base + w * 8;
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        float xv = bf16_to_f32(s_x[c0 + nib]);
                        qdot_g += (float)((wg >> (nib * 4)) & 0xFu) * xv;
                        qdot_u += (float)((wu >> (nib * 4)) & 0xFu) * xv;
                    }
                }
                acc_gate += sg * qdot_g;
                acc_up   += su * qdot_u;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int g = (col_base + sub * 16) >> 4;
                    float sg = bf16_to_f32(s_gate_row[g]);
                    float su = bf16_to_f32(s_up_row[g]);
                    float qdot_g = 0.0f;
                    float qdot_u = 0.0f;
                    #pragma unroll
                    for (int w = sub * 2; w < sub * 2 + 2; ++w) {
                        unsigned int wg = words_g[w];
                        unsigned int wu = words_u[w];
                        int c0 = col_base + w * 8;
                        #pragma unroll
                        for (int nib = 0; nib < 8; ++nib) {
                            float xv = bf16_to_f32(s_x[c0 + nib]);
                            qdot_g += (float)((wg >> (nib * 4)) & 0xFu) * xv;
                            qdot_u += (float)((wu >> (nib * 4)) & 0xFu) * xv;
                        }
                    }
                    acc_gate += sg * qdot_g;
                    acc_up   += su * qdot_u;
                }
            }
        }
    } else {
        for (int b = lane; b < packed_H; b += 32) {
            unsigned char bg = q_gate_ptr[b];
            unsigned char bu = q_up_ptr[b];
            int c0 = b * 2;
            int g = c0 / GS;
            float sg = bf16_to_f32(s_gate_row[g]);
            float su = bf16_to_f32(s_up_row[g]);
            float x0 = bf16_to_f32(s_x[c0]);
            float x1 = bf16_to_f32(s_x[c0 + 1]);
            acc_gate += sg * ((float)(bg & 0xFu) * x0 + (float)(bg >> 4) * x1);
            acc_up   += su * ((float)(bu & 0xFu) * x0 + (float)(bu >> 4) * x1);
        }
    }

    for (int g = lane; g < num_groups; g += 32) {
        float xs = s_xsum[g];
        acc_gate += bf16_to_f32(z_gate_row[g]) * xs;
        acc_up   += bf16_to_f32(z_up_row[g]) * xs;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_gate += __shfl_down_sync(0xffffffffu, acc_gate, offset);
        acc_up   += __shfl_down_sync(0xffffffffu, acc_up, offset);
    }

    if (lane == 0) {
        float silu_g = acc_gate / (1.0f + expf(-acc_gate));
        inter_out[k_slot * I + row] = silu_g * acc_up;
    }
}

/* ============================================================================
 * 2. Routed MoE Down Projection — W4A16 (GS in {16, 32, 64})
 * ============================================================================ */
extern "C" __global__ void routed_down_q4(
    const float* __restrict__ inter_in,             // [K_SUB, I]
    const long long* __restrict__ topk_idx,         // [K_SUB]
    const unsigned short* __restrict__ topk_w_bf16, // [K_SUB]
    const unsigned char* __restrict__ dn_q,         // [E, H, I/2]
    const unsigned short* __restrict__ dn_s,        // [E, H, I/GS]
    const unsigned short* __restrict__ dn_z,        // [E, H, I/GS]
    float* __restrict__ part_out,                   // [K_SUB, H]
    int H, int I, int GS
) {
    extern __shared__ float smem_f[];
    float* s_x = smem_f;                            // [I]
    int num_groups = I / GS;
    float* s_xsum = smem_f + I;                     // [I / GS]

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int k_slot = blockIdx.y;
    const float* x_in = inter_in + k_slot * I;

    for (int c = tid; c < I; c += block_threads) {
        s_x[c] = x_in[c];
    }
    __syncthreads();

    for (int g = tid; g < num_groups; g += block_threads) {
        float sum = 0.0f;
        int base = g * GS;
        for (int j = 0; j < GS; ++j) sum += s_x[base + j];
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = block_threads >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= H) return;

    int e = (int)topk_idx[k_slot];
    int packed_I = I >> 1;
    long long exp_q_off = (long long)e * H * packed_I;
    long long exp_s_off = (long long)e * H * num_groups;

    const unsigned char* q_ptr = dn_q + exp_q_off + (long long)row * packed_I;
    const unsigned short* s_row = dn_s + exp_s_off + (long long)row * num_groups;
    const unsigned short* z_row = dn_z + exp_s_off + (long long)row * num_groups;

    float acc = 0.0f;
    if (I >= 64 && (I & 31) == 0 && ((unsigned long long)q_ptr & 15ull) == 0ull) {
        const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
        int vec_count = packed_I >> 4;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 qv = q_vec[v];
            int col_base = v * 32;
            unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
            if (GS >= 32) {
                int g = col_base / GS;
                float sv = bf16_to_f32(s_row[g]);
                float qdot = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int ww = words[w];
                    int c0 = col_base + w * 8;
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        qdot += (float)((ww >> (nib * 4)) & 0xFu) * s_x[c0 + nib];
                    }
                }
                acc += sv * qdot;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int g = (col_base + sub * 16) >> 4;
                    float sv = bf16_to_f32(s_row[g]);
                    float qdot = 0.0f;
                    #pragma unroll
                    for (int w = sub * 2; w < sub * 2 + 2; ++w) {
                        unsigned int ww = words[w];
                        int c0 = col_base + w * 8;
                        #pragma unroll
                        for (int nib = 0; nib < 8; ++nib) {
                            qdot += (float)((ww >> (nib * 4)) & 0xFu) * s_x[c0 + nib];
                        }
                    }
                    acc += sv * qdot;
                }
            }
        }
    } else {
        for (int b = lane; b < packed_I; b += 32) {
            unsigned char bv = q_ptr[b];
            int c0 = b * 2;
            int g = c0 / GS;
            float sv = bf16_to_f32(s_row[g]);
            acc += sv * ((float)(bv & 0xFu) * s_x[c0] + (float)(bv >> 4) * s_x[c0 + 1]);
        }
    }

    for (int g = lane; g < num_groups; g += 32) {
        acc += bf16_to_f32(z_row[g]) * s_xsum[g];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        float rw = bf16_to_f32(topk_w_bf16[k_slot]);
        part_out[k_slot * H + row] = acc * rw;
    }
}

/* ============================================================================
 * 3. Routed MoE Gate+Up+SwiGLU — W6A16 (Q6: 4-bit lo + 2-bit hi, GS in {16,32,64})
 * ============================================================================ */
extern "C" __global__ void routed_gate_up_swiglu_q6(
    const unsigned short* __restrict__ x_bf16,      // [H]
    const long long* __restrict__ topk_idx,         // [K_SUB]
    const unsigned char* __restrict__ gu_qlo,       // [E, 2*I, H/2]
    const unsigned char* __restrict__ gu_qhi,       // [E, 2*I, H/4]
    const unsigned short* __restrict__ gu_s,        // [E, 2*I, H/GS]
    const unsigned short* __restrict__ gu_z,        // [E, 2*I, H/GS]
    float* __restrict__ inter_out,                  // [K_SUB, I]
    int H, int I, int GS
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    int num_groups = H / GS;
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    for (int c = tid; c < H; c += block_threads) {
        s_x[c] = x_bf16[c];
    }
    __syncthreads();

    for (int g = tid; g < num_groups; g += block_threads) {
        float sum = 0.0f;
        int base = g * GS;
        for (int j = 0; j < GS; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = block_threads >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    int k_slot = blockIdx.y;
    if (row >= I) return;

    int e = (int)topk_idx[k_slot];
    int two_I = 2 * I;
    int lo_H = H >> 1;
    int hi_H = H >> 2;

    long long exp_lo_off = (long long)e * two_I * lo_H;
    long long exp_hi_off = (long long)e * two_I * hi_H;
    long long exp_s_off  = (long long)e * two_I * num_groups;

    const unsigned char* qlo_g_ptr = gu_qlo + exp_lo_off + (long long)row * lo_H;
    const unsigned char* qlo_u_ptr = gu_qlo + exp_lo_off + (long long)(row + I) * lo_H;
    const unsigned char* qhi_g_ptr = gu_qhi + exp_hi_off + (long long)row * hi_H;
    const unsigned char* qhi_u_ptr = gu_qhi + exp_hi_off + (long long)(row + I) * hi_H;

    const unsigned short* s_gate_row = gu_s + exp_s_off + (long long)row * num_groups;
    const unsigned short* z_gate_row = gu_z + exp_s_off + (long long)row * num_groups;
    const unsigned short* s_up_row   = gu_s + exp_s_off + (long long)(row + I) * num_groups;
    const unsigned short* z_up_row   = gu_z + exp_s_off + (long long)(row + I) * num_groups;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;

    bool aligned = (((unsigned long long)qlo_g_ptr | (unsigned long long)qlo_u_ptr) & 15ull) == 0ull
                && (((unsigned long long)qhi_g_ptr | (unsigned long long)qhi_u_ptr) & 7ull) == 0ull;

    if (H >= 64 && (H & 31) == 0 && aligned) {
        const uint4* qlo_g_vec = reinterpret_cast<const uint4*>(qlo_g_ptr);
        const uint4* qlo_u_vec = reinterpret_cast<const uint4*>(qlo_u_ptr);
        const uint2* qhi_g_vec = reinterpret_cast<const uint2*>(qhi_g_ptr);
        const uint2* qhi_u_vec = reinterpret_cast<const uint2*>(qhi_u_ptr);
        int vec_count = lo_H >> 4;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 log = qlo_g_vec[v];
            uint4 lou = qlo_u_vec[v];
            uint2 hig = qhi_g_vec[v];
            uint2 hiu = qhi_u_vec[v];
            int col_base = v * 32;

            unsigned int wlo_g[4] = {log.x, log.y, log.z, log.w};
            unsigned int wlo_u[4] = {lou.x, lou.y, lou.z, lou.w};
            unsigned int whi_g[4] = {hig.x & 0xFFFFu, hig.x >> 16, hig.y & 0xFFFFu, hig.y >> 16};
            unsigned int whi_u[4] = {hiu.x & 0xFFFFu, hiu.x >> 16, hiu.y & 0xFFFFu, hiu.y >> 16};

            if (GS >= 32) {
                int g = col_base / GS;
                float sg = bf16_to_f32(s_gate_row[g]);
                float su = bf16_to_f32(s_up_row[g]);
                float qdot_g = 0.0f;
                float qdot_u = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int lg = wlo_g[w], lu = wlo_u[w];
                    unsigned int hg = whi_g[w], hu = whi_u[w];
                    int c0 = col_base + w * 8;
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        float xv = bf16_to_f32(s_x[c0 + nib]);
                        unsigned int qg6 = ((lg >> (nib * 4)) & 0xFu) | (((hg >> (nib * 2)) & 0x3u) << 4);
                        unsigned int qu6 = ((lu >> (nib * 4)) & 0xFu) | (((hu >> (nib * 2)) & 0x3u) << 4);
                        qdot_g += (float)qg6 * xv;
                        qdot_u += (float)qu6 * xv;
                    }
                }
                acc_gate += sg * qdot_g;
                acc_up   += su * qdot_u;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int g = (col_base + sub * 16) >> 4;
                    float sg = bf16_to_f32(s_gate_row[g]);
                    float su = bf16_to_f32(s_up_row[g]);
                    float qdot_g = 0.0f;
                    float qdot_u = 0.0f;
                    #pragma unroll
                    for (int w = sub * 2; w < sub * 2 + 2; ++w) {
                        unsigned int lg = wlo_g[w], lu = wlo_u[w];
                        unsigned int hg = whi_g[w], hu = whi_u[w];
                        int c0 = col_base + w * 8;
                        #pragma unroll
                        for (int nib = 0; nib < 8; ++nib) {
                            float xv = bf16_to_f32(s_x[c0 + nib]);
                            unsigned int qg6 = ((lg >> (nib * 4)) & 0xFu) | (((hg >> (nib * 2)) & 0x3u) << 4);
                            unsigned int qu6 = ((lu >> (nib * 4)) & 0xFu) | (((hu >> (nib * 2)) & 0x3u) << 4);
                            qdot_g += (float)qg6 * xv;
                            qdot_u += (float)qu6 * xv;
                        }
                    }
                    acc_gate += sg * qdot_g;
                    acc_up   += su * qdot_u;
                }
            }
        }
    } else {
        for (int qidx = lane; qidx < hi_H; qidx += 32) {
            int c0 = qidx * 4;
            int g = c0 / GS;
            float sg = bf16_to_f32(s_gate_row[g]);
            float su = bf16_to_f32(s_up_row[g]);
            unsigned char hg = qhi_g_ptr[qidx];
            unsigned char hu = qhi_u_ptr[qidx];
            unsigned char lg0 = qlo_g_ptr[qidx * 2];
            unsigned char lg1 = qlo_g_ptr[qidx * 2 + 1];
            unsigned char lu0 = qlo_u_ptr[qidx * 2];
            unsigned char lu1 = qlo_u_ptr[qidx * 2 + 1];

            float x0 = bf16_to_f32(s_x[c0]);
            float x1 = bf16_to_f32(s_x[c0 + 1]);
            float x2 = bf16_to_f32(s_x[c0 + 2]);
            float x3 = bf16_to_f32(s_x[c0 + 3]);

            float dg = (float)((lg0 & 0xFu) | ((hg & 0x3u) << 4)) * x0
                     + (float)((lg0 >> 4)   | (((hg >> 2) & 0x3u) << 4)) * x1
                     + (float)((lg1 & 0xFu) | (((hg >> 4) & 0x3u) << 4)) * x2
                     + (float)((lg1 >> 4)   | (((hg >> 6) & 0x3u) << 4)) * x3;
            float du = (float)((lu0 & 0xFu) | ((hu & 0x3u) << 4)) * x0
                     + (float)((lu0 >> 4)   | (((hu >> 2) & 0x3u) << 4)) * x1
                     + (float)((lu1 & 0xFu) | (((hu >> 4) & 0x3u) << 4)) * x2
                     + (float)((lu1 >> 4)   | (((hu >> 6) & 0x3u) << 4)) * x3;
            acc_gate += sg * dg;
            acc_up   += su * du;
        }
    }

    for (int g = lane; g < num_groups; g += 32) {
        float xs = s_xsum[g];
        acc_gate += bf16_to_f32(z_gate_row[g]) * xs;
        acc_up   += bf16_to_f32(z_up_row[g]) * xs;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_gate += __shfl_down_sync(0xffffffffu, acc_gate, offset);
        acc_up   += __shfl_down_sync(0xffffffffu, acc_up, offset);
    }

    if (lane == 0) {
        float silu_g = acc_gate / (1.0f + expf(-acc_gate));
        inter_out[k_slot * I + row] = silu_g * acc_up;
    }
}

/* ============================================================================
 * 4. Routed MoE Down Projection — W6A16 (Q6: 4-bit lo + 2-bit hi, GS in {16,32,64})
 * ============================================================================ */
extern "C" __global__ void routed_down_q6(
    const float* __restrict__ inter_in,             // [K_SUB, I]
    const long long* __restrict__ topk_idx,         // [K_SUB]
    const unsigned short* __restrict__ topk_w_bf16, // [K_SUB]
    const unsigned char* __restrict__ dn_qlo,       // [E, H, I/2]
    const unsigned char* __restrict__ dn_qhi,       // [E, H, I/4]
    const unsigned short* __restrict__ dn_s,        // [E, H, I/GS]
    const unsigned short* __restrict__ dn_z,        // [E, H, I/GS]
    float* __restrict__ part_out,                   // [K_SUB, H]
    int H, int I, int GS
) {
    extern __shared__ float smem_f[];
    float* s_x = smem_f;
    int num_groups = I / GS;
    float* s_xsum = smem_f + I;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int k_slot = blockIdx.y;
    const float* x_in = inter_in + k_slot * I;

    for (int c = tid; c < I; c += block_threads) s_x[c] = x_in[c];
    __syncthreads();

    for (int g = tid; g < num_groups; g += block_threads) {
        float sum = 0.0f;
        int base = g * GS;
        for (int j = 0; j < GS; ++j) sum += s_x[base + j];
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = block_threads >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= H) return;

    int e = (int)topk_idx[k_slot];
    int lo_I = I >> 1;
    int hi_I = I >> 2;
    long long exp_lo_off = (long long)e * H * lo_I;
    long long exp_hi_off = (long long)e * H * hi_I;
    long long exp_s_off  = (long long)e * H * num_groups;

    const unsigned char* qlo_ptr = dn_qlo + exp_lo_off + (long long)row * lo_I;
    const unsigned char* qhi_ptr = dn_qhi + exp_hi_off + (long long)row * hi_I;
    const unsigned short* s_row  = dn_s   + exp_s_off  + (long long)row * num_groups;
    const unsigned short* z_row  = dn_z   + exp_s_off  + (long long)row * num_groups;

    float acc = 0.0f;
    bool aligned = (((unsigned long long)qlo_ptr & 15ull) == 0ull) && (((unsigned long long)qhi_ptr & 7ull) == 0ull);
    if (I >= 64 && (I & 31) == 0 && aligned) {
        const uint4* qlo_vec = reinterpret_cast<const uint4*>(qlo_ptr);
        const uint2* qhi_vec = reinterpret_cast<const uint2*>(qhi_ptr);
        int vec_count = lo_I >> 4;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 lo4 = qlo_vec[v];
            uint2 hi2 = qhi_vec[v];
            int col_base = v * 32;
            unsigned int wlo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
            unsigned int whi[4] = {hi2.x & 0xFFFFu, hi2.x >> 16, hi2.y & 0xFFFFu, hi2.y >> 16};
            if (GS >= 32) {
                int g = col_base / GS;
                float sv = bf16_to_f32(s_row[g]);
                float qdot = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int lw = wlo[w], hw = whi[w];
                    int c0 = col_base + w * 8;
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        unsigned int q6 = ((lw >> (nib * 4)) & 0xFu) | (((hw >> (nib * 2)) & 0x3u) << 4);
                        qdot += (float)q6 * s_x[c0 + nib];
                    }
                }
                acc += sv * qdot;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int g = (col_base + sub * 16) >> 4;
                    float sv = bf16_to_f32(s_row[g]);
                    float qdot = 0.0f;
                    #pragma unroll
                    for (int w = sub * 2; w < sub * 2 + 2; ++w) {
                        unsigned int lw = wlo[w], hw = whi[w];
                        int c0 = col_base + w * 8;
                        #pragma unroll
                        for (int nib = 0; nib < 8; ++nib) {
                            unsigned int q6 = ((lw >> (nib * 4)) & 0xFu) | (((hw >> (nib * 2)) & 0x3u) << 4);
                            qdot += (float)q6 * s_x[c0 + nib];
                        }
                    }
                    acc += sv * qdot;
                }
            }
        }
    } else {
        for (int qidx = lane; qidx < hi_I; qidx += 32) {
            int c0 = qidx * 4;
            int g = c0 / GS;
            float sv = bf16_to_f32(s_row[g]);
            unsigned char hv = qhi_ptr[qidx];
            unsigned char l0 = qlo_ptr[qidx * 2];
            unsigned char l1 = qlo_ptr[qidx * 2 + 1];
            float d = (float)((l0 & 0xFu) | ((hv & 0x3u) << 4)) * s_x[c0]
                    + (float)((l0 >> 4)   | (((hv >> 2) & 0x3u) << 4)) * s_x[c0 + 1]
                    + (float)((l1 & 0xFu) | (((hv >> 4) & 0x3u) << 4)) * s_x[c0 + 2]
                    + (float)((l1 >> 4)   | (((hv >> 6) & 0x3u) << 4)) * s_x[c0 + 3];
            acc += sv * d;
        }
    }

    for (int g = lane; g < num_groups; g += 32) {
        acc += bf16_to_f32(z_row[g]) * s_xsum[g];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        float rw = bf16_to_f32(topk_w_bf16[k_slot]);
        part_out[k_slot * H + row] = acc * rw;
    }
}

/* ============================================================================
 * 5. Reduction & Conversion Helpers for Routed MoE
 * ============================================================================ */
extern "C" __global__ void reduce_k_to_bf16(
    const float* __restrict__ part_out,     // [K_SUB, H]
    unsigned short* __restrict__ out_bf16,  // [H]
    int K_SUB, int H
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= H) return;
    float sum = 0.0f;
    for (int k = 0; k < K_SUB; ++k) {
        sum += part_out[k * H + col];
    }
    out_bf16[col] = f32_to_bf16_rne(sum);
}

extern "C" __global__ void f32_to_bf16_vec(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int total_elems
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;
    dst[i] = f32_to_bf16_rne(src[i]);
}

/* ============================================================================
 * 6. Dense Linear W4A16 GEMV/Small-Batch GEMM (for Qwen3.5-9B ColibriQ4Linear)
 *    Computes y[M, N] = x[M, K] @ dequant(W[N, K])^T + bias[N] for small M.
 * ============================================================================ */
extern "C" __global__ void dense_gemv_q4_bf16(
    const unsigned short* __restrict__ x_bf16,      // [M, K]
    const unsigned char* __restrict__ qweight,      // [N, K/2]
    const unsigned short* __restrict__ scales,      // [N, K/GS]
    const unsigned short* __restrict__ zeros,       // [N, K/GS]
    const unsigned short* __restrict__ bias_bf16,   // [N] or nullptr
    unsigned short* __restrict__ y_bf16,            // [M, N]
    int M, int N, int K, int GS
) {
    extern __shared__ char smem_raw[];
    float* s_xsum = reinterpret_cast<float*>(smem_raw);
    int num_groups = K / GS;
    unsigned short* s_x = reinterpret_cast<unsigned short*>(s_xsum + num_groups);

    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int m_idx = blockIdx.y;
    if (m_idx >= M) return;

    const unsigned short* x_row = x_bf16 + (long long)m_idx * K;
    for (int c = tid; c < K; c += block_threads) {
        s_x[c] = x_row[c];
    }
    __syncthreads();

    for (int g = tid; g < num_groups; g += block_threads) {
        float sum = 0.0f;
        int base = g * GS;
        for (int j = 0; j < GS; ++j) sum += bf16_to_f32(s_x[base + j]);
        s_xsum[g] = sum;
    }
    __syncthreads();

    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps_per_block = block_threads >> 5;
    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= N) return;

    int packed_K = K >> 1;
    const unsigned char* q_ptr = qweight + (long long)row * packed_K;
    const unsigned short* s_row = scales + (long long)row * num_groups;
    const unsigned short* z_row = zeros  + (long long)row * num_groups;

    float acc = 0.0f;
    if (K >= 64 && (K & 31) == 0 && ((unsigned long long)q_ptr & 15ull) == 0ull) {
        const uint4* q_vec = reinterpret_cast<const uint4*>(q_ptr);
        int vec_count = packed_K >> 4;
        for (int v = lane; v < vec_count; v += 32) {
            uint4 qv = q_vec[v];
            int col_base = v * 32;
            unsigned int words[4] = {qv.x, qv.y, qv.z, qv.w};
            if (GS >= 32) {
                int g = col_base / GS;
                float sv = bf16_to_f32(s_row[g]);
                float qdot = 0.0f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned int ww = words[w];
                    int c0 = col_base + w * 8;
                    #pragma unroll
                    for (int nib = 0; nib < 8; ++nib) {
                        qdot += (float)((ww >> (nib * 4)) & 0xFu) * bf16_to_f32(s_x[c0 + nib]);
                    }
                }
                acc += sv * qdot;
            } else {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int g = (col_base + sub * 16) >> 4;
                    float sv = bf16_to_f32(s_row[g]);
                    float qdot = 0.0f;
                    #pragma unroll
                    for (int w = sub * 2; w < sub * 2 + 2; ++w) {
                        unsigned int ww = words[w];
                        int c0 = col_base + w * 8;
                        #pragma unroll
                        for (int nib = 0; nib < 8; ++nib) {
                            qdot += (float)((ww >> (nib * 4)) & 0xFu) * bf16_to_f32(s_x[c0 + nib]);
                        }
                    }
                    acc += sv * qdot;
                }
            }
        }
    } else {
        for (int b = lane; b < packed_K; b += 32) {
            unsigned char bv = q_ptr[b];
            int c0 = b * 2;
            int g = c0 / GS;
            float sv = bf16_to_f32(s_row[g]);
            acc += sv * ((float)(bv & 0xFu) * bf16_to_f32(s_x[c0]) + (float)(bv >> 4) * bf16_to_f32(s_x[c0 + 1]));
        }
    }

    for (int g = lane; g < num_groups; g += 32) {
        acc += bf16_to_f32(z_row[g]) * s_xsum[g];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        if (bias_bf16 != nullptr) {
            acc += bf16_to_f32(bias_bf16[row]);
        }
        y_bf16[(long long)m_idx * N + row] = f32_to_bf16_rne(acc);
    }
}
