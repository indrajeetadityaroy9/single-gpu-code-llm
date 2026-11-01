/*
 * Fused QK-norm and RoPE for training (paper §4): per-head RMSNorm, the learned per-dimension scale (QK-norm as in
 * arXiv 2505.09388 §2) and RoPE, with one kernel for the forward and one for the backward. Fusing them reads q and k
 * from the QKV projection output once and writes them once, instead of once per elementwise step.
 * For a head vector x with learned scale g, the forward computes
 *   n = x / sqrt(mean(x^2) + FLT_EPSILON), m = n * g,
 *   y1 = m1 cos + m2 sin, y2 = -m1 sin + m2 cos,
 * where m1 and m2 are the first and second halves of m (RoPE, arXiv 2104.09864 §3.2.2).
 * The backward recomputes n = x * rstd from x and applies, with dy the output gradient:
 *   dm1 = dy1 cos - dy2 sin, dm2 = dy1 sin + dy2 cos (inverse rotation),
 *   dg = sum over rows of dm * n, dn = dm * g, dx = rstd * (dn - n * mean(dn * n)).
 * The scale gradient dg avoids atomics: each block writes a partial sum, and k_gain_sum adds the partials in block
 * order, so dg is bitwise reproducible.
 * One row is spread over QK_LANES threads; each thread holds VEC dimensions from the first half and the VEC
 * dimensions HALF positions later that RoPE rotates with them.
 */
#include "common.cuh"

constexpr int HALF = HEAD_DIM / 2;    /* rotation pairs per head: dims i and i + HALF */
constexpr int QK_LANES = HALF / VEC;  /* threads per head row, one rotation pair of vectors each */

/* Shuffle mask of the QK_LANES threads that hold this thread's row; a row always occupies a whole aligned group. */
static __device__ __forceinline__ unsigned qk_row_mask() {
    return ((1u << QK_LANES) - 1) << ((threadIdx.x & (WARP - 1)) & ~(QK_LANES - 1));
}

static __device__ __forceinline__ float qk_row_sum(float v) {
    const unsigned mask = qk_row_mask();
    for (int o = QK_LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/* 1 / sqrt(mean(x^2) + FLT_EPSILON) of this lane's row. */
static __device__ __forceinline__ float qk_rstd(const float *x1, const float *x2) {
    float ss = 0.f;
    for (int i = 0; i < VEC; i++) ss += x1[i] * x1[i] + x2[i] * x2[i];
    return rsqrtf(qk_row_sum(ss) / HEAD_DIM + FLT_EPSILON);
}

/* QK-norm and RoPE of this thread's two vectors x1, x2 (HALF dimensions apart), given their scales g1, g2 and
 * their cos/sin entries. */
static __device__ __forceinline__ void qk_rotate(const float *x1, const float *x2, const float *g1, const float *g2,
                                                 const float *cs, const float *sn, float *y1, float *y2) {
    const float m = qk_rstd(x1, x2);
    for (int i = 0; i < VEC; i++) {
        const float n1 = x1[i] * m * g1[i], n2 = x2[i] * m * g2[i];
        y1[i] = n1 * cs[i] + n2 * sn[i];
        y2[i] = n2 * cs[i] - n1 * sn[i];
    }
}

__global__ void k_forward(const bf16 *__restrict__ q, const bf16 *__restrict__ k, const float *__restrict__ gq,
                          const float *__restrict__ gk, bf16 *__restrict__ qo, bf16 *__restrict__ ko,
                          const float *__restrict__ cos, const float *__restrict__ sin, int64_t rows, unsigned nb,
                          int H, int T) {
    const bool isq = blockIdx.x < nb;
    const int64_t g = (int64_t)(isq ? blockIdx.x : blockIdx.x - nb) * blockDim.x + threadIdx.x, p = g / QK_LANES;
    if (p * H >= rows) return;
    const int o = (int)(g % QK_LANES) * VEC;
    const int64_t tab = (p % T) * HALF + o;
    const bf16 *x = (isq ? q : k) + p * 3 * H * HEAD_DIM + o;
    bf16 *y = (isq ? qo : ko) + p * H * HEAD_DIM + o;
    const float *gain = (isq ? gq : gk) + o;
    float g1[VEC], g2[VEC], cs[VEC], sn[VEC];
    load(gain, g1); load(gain + HALF, g2);
    load(cos + tab, cs); load(sin + tab, sn);
    for (int h = 0; h < H; h += 2) {  /* two heads per iteration (H = HIDDEN_SIZE / HEAD_DIM is even) */
        float a1[VEC], a2[VEC], b1[VEC], b2[VEC], y1[VEC], y2[VEC];
        load(x + h * HEAD_DIM, a1); load(x + h * HEAD_DIM + HALF, a2);
        load(x + (h + 1) * HEAD_DIM, b1); load(x + (h + 1) * HEAD_DIM + HALF, b2);
        qk_rotate(a1, a2, g1, g2, cs, sn, y1, y2);
        store(y + h * HEAD_DIM, y1); store(y + h * HEAD_DIM + HALF, y2);
        qk_rotate(b1, b2, g1, g2, cs, sn, y1, y2);
        store(y + (h + 1) * HEAD_DIM, y1); store(y + (h + 1) * HEAD_DIM + HALF, y2);
    }
}

constexpr int MAX_WARPS = 1024 / WARP;  /* CUDA's maximum threads per block */

__global__ void k_backward(const bf16 *q, const bf16 *k, const float *gq, const float *gk, const bf16 *dqo,
                           const bf16 *dko, bf16 *dq, bf16 *dk, const float *cos, const float *sin, float *part,
                           int64_t rows, unsigned nb, int H, int T) {
    __shared__ float warp_part[MAX_WARPS][HEAD_DIM];  /* each warp's partial scale gradient */
    const bool isq = blockIdx.x < nb;
    const int64_t first = isq ? blockIdx.x : blockIdx.x - nb, per_pass = blockDim.x / QK_LANES;
    const int o = (int)(threadIdx.x % QK_LANES) * VEC;
    const bf16 *x_in = isq ? q : k, *dy_in = isq ? dqo : dko;
    bf16 *dx_out = isq ? dq : dk;
    float g1[VEC], g2[VEC];
    load((isq ? gq : gk) + o, g1); load((isq ? gq : gk) + o + HALF, g2);
    float a1[VEC] = {}, a2[VEC] = {};  /* this lane's sum of dm * n over its rows */
    for (int64_t i = first * per_pass + threadIdx.x / QK_LANES; i < rows; i += (int64_t)nb * per_pass) {
        const int64_t t = i / H, hd = t * 3 * H * HEAD_DIM + (i % H) * HEAD_DIM + o, at = i * HEAD_DIM + o,
                      tab = (t % T) * HALF + o;
        float x1[VEC], x2[VEC], d1[VEC], d2[VEC], c[VEC], s[VEC];
        load(x_in + hd, x1); load(x_in + hd + HALF, x2);
        load(dy_in + at, d1); load(dy_in + at + HALF, d2);
        load(cos + tab, c); load(sin + tab, s);
        const float m = qk_rstd(x1, x2);
        float sd = 0.f;  /* sum dn * n */
        for (int j = 0; j < VEC; j++) {
            const float n1 = x1[j] * m, n2 = x2[j] * m;
            const float dm1 = d1[j] * c[j] - d2[j] * s[j], dm2 = d1[j] * s[j] + d2[j] * c[j];
            a1[j] += dm1 * n1; a2[j] += dm2 * n2;
            d1[j] = dm1 * g1[j]; d2[j] = dm2 * g2[j];
            x1[j] = n1; x2[j] = n2;
            sd += d1[j] * n1 + d2[j] * n2;
        }
        const float b = qk_row_sum(sd) / HEAD_DIM;
        for (int j = 0; j < VEC; j++) {
            d1[j] = m * (d1[j] - x1[j] * b);
            d2[j] = m * (d2[j] - x2[j] * b);
        }
        store(dx_out + hd, d1); store(dx_out + hd + HALF, d2);
    }
    /* Lanes QK_LANES apart hold the same dimensions; reduce across them so lanes 0..QK_LANES-1 hold the warp's sums */
    for (int w = QK_LANES; w < WARP; w <<= 1)
        for (int j = 0; j < VEC; j++) {
            a1[j] += __shfl_xor_sync(0xffffffffu, a1[j], w);
            a2[j] += __shfl_xor_sync(0xffffffffu, a2[j], w);
        }
    if ((threadIdx.x & (WARP - 1)) < QK_LANES)
        for (int j = 0; j < VEC; j++) {
            warp_part[threadIdx.x / WARP][o + j] = a1[j];
            warp_part[threadIdx.x / WARP][o + HALF + j] = a2[j];
        }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float sum = 0.f;
        for (int w = 0; w < (int)(blockDim.x / WARP); w++) sum += warp_part[w][d];
        part[(int64_t)blockIdx.x * HEAD_DIM + d] = sum;
    }
}

/* Block 0 sums q's scale gradient, block 1 k's; thread d adds dimension d of the nb partials in block order. */
__global__ void k_gain_sum(const float *part, unsigned nb, float *dgq, float *dgk) {
    const float *p = part + (int64_t)blockIdx.x * nb * HEAD_DIM + threadIdx.x;
    float sum = 0.f;
    for (unsigned j = 0; j < nb; j++) sum += p[(int64_t)j * HEAD_DIM];
    (blockIdx.x ? dgk : dgq)[threadIdx.x] = sum;
}

/* The occupancy calculator's block size for f. */
template <typename F> static int block_size(F f) {
    int min_grid, threads;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &threads, f);
    return threads;
}

static int backward_threads() {
    static const int threads = block_size(k_backward);
    return threads;
}

extern "C" void qkp_forward(const void *q, const void *k, const void *gq, const void *gk, void *qo, void *ko,
                            const void *cos, const void *sin, long long rows, int H, int T, void *stream) {
    static const int threads = block_size(k_forward);
    const unsigned nb = (rows / H * QK_LANES + threads - 1) / threads;
    k_forward<<<2 * nb, threads, 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (bf16 *)qo, (bf16 *)ko,
        (const float *)cos, (const float *)sin, rows, nb, H, T);
}

/* Blocks per tensor in the backward: half the blocks the device can hold at once. The count is fixed per device, so
 * the summation order of the scale gradient is fixed; blocks past the last row contribute exact zeros. */
extern "C" int qkp_backward_blocks() {
    static const int half = resident_blocks(k_backward, backward_threads()) / 2;
    return half;
}

/* part: fp32 (2 * qkp_backward_blocks(), HEAD_DIM). */
extern "C" void qkp_backward(const void *q, const void *k, const void *gq, const void *gk, const void *dqo,
                             const void *dko, void *dq, void *dk, void *dgq, void *dgk, void *part, const void *cos,
                             const void *sin, long long rows, int H, int T, void *stream) {
    const unsigned nb = qkp_backward_blocks();
    k_backward<<<2 * nb, backward_threads(), 0, (cudaStream_t)stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const float *)gq, (const float *)gk, (const bf16 *)dqo, (const bf16 *)dko,
        (bf16 *)dq, (bf16 *)dk, (const float *)cos, (const float *)sin, (float *)part, rows, nb, H, T);
    k_gain_sum<<<2, HEAD_DIM, 0, (cudaStream_t)stream>>>((const float *)part, nb, (float *)dgq, (float *)dgk);
}
