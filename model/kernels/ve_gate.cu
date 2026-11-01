/*
 * Value-embedding gate and mix (paper §4), one fused kernel each for the forward and the backward.
 * Forward, per token row: gate logits z = n W^T in fp32 (n the normalized layer input, one logit per head),
 * gate = 2 sigmoid(z), and out = bf16(v + gate[head] * bf16(table[token])). One warp handles one row, so the
 * gate, the table lookup and the add need a single pass over the row instead of separate elementwise kernels.
 * Backward: dgate[h] = sum over the head's dimensions of dout * ve, dz = 2 dgate (1 - s) s with s = gate / 2,
 * dn = bf16(dz W), and the table gradient dve = bf16(gate[head] * dout). The rows are visited in token order, so
 * equal tokens are adjacent: dve is summed over each run of equal tokens and added into the table gradient with one
 * 16-byte atomic per 4 floats, instead of one atomic per row.
 * The design follows deepseek-ai/TileKernels' Engram gate kernels.
 */
#include "common.cuh"

/* VEC fp32 of the table rounded to bf16, packed in a 16-byte word. */
static __device__ __forceinline__ uint4 load_ve(const float *p) {
    float v[VEC];
    load(p, v);
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    return u;
}

/* VEC bf16 of a 16-byte word as fp32. */
static __device__ __forceinline__ void unpack(const uint4 &u, float *v) {
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x; v[2 * i + 1] = f.y;
    }
}

/* The warp's sum of v to every lane: a butterfly in a fixed order, each add rounded (never contracted). */
static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* The fastest configuration measured at d24 on the GH200; the backward runs 2 blocks per SM (128 registers per
 * thread). */
constexpr int THREADS = 256;  /* warps per block: THREADS / WARP */
constexpr int RUN = 16;       /* backward: consecutive rows (in token order) per warp */
static_assert(RUN <= WARP, "a warp's positions are held one per lane");
/* HIDDEN_SIZE and HEAD_DIM are defined by native.py's nvcc command (model/config.py) */
constexpr int C = HIDDEN_SIZE, H = C / HEAD_DIM, CHUNKS = C / (WARP * VEC);
static_assert(C % (WARP * VEC) == 0, "a row is whole warp-wide chunks");

__global__ void __launch_bounds__(THREADS)
ve_forward_kernel(const bf16 *n, const bf16 *v, int64_t v_stride, const int *idx, const float *table, const float *w,
                  bf16 *out, float *gate, int64_t rows) {
    __shared__ __align__(16) float ws[H * C];
    for (int i = threadIdx.x; i < H * C; i += blockDim.x) ws[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x % WARP;
    const int64_t warps = (int64_t)gridDim.x * (THREADS / WARP);
    for (int64_t r = (int64_t)blockIdx.x * (THREADS / WARP) + threadIdx.x / WARP; r < rows; r += warps) {
        /* Issue every load of the row (n, v and the table row) before any math, to overlap their latency */
        const float *row = table + (int64_t)idx[r] * C;
        uint4 xn[CHUNKS], xv[CHUNKS];
        float te[CHUNKS][VEC];
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            xn[k] = *reinterpret_cast<const uint4 *>(n + r * C + c);
            xv[k] = *reinterpret_cast<const uint4 *>(v + r * v_stride + c);
            load(row + c, te[k]);
        }
        float z[H] = {};
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float x[VEC];
            unpack(xn[k], x);
#pragma unroll
            for (int h = 0; h < H; h++) {
                float wv[VEC];
                load(ws + h * C + c, wv);
                for (int j = 0; j < VEC; j++) z[h] += x[j] * wv[j];
            }
        }
        float g[H];
#pragma unroll
        for (int h = 0; h < H; h++) g[h] = 2.f * (1.f / (1.f + expf(-warp_sum(z[h]))));
#pragma unroll
        for (int h = 0; h < H; h++)
            if (lane == h) gate[r * H + h] = g[h];
#pragma unroll
        for (int k = 0; k < CHUNKS; k++) {
            const int c = (k * WARP + lane) * VEC;
            float x[VEC], gh = 0.f;
            unpack(xv[k], x);
#pragma unroll
            for (int h = 0; h < H; h++)
                if (c / HEAD_DIM == h) gh = g[h];
            for (int j = 0; j < VEC; j++)
                x[j] = __fadd_rn(x[j], __fmul_rn(gh, __bfloat162float(__float2bfloat16(te[k][j]))));
            store(out + r * C + c, x);
        }
    }
}

/* Add acc (this lane's chunks of one row) into table-gradient row `row`, with one 16-byte atomic per 4 floats. */
static __device__ __forceinline__ void add_row(float *row, const float (&acc)[CHUNKS][VEC], int lane) {
#pragma unroll
    for (int k = 0; k < CHUNKS; k++) {
        float4 *t = reinterpret_cast<float4 *>(row + (k * WARP + lane) * VEC);
#pragma unroll
        for (int j = 0; j < VEC; j += 4)
            atomicAdd(t + j / 4, make_float4(acc[k][j], acc[k][j + 1], acc[k][j + 2], acc[k][j + 3]));
    }
}

__global__ void __launch_bounds__(THREADS, 2)
ve_backward_kernel(const bf16 *__restrict__ dout, int64_t dout_stride, const int *__restrict__ ids,
                   const int64_t *__restrict__ order,
                   const float *__restrict__ table, const float *__restrict__ w, const float *__restrict__ gate,
                   bf16 *__restrict__ dn, float *dtable, bf16 *__restrict__ dz_out, int64_t rows) {
    __shared__ __align__(16) float ws[H * C];
    for (int i = threadIdx.x; i < H * C; i += blockDim.x) ws[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x % WARP;
    const int64_t warps = (int64_t)gridDim.x * (THREADS / WARP);
    for (int64_t first = ((int64_t)blockIdx.x * (THREADS / WARP) + threadIdx.x / WARP) * RUN; first < rows;
         first += warps * RUN) {
        const int count = rows - first < RUN ? (int)(rows - first) : RUN;
        /* Lane j holds the row index and token of position first + j */
        long long row_j = 0;
        int tok_j = -1;
        if (lane < count) {
            row_j = order[first + lane];
            tok_j = ids[first + lane];
        }
        uint4 next[CHUNKS];
        {
            const int64_t r = __shfl_sync(0xffffffffu, row_j, 0);
#pragma unroll
            for (int k = 0; k < CHUNKS; k++)
                next[k] = *reinterpret_cast<const uint4 *>(dout + r * dout_stride + (k * WARP + lane) * VEC);
        }
        uint4 e[CHUNKS];  /* the run's table row, bf16 */
        float acc[CHUNKS][VEC];
        int cur = -1;
        for (int i = 0; i < count; i++) {
            const int64_t r = __shfl_sync(0xffffffffu, row_j, i);
            const int t = __shfl_sync(0xffffffffu, tok_j, i);
            uint4 u[CHUNKS];
#pragma unroll
            for (int k = 0; k < CHUNKS; k++) u[k] = next[k];
            if (i + 1 < count) {  /* load the next row's dout while this row is computed */
                const int64_t rn = __shfl_sync(0xffffffffu, row_j, i + 1);
#pragma unroll
                for (int k = 0; k < CHUNKS; k++)
                    next[k] = *reinterpret_cast<const uint4 *>(dout + rn * dout_stride + (k * WARP + lane) * VEC);
            }
            if (t != cur) {  /* a new token: flush the previous run's sum into dtable and load this token's row */
                if (cur >= 0) add_row(dtable + (int64_t)cur * C, acc, lane);
                cur = t;
#pragma unroll
                for (int k = 0; k < CHUNKS; k++) {
                    e[k] = load_ve(table + (int64_t)t * C + (k * WARP + lane) * VEC);
                    for (int j = 0; j < VEC; j++) acc[k][j] = 0.f;
                }
            }
            float g[H], dg[H] = {};
#pragma unroll
            for (int h = 0; h < H; h++) g[h] = gate[r * H + h];
#pragma unroll
            for (int k = 0; k < CHUNKS; k++) {
                const int c = (k * WARP + lane) * VEC;
                float d[VEC], ev[VEC], gh = 0.f, s = 0.f;
                unpack(u[k], d);
                unpack(e[k], ev);
#pragma unroll
                for (int h = 0; h < H; h++)
                    if (c / HEAD_DIM == h) gh = g[h];
                for (int j = 0; j < VEC; j++) {
                    s += d[j] * ev[j];
                    acc[k][j] += __bfloat162float(__float2bfloat16(gh * d[j]));  /* dve */
                }
#pragma unroll
                for (int h = 0; h < H; h++)
                    if (c / HEAD_DIM == h) dg[h] += s;
            }
            float dz[H];
#pragma unroll
            for (int h = 0; h < H; h++) {
                const float sg = g[h] * 0.5f;
                dz[h] = 2.f * warp_sum(dg[h]) * (1.f - sg) * sg;
                if (lane == h) dz_out[r * H + h] = __float2bfloat16(dz[h]);
            }
#pragma unroll
            for (int k = 0; k < CHUNKS; k++) {
                const int c = (k * WARP + lane) * VEC;
                float x[VEC] = {};
#pragma unroll
                for (int h = 0; h < H; h++) {
                    float wv[VEC];
                    load(ws + h * C + c, wv);
                    for (int j = 0; j < VEC; j++) x[j] += dz[h] * wv[j];
                }
                store(dn + r * C + c, x);
            }
        }
        add_row(dtable + (int64_t)cur * C, acc, lane);
    }
}

/* v_stride: elements between consecutive rows of v. */
extern "C" void ve_forward(const void *n, const void *v, long long v_stride, const void *idx, const void *table,
                           const void *w, void *out, void *gate, long long rows, void *stream) {
    static const int blocks = resident_blocks(ve_forward_kernel, THREADS);
    ve_forward_kernel<<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
        (const bf16 *)n, (const bf16 *)v, v_stride, (const int *)idx, (const float *)table, (const float *)w,
        (bf16 *)out, (float *)gate, rows);
}

/* ids: int32 tokens in visiting order (idx[order]); order: int64 permutation that sorts idx; dtable: fp32 table
 * gradient, accumulated into; dz: bf16 (rows, H) gate-logit gradients. */
extern "C" void ve_backward(const void *dout, long long dout_stride, const void *ids, const void *order,
                            const void *table, const void *w, const void *gate, void *dn, void *dtable, void *dz,
                            long long rows, void *stream) {
    static const int blocks = resident_blocks(ve_backward_kernel, THREADS);
    ve_backward_kernel<<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
        (const bf16 *)dout, dout_stride, (const int *)ids, (const int64_t *)order, (const float *)table,
        (const float *)w, (const float *)gate, (bf16 *)dn, (float *)dtable, (bf16 *)dz, rows);
}
