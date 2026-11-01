/*
 * Prepares the per-row statistics that FlashAttention-4's backward reads, replacing its preprocessing pass
 * (flash_bwd_preprocess.py). Inputs: dpsum = rowsum(dO * O) per token and head, which the o_proj backward GEMM already
 * computed in its epilogue (linear.py dx_dpsum), and the forward's logsumexp lse. The backward expects both per head,
 * with each document segment b starting at a padded row offset start_b = (cu[b] + b m) / m * m, where m is its row
 * tile (seqlen_info.py offset_batch, padded=True). Row r = cu[b] + i of segment b goes to padded row p = start_b + i:
 *   dpsum_pad[h, p] = dpsum[r, h], lse_log2[h, p] = lse[h, r] * log2(e) (0 where lse = -inf).
 * Padding rows up to the next multiple of m get dpsum 0 and lse_log2 +inf, so their attention probabilities are 0.
 * One thread block per backward row tile of m rows.
 */
#include <math.h>

/* Padded row offset of segment b */
static __device__ __forceinline__ long long start(const int *cu, int b, int m) {
    return ((long long)cu[b] + (long long)b * m) / m * m;
}

__global__ void k_rows(const float *__restrict__ dpsum, const float *__restrict__ lse, long long lse_stride,
                       const int *__restrict__ cu, int segments, int H, int m, long long padded,
                       float *__restrict__ dpsum_pad, float *__restrict__ lse_log2) {
    const long long p = (long long)blockIdx.x * m + threadIdx.x;
    int lo = 0, hi = segments - 1;  /* the last segment b with start_b <= p */
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (start(cu, mid, m) <= p) lo = mid; else hi = mid - 1;
    }
    const int b = lo, len = cu[b + 1] - cu[b];
    const long long i = p - start(cu, b, m);
    if (i < len) {
        const long long r = cu[b] + i;
        for (int h = 0; h < H; h++) {
            const float l = lse[h * lse_stride + r];
            dpsum_pad[h * padded + p] = dpsum[r * H + h];
            /* log2(e) rounded to fp32, as FlashAttention-4's preprocessing (Float32(math.log2(math.e))) */
            lse_log2[h * padded + p] = l == -INFINITY ? 0.f : l * (float)M_LOG2E;
        }
    } else if (i < (len + m - 1) / m * m) {
        for (int h = 0; h < H; h++) {
            dpsum_pad[h * padded + p] = 0.f;
            lse_log2[h * padded + p] = INFINITY;
        }
    }
}

extern "C" void fa4_rows(const void *dpsum, const void *lse, long long lse_stride, const void *cu, int segments, int H,
                         int m, long long padded, void *dpsum_pad, void *lse_log2, void *stream) {
    k_rows<<<(unsigned)(padded / m), m, 0, (cudaStream_t)stream>>>(
        (const float *)dpsum, (const float *)lse, lse_stride, (const int *)cu, segments, H, m, padded,
        (float *)dpsum_pad, (float *)lse_log2);
}
