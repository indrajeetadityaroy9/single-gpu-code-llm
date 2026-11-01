/*
 * Code shared by qk_prologue.cu and ve_gate.cu: types, 16-byte vector access and the device's launch capacity.
 */
#pragma once

#include <float.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

constexpr int WARP = 32;
constexpr int VEC = sizeof(uint4) / sizeof(bf16);  /* bf16 per 16-byte access */

/* VEC bf16 (one 16-byte load) as fp32. */
static __device__ __forceinline__ void load(const bf16 *p, float *v) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x; v[2 * i + 1] = f.y;
    }
}

/* VEC fp32; offsets are multiples of VEC, so each float4 is aligned. */
static __device__ __forceinline__ void load(const float *p, float *v) {
    for (int i = 0; i < VEC; i += sizeof(float4) / sizeof(float))
        *reinterpret_cast<float4 *>(v + i) = *reinterpret_cast<const float4 *>(p + i);
}

/* VEC fp32 rounded to bf16, one 16-byte store. */
static __device__ __forceinline__ void store(bf16 *p, const float *v) {
    uint4 u;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&u);
    for (int i = 0; i < VEC / 2; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = u;
}

/* Blocks of kernel f the current device holds at once with `threads` threads per block: a persistent grid. */
template <typename F> static int resident_blocks(F f, int threads) {
    int device, sms, per_sm;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, threads, 0);
    return per_sm * sms;
}
