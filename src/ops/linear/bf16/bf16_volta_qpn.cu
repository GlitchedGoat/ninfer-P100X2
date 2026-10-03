#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/volta_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Eight tokens share one stream of unchanged row-major BF16 weights. Volta MMA
// takes FP16 operands (BF16 -> FP32 -> FP16), accumulates in FP32, and casts only
// the final output to BF16. There is no persistent repack or transient allocation.
template <int SplitK>
__global__ __launch_bounds__(SplitK * 32, 32 / SplitK) void bf16_volta_qpn_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out, int n, int k, int t) {
    __shared__ float partial[SplitK][8 * 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col = blockIdx.x * 32 + qp * 8 + r;
    const int blocks = k / 64;
    const int begin = warp * (blocks / SplitK) + min(warp, blocks % SplitK);
    const int end = begin + blocks / SplitK + (warp < blocks % SplitK);
    float c[2][8] = {};
    for (int block = begin; block < end; ++block) {
        // Preload an entire 128-byte row segment before contraction. Each lane
        // owns an output row; the eight vector loads consume its cache line once.
        uint4 words[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            words[e] = __ldg(reinterpret_cast<const uint4*>(
                w + static_cast<std::int64_t>(col) * k + block * 64 + e * 8));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const auto* wb = reinterpret_cast<const __nv_bfloat16*>(&words[e]);
            __half b[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { b[j] = __float2half_rn(__bfloat162float(wb[j])); }
            __half a[8];
            if (r < t) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(r) * k + block * 64 + e * 8);
                const auto* xb = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) { a[j] = __float2half_rn(__bfloat162float(xb[j])); }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) { a[j] = __ushort_as_half(0); }
            }
            const auto* aa = reinterpret_cast<const unsigned*>(a);
            const auto* bb = reinterpret_cast<const unsigned*>(b);
            volta_mma_qp_n(c[0], aa[0], aa[1], bb[0], bb[1]);
            volta_mma_qp_n(c[1], aa[2], aa[3], bb[2], bb[3]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cl = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partial[warp][row * 32 + qp * 8 + cl] = c[0][i] + c[1][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 8 * 32; e += SplitK * 32) {
        const int row = e / 32;
        if (row < t) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) { sum += partial[s][e]; }
            out[static_cast<std::int64_t>(row) * n + blockIdx.x * 32 + e % 32] =
                __float2bfloat16_rn(sum);
        }
    }
}

} // namespace

void launch_bf16_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    bf16_volta_qpn_kernel<4><<<weight.n / 32, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
