#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Shared staging kernels for the pre-Ampere wide-T GGML_K route (ggml_k_prefill.h). `Elem` is the
// dense GEMM operand type: a 16-bit half type for the Volta Tensor-Core GEMM, float for the Pascal
// SIMT GEMM. Each value is formed in FP32 from the stored codes and scales and rounded once to
// `Elem`, so the FP32 instantiation materializes the exact FP32 product of the stored fields.

// One CTA owns a native K256 block.  For Q4_K a lane reads one packed byte and emits both
// values, so its two 32-wide stores are coalesced; Q6_K keeps one lane per logical value.
template <typename Elem>
__global__ void ggml_k_dequant_rows(const unsigned char* __restrict__ rows,
                                    const std::uint64_t* __restrict__ descriptors,
                                    Elem* __restrict__ out, int k) {
    const int row = static_cast<int>(blockIdx.y);
    const int block_index = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const std::uint64_t descriptor = descriptors[row];
    const bool q6 = (descriptor & 1u) != 0;
    const unsigned char* block = rows + (descriptor >> 1) + block_index * (q6 ? 210 : 144);
    auto* output = out + static_cast<std::int64_t>(row) * k + block_index * 256;
    if (q6) {
        if (tid >= 64) { return; }
        const int half = tid >> 5;
        const int lane = tid & 31;
        const unsigned low0 = __ldg(block + half * 64 + lane);
        const unsigned low1 = __ldg(block + half * 64 + 32 + lane);
        const unsigned high = __ldg(block + 128 + half * 32 + lane);
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
#pragma unroll
        for (int section = 0; section < 4; ++section) {
            const unsigned low = (section & 1) == 0 ? low0 : low1;
            const unsigned nibble = (low >> ((section >> 1) * 4)) & 15u;
            const unsigned high_bits = (high >> (section * 2)) & 3u;
            const int code = static_cast<int>(nibble | (high_bits << 4)) - 32;
            const int scale_index = half * 8 + section * 2 + (lane >> 4);
            const int scale = static_cast<int>(static_cast<signed char>(
                __ldg(block + 192 + scale_index)));
            output[half * 128 + section * 32 + lane] =
                Elem(d * static_cast<float>(scale * code));
        }
        return;
    }
    if (tid >= 128) { return; }
    const int pair = tid >> 5;
    const int lane = tid & 31;
    const unsigned code = block[16 + tid];
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dm = __half2float(*reinterpret_cast<const __half*>(block + 2));
    const unsigned char* scales = block + 4;
    const int g0 = pair * 2;
    const int g1 = g0 + 1;
    const int scale0 = g0 < 4 ? scales[g0] & 63
                              : (scales[g0 + 4] & 15) | ((scales[g0 - 4] >> 6) << 4);
    const int scale1 = g1 < 4 ? scales[g1] & 63
                              : (scales[g1 + 4] & 15) | ((scales[g1 - 4] >> 6) << 4);
    const int min0 = g0 < 4 ? scales[g0 + 4] & 63
                            : (scales[g0 + 4] >> 4) | ((scales[g0] >> 6) << 4);
    const int min1 = g1 < 4 ? scales[g1 + 4] & 63
                            : (scales[g1 + 4] >> 4) | ((scales[g1] >> 6) << 4);
    output[pair * 64 + lane] = Elem((d * scale0) * (code & 15) - dm * min0);
    output[pair * 64 + 32 + lane] = Elem((d * scale1) * (code >> 4) - dm * min1);
}

// BF16 activations -> dense operand. With TiledGdn the GGUF GDN-output column permutation is
// applied while staging (see ggml_k.cu input_column), so the weight blocks stay untouched.
template <typename Elem, bool TiledGdn>
__global__ void ggml_k_stage_input(const __nv_bfloat16* __restrict__ in, Elem* __restrict__ out,
                                   std::int64_t count, int k) {
    // Every registered projection has an even K and token count. Convert two adjacent BF16
    // values per lane so the staging pass uses one 32-bit load; keep the odd-count tail for the
    // public helper's general contract.
    const std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t index = pair * 2;
    if (index >= count) { return; }
    std::int64_t source = index;
    if constexpr (TiledGdn) {
        const int col = static_cast<int>(index % k);
        const int head = col / 128;
        const int grouped = k == 3072 ? (head & 7) * 3 + (head >> 3)
                                      : (head & 15) * 3 + (head >> 4);
        source = (index / k) * k + grouped * 128 + (col & 127);
    }
    if (index + 1 < count) {
        const float2 values =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + source));
        if constexpr (sizeof(Elem) == sizeof(float)) {
            *reinterpret_cast<float2*>(out + index) = values;
        } else {
            *reinterpret_cast<__half2*>(out + index) = __float22half2_rn(values);
        }
    } else {
        out[index] = Elem(__bfloat162float(in[source]));
    }
}

} // namespace ninfer::ops::detail
