// Pascal (SM60) prefill attention over the paged INT8-G64 KV cache.
//
// GP100 has no Tensor Cores, so the Volta flash route (FP16 staging + vendored MMA kernel) does not
// apply. This route keeps every product and sum in FP32 on the SIMT pipeline and reads the paged
// cache directly: no gathered K/V copy, no mask tensor and no split-K partials, so it needs no
// workspace beyond the cache itself.
//
// Mathematics (per query head h of KV head g, query token t at absolute position P_t):
//   out[t,h] = sum_{j<=P_t} softmax_j(scale * q[t,h] . K[j,g]) V[j,g]
// with K/V the dequantized cache values code * fp16 group scale, exactly as the decode kernels
// read them. The cache is first appended for the whole width with the same quantizer as the decode
// kernels, so every key (history and current chunk) is read back through one representation.
//
// A masked call (valid_columns) has the chunked route's semantics: columns at or past
// valid_columns[0] are neither appended nor attended, and their output is zero.
//
// Work decomposition: one CTA owns (KV head, block of kTokens query tokens) = kTokens*GroupSize
// query rows; eight warps own interleaved rows. Lane i owns head dimensions [8i, 8i+8). Keys are
// processed in 32-key tiles staged in shared memory as FP32 (K, then V, in the same 32 KiB buffer,
// below GP100's 48 KiB limit). For one row and one tile, each lane forms 32 partial dot products
// over its 8 dimensions, and a butterfly reduce-scatter (31 shuffles) leaves the full score of key
// `lane` in lane `lane`. Online softmax in base 2 then updates the row state; P.V broadcasts each
// probability by shuffle. Rows past the chunk width and keys past a row's position are masked.

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/math.cuh"
#include "ops/common/pascal_convert.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_geometry_dispatch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim      = kGqaHeadDim; // 256
constexpr int kLaneDims     = kHeadDim / 32;
constexpr int kKeyTile      = 32;
constexpr int kTokens       = 4;
constexpr int kWarps        = 8;
constexpr int kThreads      = kWarps * 32;
constexpr unsigned kFull    = 0xffffffffu;
constexpr float kLog2E      = 1.4426950408889634074f;
static_assert(kLaneDims * 8 == kGqaKvQuantGroup, "a lane's dimensions lie in one quant group");
static_assert(kKeyTile * kHeadDim * sizeof(float) <= 32 * 1024, "tile must fit GP100 smem");

// INT8 cache code -> FP32. Exact either way; the Pascal form avoids I2F (T-010 O-4).
__device__ __forceinline__ float code_to_float(std::int8_t code) {
#ifdef NINFER_PASCAL_FAST_CONVERT
    return pascal_convert::small_int_to_float(code);
#else
    return static_cast<float>(code);
#endif
}

__device__ __forceinline__ const std::int32_t* select_block_table(const std::int32_t* tables,
                                                                  const std::int32_t* table_rows,
                                                                  int table_stride) {
    const int row = table_rows == nullptr ? 0 : table_rows[0];
    return tables + static_cast<std::int64_t>(row) * table_stride;
}

// BF16 K/V of the whole chunk -> paged INT8-G64 cache. One warp quantizes one
// (token, KV head, 64-wide group), identical to the decode kernels' cache write.
template <typename Geometry>
__global__ void __launch_bounds__(256)
    append_kv_i8_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                        const std::int32_t* __restrict__ positions,
                        const std::int32_t* __restrict__ table_rows,
                        std::int8_t* __restrict__ k_pages, std::int8_t* __restrict__ v_pages,
                        __half* __restrict__ k_scales, __half* __restrict__ v_scales,
                        const std::int32_t* __restrict__ block_tables, int table_stride,
                        const std::int32_t* __restrict__ valid_columns, int width) {
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int unit  = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int units = width * Geometry::KVHeads * kGqaKvQuantGroups;
    if (unit >= units) { return; }
    const int group   = unit % kGqaKvQuantGroups;
    const int rest    = unit / kGqaKvQuantGroups;
    const int kv_head = rest % Geometry::KVHeads;
    const int token   = rest / Geometry::KVHeads;
    if (valid_columns != nullptr && token >= valid_columns[0]) { return; } // warp-uniform
    const int d0      = group * kGqaKvQuantGroup + lane;
    const int d1      = d0 + 32;
    const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
    const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
    const float k0 = __bfloat162float(k[src0]);
    const float k1 = __bfloat162float(k[src1]);
    const float v0 = __bfloat162float(v[src0]);
    const float v1 = __bfloat162float(v[src1]);
    const float kamax = warp_max(fmaxf(fabsf(k0), fabsf(k1)), kFull);
    const float vamax = warp_max(fmaxf(fabsf(v0), fabsf(v1)), kFull);
    const __half ksh  = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
    const __half vsh  = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float k_inv = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float v_inv = vs > 0.0f ? 1.0f / vs : 0.0f;

    const std::int32_t* block_table = select_block_table(block_tables, table_rows, table_stride);
    const int position              = positions[token];
    const int physical_page         = paged_kv_physical_page(block_table, position);
    const int page_offset           = position & kPagedKVPageMask;
    k_pages[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
        gqa_kv_quant_code(k0, k_inv);
    k_pages[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
        gqa_kv_quant_code(k1, k_inv);
    v_pages[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
        gqa_kv_quant_code(v0, v_inv);
    v_pages[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
        gqa_kv_quant_code(v1, v_inv);
    if (lane == 0) {
        const std::int64_t scale =
            gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, group, page_offset);
        k_scales[scale] = ksh;
        v_scales[scale] = vsh;
    }
}

// Stages one 32-key tile of K or V as FP32. Thread t owns key t/8 and dimensions
// [32*(t%8), 32*(t%8)+32), i.e. half of one 64-wide quant group.
template <typename Geometry>
__device__ __forceinline__ void stage_tile(float* __restrict__ tile,
                                           const std::int8_t* __restrict__ pages,
                                           const __half* __restrict__ scales,
                                           const std::int32_t* __restrict__ block_table,
                                           int kv_head, int key_begin, int key_end) {
    const int tid = static_cast<int>(threadIdx.x);
    const int key = tid >> 3;
    const int d0  = (tid & 7) * 32;
    float* row    = tile + key * kHeadDim + d0;
    const int absolute = key_begin + key;
    if (absolute >= key_end) {
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
            *reinterpret_cast<float4*>(row + i) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const int physical_page = paged_kv_physical_page(block_table, absolute);
    const int page_offset   = absolute & kPagedKVPageMask;
    const float scale = __half2float(scales[gqa_kv_quant_scale_index<Geometry>(
        physical_page, kv_head, d0 / kGqaKvQuantGroup, page_offset)]);
    const auto* codes = reinterpret_cast<const int4*>(
        pages + gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset));
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int4 packed = __ldg(codes + half);
        const auto* bytes = reinterpret_cast<const std::int8_t*>(&packed);
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            *reinterpret_cast<float4*>(row + half * 16 + i) =
                make_float4(code_to_float(bytes[i]) * scale, code_to_float(bytes[i + 1]) * scale,
                            code_to_float(bytes[i + 2]) * scale,
                            code_to_float(bytes[i + 3]) * scale);
        }
    }
}

// After this call lane L holds sum over all lanes of partial[L] (recursive halving).
__device__ __forceinline__ float reduce_scatter_32(float (&partial)[kKeyTile], int lane) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < offset; ++i) {
            const float send = upper ? partial[i] : partial[i + offset];
            const float keep = upper ? partial[i + offset] : partial[i];
            partial[i]       = keep + __shfl_xor_sync(kFull, send, offset);
        }
    }
    return partial[0];
}

template <typename Geometry>
__global__ void __launch_bounds__(kThreads)
    flash_prefill_i8_kernel(const __nv_bfloat16* __restrict__ q,
                            const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ table_rows,
                            const std::int8_t* __restrict__ k_pages,
                            const std::int8_t* __restrict__ v_pages,
                            const __half* __restrict__ k_scales,
                            const __half* __restrict__ v_scales,
                            const std::int32_t* __restrict__ block_tables, int table_stride,
                            const std::int32_t* __restrict__ valid_columns, int width,
                            float scale_log2, __nv_bfloat16* __restrict__ out) {
    constexpr int kRows        = kTokens * Geometry::GroupSize;
    constexpr int kRowsPerWarp = (kRows + kWarps - 1) / kWarps;
    __shared__ __align__(16) float tile[kKeyTile * kHeadDim];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * kTokens;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int valid       = valid_columns == nullptr ? width : min(width, valid_columns[0]);
    const int tokens      = max(0, min(kTokens, valid - token_begin));
    const int block_width = min(kTokens, width - token_begin);
    const std::int32_t* block_table = select_block_table(block_tables, table_rows, table_stride);
    // Positions are non-decreasing within a prefill chunk; the last token bounds every row.
    const int key_end = tokens > 0 ? positions[token_begin + tokens - 1] + 1 : 0;

    float q_reg[kRowsPerWarp][kLaneDims];
    float acc[kRowsPerWarp][kLaneDims];
    float m[kRowsPerWarp];
    float l[kRowsPerWarp];
    int limit[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row   = warp + r * kWarps;
        const int token = row / Geometry::GroupSize;
        const int head  = kv_head * Geometry::GroupSize + row % Geometry::GroupSize;
        limit[r]        = row < kRows && token < tokens ? positions[token_begin + token] : -1;
        m[r]            = -CUDART_INF_F;
        l[r]            = 0.0f;
#pragma unroll
        for (int i = 0; i < kLaneDims; ++i) {
            acc[r][i]   = 0.0f;
            q_reg[r][i] = 0.0f;
        }
        if (limit[r] >= 0) {
            const int4 packed = *reinterpret_cast<const int4*>(
                q + gqa_q_index<Geometry>(head, lane * kLaneDims, token_begin + token));
            const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
            // Fold the softmax scale and log2(e) into Q once: scores are then base-2 logits.
#pragma unroll
            for (int i = 0; i < kLaneDims; ++i) {
                q_reg[r][i] = __bfloat162float(values[i]) * scale_log2;
            }
        }
    }

    for (int key_begin = 0; key_begin < key_end; key_begin += kKeyTile) {
        float p[kRowsPerWarp];
        stage_tile<Geometry>(tile, k_pages, k_scales, block_table, kv_head, key_begin, key_end);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            p[r] = 0.0f;
            if (limit[r] < key_begin) { continue; } // warp-uniform: limit is per row
            float partial[kKeyTile];
#pragma unroll
            for (int j = 0; j < kKeyTile; ++j) {
                const float4 k0 = *reinterpret_cast<const float4*>(tile + j * kHeadDim +
                                                                   lane * kLaneDims);
                const float4 k1 = *reinterpret_cast<const float4*>(tile + j * kHeadDim +
                                                                   lane * kLaneDims + 4);
                float dot = q_reg[r][0] * k0.x;
                dot       = fmaf(q_reg[r][1], k0.y, dot);
                dot       = fmaf(q_reg[r][2], k0.z, dot);
                dot       = fmaf(q_reg[r][3], k0.w, dot);
                dot       = fmaf(q_reg[r][4], k1.x, dot);
                dot       = fmaf(q_reg[r][5], k1.y, dot);
                dot       = fmaf(q_reg[r][6], k1.z, dot);
                partial[j] = fmaf(q_reg[r][7], k1.w, dot);
            }
            // Every lane must reach every shuffle: Pascal has no independent thread scheduling.
            const float total = reduce_scatter_32(partial, lane);
            const float score = key_begin + lane <= limit[r] ? total : -CUDART_INF_F;
            const float tile_max = warp_max(score, kFull);
            const float m_new    = fmaxf(m[r], tile_max);
            const float alpha    = m[r] == -CUDART_INF_F ? 0.0f : exp2_approx(m[r] - m_new);
            p[r]                 = score == -CUDART_INF_F ? 0.0f : exp2_approx(score - m_new);
            l[r]                 = l[r] * alpha + warp_sum(p[r], kFull);
            m[r]                 = m_new;
#pragma unroll
            for (int i = 0; i < kLaneDims; ++i) { acc[r][i] *= alpha; }
        }
        __syncthreads();
        stage_tile<Geometry>(tile, v_pages, v_scales, block_table, kv_head, key_begin, key_end);
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < kKeyTile; ++j) {
            const float4 v0 =
                *reinterpret_cast<const float4*>(tile + j * kHeadDim + lane * kLaneDims);
            const float4 v1 =
                *reinterpret_cast<const float4*>(tile + j * kHeadDim + lane * kLaneDims + 4);
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const float pj = __shfl_sync(kFull, p[r], j);
                acc[r][0]      = fmaf(pj, v0.x, acc[r][0]);
                acc[r][1]      = fmaf(pj, v0.y, acc[r][1]);
                acc[r][2]      = fmaf(pj, v0.z, acc[r][2]);
                acc[r][3]      = fmaf(pj, v0.w, acc[r][3]);
                acc[r][4]      = fmaf(pj, v1.x, acc[r][4]);
                acc[r][5]      = fmaf(pj, v1.y, acc[r][5]);
                acc[r][6]      = fmaf(pj, v1.z, acc[r][6]);
                acc[r][7]      = fmaf(pj, v1.w, acc[r][7]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row   = warp + r * kWarps;
        const int token = row / Geometry::GroupSize;
        const int head  = kv_head * Geometry::GroupSize + row % Geometry::GroupSize;
        if (row >= kRows || token >= block_width) { continue; }
        // Masked columns (limit < 0) have l == 0 and are written as zero.
        const float scale = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        __nv_bfloat16 values[kLaneDims];
#pragma unroll
        for (int i = 0; i < kLaneDims; ++i) { values[i] = __float2bfloat16(acc[r][i] * scale); }
        *reinterpret_cast<int4*>(out + gqa_q_index<Geometry>(head, lane * kLaneDims,
                                                             token_begin + token)) =
            *reinterpret_cast<const int4*>(values);
    }
}

template <typename Geometry>
void launch_for(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                const Tensor& valid_columns, const Tensor& table_rows, float scale,
                const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream) {
    const int width        = q.ne[2];
    const int table_stride = cache.block_tables.ne[0];
    const auto* rows       = table_rows.data == nullptr
                                 ? nullptr
                                 : static_cast<const std::int32_t*>(table_rows.data);
    const auto* tables     = static_cast<const std::int32_t*>(cache.block_tables.data);
    const auto* pos        = static_cast<const std::int32_t*>(positions.data);
    const auto* valid      = static_cast<const std::int32_t*>(valid_columns.data);

    const int units = width * Geometry::KVHeads * kGqaKvQuantGroups;
    append_kv_i8_kernel<Geometry><<<(units + kWarps - 1) / kWarps, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data), pos,
        rows, static_cast<std::int8_t*>(cache.k_pages.data),
        static_cast<std::int8_t*>(cache.v_pages.data),
        static_cast<__half*>(cache.k_scale_pages.data),
        static_cast<__half*>(cache.v_scale_pages.data), tables, table_stride, valid, width);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(Geometry::KVHeads, (width + kTokens - 1) / kTokens);
    flash_prefill_i8_kernel<Geometry><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), pos, rows,
        static_cast<const std::int8_t*>(cache.k_pages.data),
        static_cast<const std::int8_t*>(cache.v_pages.data),
        static_cast<const __half*>(cache.k_scale_pages.data),
        static_cast<const __half*>(cache.v_scale_pages.data), tables, table_stride, valid, width,
        scale * kLog2E, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gqa_attention_pascal_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& positions, const Tensor& valid_columns,
                                       const Tensor& table_rows, float scale,
                                       PagedKVBatchLayerView cache, Tensor& out,
                                       cudaStream_t stream) {
    if (cache.dtype != DType::I8 || cache.head_dim != kHeadDim ||
        cache.quant_group != kGqaKvQuantGroup || q.ne[3] != 1) {
        throw std::invalid_argument("gqa_attention Pascal flash: requires B=1 INT8-G64 KV");
    }
    dispatch_gqa_geometry(q.ne[1], [&]<typename Geometry>() {
        launch_for<Geometry>(q, k, v, positions, valid_columns, table_rows, scale, cache, out,
                             stream);
    });
}

} // namespace ninfer::ops::detail
