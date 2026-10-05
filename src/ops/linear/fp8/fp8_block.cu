#include "ops/linear/fp8/fp8_block.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"
#include "core/layout.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
void require_shape(int n, int k) {
    const bool column = k == 5120 && (n == 14336 || n == 7168 || n == 3584 ||
        n == 16384 || n == 8192 || n == 4096 || n == 34816 || n == 17408 || n == 8704 ||
        n == 6144 || n == 3072 || n == 1536 || n == 1024 || n == 512 || n == 256);
    const bool row = n == 5120 && (k == 6144 || k == 3072 || k == 1536 ||
                                  k == 17408 || k == 8704 || k == 4352);
    if (!column && !row) { throw std::invalid_argument("FP8 block128: unsupported 27B geometry"); }
}
} // namespace

namespace {
#ifdef NINFER_VOLTA_BUILD
void fp32_projection(const Tensor& x, const Weight& w, Tensor& out,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] >= 128) {
        fp8_cutlass_sm70_unscaled_fp32_launch(x, w, out, workspace, stream);
        return;
    }
    for (int begin = 0; begin < x.ne[1]; begin += 32) {
        const int count = std::min(32, x.ne[1] - begin);
        const auto input = x.slice(1, begin, count);
        const Fp8Fp32ContiguousOutput store{
            static_cast<float*>(out.data) + static_cast<std::int64_t>(begin) * w.n, w.n};
        launch_fp8_volta_qpn_with_output<Fp8Fp32ContiguousOutput, true>(input, w, store, w.n, stream);
    }
}

template <bool SwiGLU>
__global__ void finish_fp32(const float* input, __nv_bfloat16* output, int rows,
                            std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    float result;
    if constexpr (SwiGLU) {
        const auto offset = (i / rows) * (2 * rows) + i % rows;
        const float gate = input[offset];
        result = gate / (1.0F + expf(-gate)) * input[offset + rows];
    } else {
        result = input[i] + __bfloat162float(output[i]);
    }
    output[i] = __float2bfloat16_rn(result);
}
#endif

std::size_t composite_bytes(int n, int k, int min_tokens, int max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {n, max_tokens});
    const auto projection = fp8_block_workspace_bytes(n, k, min_tokens, max_tokens);
    if (projection) { (void)layout.alloc_bytes(projection); }
    return layout.peak_bytes(1);
}

template <bool SwiGLU>
void composite_launch(const Tensor& x, const Weight& w, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    validate_fp8_block_weight(w);
#ifdef NINFER_VOLTA_BUILD
    auto scope = workspace.scope();
    Tensor projection = workspace.alloc(DType::FP32, {w.n, x.ne[1]});
    fp32_projection(x, w, projection, workspace, stream);
    const auto count = out.numel();
    finish_fp32<SwiGLU><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(projection.data), static_cast<__nv_bfloat16*>(out.data),
        out.ne[0], count);
    CUDA_CHECK(cudaGetLastError());
#else
    throw std::invalid_argument("FP8 block128 currently requires SM70");
#endif
}
} // namespace

std::size_t fp8_block_swiglu_workspace_bytes(int n, int k, int min_tokens, int max_tokens) {
    if (k != 5120 || (n != 34816 && n != 17408 && n != 8704)) {
        throw std::invalid_argument("FP8 block128 SwiGLU: unsupported geometry");
    }
    return composite_bytes(n, k, min_tokens, max_tokens);
}
void fp8_block_swiglu_launch(const Tensor& x, const Weight& w, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    (void)fp8_block_swiglu_workspace_bytes(w.n, w.k, x.ne[1], x.ne[1]);
    composite_launch<true>(x, w, out, workspace, stream);
}
std::size_t fp8_block_add_workspace_bytes(int n, int k, int min_tokens, int max_tokens) {
    if (n != 5120) { throw std::invalid_argument("FP8 block128 residual: unsupported geometry"); }
    return composite_bytes(n, k, min_tokens, max_tokens);
}
void fp8_block_add_launch(const Tensor& x, const Weight& w, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    (void)fp8_block_add_workspace_bytes(w.n, w.k, x.ne[1], x.ne[1]);
    composite_launch<false>(x, w, residual, workspace, stream);
}

void validate_fp8_block_weight(const Weight& w) {
    require_shape(w.n, w.k);
    if (w.qtype != QType::FP8_E4M3FN_BLOCK128_BF16S || w.layout != QuantLayout::BlockScale128 ||
        w.scale_dtype != DType::BF16 || w.group != 128 || w.group_size != 128 ||
        w.ndim != 2 || w.n % 128 || w.k % 128 || w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k ||
        w.scale_ne[0] != w.k / 128 || w.scale_ne[1] != w.n / 128 ||
        !w.qdata || !w.scales || w.qhigh ||
        reinterpret_cast<std::uintptr_t>(w.qdata) % 16 ||
        reinterpret_cast<std::uintptr_t>(w.scales) % 2) {
        throw std::invalid_argument("FP8 block128: invalid codes/scales or geometry");
    }
}

std::size_t fp8_block_workspace_bytes(int n, int k, int min_tokens, int max_tokens) {
    require_shape(n, k);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("FP8 block128: invalid token interval");
    }
#ifdef NINFER_VOLTA_BUILD
    return max_tokens >= 128 ? fp8_cutlass_sm70_workspace_bytes(n, k, max_tokens) : 0;
#else
    throw std::invalid_argument("FP8 block128 currently requires SM70");
#endif
}

void fp8_block_launch(const Tensor& x, const Weight& w, Tensor& out,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_block_weight(w);
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= 128 && workspace) {
        fp8_cutlass_sm70_launch(x, w, out, *workspace, stream);
        return;
    }
    for (int begin = 0; begin < x.ne[1]; begin += 32) {
        const int count = std::min(32, x.ne[1] - begin);
        Tensor input = x.slice(1, begin, count);
        Tensor output = out.slice(1, begin, count);
        const Fp8ContiguousOutput store{static_cast<__nv_bfloat16*>(output.data), w.n};
        launch_fp8_volta_qpn_with_output<Fp8ContiguousOutput, true>(input, w, store, w.n, stream);
    }
#else
    throw std::invalid_argument("FP8 block128 currently requires SM70");
#endif
}
} // namespace ninfer::ops::detail
