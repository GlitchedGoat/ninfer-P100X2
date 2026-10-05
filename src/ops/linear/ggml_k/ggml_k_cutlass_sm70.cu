#include "ops/linear/ggml_k/ggml_k_prefill.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/ggml_k/ggml_k_dequant.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::bfloat16_t;
using GemmBf16 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
using GemmFp32 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

std::size_t gemm_workspace_bytes(int n, int k, int t) {
    const cutlass::gemm::GemmCoord shape(t, n, k);
    GemmBf16::Arguments bf16_args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    GemmFp32::Arguments fp32_args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return std::max(GemmBf16::get_workspace_size(bf16_args),
                    GemmFp32::get_workspace_size(fp32_args));
}

struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, int n, int k, int t) {
    Scratch scratch;
    scratch.weight = allocator.alloc(DType::FP16, {k, n});
    scratch.input = allocator.alloc(DType::FP16, {k, t});
    const std::size_t bytes = gemm_workspace_bytes(n, k, t);
    if (bytes != 0) { scratch.gemm = allocator.alloc_bytes(bytes); }
    return scratch;
}

} // namespace

std::size_t ggml_k_prefill_workspace_bytes(std::int32_t n, std::int32_t k,
                                                std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, tokens);
    return layout.peak_bytes(1);
}

void ggml_k_prefill_launch(const Tensor& x, const Weight& w, const Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream,
                                std::int32_t weight_row_offset, bool add,
                                bool tiled_gdn_input) {
    const int n = out.ne[0];
    const int k = w.k;
    const int t = x.ne[1];
    auto scope = workspace.scope();
    const auto scratch = allocate_scratch(workspace, n, k, t);
    auto* weight = static_cast<ElementInput*>(scratch.weight.data);
    auto* input = static_cast<ElementInput*>(scratch.input.data);

    const auto* descriptors = static_cast<const std::uint64_t*>(w.qhigh) + weight_row_offset;
    ggml_k_dequant_rows<ElementInput><<<dim3(static_cast<unsigned>(k / 256), static_cast<unsigned>(n)),
                   128, 0, stream>>>(static_cast<const unsigned char*>(w.qdata), descriptors,
                                     weight, k);
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(k) * t;
    const auto input_pairs = static_cast<unsigned>((input_count + 1) / 2);
    if (tiled_gdn_input) {
        ggml_k_stage_input<ElementInput, true><<<static_cast<int>((input_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), input, input_count, k);
    } else {
        ggml_k_stage_input<ElementInput, false><<<static_cast<int>((input_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), input, input_count, k);
    }
    CUDA_CHECK(cudaGetLastError());

    const cutlass::gemm::GemmCoord shape(t, n, k);
    const bool output_fp32 = out.dtype == DType::FP32;
    cutlass::Status status = cutlass::Status::kSuccess;
    if (output_fp32) {
        const int out_ld = static_cast<int>(out.nb[1] / sizeof(float));
        GemmFp32::Arguments args{
            shape, {input, k}, {weight, k}, {static_cast<float*>(out.data), out_ld},
            {static_cast<float*>(out.data), out_ld}, {1.0F, add ? 1.0F : 0.0F}, 1};
        GemmFp32 op;
        status = op.can_implement(args);
        if (status == cutlass::Status::kSuccess) { status = op.initialize(args, scratch.gemm.data, stream); }
        if (status == cutlass::Status::kSuccess) { status = op(stream); }
    } else {
        const int out_ld = static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16));
        GemmBf16::Arguments args{
            shape, {input, k}, {weight, k}, {static_cast<ElementOutput*>(out.data), out_ld},
            {static_cast<ElementOutput*>(out.data), out_ld}, {1.0F, add ? 1.0F : 0.0F}, 1};
        GemmBf16 op;
        status = op.can_implement(args);
        if (status == cutlass::Status::kSuccess) { status = op.initialize(args, scratch.gemm.data, stream); }
        if (status == cutlass::Status::kSuccess) { status = op(stream); }
    }
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("ggml_k_prefill (SM70): CUTLASS GEMM failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
