#include "ops/linear/ggml_k/ggml_k_prefill.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/ggml_k/ggml_k_dequant.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>

// Pascal (GP100) implementation of the wide-T GGML_K projection. GP100 has no Tensor Cores and
// runs FP32 at half its FP16x2 rate, so this first route keeps every operand in FP32: weights are
// decoded from the stored Q4_K/Q6_K fields into FP32 rows, activations are widened exactly from
// BF16, and CUTLASS's SIMT SGEMM accumulates in FP32. Decoding the full [N,K] matrix in FP32 would
// double the Volta FP16 workspace, so rows are processed in bounded chunks.
namespace ninfer::ops::detail {
namespace {

using ElementOutputBf16 = cutlass::bfloat16_t;

// 128x128x8 CTA / 32x64x8 warp is CUTLASS's default SIMT SGEMM shape for Maxwell/Pascal; its
// shared-memory footprint (2 stages) stays far below GP100's 48 KiB per-block limit.
template <typename ElementOutput>
using SimtGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, ElementOutput,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

using GemmBf16 = SimtGemm<ElementOutputBf16>;
using GemmFp32 = SimtGemm<float>;

// Upper bound on one decoded FP32 weight chunk. 64 MiB keeps every registered TP1/TP2 projection
// to a handful of chunks while leaving the KV pool the bulk of a 16 GB card.
constexpr std::int64_t kWeightChunkBytes = 64ll << 20;

int chunk_rows(int n, int k) {
    const std::int64_t rows = kWeightChunkBytes / (static_cast<std::int64_t>(k) * sizeof(float));
    const int aligned = static_cast<int>(std::max<std::int64_t>(128, rows / 128 * 128));
    return std::min(n, aligned);
}

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
    scratch.weight = allocator.alloc(DType::FP32, {k, chunk_rows(n, k)});
    scratch.input = allocator.alloc(DType::FP32, {k, t});
    const std::size_t bytes = gemm_workspace_bytes(chunk_rows(n, k), k, t);
    if (bytes != 0) { scratch.gemm = allocator.alloc_bytes(bytes); }
    return scratch;
}

template <typename Gemm, typename ElementOutput>
void run_gemm(int t, int rows, int k, const float* input, const float* weight,
              ElementOutput* out, int out_ld, bool add, void* gemm_workspace,
              cudaStream_t stream) {
    const cutlass::gemm::GemmCoord shape(t, rows, k);
    typename Gemm::Arguments args{shape, {input, k}, {weight, k}, {out, out_ld}, {out, out_ld},
                                  {1.0F, add ? 1.0F : 0.0F}, 1};
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status == cutlass::Status::kSuccess) { status = op.initialize(args, gemm_workspace, stream); }
    if (status == cutlass::Status::kSuccess) { status = op(stream); }
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("ggml_k_prefill (SM60): CUTLASS SIMT GEMM failed");
    }
}

} // namespace

std::size_t ggml_k_prefill_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, tokens);
    return layout.peak_bytes(1);
}

void ggml_k_prefill_launch(const Tensor& x, const Weight& w, const Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           std::int32_t weight_row_offset, bool add, bool tiled_gdn_input) {
    const int n = out.ne[0];
    const int k = w.k;
    const int t = x.ne[1];
    auto scope = workspace.scope();
    const auto scratch = allocate_scratch(workspace, n, k, t);
    auto* weight = static_cast<float*>(scratch.weight.data);
    auto* input = static_cast<float*>(scratch.input.data);

    const std::int64_t input_count = static_cast<std::int64_t>(k) * t;
    const auto input_pairs = static_cast<unsigned>((input_count + 1) / 2);
    const auto* x_data = static_cast<const __nv_bfloat16*>(x.data);
    if (tiled_gdn_input) {
        ggml_k_stage_input<float, true><<<static_cast<int>((input_pairs + 255) / 256), 256, 0,
                                          stream>>>(x_data, input, input_count, k);
    } else {
        ggml_k_stage_input<float, false><<<static_cast<int>((input_pairs + 255) / 256), 256, 0,
                                           stream>>>(x_data, input, input_count, k);
    }
    CUDA_CHECK(cudaGetLastError());

    const bool output_fp32 = out.dtype == DType::FP32;
    const int out_ld = static_cast<int>(out.nb[1] / dtype_size(out.dtype));
    const int step = chunk_rows(n, k);
    const auto* descriptors = static_cast<const std::uint64_t*>(w.qhigh) + weight_row_offset;
    for (int begin = 0; begin < n; begin += step) {
        const int rows = std::min(step, n - begin);
        ggml_k_dequant_rows<float><<<dim3(static_cast<unsigned>(k / 256),
                                          static_cast<unsigned>(rows)),
                                     128, 0, stream>>>(static_cast<const unsigned char*>(w.qdata),
                                                       descriptors + begin, weight, k);
        CUDA_CHECK(cudaGetLastError());
        if (output_fp32) {
            run_gemm<GemmFp32>(t, rows, k, input, weight, static_cast<float*>(out.data) + begin,
                               out_ld, add, scratch.gemm.data, stream);
        } else {
            run_gemm<GemmBf16>(t, rows, k, input, weight,
                               static_cast<ElementOutputBf16*>(out.data) + begin, out_ld, add,
                               scratch.gemm.data, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
