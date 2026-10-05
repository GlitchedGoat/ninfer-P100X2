#pragma once

// Dense GEMM configuration shared by the pre-Ampere "decode once, then dense GEMM" prefill routes
// (`*_cutlass_sm70.cu`). Those routes materialize quantized weights and BF16 activations into
// caller-owned workspace and run one CUTLASS GEMM with FP32 accumulation:
//
//   - Volta (SM70): FP16 operands on SM70 Tensor Cores (mma.sync.m8n8k4).
//   - Pascal (SM60): GP100 has no Tensor Cores and its FP16 FMA is only faster with FP16
//     accumulation, so operands stay FP32 and CUTLASS's SIMT SGEMM accumulates in FP32. Operand
//     workspace is therefore twice the Volta size.
//
// Each route forms every operand value in FP32 from its stored codes/scales and rounds it once to
// `Operand`; on Pascal that rounding is the identity.

#include "core/dtype.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace ninfer::ops::detail::pre_ampere {

#if defined(NINFER_VOLTA_BUILD)
using Operand                          = cutlass::half_t;
inline constexpr DType kOperandDType   = DType::FP16;
using OpClass                          = cutlass::arch::OpClassTensorOp;
using Arch                             = cutlass::arch::Sm70;
template <int TileN = 128>
using ThreadblockShape                 = cutlass::gemm::GemmShape<128, TileN, 32>;
using WarpShape                        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape                 = cutlass::gemm::GemmShape<8, 8, 4>;
// 128-bit epilogue stores.
template <class Output>
inline constexpr int kEpilogueVector = 128 / cutlass::sizeof_bits<Output>::value;
// Fused epilogues with vector width > 1 are available.
inline constexpr bool kVectorEpilogue = true;
#elif defined(NINFER_PASCAL_BUILD)
using Operand                          = float;
inline constexpr DType kOperandDType   = DType::FP32;
using OpClass                          = cutlass::arch::OpClassSimt;
using Arch                             = cutlass::arch::Sm60;
// CUTLASS's default SIMT SGEMM tile; 2 stages use ~16 KiB of GP100's 48 KiB per block.
// Wider Volta tiles (TileN = 256) map to the same SIMT tile.
template <int TileN = 128>
using ThreadblockShape                 = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape                        = cutlass::gemm::GemmShape<32, 64, 8>;
using InstructionShape                 = cutlass::gemm::GemmShape<1, 1, 1>;
// The SIMT epilogue stores one element per access.
template <class Output>
inline constexpr int kEpilogueVector = 1;
inline constexpr bool kVectorEpilogue = false;
#else
#error "pre_ampere_gemm.cuh is only for SM60/SM70 builds"
#endif

template <class Output>
using LinearEpilogue =
    cutlass::epilogue::thread::LinearCombination<Output, kEpilogueVector<Output>, float, float>;

// D[t, n] = X[t, k] * W[n, k]^T with X row-major and W row-major (= column-major K x N).
template <class Output, class Epilogue = LinearEpilogue<Output>, int TileN = 128>
using Gemm = cutlass::gemm::device::Gemm<
    Operand, cutlass::layout::RowMajor, Operand, cutlass::layout::ColumnMajor, Output,
    cutlass::layout::RowMajor, float, OpClass, Arch, ThreadblockShape<TileN>, WarpShape,
    InstructionShape, Epilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

} // namespace ninfer::ops::detail::pre_ampere
