#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Wide-T (prefill) GGML_K projection on pre-Ampere targets: decode the native Q4_K/Q6_K rows into
// caller-owned workspace once per call, then run a dense CUTLASS GEMM. Volta implements it with
// FP16 operands on SM70 Tensor Cores (ggml_k_cutlass_sm70.cu); Pascal with FP32 operands on the
// SIMT pipeline (ggml_k_cutlass_simt.cu). Both accumulate in FP32 and preserve the code bytes.
std::size_t ggml_k_prefill_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens);
void ggml_k_prefill_launch(const Tensor& x, const Weight& w, const Tensor& out,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           std::int32_t weight_row_offset = 0, bool add = false,
                           bool tiled_gdn_input = false);

} // namespace ninfer::ops::detail
