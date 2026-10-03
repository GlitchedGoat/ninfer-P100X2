#pragma once
#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {
void validate_fp8_block_weight(const Weight& weight);
std::size_t fp8_block_workspace_bytes(int n, int k, int min_tokens, int max_tokens);
void fp8_block_launch(const Tensor& x, const Weight& weight, Tensor& out,
                      WorkspaceArena* workspace, cudaStream_t stream);
std::size_t fp8_block_swiglu_workspace_bytes(int n, int k, int min_tokens, int max_tokens);
void fp8_block_swiglu_launch(const Tensor& x, const Weight& weight, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream);
std::size_t fp8_block_add_workspace_bytes(int n, int k, int min_tokens, int max_tokens);
void fp8_block_add_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream);
} // namespace ninfer::ops::detail
