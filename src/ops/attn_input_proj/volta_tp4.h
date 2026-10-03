#pragma once

#include "ninfer/ops/linear.h"

namespace ninfer::ops::detail {

// SM70 registered [3584,5120] head-local Q|K|Gate|V profile, A16 only.
std::size_t attn_input_volta_tp4_workspace_bytes(QType type, std::int32_t tokens);
void attn_input_volta_tp4_launch(const Tensor& x, const Weight& weight, Tensor& query,
                                Tensor& gate, Tensor& key, Tensor& value,
                                WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
