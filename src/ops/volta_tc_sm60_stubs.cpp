// Pascal SM60 build: SM70 Tensor-Core entry points referenced by the shared pre-Ampere
// dispatchers. GP100 has no Tensor Cores, so their "supported" predicates are false (dispatch falls
// through to the SIMT routes) and every launch throws. Startup admission rejects the identities
// whose dispatchers still require one of these routes (see qwen3_6_27b bind_artifact), so they are
// unreachable from the Engine; reaching one is a routing bug, never a silent no-op.
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.h"
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"
#include "ops/linear_add/q5/q5_linear_add_cutlass_sm70.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_cutlass_sm70.h"
#include "ops/attn_input_proj/volta_tp4.h"
#include "ops/launcher/vision_attention.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void unavailable(const char* name) {
    throw std::logic_error(std::string(name) + ": SM70 Tensor-Core route is unavailable on SM60");
}

} // namespace

void bf16_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v,
                                         cudaStream_t stream) { unavailable("bf16_attn_input_cutlass_sm70_launch"); }

void fp8_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v,
                                        WorkspaceArena& workspace, cudaStream_t stream) { unavailable("fp8_attn_input_cutlass_sm70_launch"); }

void fp8_attn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& gate, Tensor& k, Tensor& v,
                                              WorkspaceArena& workspace, cudaStream_t stream) { unavailable("fp8_attn_input_cutlass_sm70_launch_shard"); }

std::size_t fp8_attn_input_cutlass_workspace_bytes(std::int32_t tokens) { unavailable("fp8_attn_input_cutlass_workspace_bytes"); }

bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept { return false; }

void launch_fp8_attn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     cudaStream_t stream) { unavailable("launch_fp8_attn_input_volta_qpn"); }

void launch_fp8_attn_input_volta_qpn_shard(const Tensor& x, const Weight& weight, Tensor& query,
                                           Tensor& gate, Tensor& key, Tensor& value,
                                           cudaStream_t stream) { unavailable("launch_fp8_attn_input_volta_qpn_shard"); }

void nvfp4_attn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                 Tensor& gate, Tensor& k, Tensor& v,
                                 WorkspaceArena* workspace, cudaStream_t stream) { unavailable("nvfp4_attn_input_sm70_launch"); }

std::size_t nvfp4_attn_input_sm70_workspace_bytes(int rows, int tokens) { unavailable("nvfp4_attn_input_sm70_workspace_bytes"); }

void launch_q4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                         cudaStream_t stream, std::int32_t weight_row_offset,
                         int splits_override) { unavailable("launch_q4_volta_mma"); }

void launch_q5_volta_mma(const Tensor& x, const Weight& w, Tensor& out, bool add_residual,
                         std::int32_t weight_row_offset, WorkspaceArena& ws, cudaStream_t stream,
                         int splits_override) { unavailable("launch_q5_volta_mma"); }

void q4_q5_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& query_key_weight,
                                          const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                          Tensor& k, Tensor& v, WorkspaceArena& ws,
                                          cudaStream_t stream) { unavailable("q4_q5_attn_input_cutlass_sm70_launch"); }

std::size_t q4_q5_attn_input_cutlass_workspace_bytes(std::int32_t cols) { unavailable("q4_q5_attn_input_cutlass_workspace_bytes"); }

bool q4_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept { return false; }

std::size_t q4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                      std::int32_t t) noexcept { return 0; }

bool q5_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept { return false; }

std::size_t q5_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                      std::int32_t t) noexcept { return 0; }

void fp8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) { unavailable("fp8_gdn_input_cutlass_sm70_launch"); }

void fp8_gdn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, WorkspaceArena& workspace,
                                             cudaStream_t stream) { unavailable("fp8_gdn_input_cutlass_sm70_launch_shard"); }

std::size_t fp8_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) { unavailable("fp8_gdn_input_cutlass_workspace_bytes"); }

void launch_fp8_gdn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) { unavailable("launch_fp8_gdn_input_volta_qpn"); }

void launch_fp8_gdn_input_volta_qpn_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                          Tensor& z, cudaStream_t stream) { unavailable("launch_fp8_gdn_input_volta_qpn_shard"); }

void nvfp4_gdn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                Tensor& z, WorkspaceArena* workspace, cudaStream_t stream) { unavailable("nvfp4_gdn_input_sm70_launch"); }

std::size_t nvfp4_gdn_input_sm70_workspace_bytes(int rows, int tokens) { unavailable("nvfp4_gdn_input_sm70_workspace_bytes"); }

void q4_q5_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& qk_weight,
                                         const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                         WorkspaceArena& ws, cudaStream_t stream) { unavailable("q4_q5_gdn_input_cutlass_sm70_launch"); }

std::size_t q4_q5_gdn_input_cutlass_workspace_bytes(std::int32_t cols) { unavailable("q4_q5_gdn_input_cutlass_workspace_bytes"); }

void w8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      WorkspaceArena& ws, cudaStream_t stream) { unavailable("w8_gdn_input_cutlass_sm70_launch"); }

std::size_t w8_gdn_input_cutlass_workspace_bytes(std::int32_t cols) { unavailable("w8_gdn_input_cutlass_workspace_bytes"); }

void launch_bf16_cutlass_sm70(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) { unavailable("launch_bf16_cutlass_sm70"); }

void launch_bf16_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) { unavailable("launch_bf16_volta_qpn"); }

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) { unavailable("fp8_cutlass_sm70_launch"); }

void fp8_cutlass_sm70_unscaled_fp32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           WorkspaceArena& ws, cudaStream_t stream) { unavailable("fp8_cutlass_sm70_unscaled_fp32_launch"); }

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                            std::int32_t cols) { unavailable("fp8_cutlass_sm70_workspace_bytes"); }

void launch_fp8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable("launch_fp8_volta_qpn"); }

void launch_nvfp4_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable("launch_nvfp4_volta_qpn"); }

void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) { unavailable("nvfp4_cutlass_sm70_launch"); }

std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                              std::int32_t cols) { unavailable("nvfp4_cutlass_sm70_workspace_bytes"); }

bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept { return false; }

void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                         std::int32_t weight_row_offset) { unavailable("launch_q4_volta_qpn"); }

void launch_w8_volta_mma(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable("launch_w8_volta_mma"); }

void launch_w8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable("launch_w8_volta_qpn"); }

bool w8_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept { return false; }

bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept { return false; }

void bf16_linear_add_volta_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) { unavailable("bf16_linear_add_volta_launch"); }

void q5_linear_add_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       WorkspaceArena& ws, cudaStream_t stream) { unavailable("q5_linear_add_cutlass_sm70_launch"); }

std::size_t q5_linear_add_cutlass_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t cols) { unavailable("q5_linear_add_cutlass_workspace_bytes"); }

void fp8_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        float* gate_scratch, float* up_scratch,
                                        cudaStream_t stream) { unavailable("fp8_linear_swiglu_qpn_split_launch"); }

bool fp8_linear_swiglu_qpn_split_supported(std::int32_t n, std::int32_t k,
                                                          std::int32_t t) noexcept { return false; }

void nvfp4_cutlass_sm70_fp32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& ws, cudaStream_t stream) { unavailable("nvfp4_cutlass_sm70_fp32_launch"); }

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, void* activation_scratch,
                                          cudaStream_t stream) { unavailable("nvfp4_linear_swiglu_qpn_split_launch"); }

bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t n, std::int32_t k,
                                                            std::int32_t t) noexcept { return false; }

void nvfp4_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) { unavailable("nvfp4_linear_swiglu_volta_qpn_launch"); }

void q4_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& gate_up_out,
                                          WorkspaceArena& ws, cudaStream_t stream) { unavailable("q4_linear_swiglu_cutlass_sm70_launch"); }

std::size_t q4_linear_swiglu_cutlass_workspace_bytes(std::int32_t gate_up_rows, std::int32_t k,
                                                      std::int32_t cols) { unavailable("q4_linear_swiglu_cutlass_workspace_bytes"); }

void attn_input_volta_tp4_launch(const Tensor& x, const Weight& weight, Tensor& query,
                                Tensor& gate, Tensor& key, Tensor& value,
                                WorkspaceArena* workspace, cudaStream_t stream) { unavailable("attn_input_volta_tp4_launch"); }

std::size_t attn_input_volta_tp4_workspace_bytes(QType type, std::int32_t tokens) { unavailable("attn_input_volta_tp4_workspace_bytes"); }

void vision_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const std::int32_t* segment_bounds, std::int32_t segments,
                                         Tensor& q_f32, Tensor& k_f16, Tensor& v_f16,
                                         Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                         cudaStream_t stream) { unavailable("vision_attention_volta_flash_launch"); }

std::size_t vision_attention_volta_flash_meta_elements(std::int32_t patches) { unavailable("vision_attention_volta_flash_meta_elements"); }

std::int32_t vision_attention_volta_flash_staged_rows(std::int32_t patches) { unavailable("vision_attention_volta_flash_staged_rows"); }

} // namespace ninfer::ops::detail
