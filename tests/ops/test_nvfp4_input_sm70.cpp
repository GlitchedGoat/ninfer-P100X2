#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

#ifdef NINFER_VOLTA_BUILD
namespace {

constexpr ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

int check(std::string_view name, GuardedBf16Tensor& output, const qw::PackedWeight& weight,
          int row_begin, int rows, const std::vector<float>& x, int tokens) {
    return output.verify_guards(name) + output.verify_fully_written(name) +
        compare(name, gather_rows(output.values(), rows, 0, rows, tokens),
                projection_oracle(weight, row_begin, rows, x, 5120, tokens), kA16);
}

int exercise(bool attention, int tp) {
    const int n = (attention ? 14336 : 16384) / tp;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    const auto native = qw::make_patterned_weight(QType::NVFP4, n, 5120, attention ? 431U : 437U,
                                                  options);
    const auto packed = qw::nvfp4_qpn_payload(native);
    DeviceBuffer device_weight(packed.size());
    device_weight.copy_from_host(packed.data(), packed.size());
    auto weight = native.device_weight(device_weight.p);
    weight.layout = QuantLayout::VoltaQpnPrepacked;
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "projection test stream");
    int failures = 0;
    for (const int tokens : {1, 3, 4, 8, 9, 32, 33, 127, 128, 129}) {
        const auto x_host = make_bf16_activation(5120, tokens, 449U + tokens);
        auto device_x = to_device_bf16(x_host);
        Tensor x(device_x.p, DType::BF16, {5120, tokens});
        const auto prefix = std::string(attention ? "NVFP4 attention" : "NVFP4 GDN") +
            " TP" + std::to_string(tp) + " T=" + std::to_string(tokens);
        const auto capacity = tp == 2
            ? (attention ? ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, 2)
                         : ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, 2))
            : attention
            ? ops::attn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens)
            : ops::gdn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena workspace(std::max<std::size_t>(capacity, 1));
        if (attention) {
            const int q_rows = 6144 / tp, kv_rows = 1024 / tp;
            GuardedBf16Tensor query(q_rows, tokens), key(kv_rows, tokens),
                gate(q_rows, tokens), value(kv_rows, tokens);
            auto q = query.tensor(), k = key.tensor(), g = gate.tensor(), v = value.tensor();
            auto invoke = [&] {
                if (tp == 2) {
                    // Qualify the real shard kernel directly against the same mathematical
                    // oracle; the split suite separately checks the public two-device wrapper.
                    ops::detail::nvfp4_attn_input_sm70_launch(
                        x, weight, q, g, k, v, &workspace, stream);
                } else {
                    ops::attn_input_proj(x, weight, q, g, k, v,
                                        ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            // Capture the same public route using exactly its declared scratch capacity.
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate input");
            cuda_check(cudaGraphLaunch(exec, stream), "replay input");
            cuda_synchronize(stream);
            failures += check(prefix + " q", query, native, 0, q_rows, x_host, tokens);
            failures += check(prefix + " k", key, native, q_rows, kv_rows, x_host, tokens);
            failures += check(prefix + " gate", gate, native, q_rows + kv_rows, q_rows, x_host, tokens);
            failures += check(prefix + " v", value, native, 2 * q_rows + kv_rows, kv_rows, x_host, tokens);
            cuda_check(cudaGraphExecDestroy(exec), "destroy input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy input graph");
        } else {
            const int qkv_rows = 10240 / tp, z_rows = 6144 / tp;
            GuardedBf16Tensor projected(qkv_rows, tokens), z_out(z_rows, tokens);
            auto qkv = projected.tensor(), z = z_out.tensor();
            auto invoke = [&] {
                if (tp == 2) {
                    ops::detail::nvfp4_gdn_input_sm70_launch(
                        x, weight, qkv, z, &workspace, stream);
                } else {
                    ops::gdn_input_proj(x, weight, qkv, z,
                                       ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture GDN input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish GDN input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate GDN input");
            cuda_check(cudaGraphLaunch(exec, stream), "replay GDN input");
            cuda_synchronize(stream);
            failures += check(prefix + " qkv", projected, native, 0, qkv_rows, x_host, tokens);
            failures += check(prefix + " z", z_out, native, qkv_rows, z_rows, x_host, tokens);
            cuda_check(cudaGraphExecDestroy(exec), "destroy GDN input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy GDN input graph");
        }
        std::cout << prefix << '\n';
    }
    if (from_device<std::uint8_t>(device_weight.p, packed.size()) != packed) {
        std::cerr << "projection modified its packed weight\n";
        ++failures;
    }
    cuda_check(cudaStreamDestroy(stream), "destroy projection test stream");
    return failures;
}

} // namespace
#endif

int main() {
#ifndef NINFER_VOLTA_BUILD
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0), "query projection device");
    if (props.major != 7 || props.minor != 0) { return 77; }
    const int failures = exercise(true, 1) + exercise(false, 1) +
        exercise(true, 2) + exercise(false, 2);
    std::cout << (failures ? "FAIL" : "OK") << " NVFP4 SM70 input projections\n";
    return failures ? 1 : 0;
#endif
}
