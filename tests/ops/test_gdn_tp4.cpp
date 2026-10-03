// Four-rank quarter projections checked directly against independent FP64 formulas.
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <iostream>
#include <memory>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {
int run_control(const ExecutionContext& ec, int tokens) {
    constexpr int heads=12, hidden=5120;
    std::array<DeviceBuffer,4> dx,dw,da,dd,dg,db,scratch;
    std::array<std::unique_ptr<WorkspaceArena>,4> arenas;
    std::array<Tensor,4> x,a_log,dt_bias,g,beta;
    std::array<Weight,4> weights;
    std::array<WorkspaceArena*,4> workspace;
    std::array<std::vector<float>,4> activation,weight,logs,bias;
    const auto bytes=ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(tokens,tokens);
    for(int rank=0;rank<4;++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        activation[rank]=make_bf16_activation(hidden,tokens,2111+rank);
        weight[rank]=make_bf16_activation(hidden,2*heads,2121+rank);
        logs[rank].resize(heads);bias[rank].resize(heads);
        fill_uniform(logs[rank],2131+rank,-2,1);
        fill_uniform(bias[rank],2141+rank,-1,1);
        dx[rank]=to_device_bf16(activation[rank]);dw[rank]=to_device_bf16(weight[rank]);
        da[rank]=to_device(logs[rank]);dd[rank]=to_device(bias[rank]);
        dg[rank]=DeviceBuffer(heads*tokens*sizeof(float));
        db[rank]=DeviceBuffer(heads*tokens*sizeof(float));
        scratch[rank]=DeviceBuffer(std::max<std::size_t>(bytes,256));
        arenas[rank]=std::make_unique<WorkspaceArena>(
            DeviceSpan{scratch[rank].p,std::max<std::size_t>(bytes,256)});
        workspace[rank]=arenas[rank].get();
        x[rank]=Tensor(dx[rank].p,DType::BF16,{hidden,tokens});
        a_log[rank]=Tensor(da[rank].p,DType::FP32,{heads});
        dt_bias[rank]=Tensor(dd[rank].p,DType::FP32,{heads});
        g[rank]=Tensor(dg[rank].p,DType::FP32,{heads,tokens});
        beta[rank]=Tensor(db[rank].p,DType::FP32,{heads,tokens});
        auto& w=weights[rank];
        w.qtype=QType::BF16_CTRL;w.layout=QuantLayout::Contiguous;
        w.payload=w.qdata=dw[rank].p;w.payload_bytes=dw[rank].bytes;
        w.ndim=2;w.n=w.shape[0]=w.padded_shape[0]=2*heads;
        w.k=w.shape[1]=w.padded_shape[1]=hidden;
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::gdn_gating_proj_column_parallel(x,weights,a_log,dt_bias,workspace,g,beta,ec);
    int failures=0;
    for(int rank=0;rank<4;++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));ec.dev[rank]->synchronize();
        std::vector<double> reference_g(heads*tokens),reference_beta(heads*tokens);
        for(int token=0;token<tokens;++token) {
            for(int head=0;head<heads;++head) {
                double pa=0,pb=0;
                for(int k=0;k<hidden;++k) {
                    const double value=activation[rank][token*hidden+k];
                    pa+=value*weight[rank][head*hidden+k];
                    pb+=value*weight[rank][(heads+head)*hidden+k];
                }
                const double v=pa+bias[rank][head];
                reference_g[token*heads+head]=-std::exp(double(logs[rank][head]))*
                    (std::max(v,0.0)+std::log1p(std::exp(-std::abs(v))));
                reference_beta[token*heads+head]=1/(1+std::exp(-pb));
            }
        }
        const std::string label="TP4 GDN control rank="+std::to_string(rank)+
                                " T="+std::to_string(tokens);
        const ReductionCriterion criterion{1.4e-6,5e-7,2.5e-6};
        const auto actual_g=from_device<float>(dg[rank],reference_g.size());
        const auto actual_beta=from_device<float>(db[rank],reference_beta.size());
        failures+=compare(label+" g",{actual_g.begin(),actual_g.end()},reference_g,criterion);
        failures+=compare(label+" beta",{actual_beta.begin(),actual_beta.end()},reference_beta,criterion);
        if(workspace[rank]->used()!=0 || workspace[rank]->peak_used()>bytes) {++failures;}
    }
    return failures;
}

int run_attention(const ExecutionContext& ec, QType type, int tokens) {
    std::array<std::unique_ptr<DevicePackedWeight>,4> weights;
    std::array<std::unique_ptr<GuardedBf16Tensor>,4> inputs,q,k,g,v;
    std::array<std::unique_ptr<DeviceBuffer>,4> storage;
    std::array<std::unique_ptr<WorkspaceArena>,4> arenas;
    std::array<Tensor,4> xv,qv,kv,gv,vv;
    std::array<Weight,4> wv;
    std::array<WorkspaceArena*,4> ws;
    std::array<std::vector<float>,4> activation;
    const auto bytes=ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
        type,ops::LinearPolicy::A16Only,tokens,tokens,4);
    for(int r=0;r<4;++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        quantized_weight::PatternedWeightOptions options;
        if(type==QType::NVFP4) {options.weight_scale_divisor=.125F;options.input_scale_divisor=3.5F;}
        weights[r]=std::make_unique<DevicePackedWeight>(quantized_weight::make_patterned_weight(
            type,3584,5120,2011+r,options));
        inputs[r]=std::make_unique<GuardedBf16Tensor>(5120,tokens);
        q[r]=std::make_unique<GuardedBf16Tensor>(1536,tokens);
        k[r]=std::make_unique<GuardedBf16Tensor>(256,tokens);
        g[r]=std::make_unique<GuardedBf16Tensor>(1536,tokens);
        v[r]=std::make_unique<GuardedBf16Tensor>(256,tokens);
        activation[r]=make_bf16_activation(5120,tokens,2021+r);
        std::vector<std::uint16_t> bits(activation[r].size());
        std::transform(activation[r].begin(),activation[r].end(),bits.begin(),f32_to_bf16);
        inputs[r]->copy_from_bits(bits);
        storage[r]=std::make_unique<DeviceBuffer>(std::max<std::size_t>(bytes,256));
        arenas[r]=std::make_unique<WorkspaceArena>(
            DeviceSpan{storage[r]->p,std::max<std::size_t>(bytes,256)});
        xv[r]=inputs[r]->tensor();qv[r]=q[r]->tensor();kv[r]=k[r]->tensor();
        gv[r]=g[r]->tensor();vv[r]=v[r]->tensor();wv[r]=weights[r]->view();ws[r]=arenas[r].get();
        // Fixtures use the legacy default stream; the Op uses nonblocking rank streams.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::attn_input_proj_column_parallel(xv,wv,qv,gv,kv,vv,ops::LinearPolicy::A16Only,ws,ec);
    int failures=0;
    for(int r=0;r<4;++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));ec.dev[r]->synchronize();
        const std::string label="TP4 attention type="+std::to_string(static_cast<int>(type))+
            " rank="+std::to_string(r)+" T="+std::to_string(tokens);
        const ReductionCriterion criterion{1.0/256,1.0/256,2.0/256};
        for(const auto section:{std::array{0,1536,0},std::array{1536,256,1},
                                std::array{1792,1536,2},std::array{3328,256,3}}) {
            const GuardedBf16Tensor* output=section[2]==0?q[r].get():section[2]==1?k[r].get():
                                            section[2]==2?g[r].get():v[r].get();
            failures+=compare(label,gather_rows(output->values(),section[1],0,section[1],tokens),
                projection_oracle(weights[r]->host,section[0],section[1],activation[r],5120,tokens),criterion);
            failures+=output->verify_guards(label);
            failures+=output->verify_fully_written(label);
        }
        failures+=weights[r]->verify_preserved(label);
        if(ws[r]->used()!=0 || ws[r]->peak_used()>bytes) {++failures;}
    }
    return failures;
}

int run_projection(const ExecutionContext& ec, QType type, int tokens) {
    std::array<std::unique_ptr<DevicePackedWeight>, 4> weights;
    std::array<std::unique_ptr<GuardedBf16Tensor>, 4> inputs, qkv, z;
    std::array<std::unique_ptr<DeviceBuffer>, 4> storage;
    std::array<std::unique_ptr<WorkspaceArena>, 4> arenas;
    std::array<Tensor, 4> x_view, qkv_view, z_view;
    std::array<Weight, 4> w_view;
    std::array<WorkspaceArena*, 4> ws;
    std::array<std::vector<float>, 4> activation;
    const auto bytes = ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
        type, ops::LinearPolicy::A16Only, tokens, tokens, 4);
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        quantized_weight::PatternedWeightOptions options;
        if (type == QType::NVFP4) {
            options.weight_scale_divisor = 0.125F;
            options.input_scale_divisor = 3.5F;
        }
        weights[r] = std::make_unique<DevicePackedWeight>(
            quantized_weight::make_patterned_weight(type, 4096, 5120, 1911U + r, options));
        inputs[r] = std::make_unique<GuardedBf16Tensor>(5120, tokens);
        qkv[r] = std::make_unique<GuardedBf16Tensor>(2560, tokens);
        z[r] = std::make_unique<GuardedBf16Tensor>(1536, tokens);
        activation[r] = make_bf16_activation(5120, tokens, 1917U + r);
        std::vector<std::uint16_t> bits(activation[r].size());
        std::transform(activation[r].begin(), activation[r].end(), bits.begin(), f32_to_bf16);
        inputs[r]->copy_from_bits(bits);
        storage[r] = std::make_unique<DeviceBuffer>(std::max<std::size_t>(bytes, 256));
        arenas[r] = std::make_unique<WorkspaceArena>(DeviceSpan{storage[r]->p, bytes});
        x_view[r] = inputs[r]->tensor();
        qkv_view[r] = qkv[r]->tensor();
        z_view[r] = z[r]->tensor();
        w_view[r] = weights[r]->view();
        ws[r] = arenas[r].get();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::gdn_input_proj_column_parallel(x_view, w_view, qkv_view, z_view,
                                         ops::LinearPolicy::A16Only, ws, ec);
    int failures = 0;
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        ec.dev[r]->synchronize();
        const std::string label = "TP4 GDN type=" + std::to_string(static_cast<int>(type))+
                                  " rank=" + std::to_string(r) +
                                  " T=" + std::to_string(tokens);
        const ReductionCriterion criterion = type == QType::NVFP4
            ? ReductionCriterion{3.0e-3, 4.0e-3, 3.5e-3}
            : ReductionCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
        const auto values = qkv[r]->values();
        for (const auto section : {std::array{0, 512}, std::array{512, 512},
                                    std::array{1024, 1536}}) {
            failures += compare(label, gather_rows(values, 2560, section[0], section[1], tokens),
                projection_oracle(weights[r]->host, section[0], section[1], activation[r],
                                  5120, tokens), criterion);
        }
        failures += compare(label + " z", gather_rows(z[r]->values(), 1536, 0, 1536, tokens),
            projection_oracle(weights[r]->host, 2560, 1536, activation[r], 5120, tokens), criterion);
        failures += qkv[r]->verify_guards(label);
        failures += z[r]->verify_guards(label);
        failures += qkv[r]->verify_fully_written(label);
        failures += z[r]->verify_fully_written(label);
        failures += weights[r]->verify_preserved(label);
        if (ws[r]->used() != 0 || ws[r]->peak_used() > bytes) { ++failures; }
    }
    return failures;
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 4) {
        std::cout << "SKIP: four CUDA devices required\n";
        return 77;
    }
    try {
        ExecutionContext ec({0, 1, 2, 3});
        int failures = 0;
        for(const int tokens:{1,4,128,2560}) {failures+=run_control(ec,tokens);}
        for (const auto type : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S,
                                QType::FP8_E4M3FN_BLOCK128_BF16S}) {
            for (const int tokens : {1, 4, 128}) {
                failures += run_projection(ec, type, tokens);
                failures += run_attention(ec, type, tokens);
            }
        }
        std::cout << (failures ? "FAIL" : "OK") << " four-rank GDN/attention projection\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
