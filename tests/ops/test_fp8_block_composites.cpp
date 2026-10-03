#include "ninfer/ops/linear_add.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <array>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {
int residual_case(int k, int tokens) {
    DevicePackedWeight weight(quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_BLOCK128_BF16S,5120,k,3301U+k));
    GuardedBf16Tensor input(k,tokens), output(5120,tokens);
    const auto activation = make_bf16_activation(k,tokens,3313);
    const auto residual = make_bf16_activation(5120,tokens,3319);
    std::vector<std::uint16_t> xb(activation.size()), rb(residual.size());
    std::transform(activation.begin(),activation.end(),xb.begin(),f32_to_bf16);
    std::transform(residual.begin(),residual.end(),rb.begin(),f32_to_bf16);
    input.copy_from_bits(xb);
    output.copy_from_bits(rb);
    const auto bytes = ops::linear_add_workspace_capacity_bytes(weight.view().qtype,5120,k,
        ops::LinearPolicy::A16Only,tokens,tokens);
    DeviceBuffer storage(std::max<std::size_t>(bytes,256));
    WorkspaceArena workspace(DeviceSpan{storage.p,bytes});
    auto out = output.tensor();
    ops::linear_add(input.tensor(),weight.view(),out,ops::LinearPolicy::A16Only,workspace,nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto actual = output.values();
    std::vector<double> got, expected;
    for (const int row : sampled_rows(5120,11)) {
        for (const int token : sampled_rows(tokens,11)) {
            double sum = 0;
            for (int c=0;c<k;++c) {
                sum += quantized_weight::logical_weight_fp64(weight.host,row,c) *
                       static_cast<double>(activation[static_cast<std::size_t>(token)*k+c]);
            }
            const auto at = static_cast<std::size_t>(token)*5120+row;
            got.push_back(actual[at]);
            expected.push_back(sum+residual[at]);
        }
    }
    const std::string label="FP8 block residual K="+std::to_string(k)+" T="+std::to_string(tokens);
    int failures=compare(label,got,expected,{1.0/256,1.0/256,2.0/256});
    failures += output.verify_guards(label);
    failures += weight.verify_preserved(label);
    if(workspace.used()!=0 || workspace.peak_used()>bytes) {++failures;}
    return failures;
}
}

int main() {
    if(cuda_unavailable()) {return 77;}
    try {
        constexpr std::array<std::int32_t,5> tokens{1,4,33,128,2560};
        const linear_swiglu::Profile profile{QType::FP8_E4M3FN_BLOCK128_BF16S,8704,5120,4352,
            3323,linear_swiglu::ActivationCompute::A16};
        int failures=linear_swiglu::run_profile("FP8 block SwiGLU",profile,tokens);
        for(const int k:{1536,4352}) {
            for(const int t:tokens) {failures+=residual_case(k,t);}
        }
        std::cout<<(failures?"FAIL":"OK")<<" FP8 block fused mathematical contracts\n";
        return failures?1:0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
