#include "ops/linear/linear_test_common.h"
#include <array>
#include <iostream>

int main() {
    using namespace ninfer::test::linear;
    if (!cuda_available()) { return 77; }
    try {
        constexpr std::array calls{Invocation{1}, Invocation{4}, Invocation{33},
                                   Invocation{128}, Invocation{2560}};
        int failures = 0;
        for (const auto shape : {std::array{3584,5120}, std::array{4096,5120},
                                 std::array{8704,5120}, std::array{5120,1536},
                                 std::array{5120,4352}, std::array{256,5120}}) {
            failures += run_shape("FP8_BLOCK128_A16", ActivationCompute::A16,
                make_fp8_block_weight, {shape[0], shape[1], 2909, Comparison::Sampled, true, calls});
        }
        std::cout << (failures ? "FAIL" : "OK") << " native FP8 block128 Linear\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
