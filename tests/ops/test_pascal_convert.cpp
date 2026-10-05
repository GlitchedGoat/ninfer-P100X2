// Exhaustive exactness of the Pascal bit-construction conversions (src/ops/common/pascal_convert.cuh)
// against CUDA's own FP8/FP4 conversions. Host-only: runs without a GPU.
#include "ops/common/pascal_convert.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {
using namespace ninfer::ops::detail::pascal_convert;

bool same(float expected, float actual) {
    if (std::isnan(expected) || std::isnan(actual)) { return std::isnan(expected) && std::isnan(actual); }
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::memcpy(&a, &expected, sizeof(a));
    std::memcpy(&b, &actual, sizeof(b));
    return a == b; // bit-exact, including signed zero
}
} // namespace

int main() {
    int failures = 0;
    for (std::uint32_t storage = 0; storage < 65536; ++storage) {
        __nv_fp8x2_e4m3 pair;
        pair.__x             = static_cast<__nv_fp8x2_storage_t>(storage);
        const float2 expected = static_cast<float2>(pair);
        if (!same(expected.x, e4m3fn_to_float(storage & 0xFFu)) ||
            !same(expected.y, e4m3fn_to_float(storage >> 8))) {
            if (failures++ < 8) { std::cerr << "e4m3fn x2 mismatch at 0x" << std::hex << storage << std::dec << '\n'; }
        }
    }
    for (std::uint32_t storage = 0; storage < 256; ++storage) {
        __nv_fp4x2_e2m1 pair;
        pair.__x             = static_cast<__nv_fp4x2_storage_t>(storage);
        const float2 expected = static_cast<float2>(pair);
        if (!same(expected.x, e2m1_to_float(storage & 0xFu)) ||
            !same(expected.y, e2m1_to_float(storage >> 4))) {
            if (failures++ < 8) { std::cerr << "e2m1 x2 mismatch at 0x" << std::hex << storage << std::dec << '\n'; }
        }
    }
    for (std::uint32_t value = 0; value < (1u << 16); ++value) {
        if (!same(static_cast<float>(value), small_uint_to_float(value))) { ++failures; }
    }
    for (std::int32_t value = -(1 << 16); value < (1 << 16); ++value) {
        if (!same(static_cast<float>(value), small_int_to_float(value))) { ++failures; }
    }
    if (failures != 0) { std::cerr << failures << " conversion mismatches\n"; }
    return failures == 0 ? 0 : 1;
}
