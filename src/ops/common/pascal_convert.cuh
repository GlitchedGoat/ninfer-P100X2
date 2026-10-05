#pragma once

// Exact conversions built from integer bit operations (T-010 O-8/O-14/O-4).
//
// GP100 runs type conversions (I2F, F2F, and the half-precision steps CUDA's software FP8/FP4
// decoders use below SM89/SM100) at a quarter of its FP32 rate. Every value produced here is
// exactly representable in FP32, so constructing the IEEE bits directly is bit-identical to the
// conversion it replaces while using only integer/FP32 ALU operations.
//
// Selected by the CMake option NINFER_PASCAL_FAST_CONVERT (Pascal builds only, default ON) so the
// two forms can be A/B measured on hardware; tests/ops/test_pascal_convert.cpp checks exactness
// exhaustively against CUDA's own conversions on the host.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops::detail::pascal_convert {

__host__ __device__ __forceinline__ float from_bits(std::uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// Exact for 0 <= value < 2^23: place the integer in the mantissa of 2^23 and subtract 2^23.
__host__ __device__ __forceinline__ float small_uint_to_float(std::uint32_t value) {
    return from_bits(0x4B000000u | value) - 8388608.0f;
}

// Exact for -2^22 <= value < 2^22.
__host__ __device__ __forceinline__ float small_int_to_float(std::int32_t value) {
    return from_bits(0x4B000000u | static_cast<std::uint32_t>(value + 0x400000)) - 12582912.0f;
}

// OCP FP8 E4M3FN (bias 7, no infinities, S.1111.111 is NaN) -> FP32.
__host__ __device__ __forceinline__ float e4m3fn_to_float(std::uint32_t byte) {
    const std::uint32_t sign     = (byte & 0x80u) << 24;
    const std::uint32_t exponent = (byte >> 3) & 0xFu;
    const std::uint32_t mantissa = byte & 0x7u;
    if (exponent == 0xFu && mantissa == 0x7u) { return from_bits(sign | 0x7FC00000u); }
    if (exponent == 0u) {
        // Subnormal: mantissa * 2^-9 (exact product of small integer and power of two).
        const float magnitude = small_uint_to_float(mantissa) * 0.001953125f;
        return sign != 0u ? -magnitude : magnitude;
    }
    return from_bits(sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

// OCP FP4 E2M1 nibble (bias 1, values {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign) -> FP32.
__host__ __device__ __forceinline__ float e2m1_to_float(std::uint32_t nibble) {
    const std::uint32_t sign     = (nibble & 0x8u) << 28;
    const std::uint32_t exponent = (nibble >> 1) & 0x3u;
    const std::uint32_t mantissa = nibble & 0x1u;
    const std::uint32_t magnitude =
        exponent != 0u ? ((exponent + 126u) << 23) | (mantissa << 22)
                       : (mantissa != 0u ? 0x3F000000u : 0u);
    return from_bits(sign | magnitude);
}

} // namespace ninfer::ops::detail::pascal_convert
