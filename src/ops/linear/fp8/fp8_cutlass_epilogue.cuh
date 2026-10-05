#pragma once

#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace ninfer::ops::detail {

// C is a row-scale vector broadcast over tokens with a zero leading stride. Keep the
// qualified route's BF16 accumulator rounding before multiplying by that scale, then
// round the scaled result to BF16. This removes a full output read/write and launch.
// The owning GEMMs use alpha=1 and a single K slice; C is never an old output tensor.
template <int Count = 8>
class Fp8RowScaledBf16Epilogue
    : public cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, Count, float, float> {
    using Base = cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, Count, float, float>;

public:
    using typename Base::FragmentAccumulator;
    using typename Base::FragmentCompute;
    using typename Base::FragmentOutput;
    using typename Base::FragmentSource;
    using Base::kCount;
    using Base::Base;
    using Base::operator();

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return true; }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(const FragmentAccumulator& accumulator,
                              const FragmentSource& row_scales) const {
        const FragmentOutput rounded = Base::operator()(accumulator);
        cutlass::NumericArrayConverter<float, cutlass::bfloat16_t, kCount> to_float;
        cutlass::NumericArrayConverter<cutlass::bfloat16_t, float, kCount> to_bf16;
        cutlass::multiplies<FragmentCompute> multiply;
        return to_bf16(multiply(to_float(rounded), to_float(row_scales)));
    }
};

} // namespace ninfer::ops::detail
