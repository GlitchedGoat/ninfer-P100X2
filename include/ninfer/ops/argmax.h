#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

/**
 * Computes one exact argmax value and local row index per column.
 *
 * `values` is contiguous F32 [T] and `indices` is contiguous I32 [T].  The value is the
 * represented BF16 logit converted to FP32; ties select the lowest local row index.  This is
 * the vocabulary-shard primitive used by tensor-parallel greedy verification, where the
 * local winners are merged after a small cross-device transfer instead of gathering all logits.
 */
void argmax_with_value(const Tensor& logits, Tensor& values, Tensor& indices,
                       std::int32_t valid_rows, cudaStream_t stream);

/**
 * Merges local argmax results gathered as values/indices [R,T], R=2 or 4, into token IDs [T].
 * Rank r's local indices are offset by r * `first_shard_rows`; equal values select the
 * lower global token ID.  Inputs and output are contiguous and non-overlapping.
 */
void merge_argmax_shards(const Tensor& values, const Tensor& indices, Tensor& out,
                         std::int32_t first_shard_rows, cudaStream_t stream);

} // namespace ninfer::ops
