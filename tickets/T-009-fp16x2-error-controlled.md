# T-009 — FP16x2 arithmetic with controlled error (after the FP32 baseline)

**Status:** blocked on the host agent's summary of the owner's llama.cpp P100 branch (T-013 step 9a) and the T-008 FP32 baseline
**Depends on:** T-005, T-008

## Motivation

GP100 executes `HFMA2` at 2× the FP32 FMA rate (19 vs 9.5 TFLOPS). Decode at 1 token is
bandwidth-bound, but MTP verification (4 tokens) and prefill are ALU-bound in FP32 (T-001 §4).

## Numerical design (proposed)

1. **Blocked accumulation, FP32 outer sum.** Accumulate one Q4_K/Q6_K 32-element sub-block in
   `half2`, then add the sub-block result into an FP32 accumulator. Rounding error is bounded by
   the 32-term run instead of K = 5120…17408 terms. This is the practical equivalent of ordering
   the summation: summation-order effects are dominated by run length, and sorting products by
   magnitude per dot product would cost more than the FP16 speedup gains.
2. **Integer-code factorisation.** `Σ w·x = d·sc·Σ(q·x) − dmin·m·Σx` (Q4_K), `d·sc·Σ(q·x)` (Q6_K).
   The FP16 inner loop multiplies small exact integers (|q| ≤ 15 or ≤ 32) by activations; the
   FP16 scales are applied once per sub-block in FP32. `Σx` per sub-block is computed once per
   token and reused by every row.
3. **Range control.** BF16→FP16 is exact for |x| ∈ [6.1e-5, 65504] but BF16's range is larger.
   Per-token power-of-two pre-scale (exact) so that |Σ q·x| over a 32-run stays below 65504
   with margin (outlier "massive activation" channels make this mandatory). Values below FP16's
   normal range lose precision; the pre-scale also lifts them.
4. **Avoid conversion units.** GP100 type conversions run at 1/4 FP32 rate. Build FP16 code values
   with the `0x6400 | q` magic-number trick (llama.cpp does this) instead of `I2F`.
5. **Oracle and criterion.** Same FP64 oracle as FP32 routes; acceptance by per-row relative
   error distribution at real shapes plus end-to-end greedy-token agreement and perplexity delta
   vs the FP32 route on a fixed corpus (decision on thresholds → conversation queue).

## Candidates (in order of expected gain)

1. GGML_K GEMV for MTP verify (T=2..4).
2. GGML_K wide-T prefill (CUTLASS SIMT HGEMM with FP32 accumulate is *not* faster on GP100 —
   2× only applies to FP16 accumulate — so this needs the blocked scheme in a custom kernel).
3. Attention QK/PV dot products (INT8 KV → FP16 is exact for codes; scales in FP32).

## Owner's llama.cpp P100 branch (C-3, 2026-10-05)

Lives on the P100 machine. Owner's recollection: FP16 accumulation arranged to minimize FP16
addition error, possibly by sorting and/or adding in batches. Unconfirmed until the host agent
reads the diff (T-013 step 9a). Our current reasoning (above): batched/blocked accumulation with
FP32 flush gives the error bound that matters; full sorting is unlikely to pay for itself in a
bandwidth-bound GEMV, but if the branch does sort (or partially orders by exponent), measure its
accuracy/speed trade-off against blocked accumulation before choosing.
