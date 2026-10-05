# T-010 — P100 performance tuning

**Status:** blocked on hardware (T-008 baseline)
**Depends on:** T-008

## Candidates (prioritise from the T-008 whole-inference profile, not by guess)

- Pascal SIMT flash prefill attention (T-005 W-5a) — expected dominant prefill cost otherwise.
- GGML_K GEMV launch geometry for 56 SMs (currently 4 rows/CTA, 128 threads, Volta-tuned).
- Remove `I2F` from the GEMV inner loop (`static_cast<float>(code)`): magic-number float build.
- Fused dequant + SIMT GEMM for prefill to drop the FP32 workspace pass and its memory traffic.
- Pascal `gemm` tile (5–127 tokens): bank conflicts on the k-major B store; 2-row register
  blocking.
- Small-message all-reduce direct sum on sm 60 (T-007 item 4).
- `kSmallTChunkTokens`, split-KV counts, attention warps per CTA for 56 SMs.
- CUDA Graph capture is architecture-independent; confirm graph replay works on sm 60.

## Method

Per AGENTS.md: whole-inference `nsys` first; kernel-level `ncu` only for a kernel that the
profile shows matters. Respect the ten-minute active / three-minute idle GPU cadence.
