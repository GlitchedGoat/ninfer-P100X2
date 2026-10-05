# T-005 — Pascal SIMT routes for the Q4_K_M TP2 Text/MTP path

**Status:** implemented and builds for sm_60 (`4315bc4`, `9ac42c7`); numerics pending on the P100 host (T-008 step 3)
**Depends on:** T-004

## Scope

Every Op the `qwen3.8-27b/gguf-q4-k-m` TP1/TP2 Text/None/MTP route executes must have a correct
GP100 route. Weight matrices in this identity are all `GGML_K` (Q4_K/Q6_K); BF16/FP32 objects are
only norms, convolution taps and biases (checked in `convert_gguf.py::direct`).

## Route table

| Op / site | V100 | P100 (this ticket) | State |
|---|---|---|---|
| GGML_K T=1–4 (`ggml_k.cu::gemv`) | SIMT FP32 | unchanged | ok |
| GGML_K T=5–127 (`ggml_k.cu::gemm`) | WMMA FP16×FP16→FP32 | new SIMT FP32 32×{16,32} tile, decoded rows in smem (33 KiB) | written |
| GGML_K T≥128 with workspace | CUTLASS Sm70 TensorOp, FP16 operands | `ggml_k_cutlass_simt.cu`: FP32 dequant in ≤64 MiB row chunks + CUTLASS SIMT SGEMM | written |
| Shared Q4_K/Q6_K tile decoder | inline in WMMA kernel | `decode_half_block<Elem>` used by both | written |
| Staging kernels | private to sm70 TU | `ggml_k_dequant.cuh`, templated on operand type | written |
| GQA decode/verify INT8 KV | TC kernel for width ≥3 | pre-Ampere SIMT i8 kernel at all widths | routed |
| GQA prefill (wide, B=1) | VoltaFlash TC (+368 MB FP16 staging at 180k) | PascalFlash: FP32 SIMT, paged INT8 in place, 0 workspace (T-011) | implemented |
| GDN recurrent | SIMT FP32 sequential | unchanged | ok |
| TP2 all-reduce | UVA D2D (+ direct sum ≤80 KiB on sm70) | UVA D2D | ok (T-007 measures direct sum) |

## Open work

- Pascal flash prefill attention: done in T-011 (correction: the chunked fallback was slow, not
  "impractical"; see T-011's analysis).
- Tests: route `tests/ops/test_ggml_k.cpp` prefill checks through `ggml_k_prefill_*` on
  pre-Ampere and run the T=5..127 tile against the FP64 oracle on sm_60.

## Log

- 2026-10-05: GGML_K Pascal routes written (`ggml_k.cu`, `ggml_k_cutlass_simt.cu`,
  `ggml_k_dequant.cuh`, `ggml_k_prefill.h` replacing `ggml_k_cutlass_sm70.h`).
- 2026-10-05: attention route on Pascal = SmallT/ChunkedSmallT only.
