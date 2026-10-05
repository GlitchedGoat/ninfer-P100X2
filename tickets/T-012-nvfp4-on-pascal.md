# T-012 — NVFP4 `.ninfer` (`qwen3.8-27b/nvfp4`) on Pascal

**Status:** implemented and builds for sm_60 (`9ac42c7`); numerics and real-model smoke need the P100 host (T-008 steps 3–4)
**Depends on:** T-004, T-005, T-011

## Goal

Owner direction (2026-10-05): NVFP4 `.ninfer` is the main incoming quant, as on V100X2.
Admit `qwen3.8-27b/nvfp4` on SM60 for TP1/TP2 Text/None/MTP with FP32 SIMT arithmetic.

## Target artifact

The official upstream v3 container `neroued/Qwen3.8-27B-nvfp4-NInfer`, file
`qwen3_8_27b_nvfp4.ninfer` (23.72 GB, `NINFER\x00\x03`), read as `qwen3.8-27b/nvfp4` without
repacking (artifact doc §1.1). Not QUASAR. Its optional DFlash2 package (~2.2 GB) and Vision objects
are not materialized on Pascal (Text/None/MTP only).

## What the identity actually contains (artifact doc §3.1, §9.2)

It is **mixed-format**, not NVFP4 everywhere:

| Role | Format |
|---|---|
| MLP gate/up + down, layers 0–55 | NVFP4 (`blockscale-k16-m128x4-v1`) — 112 parents |
| attention in/out, GDN in/out, MLP 56–63, output head, token embedding | `FP8_E4M3FN_ROW_BF16S` — 146 parents |
| GDN fused A/B control, norms, conv | BF16 |
| MTP matrices | `W8G32_F16S` |
| optimized draft head | `Q4G64_F16S` |

So Pascal needs SIMT routes for NVFP4 **and** row-FP8, W8, Q4G64 and BF16.

## Route analysis on SM60 (every format: decode/small-T + wide-T prefill)

| Format / Op family | V100 | P100 |
|---|---|---|
| NVFP4/FP8 decode & small T | QPN Tensor-Core when `*_volta_qpn_supported`, else SIMT `decode`/`small_t` | `*_supported` → false (stubs) → SIMT `decode`/`small_t` |
| NVFP4/FP8 prefill T ≥ 128 (`nvfp4_cutlass_sm70`, `fp8_cutlass_sm70`, FP8 attn/GDN input CUTLASS) | dequant → FP16 → CUTLASS SM70 TensorOp | **same TUs, now dual-arch** via `ops/common/pre_ampere_gemm.cuh`: dequant → FP32 → CUTLASS SIMT SGEMM |
| FP8 fused row-scale epilogue | vector-8 epilogue | not fused on Pascal (separate scale kernel); epilogue templated on width |
| QPN-prepacked weights | load-time prepack | never prepacked (bindings Volta-only); QPN dequant kernel Volta-only |
| W8 (MTP) | Volta MMA/QPN bands, else SIMT r8_c8 | SIMT `r8_c8` at all T (correct; prefill re-reads weights per 8 tokens → T-010) |
| Q4G64 draft head | QPN/MMA bands, else SIMT | SIMT `r8_c4`/`r8_c8`/GEMV |
| BF16 GDN gating | pre-Ampere SIMT schedules | same |
| Attention | VoltaFlash prefill / TC verify | PascalFlash prefill (T-011) / SIMT verify |

Workspace note: FP32 operands double the dense-prefill workspace relative to Volta (e.g. TP2
MLP gate/up shard 17408×5120: 178 MB FP16 → 356 MB FP32). The capacity planner accounts for it
through the same workspace queries; whether 180k context still fits is C-6 / T-008.

## Remaining risks to check on hardware

- Any SIMT `small_t`/`decode` kernel that was never exercised on Volta (because QPN covered T=2–32)
  must not trap below SM80: run the SASS trap audit (`BPT.TRAP` per kernel, T-006) and the FP8/NVFP4
  operator tests on sm_60.
- E2M1/E4M3 conversions use `cuda_fp4.h`/`cuda_fp8.h` software paths on SM60 (correct, slower).

## Log

- 2026-10-05: admission widened (`bindings.cpp`); `pre_ampere_gemm.cuh` added; `fp8_cutlass_sm70.cu`,
  `nvfp4_cutlass_sm70.cu`, FP8 attention/GDN-input CUTLASS TUs moved to the shared SM60/SM70
  source list; corresponding stubs removed.
- 2026-10-05: **bug fixed** — NVFP4 SwiGLU on pre-Ampere routed every non-QPN width (2..∞) to the
  dense dequant+CUTLASS route; with no QPN on Pascal that meant MTP-verify widths (T=2–4)
  dequantized the full gate/up matrix to FP32 every call. Pascal now uses the fused SIMT
  small-T kernel for T ≤ 16 (decode T=1 already used the fused SIMT decode kernel).
- 2026-10-05: audited FP8 SwiGLU (falls back to chunked SIMT), NVFP4/FP8 residual add
  (`LinearThenAdd` → `linear()` → SIMT/dense): correct on Pascal. Tuning candidate: on Pascal
  NVFP4 weights are never QPN-prepacked, so the fused A16 residual-add kernels could replace
  `LinearThenAdd` (T-010).
