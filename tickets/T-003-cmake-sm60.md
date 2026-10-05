# T-003 — Admit `sm_60` in CMake; Pascal build flags

**Status:** done
**Depends on:** T-002

## Change

- `CMAKE_CUDA_ARCHITECTURES` accepts `60|70|86|89` (still exactly one per build).
- `60` and `70` both require CUDA 12.x and define `NINFER_PRE_AMPERE_BUILD`; `70` adds
  `NINFER_VOLTA_BUILD`, `60` adds `NINFER_PASCAL_BUILD`. Both fetch CUTLASS v4.4.2 and pass
  `-Wno-deprecated-gpu-targets`.
- `src/CMakeLists.txt`: `60` gets the same W4A4/A8 source exclusion + sm86 stubs as `70/86/89`
  (Blackwell-only instructions), but **not** `NINFER_SM8X_COMPAT`. Pascal-only sources:
  `ggml_k_cutlass_simt.cu`, `swiglu_fp32.cu`, `volta_tc_sm60_stubs.cpp`. Volta-only Tensor-Core
  sources stay in the `70` list.

## Flag semantics (contract for all later tickets)

| Macro | sm_60 | sm_70 | Meaning |
|---|---|---|---|
| `NINFER_PRE_AMPERE_BUILD` | ✓ | ✓ | no `ldmatrix`/`cp.async`/BF16 MMA → pre-Ampere SIMT routes, tuning |
| `NINFER_VOLTA_BUILD` | | ✓ | SM70 Tensor-Core routes (`m8n8k4`, WMMA, CUTLASS `Sm70` TensorOp, QPN prepack) |
| `NINFER_PASCAL_BUILD` | ✓ | | GP100 SIMT replacements, Pascal admission limits |

For sm_70 the rename is a no-op: both macros are defined, so every former
`NINFER_VOLTA_BUILD` site behaves as before.

## Log

- 2026-10-05: `cmake -DCMAKE_CUDA_ARCHITECTURES=60` configures (CUDA 12.8.93, GCC 13.3).
