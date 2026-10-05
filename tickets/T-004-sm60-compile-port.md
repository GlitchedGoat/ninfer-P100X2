# T-004 — Make every translation unit compile and link for `sm_60`

**Status:** all sm_60 targets build (build of record `4315bc4`); sm_70 regression running
**Depends on:** T-003

## Approach

1. **Classify** every `NINFER_VOLTA_BUILD` site. Mechanical first pass: rename to
   `NINFER_PRE_AMPERE_BUILD` in all shared files (82 files); keep `NINFER_VOLTA_BUILD` in the 14
   Volta-only TUs/headers. Then restore `NINFER_VOLTA_BUILD` where the site is a Tensor-Core
   route (decode launcher TC kernels, VoltaFlash attention, QPN prepack, QUASAR/TP4 gate).
2. **Make misuse loud at compile time:** every Volta Tensor-Core header
   (`volta_mma.cuh`, `*_tc_volta.cuh`, `*_prefill_volta.cuh`, `*_volta_{qpn,mma}_gemm.cuh`,
   `*_volta_qpn.cuh`, `*_qpn_split.cuh`) now `#error`s under `NINFER_PASCAL_BUILD`. Their
   device bodies are guarded by `__CUDA_ARCH__ == 700` and would otherwise compile to **empty
   kernels on sm_60 — silent garbage**.
3. **Unreachable TC entry points** of non-Q4_K_M formats (Q4G64/Q5/W8/FP8/NVFP4/BF16/Vision)
   are defined in `src/ops/volta_tc_sm60_stubs.cpp` (same pattern as `nvfp4_sm86_stubs.cpp`):
   `*_supported` → `false`, launches throw `std::logic_error`. Pascal admission (T-007)
   rejects those identities, so the stubs are unreachable from the Engine.

## Acceptance

- `cmake --build build-p100 --target ninfer ninfer-serve` succeeds.
- Test targets build for sm_60 (they will run on the P100 host, T-008).
- `build-v100` (sm_70) still builds after the rename (regression check).

## Log

- 2026-10-05: rename pass (82 files) + `#error` guards (14 headers).
- 2026-10-05: Volta-only restored in `gqa_attention_decode.cu` (TC i8/bf16 small-T kernels),
  `gqa_attention.cpp` (VoltaFlash route/workspace/launch), `bindings.cpp` (QPN prepack,
  QUASAR/TP4 SM70 gate).
- 2026-10-05: first sm_60 build pass started (`-k 0`, `-j3`); see Results when finished.

## Results

- **Build of record, source `acb5e75`:** `ninfer` and `ninfer-serve` compile and link for sm_60
  (CUDA 12.8.93, GCC 13.3). `cuobjdump --list-elf` shows `sm_60` cubins only. `ninfer --help` runs
  (host side; container has no driver, checked with the toolkit's stub `libcuda.so`).
- Diagnostic passes found: WMMA in `ggml_k.cu` (fixed: Pascal SIMT tile), Volta TC header include in
  `fp8_block.cu` (fixed: Volta-only), and one missing link symbol `q4_volta_qpn_supported` (stubbed
  `false`). The `#error` guards caught every Tensor-Core include at compile time as designed.
- Static audit of `libninfer_ops.a` (7,764 sm_60 kernels): **0** kernels over 48 KiB static smem,
  **0** with local-memory spills; 916 use >128 registers (occupancy, not correctness).
  1,199 kernels contain `BPT.TRAP` (Ampere-only bodies compiled as trap stubs); reachability from
  Pascal routes is argued per route in T-005/T-012 and must be confirmed by T-008 step 3.
- **All targets, build of record `4315bc4`:** apps + all 133 test executables link for sm_60. The only
  test failure was inherited, not Pascal-related: `tests/targets/qwen3_6_27b/test_load_plan.cpp`
  passed `nullptr` to `create_program`'s TP2 `std::span` peer-model parameter (broken since
  `d81c49b`); fixed by passing `{}` as production does.
- Test executables (~280 MB each, 26 GB total) were deleted afterwards to free disk for the sm_70
  regression build; they relink in minutes from the retained objects.
