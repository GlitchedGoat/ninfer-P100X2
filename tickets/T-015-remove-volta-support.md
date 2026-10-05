# T-015 — Remove Volta (V100) support

**Status:** backlog. Do this only if the owner asks or Volta upkeep starts to block Pascal work (C-5, D-13).
**Depends on:** none

## Why it is not done now (C-5, 2026-10-05)

The owner chose to keep sm_70 (and sm_86/sm_89) for now. The Volta code is close to upstream
`geoffwatts/ninfer-v100` and upstream NInfer. Keeping it unchanged makes upstream merges cheap:
upstream changes land on files we have not restructured. Removing it would make every later merge
in those files a conflict.

## Maintenance cost of keeping Volta (this is the inventory to remove)

Ongoing duties while Volta stays:

- **Regression build.** After any change to a shared pre-Ampere file, build `build-v100`
  (`-DCMAKE_CUDA_ARCHITECTURES=70`, `--target ninfer`).
  - A fresh build takes about 17 min; an incremental one takes minutes.
  - It needs disk alongside `build-p100` (see the ENV-SETUP disk gotcha).
- **Two-way flag discipline.**
  - Each new site must be classified as either `NINFER_PRE_AMPERE_BUILD` (shared by sm_60 and sm_70) or `NINFER_VOLTA_BUILD` (Tensor-Core only).
  - Each Pascal replacement goes under `NINFER_PASCAL_BUILD` with the Volta branch left untouched.
- **Upstream merges.** Re-run the classification on merged files. A new Volta Tensor-Core include in a shared TU will hit the `#error` guard in the sm_60 build. That is intended; route it as T-004 did.

Code that exists only for Volta, or only because both archs share a file:

| Area | Items |
|---|---|
| CMake | The `CMAKE_CUDA_ARCHITECTURES STREQUAL "70"` source list in `src/CMakeLists.txt`: 33 Volta-only TUs (QPN/MMA GEMMs, prepack, `*_cutlass_sm70.cu` for BF16/Q4/Q5/W8/GGML_K, NVFP4 attn/GDN input, `volta_tp4.cu`, both `*_volta_flash.cu`) and the `third_party/llama_cpp_fattn` include and compile options. The `70` branch in the root `CMakeLists.txt` (`NINFER_VOLTA_BUILD`). |
| Vendored code | `third_party/llama_cpp_fattn/` (used only by the Volta flash routes). |
| `#error` guards | 15 Volta Tensor-Core headers (`volta_mma.cuh`, `*_tc_volta.cuh`, `*_prefill_volta.cuh`, `*_volta_{qpn,mma}_gemm.cuh`, `*_volta_qpn.cuh`, `*_qpn_split.cuh`). The headers go away together with their TUs. |
| Pascal stubs | `src/ops/volta_tc_sm60_stubs.cpp`. These are throwing stubs for Volta entry points that shared code references. Once the call sites are gone, the stubs are deleted. |
| Dual-arch configs | `src/ops/common/pre_ampere_gemm.cuh` Volta branch (FP16 TensorOp tiles, `weight_chunk_rows` = n). The files `fp8_cutlass_sm70.cu`, `nvfp4_cutlass_sm70.cu`, `fp8_{attn,gdn}_input_cutlass_sm70.cu` now serve both archs, so rename them to `*_pre_ampere` (or `*_pascal`). |
| `NINFER_VOLTA_BUILD` sites in shared files | `gqa_attention.h` (2), `gqa_attention_decode.cu` (5, Tensor-Core small-T decode), `wrapper/gqa_attention.cpp` (7, VoltaFlash route, workspace, launch), `nvfp4_cutlass_sm70.cu` (3, QPN dequant), `fp8_block.cu` (5), `pre_ampere_gemm.cuh` (1), `qwen3_6_27b/.../bindings.cpp` (8, QPN prepack, QUASAR/TP4 gate). |
| Plan routing | The Volta branches next to the Pascal ones in `nvfp4_linear_swiglu_plan.cpp` (QPN routes), `nvfp4_linear_add_plan.cpp` and `fp8_linear_add_plan.cpp` (always LinearThenAdd), and the GGML_K Volta WMMA tile in `ggml_k.cu`. |
| Volta-only product scope | QUASAR (`qwen3.8-27b/quasar-nvfp4`), TP4, RAM-KV, Volta DFlash/Vision, the V100X2 Q4_K_M acceptance profile. Removing these means dropping identities from the registry and admission, plus their docs (AGENTS.md product contract, `docs/maintainer/paged-kv-cache.md` §15). |
| Tests | `test_nvfp4_input_sm70.cpp`, `linear/test_volta_tp4_a16.cpp`, `linear_swiglu/test_volta_tp4.cpp`, and the `NINFER_VOLTA_BUILD` blocks in `linear_test_common.cpp`, `test_bf16_a16.cpp`, `test_fp8_a16.cpp`, `test_nvfp4_a16.cpp`, `linear_add/test_nvfp4.cpp`, `linear_add/test_q5_a16.cpp`, `linear_swiglu_test_common.cpp`, `test_linear_swiglu_split.cpp`. |
| Docs | README V100X2 sections and commands; AGENTS.md V100 contract, acceptance and resource rows; `Dockerfile` default `CUDA_ARCH`; `build-v100/` references in tickets. |

Not Volta-only (keep): `NINFER_PRE_AMPERE_BUILD` SIMT routes, chunked dense prefill, Pascal flash
attention, and the sm_86/sm_89 builds. Those builds are a separate decision.

## Removal plan (when picked up)

1. Delete the `70` CMake branch and its TUs.
2. Collapse `NINFER_PRE_AMPERE_BUILD` and `NINFER_PASCAL_BUILD` into one Pascal macro, or keep `PRE_AMPERE` and delete `VOLTA`.
3. Delete the `#error` headers, the stubs and the Volta plan branches.
4. Rename the `*_sm70` dual-arch files.
5. Remove the Volta-only identities from admission and docs.
6. Build sm_60 apps plus `p100_op_tests`/`p100_model_tests` as the build of record. Run T-008 step 3 on the host to confirm nothing on Pascal routes changed.

## Log

- 2026-10-05: filed from C-5 (keep both for now; inventory recorded).
