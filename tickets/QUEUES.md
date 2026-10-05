# Queues

## Conversation queue (needs the owner)

| ID | Question | Agent recommendation | Status |
|---|---|---|---|
| C-0 | **Push blocked:** the Claude GitHub App has no write access to `GlitchedGoat/ninfer-P100X2` (HTTP 403). | Reconnect GitHub at https://claude.ai/connect-github or install the Claude app on the repo; commits are local until then. | resolved (2026-10-05, app installed; push works) |
| C-1 | Which identities/features must the P100 build admit first? | **Answered 2026-10-05: NVFP4 `.ninfer` (`qwen3.8-27b/nvfp4`) is the main quant.** Admitted on SM60: `qwen3.8-27b/nvfp4` + `gguf-q4-k-m`, TP1/TP2, Text/None/MTP. | resolved |
| C-2 | P100 host details: `nvidia-smi topo -m`, IOMMU mode, driver version, host RAM, OS. | Needed for T-007/T-008. Same root complex + `iommu=pt` gives direct P2P. | deferred to host agent (T-013); info lives on the P100 machine |
| C-3 | Owner's llama.cpp P100 branch | **Answered 2026-10-05:** lives on the P100 machine. It uses FP16 accumulation arranged to limit FP16 addition error (owner recalls sorting and/or adding in batches; unconfirmed). Host agent locates it, summarizes the technique into T-009 and measures it as the C-4 baseline (T-013 step 9a). | resolved (details via host agent) |
| C-4 | Acceptance target? | **Answered 2026-10-05:** committed **decode** tok/s above the owner's llama.cpp P100 setup at the same occupied context (3k/32k/85k; slightly reduced capacity OK). Q4_K_M vs Q4_K_M like for like; NVFP4 `.ninfer` must beat llama.cpp Q4_K_M. Prefill above llama.cpp desired, secondary ("faster on all fronts" ideal). Recorded in AGENTS.md "Performance work". | resolved |
| C-5 | Keep `sm_70`/`sm_86`/`sm_89` builds in this fork, or make it P100-only? | **Answered 2026-10-05:** keep them while upkeep stays minor (eases upstream merges). Volta upkeep inventory and removal plan in backlog ticket T-015. | resolved |
| C-6 | Is 180000-token capacity still the goal on 2×16 GB with FP32 routes (more workspace)? | **Answered 2026-10-05 (with C-4):** 180000 is the target; a slightly reduced capacity is acceptable if it does not fit. The host agent measures the largest loading capacity (T-013 step 8); it must still cover the 85k occupancy point. | resolved (number via host agent) |
| C-7 | Flash attention: owner asked for it (VRAM). | Implemented a native direct-paged FP32 kernel (zero workspace) instead of vendoring llama.cpp's tile kernel, which would need the 368 MB FP16 staging (T-011). | resolved |
| C-8 | Which NVFP4 artifact? | **Answered 2026-10-05:** the official upstream v3 container `neroued/Qwen3.8-27B-nvfp4-NInfer` → `qwen3_8_27b_nvfp4.ninfer` (23.72 GB, identity `qwen3.8-27b/nvfp4`). Its DFlash2 package and Vision objects are not loaded on Pascal. Still needed: its path on the P100 host. | resolved (path pending) |

## Work queue (executable now, in order)

| ID | Item | Ticket | Status |
|---|---|---|---|
| W-1 | CUDA 12.8 toolchain in container | T-002 | done |
| W-2 | Admit `CMAKE_CUDA_ARCHITECTURES=60`, add Pascal flags | T-003 | done |
| W-3 | Build `ninfer` for sm_60; fix compile/link errors | T-004 | done (`acb5e75`) |
| W-4 | Classify every `NINFER_VOLTA_BUILD` site (pre-Ampere vs Tensor Core) | T-004 | done (first pass) |
| W-5 | Pascal GGML_K routes (T=5–127 tile, prefill SIMT SGEMM) | T-005 | done (builds; numerics on host) |
| W-6 | sm_70 regression build after the rename | T-004 | done (`4315bc4`) |
| W-7 | Build all tests for sm_60 | T-004/T-008 | done (`4315bc4`, 133 tests) |
| W-8 | Static audit: resource usage dump, warp-sync patterns | T-006 | done (sanitizer runs on host) |
| W-9 | Pascal flash prefill attention | T-011 | implemented, builds (155 regs, 32 KiB smem) |
| W-11 | NVFP4 identity on Pascal: dual-arch dense prefill GEMMs (FP8/NVFP4), admission | T-012 | done (builds; numerics on host) |
| W-14 | O-12: chunk FP32 dense-prefill workspace for NVFP4/FP8 (capacity) | T-010 | done (`9ac42c7`; sm_60 + sm_70 builds of record pass) |
| W-15 | O-16: fused A16 residual add for NVFP4/FP8 on Pascal | T-010 | done (`9ac42c7`; sm_60 + sm_70 builds of record pass) |
| W-17 | O-8/O-14: avoid slow conversions behind CMake `NINFER_PASCAL_FAST_CONVERT` (default ON, exact) for host A/B | T-010 | done (`9ac42c7`; exactness test passes; sm_60 + sm_70 builds pass) |
| W-16 | Review SIMT small-T kernels now on Pascal routes vs trap audit | T-006/T-012 | done (no trap kernel reachable) |
| W-10 | Dockerfile variant on CUDA 12.8 for Pascal | T-008 | done (parameterized `CUDA_VERSION`/`CUDA_ARCH`) |
| W-12 | Umbrella targets `p100_op_tests` (40) / `p100_model_tests` (5): ~12 GB instead of 26 GB | T-008 | done |
| W-13 | Opt-in `NINFER_SHARED_LIBS` for dev/test builds (~1 GB of tests instead of 26 GB) | T-014 | deferred (later improvement) |
| W-18 | Remove Volta (V100) support — only if needed | T-015 | backlog |
