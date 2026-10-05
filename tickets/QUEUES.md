# Queues

## Conversation queue (needs the owner)

| ID | Question | Agent recommendation | Status |
|---|---|---|---|
| C-0 | **Push blocked:** the Claude GitHub App has no write access to `GlitchedGoat/ninfer-P100X2` (HTTP 403). | Reconnect GitHub at https://claude.ai/connect-github or install the Claude app on the repo; commits are local until then. | open |
| C-1 | Which identities/features must the P100 build admit first? | Only `qwen3.8-27b/gguf-q4-k-m`, TP1+TP2, Text/None/MTP (implemented as the default). Reject NVFP4/FP8/QUASAR/DFlash/Vision/TP4/35B/RAM-KV until a later ticket ports them. | open (default implemented) |
| C-2 | P100 host details: `nvidia-smi topo -m`, IOMMU mode, driver version, host RAM, OS. | Needed for T-007/T-008. Same root complex + `iommu=pt` gives direct P2P. | open |
| C-3 | Link/branch of your llama.cpp P100 FP16 work. | Reference for T-009 (FP16x2 with FP32 flush) and T-011 tile configs. | open |
| C-4 | Acceptance target for P100X2 (tok/s at what occupied context, vs which baseline)? | Measure your llama.cpp P100 numbers at 3k/32k/85k occupied tokens with the same Q4_K_M + Q8 KV + 180k capacity, then aim above them. | open |
| C-5 | Keep `sm_70`/`sm_86`/`sm_89` builds in this fork, or make it P100-only? | Keep them; Pascal is an added single-arch build, V100 routes untouched. | open |
| C-6 | Is 180000-token capacity still the goal on 2×16 GB with FP32 routes (more workspace)? | Keep 180k as target; fall back to 131072 if FP32 prefill workspace does not fit — decided by measurement. | open |
| C-7 | OK to vendor llama.cpp `fattn-tile.cuh` (MIT) for Pascal prefill attention, built FP32-only? | Yes — same provenance pattern as the vendored Volta kernel (T-011). | open |

## Work queue (executable now, in order)

| ID | Item | Ticket | Status |
|---|---|---|---|
| W-1 | CUDA 12.8 toolchain in container | T-002 | done |
| W-2 | Admit `CMAKE_CUDA_ARCHITECTURES=60`, add Pascal flags | T-003 | done |
| W-3 | Build `ninfer` for sm_60; fix compile/link errors | T-004 | in progress |
| W-4 | Classify every `NINFER_VOLTA_BUILD` site (pre-Ampere vs Tensor Core) | T-004 | done (first pass) |
| W-5 | Pascal GGML_K routes (T=5–127 tile, prefill SIMT SGEMM) | T-005 | written, compiling |
| W-6 | sm_70 regression build after the rename | T-004 | todo |
| W-7 | Build all tests for sm_60 | T-004/T-008 | todo |
| W-8 | Static audit: resource usage dump, warp-sync patterns | T-006 | todo |
| W-9 | Pascal flash prefill attention (after C-7) | T-011 | todo |
| W-10 | Dockerfile variant on CUDA 12.8 for Pascal | T-008 | todo |
