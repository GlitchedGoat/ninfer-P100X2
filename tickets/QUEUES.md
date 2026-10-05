# Queues

## Conversation queue (needs the owner)

| ID | Question | Agent recommendation | Status |
|---|---|---|---|
| C-1 | Which identities/features must the P100 build admit first? | Only `qwen3.8-27b/gguf-q4-k-m`, TP1+TP2, Text/None/MTP. Reject NVFP4/FP8/QUASAR/DFlash/Vision/TP4/35B/RAM-KV until a later ticket ports them. | open |
| C-2 | P100 host details: motherboard/PCIe topology (`nvidia-smi topo -m`), IOMMU mode, driver version, host RAM, OS. | Needed for T-007/T-008. Same root complex + `iommu=pt` gives direct P2P. | open |
| C-3 | Link/branch of your llama.cpp P100 FP16 work. | Use it as reference for T-009 (FP16x2 with FP32 flush). | open |
| C-4 | Acceptance target for P100X2 (tok/s at what occupied context, vs which baseline)? | Measure your llama.cpp P100 numbers at 3k/32k/85k occupied tokens with the same Q4_K_M + Q8 KV + 180k capacity, then aim above them. | open |
| C-5 | Keep `sm_70`/`sm_86`/`sm_89` builds in this fork, or make it P100-only? | Keep them; Pascal is an added single-arch build, existing V100 routes untouched. | open |
| C-6 | Is 180000-token capacity still the goal on 2×16 GB with FP32 routes (more workspace)? | Keep 180k as target; fall back to 131072 if FP32 prefill workspace does not fit — decided by measurement. | open |

## Work queue (executable now, in order)

| ID | Item | Ticket | Status |
|---|---|---|---|
| W-1 | CUDA 12.8 toolchain in container; sm_70 baseline build of `ninfer` | T-002 | in progress |
| W-2 | Admit `CMAKE_CUDA_ARCHITECTURES=60`, add Pascal flags | T-003 | todo |
| W-3 | Build `ninfer` for sm_60; record failing TUs; fix compile errors | T-004 | todo |
| W-4 | Classify every `NINFER_VOLTA_BUILD` site (pre-Ampere vs Tensor Core) | T-004/T-005 | todo |
| W-5 | Pascal SIMT routes: ggml_k T=5–127, ggml_k prefill SIMT SGEMM, attention route selection | T-005 | todo |
| W-6 | Static audit: dynamic smem >48 KB, scalar bf16 atomics, warp-sync assumptions | T-006 | todo |
| W-7 | Runtime admission for sm 60; allreduce direct-sum gate | T-007 | todo |
| W-8 | Build tests for sm_60 (operator oracles) so they can run on the P100 host | T-008 | todo |
| W-9 | Write P100 validation runbook | T-008 | todo |
