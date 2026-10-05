# P100X2 port — agent entry point

Read this file first, then `tickets/QUEUES.md`, then the ticket you pick up.

## Goal

Port this NInfer fork (tuned for 2 × Tesla V100-SXM2 16 GB, `sm_70`) to the owner's
**2 × Tesla P100 PCIe 16 GB** (`sm_60`, Pascal, no Tensor Cores, no NVLink), keeping the
property the owner values: **tensor-parallel TP2 that keeps both GPUs busy on every layer**
(not llama.cpp-style layer split). First numerical target is **FP32 SIMT arithmetic**; FP16x2
(`HFMA2`, 2× FP32 rate on GP100) comes later with explicit error control.

Primary workload: **`qwen3.8-27b/nvfp4`** (`.ninfer`, owner direction 2026-10-05), also `gguf-q4-k-m`; TP2, Text + MTP (≤3 drafts),
INT8 group-64 KV, CUDA Graphs, single request. Other identities are out of first scope
(see conversation queue C-1).

## Ground rules

- `AGENTS.md` at the repo root governs all work (scope, numerics, tests, commits). Its "current
  product contract" still describes V100X2; T-003 onward updates it as the P100 contract lands.
- Develop on branch `claude/ninfer-p100-adaptation-xicv3d` (or the branch the owner names).
- Every unit of work is a ticket `tickets/T-NNN-<slug>.md`. Keep its **Status**, **Log** and
  **Results** current as you work; log real commands and outcomes, not intentions.
- `tickets/QUEUES.md` holds the **conversation queue** (needs the owner) and the **work queue**
  (executable now). Move items between them; never silently drop one.
- The build host has **no P100**. Anything needing hardware goes into
  `T-008` (validation plan) as a concrete step, not a guess presented as a result.

## Build environment (no NVIDIA apt repo needed)

Full procedure and pitfalls: [`ENV-SETUP.md`](ENV-SETUP.md). Session summaries live in
`session-summaries/` at the repo root; write one at the end of each session and commit it.

```bash
scripts/p100/setup_cuda_toolchain.sh /opt/cuda-12.8      # CUDA 12.8 from conda-forge
export PATH=/opt/cuda-12.8/bin:$PATH CUDAToolkit_ROOT=/opt/cuda-12.8
cmake -S . -B build-p100 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DBUILD_TESTING=ON
cmake --build build-p100 -j3 --target ninfer        # keep host CPU < ~85 % (4-core box: -j3)
```

CUDA 12.x is mandatory: CUDA 13 removed offline compilation for Maxwell/Pascal/Volta. The
P100 host needs driver R525+ for 12.x; R580 is the last branch supporting Pascal.

## Ticket index

| Ticket | Title | Status |
|---|---|---|
| T-001 | Research and port plan | done |
| T-002 | Build environment (CUDA 12.8, sm_70 baseline) | see ticket |
| T-003 | Admit `sm_60` in CMake; Pascal build flags | see ticket |
| T-004 | Make every translation unit compile for `sm_60` | see ticket |
| T-005 | Pascal SIMT routes for the Q4_K_M TP2 Text/MTP path | see ticket |
| T-006 | Shared-memory / occupancy / ISA audit for GP100 | see ticket |
| T-007 | Runtime admission and TP2 over PCIe on P100 | see ticket |
| T-008 | P100 hardware validation plan | see ticket |
| T-009 | FP16x2 error-controlled arithmetic (post-FP32) | blocked on C-3 |
| T-010 | P100 performance tuning | blocked on hardware |
| T-011 | Pascal flash-attention prefill (paged INT8, 0 workspace) | implemented |
| T-012 | NVFP4 `.ninfer` identity on Pascal | in progress |
