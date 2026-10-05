# T-002 — Build environment (CUDA 12.8 without P100 hardware)

**Status:** done
**Depends on:** —

## Goal

A reproducible way to build NInfer for `sm_60`/`sm_70` on a host without GPUs or NVIDIA's apt
repository (the cloud agent container), plus the facts the real P100 host needs.

## Decisions

- **CUDA 12.8** (pin shared with V100X2). CUDA 13 removed Pascal/Volta offline compilation.
  12.9 would also work; nothing in the tree needs it.
- Agent container: NVIDIA's apt/download hosts are blocked by the egress proxy (HTTP 403);
  PyPI's `nvidia-cuda-nvcc-cu12` wheels ship only `ptxas` (no `nvcc` driver, `cicc`,
  `fatbinary`). conda-forge is reachable and ships the full 12.8 compiler →
  `scripts/p100/setup_cuda_toolchain.sh`.
- Host compiler: system GCC 13.3 (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++`).
- Extra package found missing: `cuda-nvml-dev` (`allreduce.cu` includes `nvml.h`; NVML itself
  is `dlopen`ed at runtime, not linked).

## P100 host requirements (for T-008)

- NVIDIA driver R525–R580 (R580 is the last branch with Pascal support), CUDA 12.8 toolkit
  (`cuda-toolkit-12-8` from NVIDIA's repo, or this script).
- Ubuntu 24.04 packages: `cmake ninja-build pkg-config libavcodec-dev libavformat-dev
  libavutil-dev libswscale-dev libcurl4-openssl-dev`.
- Dockerfile currently uses `nvidia/cuda:13.1.2` images → cannot target Pascal. A P100 image
  must use `nvidia/cuda:12.8.1-devel-ubuntu24.04` / `-runtime-` (tracked in T-008).

## Log

- 2026-10-05: apt `developer.download.nvidia.com` → 403 via proxy. PyPI nvcc wheel lacks driver.
- 2026-10-05: micromamba 1.5.10 from conda-forge → `/opt/cuda-12.8`, nvcc V12.8.93.
- 2026-10-05: `nvcc -arch=sm_60` smoke test OK (bf16x2, half2 FMA, fp8 conversions).
- 2026-10-05: sm_70 baseline configure OK (CUTLASS v4.4.2 FetchContent works through proxy).
  Baseline build compiled ~100 of 601 TUs cleanly before being stopped to free CPU for sm_60;
  first failure was the missing `nvml.h`, fixed by `cuda-nvml-dev`. A full sm_70 regression
  build is part of T-004's acceptance instead.

## Results

Toolchain: `/opt/cuda-12.8` (conda-forge CUDA 12.8.93) + GCC 13.3 + CMake 3.28 + Ninja.
Build concurrency `-j3` on the 4-core container (AGENTS.md 85 % CPU ceiling).
