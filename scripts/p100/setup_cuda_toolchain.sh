#!/usr/bin/env bash
# Install a CUDA 12.8 toolkit (nvcc, cudart, NVML/NVTX headers, CCCL) from conda-forge into a
# prefix, for build hosts without NVIDIA's apt repository (e.g. cloud agent containers).
# Pascal sm_60 needs CUDA 12.x: CUDA 13 removed offline compilation for Maxwell/Pascal/Volta.
# A real P100 host may instead use NVIDIA's cuda-toolkit-12-8 package with driver R570-R580.
#
# Usage: scripts/p100/setup_cuda_toolchain.sh [prefix]   (default /opt/cuda-12.8)
set -euo pipefail
PREFIX=${1:-/opt/cuda-12.8}
WORK=${TMPDIR:-/tmp}/ninfer-micromamba
mkdir -p "$WORK"
if [[ ! -x "$WORK/bin/micromamba" ]]; then
  curl -sSL -o "$WORK/mm.tar.bz2" \
    https://conda.anaconda.org/conda-forge/linux-64/micromamba-1.5.10-0.tar.bz2
  tar -C "$WORK" -xjf "$WORK/mm.tar.bz2" bin/micromamba
fi
MAMBA_ROOT_PREFIX="$WORK/root" "$WORK/bin/micromamba" create -y -q -p "$PREFIX" -c conda-forge \
  "cuda-version=12.8" "cuda-nvcc=12.8.*" "cuda-cudart-dev=12.8.*" "cuda-nvtx-dev=12.8.*" \
  "cuda-driver-dev=12.8.*" "cuda-nvml-dev=12.8.*" "cuda-cccl=12.8.*" "cuda-cuobjdump=12.8.*"
"$PREFIX/bin/nvcc" --version | tail -2
cat <<MSG
Configure with:
  export PATH=$PREFIX/bin:\$PATH CUDAToolkit_ROOT=$PREFIX
  cmake -S . -B build-p100 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60 \\
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DBUILD_TESTING=ON
Host packages (Ubuntu 24.04): libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
  libcurl4-openssl-dev pkg-config ninja-build cmake
MSG
