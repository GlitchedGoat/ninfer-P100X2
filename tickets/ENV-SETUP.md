# Build environment setup (cloud agent container, no GPU)

What a fresh Claude Code cloud container needed to build NInfer for `sm_60`, in order. Everything
here was hit in practice on 2026-10-05 (Ubuntu 24.04, 4 cores, 15 GB RAM, no GPU).

## 1. CUDA 12.8 toolkit — use conda-forge

| Source | Result |
|---|---|
| `developer.download.nvidia.com` (NVIDIA apt repo) | **blocked** by the egress proxy (HTTP 403) |
| Ubuntu `nvidia-cuda-toolkit` | CUDA 12.0 — too old (CMake requires ≥ 12.8) |
| PyPI `nvidia-cuda-nvcc-cu12` | ships `ptxas` only — no `nvcc` driver, `cicc`, `fatbinary` |
| **conda-forge** (`conda.anaconda.org`) | **works**: full 12.8 compiler + headers |

```bash
scripts/p100/setup_cuda_toolchain.sh /opt/cuda-12.8
```

The script downloads micromamba 1.5.10 from conda-forge and installs `cuda-nvcc`, `cuda-cudart-dev`,
`cuda-nvtx-dev`, `cuda-driver-dev`, **`cuda-nvml-dev`** (`allreduce.cu` includes `nvml.h`),
`cuda-cccl` and `cuda-cuobjdump` (for `scripts/p100/audit_resources.py`), all pinned to 12.8.
Do **not** use CUDA 13: it removed Pascal/Volta offline compilation.

## 2. Host packages (apt works through the proxy)

```bash
apt-get install -y --no-install-recommends libavcodec-dev libavformat-dev libavutil-dev \
  libswscale-dev libcurl4-openssl-dev pkg-config     # cmake 3.28, ninja, gcc 13 preinstalled
```

## 3. Configure and build

```bash
export PATH=/opt/cuda-12.8/bin:$PATH CUDAToolkit_ROOT=/opt/cuda-12.8
cmake -S . -B build-p100 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DBUILD_TESTING=ON
nice cmake --build build-p100 -j3 --target ninfer -- -k 0 > build-p100.log 2>&1
```

- Use the **system** g++ 13 as host compiler (the conda env also contains a gcc; don't mix them).
- CUTLASS v4.4.2 is fetched by CMake `FetchContent` from GitHub; git through the proxy works.
- `-j3` on 4 cores respects the owner's ~85 % CPU ceiling. `ninfer` from scratch (~300 steps)
  took ~15–20 min here; a few CUTLASS/attention TUs take several minutes each.
- `-- -k 0` keeps compiling past failures so one pass lists every broken TU.

## 4. Pitfalls

- **Stopping a build:** `pkill ninja`/`pkill cmake` leaves orphaned `nvcc`/`cicc`/`ptxas` children
  running (they kept the load at 9 on 4 cores). Also kill them:
  `pkill -f 'cicc|ptxas|nvcc'` (only when no other build is meant to run).
- **No edits to `src/`, `tests/`, `apps/`, `bench/` or any CMakeLists while a build runs.** Ninja
  fixes its work list at the start of a pass: files edited after they compiled leave stale objects
  in that pass, and CMake changes are not seen until the next pass. A pass that saw edits is
  *diagnostic only* (use it to collect errors). Claims such as "builds for sm_60" require a
  **build of record**: a pass started after the last source edit with no edits during it.
  Editing `tickets/` or other docs during a build is fine.
- **Disk:** the container's writable allowance is ~39 GB. A full sm_60 build with tests is ~27 GB,
  of which 26 GB is the 133 statically linked test executables (~280 MB each). Build only the
  targets you need, and delete linked test binaries (`find build-p100/tests -maxdepth 1 -type f
  -name 'ninfer_*' -executable -delete`) before starting a second build tree. "No space left on
  device" at link time is this, not a code error.
- Long foreground `sleep` is blocked in the agent harness; run the build with
  `run_in_background` and use a Monitor on `^FAILED|ninja: build stopped|Linking` in the log.
- **Pushing:** `git push` returned 403 until the owner installed the Claude GitHub App on the
  repository (C-0, resolved). If it recurs, commits stay local until access is fixed.
- Docs at `docs.nvidia.com` are blocked for WebFetch; WebSearch summaries work.

## 5. Useful checks without a GPU

```bash
cuobjdump --list-elf build-p100/apps/ninfer | head            # sm_60 cubins present
python3 scripts/p100/audit_resources.py <object-or-binary>    # smem > 48 KiB, spills, regs
nvcc -arch=sm_60 -std=c++20 smoke.cu                           # quick intrinsic availability test
```

## 6. On the P100 host

The cloud container steps above do not all apply on the owner's machine.
- **Toolchain and packages:** use an installed CUDA 12.x (≥ 12.8) if present; otherwise use the same script. Install the §2 packages with the host's package manager.
- **Driver:** must be R525–R580 (R580 is the last branch supporting Pascal).
- **Jobs:** use ~80 % of `nproc`.
- **Disk:** apps plus `p100_op_tests` and `p100_model_tests` need ~12 GB; all 133 tests need ~26 GB.

The runbook is `T-008-p100-validation-plan.md`, started from the prompt in `T-013-host-agent-handoff.md`.
