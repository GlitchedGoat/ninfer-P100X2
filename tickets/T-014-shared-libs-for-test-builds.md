# T-014 — Opt-in shared libraries for development/test builds

**Status:** todo (later improvement; owner decision 2026-10-05)
**Depends on:** —

## Problem

Every CUDA library is an archive with `CUDA_SEPARABLE_COMPILATION ON` and
`CUDA_RESOLVE_DEVICE_SYMBOLS ON` (`ninfer_cuda_archive()` in `src/CMakeLists.txt`), so each archive
carries one device-link object referencing all of its kernels (~7,700 on sm_60). Any executable
linking `ninfer_ops`/`ninfer_engine` therefore embeds the whole device image: every test is
~280 MB, 133 tests ≈ 26 GB, even host-only tests (e.g. HTTP error handling). Tests contain no
device code (`tests/` has no `.cu` files).

Interim mitigation (done): `p100_op_tests` / `p100_model_tests` umbrella targets build only the
45 tests relevant to Pascal (≈ 12 GB).

## Proposal

- CMake option `NINFER_SHARED_LIBS` (default **OFF**). When ON, the internal libraries
  (`ninfer_core`, `ninfer_artifact`, `ninfer_ops`, `ninfer_text`, `ninfer_engine`, …) become
  `SHARED` with `POSITION_INDEPENDENT_CODE ON`, keeping `CUDA_RESOLVE_DEVICE_SYMBOLS ON` so each
  `.so` is device-self-contained (already true: no device symbol crosses a library boundary today).
- Tests (and optionally dev apps) link the `.so`s: expected ~1 GB total instead of 26 GB, and much
  faster test links.
- Product binaries (`ninfer`, `ninfer-serve`, Docker image) keep static archives.
- Keep the shared CUDA runtime (`CUDA::cudart`), already the Linux default; static cudart across
  several `.so`s would be unsafe.

## Risks / checks

- First build with the option ON recompiles everything (PIC).
- A test relying on a symbol with hidden visibility would fail to link — none expected with GCC's
  default visibility; the build shows it.
- Verify: full `BUILD_TESTING=ON` build with the option ON, `du -sh build/tests`, run a CPU-only
  test and (on the P100 host) `ninfer_ggml_k_test` against the `.so`.
