# T-006 — GP100 shared-memory, ISA and warp-synchronization audit

**Status:** todo (partially covered while porting)
**Depends on:** T-004

## Why

Three GP100 limits do not show up as compile errors everywhere:

1. **48 KiB per-block shared memory.** Static overflows fail in `ptxas`; *dynamic* requests
   (`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, >48 KiB)`) fail only at runtime.
2. **No independent thread scheduling.** Volta code may rely on per-thread progress inside a
   warp (spin locks, `__shfl_sync`/`__syncwarp` in divergent branches). On Pascal these hang
   or return garbage.
3. **No `__dp4a`, no scalar BF16 `atomicAdd`, no `__nanosleep`** — compile errors, caught by
   the build.

## Findings so far

| Site | Dynamic smem | Reached on Pascal? |
|---|---|---|
| `gqa_attention_decode.cu` i8 tiled, `TokenTile==6, KeyBlock=64` | 64 KiB | no — pre-Ampere chunk is 5 tokens |
| `gqa_attention_prefill.cu` Prompt route | >48 KiB | no — Prompt route is Ampere-only |
| GDN chunked (`prepare_wy_wu`, `state_passing`, `output`) | >48 KiB, MMA | no — pre-Ampere uses recurrent FP32 |
| VoltaFlash / Vision flash | opt-in | no — Volta-only / Vision rejected |
| New Pascal `ggml_k` tile | 33 KiB static | yes, within limit |
| CUTLASS SIMT SGEMM 128×128×8, 2 stages | ~16 KiB | yes |

## To do

- Script: list every kernel's static smem/registers from the sm_60 binary
  (`cuobjdump --dump-resource-usage build-p100/apps/ninfer`) and flag >48 KiB or spills.
- Grep audit of `__shfl_*_sync`/`__syncwarp`/`__ballot_sync` inside lane-dependent branches in
  kernels reachable on the Q4_K_M route (attention SIMT decode, GDN recurrent, sampling, argmax,
  allreduce, MTP round).
- On the P100 host: `compute-sanitizer --tool synccheck` and `--tool racecheck` on
  `ninfer_gqa_attention_test`, `ninfer_ggml_k_test`, `ninfer_gdn_input_proj_test` (T-008 step 3).
