# T-006 — GP100 shared-memory, ISA and warp-synchronization audit

**Status:** static audits done (resources, trap reachability, warp-sync review); sanitizer runs pending on the P100 host (T-008 step 3)
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

## Results (2026-10-05, build of record `acb5e75`)

`scripts/p100/audit_resources.py build-p100/src/libninfer_ops.a`: 7,764 kernels; 0 over 48 KiB
static smem; 0 spills; 916 with >128 registers. `--traps`: 1,199 kernels contain `BPT.TRAP`.
New Pascal kernels: flash prefill 155 regs / 32 KiB smem (1 CTA/SM — tuning item); GGML_K SIMT tile
48–56 regs / 26–35 KiB smem.

## Warp-synchronization review (2026-10-05)

Heuristic: in kernels outside Volta/Ampere-only files, find a thread- or lane-dependent early
`return`/`continue` followed by a full-mask warp collective (`__shfl*_sync`, `warp_sum/max`,
`__syncwarp`). Ten candidates; all are safe:

| Site | Exit condition | Verdict |
|---|---|---|
| `l2norm.cuh:24,63`, `rmsnorm.cuh:42,94`, `layer_norm.cuh:112` | `row = block*warps + warp` | warp-uniform |
| `ggml_k.cu` GEMV | `row = blockIdx.x*4 + threadIdx.x/32` | warp-uniform |
| `q5_rowsplit_gemm_simt.cuh:176,439`, `q6_…:216`, `w8_…:186` | row from block/warp index | warp-uniform |
| `gqa_attention_prefill_bf16.cuh:34` | per-thread `idx < tokens*KVHeads*32` | multiple of 32 → whole warps; Ampere prompt route only |

New Pascal flash kernel: every shuffle sits outside lane-dependent branches (masking is applied to
the reduced score, not around the reduction). Remaining evidence: `compute-sanitizer --tool
synccheck` on the P100 host (T-008 step 3).

## W-16: trap-bearing kernels vs Pascal routes (2026-10-05)

`audit_resources.py --traps` grouped by kernel family (1,199 instantiations). Every family is an
Ampere `mma.m16n8k16`/`ldmatrix` kernel compiled as a trap stub below SM80. Reachability on the
SM60-admitted identities (NVFP4 artifact: NVFP4, row-FP8, W8 MTP, Q4G64 draft head, BF16 GDN
control; GGUF Q4_K_M):

| Trapping family (count) | Used by | Pascal route instead | Verdict |
|---|---|---|---|
| `w8_small_t_mma` (563), `w8_rowsplit_gemm_mma` (120), `w8_rowsplit_medium_t_splitk` (42), `w8_pair_gemm_mma` (10), W8 GDN/SwiGLU split-K | W8 (MTP layer) | `launch_w8_small_t` → SIMT `r8_c8` under `NINFER_PRE_AMPERE_BUILD`; W8 attn-input/linear-add/pair/SwiGLU plans all have pre-Ampere SIMT tables | not reachable |
| `fp8_a16_mma` (96) | FP8 vocabulary head | only launch site `fp8_dispatch.cpp:142` is guarded by `!kVoltaBuild` (true for all pre-Ampere) → SIMT chunks | not reachable |
| `q4_small_t_mma` (26 + 31 in SwiGLU GEMV), `q4_rowsplit_gemm_mma` (27), `q4_linear_swiglu_mma_*` | Q4G64 draft head; groupwise-int | draft head small-T redirects to SIMT `r8_c8` on pre-Ampere; groupwise-int not admitted | not reachable |
| `bf16_gdn_gating_proj_gemm_mma` (28) | BF16 GDN control | pre-Ampere route table uses SIMT `GemvPairedRows`/`SmallTSplit10` | not reachable |
| `bf16_gemm_mma` (12) | BF16 linear/attn-input/linear-add (DFlash, 35B) | not admitted on SM60 | not reachable |
| `q5_/q6_rowsplit_gemm_mma`, `rowsplit_grouped_mma` | groupwise-int identities | not admitted | not reachable |
| `gqa_attention_prefill_{bf16,i8}` (24) | Ampere `Prompt` route | pre-Ampere resolves to SmallT/ChunkedSmallT/PascalFlash | not reachable |
| GDN `chunked::*` (6) | Ampere chunked GDN | pre-Ampere uses the recurrent FP32 kernels | not reachable |
| `vision_attention_flash` (3) | Vision | Vision rejected on SM60 | not reachable |

Conclusion: no trap-stub kernel is on an admitted Pascal route by static routing analysis. The
P100 operator tests (`p100_op_tests`) are the runtime confirmation.
