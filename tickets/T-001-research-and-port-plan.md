# T-001 — Research and P100 port plan

**Status:** done (owner answers C-1…C-6 recorded in QUEUES.md)
**Owner:** agent · **Created:** 2026-10-05

## Deliverable

Research of (a) what this fork changed relative to upstream NInfer, (b) how it adapts to V100,
(c) the P100 hardware, and a concrete plan + ticket breakdown for a P100X2 build.

## 1. Fork lineage (from `git log`)

| Stage | Commits | What it added |
|---|---|---|
| Upstream `Neroued/ninfer` (≤ v0.6.1) | before `75d94ea` | RTX 5090 `sm_120a` engine: `.ninfer` artifacts, Qwen3.6 family runtime, NVFP4/FP8/W8/Q4 Ops, MTP, paged INT8 KV, CUDA Graphs, OpenAI/Anthropic serving. Single GPU. |
| RTX 3060 TP2 | `777ac1b`, `70f9a23`, `62f0ab2`, `32eca5f`, `d81c49b` | **Tensor parallelism**: per-GPU weight arenas, row/column-sharded projections with all-reduce, `sm_86` compatibility routes (`NINFER_SM8X_COMPAT`). |
| Volta merge (`geoffwatts/ninfer-v100`) | `6b86bab` (253 files, +21k) | `sm_70` build (`NINFER_VOLTA_BUILD`), GGUF Q4_K_M identity, Volta Tensor-Core kernels (`mma.sync.m8n8k4`, WMMA, CUTLASS `Sm70` TensorOp), SIMT fallbacks for every Ampere-only `mma.m16n8k16`/`ldmatrix`/`cp.async` kernel, UVA-copy TP2 collectives. |
| V100X2 follow-ups | `a7fb634` … `e27de9b` | TP2 MTP prefix reuse, DFlash2, NVFP4 v3 container, QUASAR, TP4 NVLink, RAM-KV, P2P qualification. |

## 2. How the V100 adaptation is built (the pattern we extend)

1. **One architecture per build.** `CMAKE_CUDA_ARCHITECTURES` must be exactly one of `70|86|89`;
   `70` defines `NINFER_VOLTA_BUILD`, fetches CUTLASS v4.4.2, and refuses CUDA ≥ 13.
2. **Device-side guards** (`#if __CUDA_ARCH__ >= 800 … #else`) keep Ampere-only kernels
   compilable on older targets; the `#else` bodies are either real SIMT paths
   (e.g. `gqa_attention_decode_i8.cuh` "Volta SIMT decode, no tensor cores") or traps.
3. **Host-side routing** (`#ifdef NINFER_VOLTA_BUILD`, ~250 sites in ~60 files) picks a route
   per Op at plan time. Volta routes are of two kinds:
   - **pre-Ampere SIMT** routes (valid on Pascal too), and
   - **Volta Tensor-Core** routes: `ggml_k` WMMA GEMM (5–127 tokens), `ggml_k_cutlass_sm70`
     (≥128 tokens, prefill), `*_volta_qpn_*`, `*_volta_mma_*`, `gqa_attention_decode_i8_tc_volta`,
     `gqa_attention_prefill_volta` / VoltaFlash, `bidirectional_gqa_attention_volta`, `swa_volta`,
     CUTLASS `Sm70` BF16/FP8/NVFP4 GEMMs and their QPN prepacks.
4. **Runtime gates** check `DeviceContext::sm()`: family admission accepts `70|86|89|120`
   (`layouts_impl.h:747`), RAM-KV requires 70, direct-peer small all-reduce requires 70
   (`allreduce.cu:515`).
5. **TP2 transport**: graph-capturable UVA D2D copies + event pairs; direct PCIe P2P when
   the IOMMU is in identity/passthrough mode, else verified CUDA-managed staging; small
   (≤80 KiB) all-reduces read the peer buffer directly.

## 3. P100 vs V100 facts that drive the design

| | V100-SXM2 16 GB (`sm_70`) | P100 PCIe 16 GB (`sm_60`) | Consequence |
|---|---|---|---|
| Tensor Cores | yes (m8n8k4) | **none** | every TC route needs a SIMT route |
| FP32 / FP16 | 15.7 / 31.4 TF (+125 TC) | **9.5 / 19.1 TF** | FP16x2 is the only 2× lever |
| HBM2 BW | 900 GB/s | **732 GB/s** | decode floor ≈ 12 ms/token-pass for ~8.6 GB/GPU |
| SMs / L2 | 80 / 6 MB | 56 / 4 MB | fewer CTAs per wave; retune grids |
| Shared mem | 96 KB/SM, 96 KB/block opt-in | **64 KB/SM, 48 KB/block max** | any >48 KB dynamic smem launch fails |
| `__dp4a` | yes | **no** (needs 6.1) | INT8 dot products must unpack |
| Independent thread scheduling | yes | **no** | intra-warp spin/lock patterns can deadlock |
| BF16 | conversion only | conversion only (software); **no scalar bf16 `atomicAdd`** (verified with nvcc 12.8) | same BF16 activation boundary works |
| Type conversions (I2F/F2F) | 16/clk/SM of 64 lanes | 16/clk/SM of 64 lanes | avoid per-weight `int→float`; use magic-number bit tricks |
| Interconnect | NVLink2 (or PCIe) | **PCIe 3.0 x16 only** | P2P iff same root complex & IOMMU passthrough; else staging |
| Toolkit | CUDA ≤ 12.9 | **CUDA ≤ 12.9** (13 dropped sm_50–72) | keep CUDA 12.8 pin |

Sources: CUDA 13.0 release notes; Pascal Tuning Guide (48 KB/block, 64 KB/SM on GP100);
llama.cpp #2159 (`__dp4a` requires 6.1); P100 PCIe datasheet figures.

## 4. Performance model (why FP32 first is viable and where FP16 matters)

Q4_K_M 27B text weights ≈ 17 GB → ≈ 8.6 GB per GPU at TP2.
- **Decode, 1 token:** memory-bound. 8.6 GB / (~0.8 × 732 GB/s) ≈ **15 ms** per pass.
- **MTP verify, 4 tokens:** per weight ≈ 3–4 decode instructions + 4 FMAs ≈ 8 instr;
  13–14 G weights/GPU × 8 ≈ 110 G instr / 4.8 T instr/s ≈ **23 ms** → *ALU-bound in FP32*.
  This is exactly where HFMA2 (and removing I2F conversions) pays off.
- **Prefill:** compute-bound; FP32 SIMT GEMM ≈ 6–7 TF achievable →
  ≈ 27 GFLOP/token/GPU ⇒ **~200–250 tok/s** ceiling before attention.
- **TP2 collectives:** 2 all-reduces/layer × 64 layers; PCIe P2P latency ~10–20 µs ⇒ 1.3–2.6 ms
  per pass — significant vs 15 ms, so the small-message direct path matters (T-007).

These are estimates to order the work, not measurements.

## 5. Architecture proposal

**Build flags** (single arch per build, as today):
- `60` → `NINFER_PASCAL_BUILD=1` and `NINFER_PRE_AMPERE_BUILD=1`; CUTLASS fetched (SIMT GEMM).
- `70` → `NINFER_VOLTA_BUILD=1` and `NINFER_PRE_AMPERE_BUILD=1`.
- Every existing `#ifdef NINFER_VOLTA_BUILD` site is classified once (T-004/T-005):
  "pre-Ampere" semantics → `NINFER_PRE_AMPERE_BUILD`; "Volta Tensor Core" semantics stays
  `NINFER_VOLTA_BUILD` with an explicit Pascal SIMT route beside it. No runtime arch branching
  inside family scheduling (AGENTS.md).

**Pascal routes for the Q4_K_M TP2 Text/MTP path** (T-005):
| Op site | V100 route | P100 route |
|---|---|---|
| `ggml_k` T=1–4 | SIMT FP32 GEMV | same kernel, retuned grid; later remove I2F |
| `ggml_k` T=5–127 | WMMA FP16 | new SIMT FP32 tiled GEMM (decode-in-smem, ≤48 KB) |
| `ggml_k` T≥128 (prefill) | dequant → FP16 → CUTLASS Sm70 TensorOp | dequant → **FP32** workspace → CUTLASS `Sm50/Sm60` SIMT SGEMM (FP32 accumulate) |
| GQA decode INT8 KV, verify widths | Volta TC i8 kernel (width ≥3) | existing SIMT i8 kernel at all widths |
| GQA prefill | VoltaFlash TC | existing ChunkedSmallT SIMT route (then a SIMT flash kernel if prefill attention dominates) |
| GDN chunked prefill / recurrent | SIMT + >48 KB smem? | audited ≤48 KB variants (T-006) |
| BF16 GDN gating / small BF16 GEMMs | CUTLASS Sm70 / Volta QPN | CUTLASS SIMT or existing SIMT GEMV |
| TP2 all-reduce | UVA D2D + direct sum on sm70 | same; extend direct-sum qualification to sm60 after measurement |

**Identity admission:** P100 build admits `qwen3.8-27b/gguf-q4-k-m` TP1/TP2 Text/None/MTP first;
NVFP4/FP8/QUASAR/DFlash/Vision/TP4/35B/RAM-KV are rejected with explicit messages until ported
(decision C-1).

## 6. Numerical policy

- **Oracle unchanged:** FP64 naive formula decoded from exact Q4_K/Q6_K codes and scales.
- **FP32 phase:** all Pascal matmuls accumulate in FP32; activations remain BF16 at Op
  boundaries; BF16→FP32 is an exact bit shift. Expected to match the V100 SIMT routes to
  reduction-order tolerance.
- **FP16 phase (T-009), proposed design — sorting is not the practical tool:** sorting
  products by magnitude per dot product is too expensive in a bandwidth-bound GEMV. The
  standard equivalent guarantees come from **blocked / pairwise summation**: accumulate short
  runs (one Q4_K 32-element sub-block) in `half2`, flush each run into an FP32 accumulator.
  Error grows with the run length (≤32) instead of K (5120–17408). Two further GGUF-specific
  facts make this attractive:
  1. BF16→FP16 is **exact** for |x| in [6.1e-5, 65504] (FP16 has more mantissa bits than BF16);
     only range is lost. Apply an exact per-token power-of-two pre-scale to keep partial sums
     of `q·x` (q ≤ 63 for Q6_K, ≤ 15 for Q4_K) below 65504; activations with outlier channels
     ("massive activations") make this mandatory, not optional.
  2. Q4_K factorises: `Σ w·x = d·sc·Σ(q·x) − dmin·m·Σx`, so the FP16 inner loop multiplies
     small exact integers by activations and the scales are applied once per sub-block in FP32.
- Every Pascal route is checked against the FP64 oracle at real Qwen3.8 shapes (T-008).

## 7. Build plan

CUDA **12.8** (same pin as V100X2; 12.9 also fine). Host: Ubuntu 24.04, GCC 13, CMake ≥3.28,
Ninja, FFmpeg ≥6 and libcurl dev packages. Agent containers install the toolkit from conda-forge
(`scripts/p100/setup_cuda_toolchain.sh`) because NVIDIA's apt repo is blocked there.
`-DCMAKE_CUDA_ARCHITECTURES=60` → `build-p100/`. Docker image must move to a
`nvidia/cuda:12.8.x-devel-ubuntu24.04` base for Pascal (current image is CUDA 13.1).

## 8. Ticket breakdown

T-002 env → T-003 CMake → T-004 compile port → T-005 SIMT routes → T-006 audit →
T-007 runtime/TP2 → T-008 validation plan (executed on the P100 host) → T-009 FP16 →
T-010 tuning. See `tickets/QUEUES.md` for order and open questions.

## Log

- 2026-10-05: Surveyed history, arch guards (`__CUDA_ARCH__`, `NINFER_VOLTA_BUILD`, runtime
  `sm()` gates), Q4_K_M route (`ggml_k.cu`, wrappers), TP2 transport (`allreduce.cu`).
- 2026-10-05: nvcc 12.8 `-arch=sm_60` smoke test: `__nv_bfloat162` arithmetic, `__hfma2`,
  `__nv_fp8_e4m3` conversions compile; scalar `atomicAdd(__nv_bfloat16*)` does **not**.
