# T-010 — P100 (GP100, sm_60) optimization backlog

**Status:** open backlog — measurements blocked on hardware (T-008 baseline first)
**Depends on:** T-008 for any claim

**Rule for every agent:** whenever you notice a Pascal optimization opportunity, add it here (ID,
location, why, expected effect, how to verify). Do not implement items without a baseline
measurement at the claimed scope (AGENTS.md "Performance work"): profile the whole request with
`nsys` first, and use `ncu` only on a kernel the profile shows matters.

## GP100 facts that drive the items

56 SMs, 64 FP32 lanes/SM, 9.5 TF FP32 / 19 TF FP16x2 (HFMA2) / no Tensor Cores; 732 GB/s HBM2;
4 MB L2; **64 KiB smem/SM, 48 KiB/block**; 64K regs/SM; **type conversions (I2F/F2F/F2I) run at
1/4 of FP32 rate**; no `__dp4a`; no independent thread scheduling; PCIe 3.0 x16 only.

## Backlog

Effect estimates are reasoning, not measurements. "Scope" is where to measure.

| ID | Area / location | Opportunity | Why on P100 | Scope to verify |
|---|---|---|---|---|
| O-1 | Flash prefill `gqa_attention_pascal_flash.cu` | 155 regs → 1 CTA (8 warps)/SM. Cut registers (fewer rows/warp, `__launch_bounds__(256,2)`) or restructure to reach 2 CTAs/SM | latency hiding on SIMT FMA pipe | kernel (`ncu` achieved occupancy, FMA util) |
| O-2 | Flash prefill | Each warp re-reads the K tile from smem once per row (3×); process 2 rows per pass sharing each `float4` K load | smem-bandwidth bound (~2× FMA time) by estimate | kernel |
| O-3 | Flash prefill | Query block is 4 tokens → KV re-read per 4 tokens; try 8/16 tokens (registers permitting) or split K/V tiles across two CTAs | DRAM/L2 KV traffic at long context | kernel at 32k/85k keys |
| O-4 | Flash prefill staging | `static_cast<float>(int8)` is I2F (1/4 rate); use the magic-number trick (`__int_as_float(0x4B400000 | (code+128))` style) | conversion throughput | kernel |
| O-5 | Flash prefill scheduling | Later query blocks see more keys; order `blockIdx.y` longest-first to cut tail | load balance | kernel |
| O-6 | Attention decode/verify SIMT i8 kernel (`gqa_attention_decode_i8.cuh` `#else` branch) | One `warp_sum` (5 shuffles) per (row, key) for 8 FMAs; reuse the flash kernel's 32-key reduce-scatter (31 shuffles per 32 keys) for MTP-verify widths | shuffle-bound SIMT | kernel at T=1..4, 8k–85k keys |
| O-7 | Attention split policy | `DecodeSplits = 560*scale` and `kSmallTChunkTokens=5` were tuned on 80-SM V100; retune for 56 SMs | grid sizing | decode step |
| O-8 | GGML_K GEMV `ggml_k.cu::gemv` | Per-weight `static_cast<float>(code)` (I2F); use magic-number float build or factor `d·sc·Σ(q·x) − dmin·m·Σx` so codes stay integer until a per-sub-block scale | conversion-bound at T=2..4 | kernel |
| O-9 | GGML_K GEMV launch | 4 rows/CTA, 128 threads (Volta-tuned) → sweep for 56 SMs | occupancy/wave quantization | kernel |
| O-10 | GGML_K Pascal tile (`gemm`, T=5–127) | k-major B store has bank conflicts (stride 36/20 floats); one row per lane — add 2-row register blocking | smem conflicts, FMA:LDS ratio | kernel |
| O-11 | Dense prefill GEMMs (`pre_ampere_gemm.cuh`, `ggml_k_cutlass_simt.cu`) | Separate dequant-to-FP32 pass + GEMM; fuse dequant into a SIMT GEMM mainloop (decode Q4_K/NVFP4/FP8 tiles straight into smem) | removes a full FP32 weight write+read per call and the workspace | prefill phase |
| O-12 | Dense prefill workspace | FP32 operands double Volta's workspace (e.g. 356 MB for a TP2 gate/up shard); chunk rows like `ggml_k_cutlass_simt.cu` does (64 MiB) in `nvfp4_cutlass_sm70.cu` / `fp8_cutlass_sm70.cu` | frees VRAM for KV capacity (C-6) | capacity at load |
| O-13 | CUTLASS SIMT tile config | Default 128×128×8 / 32×64×8, 2 stages; sweep 128×64, 64×128, 3 stages for GP100 | 48 KiB smem, 56 SMs | operator |
| O-14 | NVFP4/FP8 SIMT decode & small-T kernels (`nvfp4_gemv`, `nvfp4_small_t`, `fp8_gemv`, `fp8_small_t`, attn/GDN-input variants) | E2M1/E4M3 decode goes through `cuda_fp4.h`/`cuda_fp8.h` software paths on SM60; replace with a 16-entry (E2M1) / 256-entry (E4M3) LUT in registers or smem | conversion-heavy on sm_60 | kernel at T=1..4 |
| O-15 | NVFP4/FP8 chunk sizes | `kNvfp4LastSmallT`, `kFp8LinearSmallTMax` were set with QPN covering T=2–32 on Volta; Pascal runs the SIMT small-T kernels there — retune chunk widths | routing | operator at T=2..32 |
| O-16 | NVFP4/FP8 residual add (`*_linear_add_plan.cpp`) | Pascal uses `LinearThenAdd` (because Volta weights are QPN-prepacked); Pascal weights are row-major, so the fused A16 add kernels are usable | one fewer launch + output round trip | layer |
| O-17 | FP8 fused row-scale epilogue (`fp8_cutlass_sm70.cu`) | Disabled on Pascal (SIMT epilogue is 1 element/access); write a SIMT-compatible fused epilogue | removes a full output read/write | prefill |
| O-18 | W8 MTP projections (`w8_dispatch.cpp`) | Pascal uses SIMT `r8_c8` at every T, re-reading weights per 8 tokens during prefill; add a dual-arch dense route (as for FP8/NVFP4) above ~128 tokens | prefill of the MTP layer | prefill phase |
| O-19 | Hot kernels with >128 registers (916 found by `scripts/p100/audit_resources.py`) | Occupancy review of those on the critical path (decode GEMVs, attention, GDN) | 64K regs/SM | per kernel |
| O-20 | GDN recurrent FP32 (`recurrent.cu`) | Re-check grid/occupancy for 56 SMs; consider FP32 state in registers longer | sequential, latency-bound | decode step |
| O-21 | TP2 all-reduce (`allreduce.cu:515`) | Small (≤80 KiB) direct peer-read sum is gated to sm 70; measure on P100 PCIe P2P and enable if faster | ~128 all-reduces per token | decode step |
| O-22 | PCIe topology | If the GPUs are under different root complexes, P2P may be off → managed staging; consider pinned host double-buffering tuned for PCIe 3.0 | collective latency | decode step |
| O-23 | FP16x2 arithmetic (T-009) | HFMA2 with blocked FP32 flush, exact power-of-two pre-scale, integer-code factorisation | 2× FMA rate where ALU-bound (verify widths, prefill, attention) | operator + end-to-end quality |
| O-24 | Global conversion audit | Grep hot kernels for I2F/F2F (`cuobjdump -sass | grep -c "I2F\|F2F"`) and remove avoidable ones | 1/4-rate conversions | per kernel |
| O-25 | CUDA Graph / launch overhead | Pascal launch latency per kernel; count launches per decode step and fuse small ones (norm+residual, etc.) | many small launches | decode step |

## Method

1. T-008 baseline: prefill tok/s and committed decode tok/s at stated occupancy, MTP acceptance.
2. `nsys` whole-request profile → rank the backlog by measured share of time.
3. Implement the top item, re-check its oracle test, re-measure at the same scope, log here.

## Implemented, not yet measured

| ID | Commit | What | How to A/B on the host |
|---|---|---|---|
| O-12 | `9ac42c7` | FP8/NVFP4 dense prefill decodes ≤64 MiB FP32 row chunks (`pre_ampere::weight_chunk_rows`) | capacity at load (`--max-context` that fits); prefill tok/s vs chunk size |
| O-16 | `9ac42c7` | NVFP4/FP8 residual add uses fused SIMT A16 kernels for T < 128 on Pascal | decode/verify step time; `ninfer_linear_add_{nvfp4,fp8}_test` for correctness |
| O-4, O-8, O-14 | `9ac42c7` | `NINFER_PASCAL_FAST_CONVERT` (default ON): exact bit-construction for E4M3FN/E2M1 decode, GGML_K codes, INT8 KV staging | build twice (`-DNINFER_PASCAL_FAST_CONVERT=ON/OFF`), compare decode/prefill tok/s; outputs must be bit-identical (exhaustively checked by `ninfer_pascal_convert_test`) |

## Log

- 2026-10-05: backlog consolidated from the porting work (T-005, T-006, T-011, T-012).
