# T-011 — Pascal flash-attention prefill route

**Status:** implemented and builds for sm_60 (155 regs, 32 KiB smem); numerical validation on P100 pending (T-008 step 3)
**Depends on:** T-004

## Why (corrected analysis)

Owner requirement (2026-10-05): a flash-attention implementation, for its VRAM savings.

- Attention prefill is O(T × visible keys) on every route; on GP100 (no Tensor Cores) it is
  **compute-bound**: ≈ width × QHeads × keys × 512 FMA ≈ 0.5 TFMA per attention layer per
  1024-token chunk at 85k keys (TP2), i.e. ~0.1 s/layer at FP32 peak.
- The inherited pre-Ampere fallback (`ChunkedSmallT`) does **not** materialize a score matrix, so it
  is not a VRAM problem. Its cost is efficiency: one 5-shuffle warp reduction per (row, key), split
  partials and a launch per 5 tokens. (An earlier note in T-005 called it "quadratic re-reads,
  impractical" — that overstated it; both routes are quadratic, the difference is a constant factor.)
- Volta's flash route stages the INT8 KV into a contiguous **FP16 copy** each layer: at 180k
  context, TP2, that staging is ≈ 368 MB of workspace — the real VRAM cost to avoid.

## Implementation (`src/ops/launcher/gqa_attention_pascal_flash.cu`)

- Route `GqaAttentionRoute::PascalFlash` (Pascal builds only): 27B geometries (24/12/6 Q heads),
  B=1, INT8-G64 cache, width ≥ 64, exact prefill envelope. Honors `valid_columns` with the chunked
  route's semantics (masked columns: not appended, output 0). Workspace: **0 bytes**.
- Appends the chunk's K/V to the paged cache with the decode kernels' quantizer, then attends
  reading the paged INT8 cache **in place** (no gather, no mask tensor, no split partials).
- CTA = (KV head, 4 query tokens) = 24 rows; 8 warps × 3 rows; lane owns 8 head dims.
  32-key K tile then V tile staged as FP32 in one 32 KiB smem buffer (< 48 KiB).
  QK: 256 FMA/lane/row, butterfly reduce-scatter (31 shuffles) gives key `lane`'s score to lane
  `lane`; base-2 online softmax (scale·log2e folded into Q); PV broadcasts p by shuffle.
- All arithmetic FP32; `ex2.approx` as in the decode kernels.

## Verification plan

- `ninfer_gqa_attention_test` case `{66, 63, 129}` already routes here on sm_60 (width ≥ 64, exact
  envelope) and is compared with the FP64 oracle, including the workspace high-water contract.
- Add wider cases (width 256/1024, prefix 0/300/4000, masked) on the P100 host.
- Measure prefill tok/s vs `ChunkedSmallT` at 8k/32k/85k (T-008 step 6).

## Tuning backlog

Rows processed one at a time re-read K from smem 3×; process row pairs. I2F in staging →
magic-number conversion. FP16x2 QK/PV under T-009 rules.
