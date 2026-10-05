# T-011 — Pascal SIMT flash-attention prefill route

**Status:** design ready; implementation next (decision C-7)
**Depends on:** T-004 green build

## Problem

On V100, wide single-request prefill attention uses **VoltaFlash**: stage the visible INT8 KV once per
layer into contiguous FP16, build a causal mask from device positions, run the vendored llama.cpp
`fattn-mma-f16` kernel (SM70 Tensor Cores), scatter the output. Pascal currently falls back to
`ChunkedSmallT`, which re-reads the whole visible history for every 5-token chunk:
≈ T²/10 key reads per attention layer — fine for short prompts, impractical at 85k tokens.

## Proposed design

Reuse VoltaFlash's staging and swap only the arithmetic kernel:

- **Kernel:** vendor llama.cpp `ggml/src/ggml-cuda/fattn-tile.cuh` (MIT, SIMT, supports
  DKQ=DV=256, runs on Pascal), pinned to one commit, beside the existing
  `third_party/llama_cpp_fattn/` (same provenance pattern, same `common.cuh` shim).
  Build it with `FAST_FP16_AVAILABLE` **undefined** so the KQ and VKQ accumulations run in FP32
  (owner's FP32-first requirement); FP16 tile becomes a T-009 candidate.
- **Staging (shared with Volta):** factor `volta_flash_{append_kv,append_kv_i8,gather_kv,
  gather_kv_i8,convert_q,convert_out,build_mask}` out of `gqa_attention_volta_flash.cu` into a
  `gqa_attention_flash_staging.cuh` used by both routes. Gather INT8-G64 → FP16 rounds
  `code × fp16 scale` once (same boundary as V100).
- **Route:** rename `GqaAttentionRoute::VoltaFlash` → `Flash` with arch-specific launchers
  (`gqa_attention_volta_flash.cu` sm70, new `gqa_attention_pascal_flash.cu` sm60); the wrapper's
  workspace contract (`allocate_volta_flash_workspace`) is shared.
- **Tile config for GP100:** 48 KiB smem/block; pick `ncols2=2` (GQA ratio 6) and the smallest
  `ncols1` that fits, using llama.cpp's FP32 Nvidia config table as the starting point.

## Verification

- `ninfer_gqa_attention_test` / `_long_context_test` already compare routes against an FP64
  attention oracle; add Pascal widths ≥ 64 (route switch) on sm_60.
- End-to-end: 8k/32k/85k prefill tok/s before/after (T-008 step 6).

## Alternative considered

Hand-written SIMT kernel reading paged INT8 directly (no FP16 staging): saves the staging pass but
must fit Q/K/V tiles for D=256 in 48 KiB and avoid per-element `I2F` (1/4 rate on GP100). Higher
risk without hardware; revisit only if staging shows up in the profile.
