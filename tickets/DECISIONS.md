# Decision log — P100X2 port

Why each non-obvious choice was made, so a later agent can reconstruct the reasoning and know what
evidence would overturn it. Newest last. Format: **decision** — context — reasoning — what would
change it.

## D-1 CUDA 12.8, single-arch `sm_60` build
- **Decision:** target `sm_60` only, with CUDA 12.8.
- **Reasoning:** CUDA 13 removed Pascal/Volta offline compilation; 12.8 is the V100X2 pin, so
  one toolkit serves both. The repo already enforces one architecture per build, and P100 is
  `sm_60` (not 6.1: no `__dp4a`, but full-rate FP16x2).
- **Would change if:** a host only has CUDA ≤ 12.7 (CMake requires ≥ 12.8; could be relaxed after
  checking what needs 12.8).

## D-2 Three build macros instead of reusing `NINFER_VOLTA_BUILD`
- **Decision:** `NINFER_PRE_AMPERE_BUILD` (sm_60 + sm_70), `NINFER_VOLTA_BUILD` (sm_70 Tensor-Core
  routes only), `NINFER_PASCAL_BUILD` (GP100 replacements).
- **Reasoning:** the inherited `NINFER_VOLTA_BUILD` meant two things: "no Ampere instructions,
  use SIMT routes" and "use SM70 Tensor Cores". Pascal needs the first and must never get the
  second. Defining `NINFER_VOLTA_BUILD` on Pascal would have compiled Volta TC kernels whose bodies
  are `#if __CUDA_ARCH__ == 700` → **empty kernels on sm_60, silent garbage**. A mechanical rename
  (82 files) to `PRE_AMPERE`, then restoring `VOLTA` only at Tensor-Core sites, is a no-op for
  sm_70 (both macros defined) — confirmed by the sm_70 regression build.

## D-3 `#error` in Volta Tensor-Core headers under Pascal
- **Reasoning:** turns the "silent empty kernel" failure above into a compile error at the
  include site. It found the unguarded include in `fp8_block.cu`.

## D-4 Throwing stubs for unreachable Volta entry points
- **Decision:** `src/ops/volta_tc_sm60_stubs.cpp`: `*_supported` → `false`, launches throw.
- **Reasoning:** shared dispatchers for formats Pascal does not admit still reference Volta TC
  functions; the repo already uses this pattern (`nvfp4_sm86_stubs.cpp`). Returning `false` from
  predicates makes dispatch fall through to the existing SIMT routes rather than failing. Startup
  admission (D-6) keeps the throwing launches unreachable; reaching one is a loud routing bug.

## D-5 FP32 operands first (not FP16) on Pascal
- **Reasoning:** owner asked for FP32 first. Also: GP100's 2× FP16 rate applies to `HFMA2` with
  FP16 accumulation; FP16 operands with FP32 accumulation run at FP32 rate, so CUTLASS "HGEMM with
  FP32 accumulate" would bring rounding without speed. FP16 work is a separate, error-controlled
  design (T-009).

## D-6 Admission on SM60: NVFP4 + Q4_K_M only, TP1/TP2, Text/None/MTP
- **Reasoning:** every other identity/feature still depends on Tensor-Core or Ampere kernels.
  Rejecting at startup is honest and cheap to widen later. NVFP4 is the owner's main quant; the
  official v3 artifact is mixed-format, so FP8/W8/Q4G64/BF16 routes were included (T-012).

## D-7 Dual-arch dense prefill GEMMs via `pre_ampere_gemm.cuh`
- **Decision:** one config header selects FP16 TensorOp (Volta, unchanged tile/epilogue widths) or
  FP32 SIMT (Pascal); the existing `*_cutlass_sm70.cu` routes compile for both.
- **Reasoning:** reuses the measured routing/workspace logic instead of duplicating ten files; the
  only Pascal-specific differences are operand type, op class, tile shape and epilogue width.
  The fused FP8 row-scale epilogue needs vector width > 1, so Pascal uses the separate scale kernel.
- **Known cost:** file/function names still say `sm70` though they now serve sm_60 too.

## D-8 Chunked FP32 weight decode on Pascal (64 MiB, multiples of 128 rows)
- **Reasoning:** FP32 operands double Volta's workspace (356 MB for a TP2 gate/up shard), which
  competes with KV capacity on 16 GB. Each chunk re-reads the staged activations but not the
  weights, so extra traffic is small. 64 MiB keeps chunk GEMMs large enough (≥ ~1–3k rows) for good
  SIMT efficiency; 128-row multiples keep NVFP4 M128 scale tiles and FP8 128-row blocks whole.
- **Would change if:** measurements show prefill slower than unchunked at the same capacity (try
  larger chunks), or O-11 (fused decode-in-GEMM) lands and removes the workspace entirely.

## D-9 Pascal flash prefill reads the paged INT8 cache in place
- **Context:** owner asked for flash attention for VRAM. Volta's flash route stages INT8 KV into a
  contiguous FP16 copy (≈ 368 MB at 180k, TP2) and runs a vendored MMA kernel.
- **Reasoning:** the inherited chunked fallback never materializes scores, so VRAM was not its
  problem — staging would *add* VRAM. Its problem is efficiency (one 5-shuffle reduction per
  (row, key), split partials, a launch per 5 tokens). The new kernel: FP32 throughout; 32-key tiles
  of K then V share one 32 KiB smem buffer (GP100's 48 KiB/block cannot hold both); a butterfly
  reduce-scatter gives each lane one key's full score in 31 shuffles per 32 keys (vs 160); base-2
  online softmax with scale·log2e folded into Q; masked columns follow the chunked route's
  semantics so the route always needs zero workspace (keeps the exact workspace-contract tests).
  Query block = 4 tokens × 6 heads = 24 rows (register budget; O-1/O-3 to tune).
- **Correction recorded:** an early note called the chunked fallback "quadratic, impractical";
  both routes are quadratic — the difference is a constant factor.

## D-10 Fused small-T residual add on Pascal (T < 128)
- **Reasoning:** Volta always used `linear()` + add because its NVFP4 weights are QPN-prepacked and
  the fused row-major kernels cannot read that layout. Pascal never prepacks, and below 128 tokens
  `linear()` runs the same SIMT kernel families, so the fused kernels strictly remove a projected
  round trip. 128 matches `linear()`'s dense-GEMM threshold.

## D-11 Exact bit-construction conversions behind a build flag
- **Reasoning:** GP100 converts (I2F/F2F) at ¼ FP32 rate, and CUDA's FP8/FP4 decoders below
  SM89/SM100 go through half-precision conversions. Every decoded value (E4M3FN, E2M1, small
  integers) is exactly representable in FP32, so building IEEE bits directly is **bit-identical**
  — verified exhaustively on the host against CUDA's own conversions (all 65,536 FP8 pairs, all 256
  FP4 pairs, integer ranges). A **build** flag (not runtime) keeps one kernel per setting and follows
  the repo's no-runtime-switch rule; default ON because it is exact; OFF exists only for the A/B.
- **Would change if:** the A/B shows no gain (then delete the flag and keep whichever is simpler).

## D-12 Build-of-record rule and test subsets
- **Reasoning:** Ninja fixes its work list at pass start, so edits mid-build mix old/new objects;
  only an edit-free pass supports "builds" claims. Each test statically links the full device
  image (~280 MB), so the runbook builds `p100_op_tests`/`p100_model_tests` (45 tests) instead of
  all 133; shared-library test builds are deferred (T-014).
