# T-008 — P100X2 hardware validation plan (runbook for the P100 host)

**Status:** plan written; execution blocked on hardware access
**Depends on:** T-004 (build), T-005 (routes), T-007 (admission)

Run the steps in order; each has a pass criterion. Record results in the **Results** table below
(hardware, driver, command, outcome). Respect AGENTS.md GPU cadence: ≤10 min active, 3 min idle.

## 0. Host facts (answers C-2)

```bash
nvidia-smi; nvidia-smi -q | grep -E "Driver Version|CUDA Version|Product Name|Bus Id|Power Limit"
nvidia-smi topo -m                         # PIX/PXB = same switch; SYS/NODE = across CPU
dmesg | grep -iE "iommu|dmar" | head; for g in /sys/kernel/iommu_groups/*/type; do cat $g; done | sort | uniq -c
free -g; nproc; lsb_release -a
```
Pass: two `Tesla P100-PCIE-16GB`, driver ≥ 525 and ≤ 580 branch, compute capability 6.0.

## 1. Build

```bash
scripts/p100/setup_cuda_toolchain.sh /opt/cuda-12.8      # or NVIDIA's cuda-toolkit-12-8
export PATH=/opt/cuda-12.8/bin:$PATH CUDAToolkit_ROOT=/opt/cuda-12.8
cmake -S . -B build-p100 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-p100 -j"$(( $(nproc) * 8 / 10 ))" --target ninfer ninfer-serve ninfer_bench p100_op_tests
# later, once a model artifact is present: --target p100_model_tests
# (all 133 tests need ~26 GB of disk; the subsets ~12 GB — see PROMPT.md gotchas)
cuobjdump --list-elf build-p100/apps/ninfer | head          # expect sm_60 cubins
```
Pass: build succeeds; binary contains `sm_60` code.

## 2. Device sanity and transport (T-007)

```bash
build-p100/tests/ninfer_device_test
build-p100/tests/ninfer_allreduce_test                     # exact-value all-reduce, TP2
nvcc -arch=sm_60 -o /tmp/p2p tools/tp2/p2p_probe.cu && /tmp/p2p
nvcc -arch=sm_60 -o /tmp/tp tools/tp2/transport_probe.cu && /tmp/tp
```
Pass: tests green; record P2P on/off, bandwidth, 10 KiB all-reduce latency (V100 staging ref:
48 µs mean).

## 3. Operator correctness against FP64 oracles (sm_60 routes)

```bash
for t in ggml_k gqa_attention gqa_attention_long_context gdn_input_proj gated_delta_net_replay_record \
         gdn_replay_fold kv_cache_append_prefix mtp_pack mtp_round speculative_round \
         linear_split linear_swiglu_split attn_input_proj_split gdn_projections_split \
         output_head_split attention_headlocal gdn_headsplit mtp_split; do
  build-p100/tests/ninfer_${t}_test || echo "FAIL $t"; done
compute-sanitizer --tool synccheck build-p100/tests/ninfer_gqa_attention_test
compute-sanitizer --tool racecheck build-p100/tests/ninfer_ggml_k_test
```
Pass: every GGML_K case (T=1–4 GEMV, 5–127 SIMT tile, ≥128 CUTLASS SIMT prefill, GDN tiled
input, residual add, FP32/BF16 outputs) within the test's FP64-oracle criterion; tests for
formats that Pascal does not admit are expected to report "unavailable on SM60", not wrong values;
no sanitizer errors.

## 4. Real-model smoke (TP1, then TP2)

```bash
W=/path/to/qwen3_8_27b_q4_k_m.ninfer      # converted from the LM Studio Q4_K_M GGUF (artifact doc §14)
build-p100/apps/ninfer $W --tp 2 --devices 0,1 --max-context 8192 --kv-dtype int8 \
  --prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking --greedy \
  --max-new 256 --prompt 'Write a bounded blocking queue in C++.'
build-p100/tests/ninfer_qwen3_8_27b_v100x2_real_test        # same identity, TP2 real checks
build-p100/tests/ninfer_qwen3_8_27b_tp2_parity_test          # TP1 vs TP2 greedy parity
```
Pass: coherent output; MTP acceptance reported; TP2 parity test green. Unsupported identities
(NVFP4/FP8/QUASAR/35B) must fail at startup with "SM60 supports only qwen3.8-27b/gguf-q4-k-m".

## 5. Quality vs reference (FP32 route)

- Greedy token agreement P100 vs the V100X2 run of the same prompts (fixed corpus, 512 tokens):
  report first divergence position per prompt. FP32 SIMT vs V100 FP16-operand prefill can
  legitimately diverge late; early divergence (<32 tokens) is a bug signal.
- Optional: perplexity on a fixed text with the CLI's scoring path vs llama.cpp on the same GGUF.

## 6. Capacity and performance

```bash
python3 tools/v100/bench_tp.py --weights $W --bench build-p100/bench/ninfer_bench --tp 2 \
  --devices 0,1 --corpus-dir <corpus> --output-dir results/p100x2 --prefill-chunk 1024 --suite occupancy
```
- Capacity: confirm 180000-token INT8 KV allocates on 2×16 GB with the FP32 prefill workspace
  (C-6); otherwise find the largest capacity that loads.
- Record prefill tok/s and committed decode tok/s at 3k/8k/32k/85k occupied tokens, MTP
  acceptance, and the same for the owner's llama.cpp P100 baseline (C-4).
- Attention prefill uses the Pascal flash route (T-011); confirm the load log or profile shows
  `pascal_flash` for long prompts.
- `nsys profile` one 8k-prompt request to attribute time (attention vs GGML_K vs all-reduce).

### 6a. A/B: exact fast conversions (`NINFER_PASCAL_FAST_CONVERT`, D-11, W-17)

The default build (`build-p100`) has the flag ON. Build a second tree with it OFF, apps only:

```bash
cmake -S . -B build-p100-cvt-off -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PASCAL_FAST_CONVERT=OFF
cmake --build build-p100-cvt-off -j"$(( $(nproc) * 8 / 10 ))" --target ninfer ninfer_bench
```
- Same prompts, TP2, NVFP4 and Q4_K_M: committed decode tok/s at 3k and 32k, prefill tok/s at
  8k. Greedy output must be **token-identical** between ON and OFF (the conversions are bit-exact).
- Record both in T-010 (implemented table, W-17). If ON gives no measurable gain, note it there:
  D-11 then says delete the flag. Delete `build-p100-cvt-off` afterwards (disk).

## Results

| Step | Date | Host / driver | Outcome |
|---|---|---|---|
| — | | | |
