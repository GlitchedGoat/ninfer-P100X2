# T-008 — P100X2 hardware validation plan (runbook for the P100 host)

**Status:** ready to run on the P100 host (not yet executed; the cloud sessions have no GPUs)
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

`p100_op_tests` (step 1) builds exactly the operator, TP2-split, attention and device/runtime tests
covering the SM60-admitted identities (list: `NINFER_P100_OP_TESTS` in `tests/CMakeLists.txt`).

```bash
for t in build-p100/tests/ninfer_*_test; do "$t" > "${t##*/}.log" 2>&1 || echo "FAIL ${t##*/}"; done
compute-sanitizer --tool synccheck build-p100/tests/ninfer_gqa_attention_test
compute-sanitizer --tool racecheck build-p100/tests/ninfer_ggml_k_test
compute-sanitizer --tool memcheck  build-p100/tests/ninfer_linear_nvfp4_a16_test
```
Pass: every case within the test's FP64-oracle criterion. This covers:
- **GGML_K:** T=1–4 GEMV, T=5–127 SIMT tile, T≥128 chunked SIMT prefill.
- **NVFP4/row-FP8/W8/Q4/BF16 linear:** small-T and dense prefill routes.
- **Fused forms:** SwiGLU and residual add.
- **Attention:** the `pascal_flash` wide-prefill cases (T-011).

A case for a format/route Pascal does not admit may throw "SM70 Tensor-Core route is unavailable
on SM60"; record it as expected, do not "fix" it. No sanitizer errors.

## 4. Real-model smoke (TP1, then TP2)

The main target is NVFP4; repeat with the Q4_K_M `.ninfer` if present.

```bash
W=/path/to/qwen3_8_27b_nvfp4.ninfer        # official v3 container (C-8)
for tp in "--device 0" "--tp 2 --devices 0,1"; do
  build-p100/apps/ninfer $W $tp --max-context 8192 --kv-dtype int8 --prefill-chunk 1024 \
    --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking --greedy --max-new 256 \
    --prompt 'Write a bounded blocking queue in C++.'
done
cmake --build build-p100 -j"$(( $(nproc) * 8 / 10 ))" --target p100_model_tests
NINFER_QWEN3_8_27B_WEIGHTS=$W build-p100/tests/ninfer_qwen3_8_27b_tp2_real_test
NINFER_QWEN3_8_27B_WEIGHTS=$W build-p100/tests/ninfer_qwen3_8_27b_mtp_tp2_real_test
NINFER_QWEN3_8_27B_WEIGHTS=$W build-p100/tests/ninfer_qwen3_8_27b_tp2_parity_test   # TP1 vs TP2
NINFER_QWEN3_8_27B_WEIGHTS=$W build-p100/tests/ninfer_qwen3_8_27b_graph_tp2_test
NINFER_V100X2_ARTIFACT=/path/to/qwen3_8_27b_q4_k_m.ninfer build-p100/tests/ninfer_qwen3_8_27b_v100x2_real_test
```
Model tests skip (exit 77) when their artifact variable is unset; check each test's source for the
variable it reads if one does not run. These tests were written for V100X2 artifacts, so one may
assert expectations specific to another identity. Read the assertion before calling it a Pascal bug.

Pass:
- coherent output and MTP acceptance reported;
- the real/parity/graph tests are green;
- any other identity (FP8, QUASAR, groupwise, 35B) fails at startup with "SM60 supports only
  qwen3.8-27b/nvfp4 and qwen3.8-27b/gguf-q4-k-m". Confirm one if such an artifact is at hand.

## 5. Quality (FP32 route)

There is no V100 on this host, so the references are:
- **TP1 vs TP2 greedy parity** (step 4).
- **Q4_K_M GGUF in the owner's llama.cpp vs Q4_K_M `.ninfer` in NInfer:** greedy token agreement on
  the same corpus prompts (512 tokens; report the first divergence position per prompt). Late
  divergence is expected from different reduction orders. Divergence within the first 32 tokens
  is a bug signal.
- **NVFP4:** coherent output plus MTP acceptance comparable to Q4_K_M. NVFP4 is a different
  quantization, so tokens are not expected to match.

## 6. Capacity and performance

Build exact-token corpora once with the corpus tool (built with `NINFER_BUILD_BENCHMARKS=ON`).
List plenty of distinct C++/CUDA sources, because the tool never repeats text. The file names below
are the ones `tools/v100/bench_tp.py` expects:

```bash
mkdir -p profiles/bench/p100
SRC=$(git ls-files 'src/*.cpp' 'src/*.cu' 'src/*.h' 'src/*.cuh')
for n in 3072 8192 16384 32768 65536 85000; do
  out=profiles/bench/p100/code-$n.ids; [ $n = 85000 ] && out=profiles/bench/p100/v100-code-85000-iommu-pt.ids
  build-p100/bench/ninfer_v100_corpus $W $out --code-chat $n --output-tokens 1024 $SRC
done
python3 tools/v100/bench_tp.py --weights $W --bench build-p100/bench/ninfer_bench --tp 2 \
  --devices 0,1 --corpus-dir profiles/bench/p100 --output-dir profiles/bench/p100x2 \
  --prefill-chunk 1024 --capacity <step-8 capacity> --suite occupancy
```
- **Capacity:** find the largest `--max-context` that loads on 2×16 GB with INT8 KV, MTP3 and the
  FP32 prefill workspace (C-6). The target is 180000, and slightly less is acceptable. It must
  leave room for 85000 + 1024 tokens. Pass it as `--capacity`.
- **Acceptance (C-4):** committed decode tok/s at **3k, 32k and 85k** occupied tokens. The suite
  also runs 8k/16k/64k. Record prefill tok/s and MTP acceptance as well, for NVFP4 and Q4_K_M.
- **llama.cpp baseline:** run the owner's P100 llama.cpp `llama-server` with the Q4_K_M GGUF on
  both GPUs. Feed it the same `.ids` corpora via `tools/v100/compare_llama.py --url <server>
  --corpus <file> --prompt-tokens N --decode-tokens 512 --output <json>`. The script needs a
  llama-server that accepts token-array prompts and `return_tokens`; if the owner's branch is
  older, record the gap and use `llama-bench -p N -n 512` instead, noting it in the result.
  Record its exact build flags and split mode.
- Attention prefill uses the Pascal flash route (T-011); confirm `pascal_flash` appears in the
  load log or profile for long prompts.
- `nsys profile` one 8k-prompt request to attribute time (attention vs GEMV vs all-reduce).

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
