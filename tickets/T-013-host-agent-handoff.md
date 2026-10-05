# T-013 — Handoff prompt for the P100 host agent

**Status:** ready to send (host offline as of 2026-10-05)
**Depends on:** T-004 (sm_60 build of record), T-008 (runbook)

The cloud sessions have no GPUs. Paste the prompt below into Claude Code on the P100 host.
It collects the host-side facts the queue still needs (C-2 topology, C-6 measured capacity, C-8
artifact path, the C-3/C-4 llama.cpp baseline), runs the T-008 runbook, and writes every result back into `tickets/` so the next session (cloud or host) can
continue from the repository alone.

## Prompt

````markdown
You are working on the NInfer P100X2 port, branch `claude/ninfer-p100-adaptation-xicv3d` of
`GlitchedGoat/ninfer-P100X2`, on the machine that has the two Tesla P100 PCIe 16 GB GPUs.

Read first, in order: `AGENTS.md`, `tickets/PROMPT.md`, `tickets/QUEUES.md`,
`tickets/T-008-p100-validation-plan.md`, `tickets/ENV-SETUP.md`. Follow AGENTS.md's rules
(host CPU < ~85 %, GPU work in ≤10 min active / 3 min idle blocks, never change power limits or
clocks, no unrequested downloads of large artifacts).

Goal: collect the hardware facts and first on-GPU evidence that the cloud session could not.
Record everything in the tickets (not only in chat), commit after each step, and push.

1. **Host facts (C-2).** Run T-008 step 0 and paste the outputs (trimmed) into a new
   section "Host facts" in `tickets/T-008-p100-validation-plan.md`: GPU names, driver and CUDA
   driver versions, `nvidia-smi topo -m`, IOMMU mode/groups, kernel cmdline (`cat /proc/cmdline`),
   CPU/RAM, OS, free disk. Say whether the two P100s share a PCIe switch/root complex.
2. **Toolchain.** If CUDA 12.8 (or any 12.x ≥ 12.8) is installed, use it; otherwise run
   `scripts/p100/setup_cuda_toolchain.sh /opt/cuda-12.8`. Do not use CUDA 13.
3. **Build of record.** T-008 step 1 (`build-p100/`, `-DCMAKE_CUDA_ARCHITECTURES=60`,
   `-DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON`), building only
   `--target ninfer ninfer-serve ninfer_bench p100_op_tests` (all tests need ~26 GB of disk;
   add `p100_model_tests` once an artifact is present). No source edits while it builds. Record the
   source commit and result in T-004. If a test target fails to compile, record it and continue
   with the rest (`-- -k 0`).
4. **Device + transport (T-007).** T-008 step 2. Record P2P on/off, bandwidth both directions and
   small all-reduce latency in T-007.
5. **Operator correctness (T-008 step 3).** Run the listed tests plus
   `ninfer_gqa_attention_test` (covers the new Pascal flash prefill route) and the FP8/NVFP4 linear
   tests. Record pass/fail per test in T-008. For any failure: keep the full error text in the
   ticket, then diagnose with `compute-sanitizer` (memcheck, synccheck, racecheck) before
   changing code. A test for a format Pascal does not admit may fail with
   "SM70 Tensor-Core route is unavailable on SM60" — record it as expected, do not "fix" it.
6. **Artifact path (C-8).** Report whether `qwen3_8_27b_nvfp4.ninfer` (official v3 container
   from `neroued/Qwen3.8-27B-nvfp4-NInfer`, 23.72 GB) and/or the Q4_K_M `.ninfer` exist on this
   host and their absolute paths. Do not download them unless the owner asks.
7. **Real-model smoke (only if the NVFP4 artifact exists).** T-008 step 4 with `$W` = the NVFP4
   file: TP1 first, then TP2, `--max-context 8192 --kv-dtype int8 --spec mtp --draft-tokens 3
   --greedy --max-new 256`. Record load summary, output text, MTP acceptance and tok/s.
   Unsupported identities must fail at startup with "SM60 supports only …" — confirm one.
8. **Capacity (C-6).** With TP2, NVFP4, INT8 KV, MTP3: find the largest `--max-context`
   that loads (try 180000, then 163840, 131072, 98304) with `--prefill-chunk 1024`. Record it.
   The owner accepts slightly less than 180000, but it must cover the 85k occupancy point.
9. **Acceptance numbers (C-4).** T-008 step 6 occupancy suite at **3k, 32k and 85k** occupied
   tokens (8k too if cheap): committed decode tok/s (the acceptance metric), prefill tok/s, MTP
   acceptance, for NVFP4 and Q4_K_M `.ninfer`. One `nsys` profile of an 8k request.
   Then T-008 step 6a: the `NINFER_PASCAL_FAST_CONVERT` ON/OFF A/B (token-identical output
   expected; record speed in T-010).
9a. **Owner's llama.cpp P100 branch (C-3).** It is on this machine. Find it (ask the owner if not
    obvious; e.g. `find / -name ggml-cuda -type d 2>/dev/null`), record its path, remote/branch and
    `git log -5 --oneline`, and diff it against upstream llama.cpp. Summarize in
    `tickets/T-009-fp16x2-error-controlled.md` **how it accumulates in FP16 while limiting error**
    (blocked/pairwise sums? sorting? FP32 flush points? scaling?), which kernels it changes (matmul,
    MMQ/dequant, flash attention, prefill), and any build flags. Then measure it as the C-4
    baseline: decode and prefill tok/s with the Q4_K_M GGUF at the same occupied contexts as step 9.
10. **Report.** Update `tickets/QUEUES.md` (close answered items, add new questions), write
    `session-summaries/<date>-p100-host-session-<n>.md`, commit, and `git push`. Stop and ask the
    owner before any change to drivers, BIOS/IOMMU/GRUB settings, power or clocks.

Do not claim performance or correctness that was not measured; label estimates as estimates.
````

## Expected outputs (for the next session to look for)

- T-008 "Host facts" + per-step results table filled.
- T-004 build-of-record entry from the host.
- T-007 transport numbers; T-012/T-011 notes for any NVFP4 or flash-attention failure.
- T-010: fast-conversion A/B result; T-009: llama.cpp branch technique summary.
- QUEUES.md: C-2 facts, C-6 capacity, C-8 path filled in; new items for anything blocking.
