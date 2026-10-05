# T-013 — Handoff prompt for the P100 host agent

**Status:** ready to send (host offline as of 2026-10-05)
**Depends on:** T-004 (sm_60 build of record), T-008 (runbook)

The cloud sessions have no GPUs. Paste the prompt below into Claude Code on the P100 host.
It collects the facts the open conversation-queue items need (C-2, C-4, C-6, C-8), runs the T-008
runbook, and writes every result back into `tickets/` so the next session (cloud or host) can
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

1. **Host facts (answers C-2).** Run T-008 step 0 and paste the outputs (trimmed) into a new
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
6. **Artifact (answers C-8).** Report whether `qwen3_8_27b_nvfp4.ninfer` (official v3 container
   from `neroued/Qwen3.8-27B-nvfp4-NInfer`, 23.72 GB) and/or the Q4_K_M `.ninfer` exist on this
   host and their absolute paths. Do not download them unless the owner asks.
7. **Real-model smoke (only if the NVFP4 artifact exists).** T-008 step 4 with `$W` = the NVFP4
   file: TP1 first, then TP2, `--max-context 8192 --kv-dtype int8 --spec mtp --draft-tokens 3
   --greedy --max-new 256`. Record load summary, output text, MTP acceptance and tok/s.
   Unsupported identities must fail at startup with "SM60 supports only …" — confirm one.
8. **Capacity (answers C-6).** With TP2, NVFP4, INT8 KV, MTP3: find the largest `--max-context`
   that loads (try 180000, then 131072, 98304, 65536) with `--prefill-chunk 1024`. Record it.
9. **Baseline numbers (helps C-4).** If time allows, T-008 step 6 occupancy suite at 3k/8k/32k
   occupied tokens (85k only if capacity permits), and one `nsys` profile of an 8k request.
   If the owner's llama.cpp P100 setup is on this host, measure it on the same prompts.
10. **Report.** Update `tickets/QUEUES.md` (close answered items, add new questions), write
    `session-summaries/<date>-p100-host-session-<n>.md`, commit, and `git push`. Stop and ask the
    owner before any change to drivers, BIOS/IOMMU/GRUB settings, power or clocks.

Do not claim performance or correctness that was not measured; label estimates as estimates.
````

## Expected outputs (for the next session to look for)

- T-008 "Host facts" + per-step results table filled.
- T-004 build-of-record entry from the host.
- T-007 transport numbers; T-012/T-011 notes for any NVFP4 or flash-attention failure.
- QUEUES.md: C-2, C-6, C-8 answered; new items for anything blocking.
