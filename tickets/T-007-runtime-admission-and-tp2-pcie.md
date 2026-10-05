# T-007 — Runtime admission and TP2 over PCIe on P100

**Status:** in progress (admission done; transport needs hardware)
**Depends on:** T-004

## Admission (done)

- Family device gate (`layouts_impl.h`) admits `sm()==60`.
- `qwen3_6_27b::bind_artifact` under `NINFER_PASCAL_BUILD`: only `Qwen38GgmlK`
  (`qwen3.8-27b/gguf-q4-k-m`); Vision and DFlash rejected. TP1 and TP2 allowed
  (TP4 already requires the Q38 NVFP4/FP8 profiles).
- `qwen3_6_35b_a3b::bind_artifact` rejects on Pascal.
- RAM-KV stays SM70-only (existing `device.sm() != 70` check).
- Pending decision C-1 may widen this.

## TP2 transport (needs the P100 host)

Inherited mechanism (`src/ops/common/allreduce.cu`): graph-capturable UVA D2D copies + event pairs;
startup qualifies direct peer access, falls back to CUDA-managed staging on translated IOMMU
domains; NVLink detection via NVML (absent on P100 PCIe). Small (≤80 KiB) all-reduces use direct
peer reads **only when every rank is sm 70** (`allreduce.cu:515`).

Questions to answer on hardware:

1. `nvidia-smi topo -m`: are the two P100s under one PCIe switch/root complex (`PIX`/`PXB`) or
   across the CPU (`SYS`/`NODE`)? P2P across sockets is often disabled or slow.
2. `dmesg | grep -i iommu`, `cat /sys/kernel/iommu_groups/*/type`: identity vs DMA/DMA-FQ.
   Identity (`iommu=pt`) is needed for direct P2P; otherwise startup selects staging.
3. Run `tools/tp2/p2p_probe.cu` and `tools/tp2/transport_probe.cu` (build with
   `-arch=sm_60`): P2P bandwidth/latency both directions, 4 KiB–80 KiB message latency.
4. Re-measure the small-message direct-sum path on sm 60; if it beats DMA as on V100, extend
   the qualification to `sm()==60` (one-line change at `allreduce.cu:515`) with evidence.

## Log

- 2026-10-05: admission changes implemented (see `bindings.cpp`, `layouts_impl.h`).
