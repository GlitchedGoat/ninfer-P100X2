# NInfer V100X2

[English](README.md) · [详细评测方法](docs/performance.md) · [HTTP API](docs/serving.md)

面向 **两张 Tesla V100-SXM2 16 GB、CUDA 12.8、SM70** 的 Qwen3.8-27B 高速推理分支，
重点优化单个活跃请求。由上游 RTX 3060 TP2 路线与 `geoffwatts/ninfer-v100` 的 Volta
实现结合而来。本文只介绍本分支的改动和本机测量，不把上游 RTX 5090 等平台的结果当作 V100 成绩。

采用 TP2 **分张量**执行，不是 32/32 层的分层卸载。默认启动脚本使用 LM Studio GGUF
转换得到的 Q4_K_M `.ninfer`；也支持官方 Qwen3.8-27B NVFP4 v3 容器。
常用配置为 180000-token 容量、完整 INT8 group-64 KV、CUDA Graph 和 MTP3。
“MTP 3-0”在这里是最多提出三个草稿 token、允许接受零个，不保证每轮接受三个。

**容量不等于实际上下文占用。** 下表中的 85K 是实际输入 85000 token；180K 是容量上限。
Decode 只统计已提交输出，不把被拒绝的草稿算入吞吐。NVFP4 和 Q4_K_M 使用不同量化权重，
这些表不是两种量化的质量等价证明。

## 本分支的改动

- Q4_K_M prefill：协作解码 GGML-K 块，写入调用方 FP16 workspace，再交给 Volta CUTLASS
  Tensor Core GEMM；GDN 控制投影保留 FP32 输出。源 GGUF 的 Q4_K/Q6_K 编码和 scale 不变。
- NVFP4/FP8：宽 prefill 每次调用只解码一次权重，支持 TP2 预打包矩阵；窄投影走 QPN
  Tensor Core 路线。SwiGLU 的 gate/up 中间值保留 FP32，仅最终激活写回 BF16。
  修复了预打包 FP8 词表头在单 token 和 chunk 尾部错误走 row-major GEMV 的问题。
- 长上下文注意力：共享 QK 分块和 softmax 权重，协作读取 split 向量，保留原有 FP32 累加顺序。
  不裁剪历史，不用检索子集代替完整注意力，不额外降低 KV 精度。
- GDN prefill：Q/K 每行只做一次 FP32 归一化，供状态 tile 复用；FP32 递推顺序不变。
  3072-token TP2 的 GDN scratch 为 24 MiB。
- TP2 通信：图可捕获的 UVA D2D copy 加双向事件顺序；启动时检查双向精确传输。
  支持 direct P2P 和经过验证的 CUDA-managed staging，不另设显式 pinned-host 通信分支。
- Volta 残差路径：Q5 与 BF16 TP2 row-shard 使用可在 SM70 执行的 kernel，并通过独立数值检查。
- NVFP4 v3：读取官方 `NINFER\x00\x03` 单文件容器，映射到已注册的
  `qwen3.8-27b/nvfp4` 身份，不重打包权重字节；使用容器原始 chat template。
  容器可选 DFlash2 组件不是这组 NVFP4/MTP 测量的执行路径。
- 前缀复用：TP2 与 optimized MTP3 可以恢复当前前沿或完整 turn/response checkpoint，
  只计算后续新增提示；支持重复恢复、精确命中和提前停止后的继续生成。

## 性能：P2P 开启

2026-10-01 实测，双卡各 300 W，PCIe 3.0 ×16、PHB 拓扑。重启后使用 `iommu=pt`，
两 GPU 的 IOMMU domain 均为 `identity`，NVIDIA P2P read/write 检查通过，
NInfer 自动启用已有 direct P2P 路径。**这是 PCIe P2P，不是 NVLink。**
该配置在本机通过验证，不代表所有主板或转接方案都能启用 P2P。

### NVFP4 v3：实际上下文占用

官方 NVFP4 v3、TP2、完整 INT8 group-64 KV、greedy（temperature=0）、MTP3、
optimized draft head、CUDA Graph。容量统一 180000（实际 KV 分配 180032），prefill chunk=3072。
每档两轮冷提示，无前缀复用、无额外请求 warmup；图预热在计时之外。
每轮生成 513 token，其中首个来自 prefill，后续 **512 个是计时 decode 输出**。
吞吐为均值 ± 样本标准差。

| 实际输入 token | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 3072 | 1783.96 ± 43.38 | 105.48 ± 0.01 | 72.97% |
| 8192 | 1754.43 ± 18.24 | 114.49 ± 0.01 | 81.98% |
| 16384 | 1699.06 ± 9.73 | 108.29 ± 0.003 | 79.47% |
| 32768 | 1596.24 ± 2.94 | 101.63 ± 0.03 | 79.69% |
| 65536 | 1397.83 ± 0.72 | 88.34 ± 0.04 | 79.12% |
| 85000 | **1306.48 ± 3.95** | **83.18 ± 0.13** | **79.12%** |

每档两轮输出 IDs 和 MTP 统计一致，无 EOS/EOG，均完成指定窗口。
不同长度的提示保留完整任务及 assistant 后缀，但代码正文、接受率不同，
不能由 8K 档更快推断上下文越长越快。这些是固定窗口吞吐，不是完整代码任务得分。

85K 输入、180K 容量满足此前提出的 1000 prefill / 70 decode tok/s 目标。
同输入关闭 P2P 的控制组为 79.18 decode tok/s，开启后为 83.18，提升 **5.05%**；
两路径各两轮的所有 513 个输出 IDs 和 speculative 字段一致，每轮 accepted/drafted=360/455，
152 rounds。不能拿下方旧语料的 78.424 tok/s 计算严格的 P2P 因果增益。

85K resident 请求平均 prefill 65.061 秒、decode 6.155 秒、总计 71.220 秒。
模型加载 19.662 秒，只在进程启动时进行一次，不计入请求吞吐。
全部上下文的阶段时间、wall decode、接受数量和测量限制见[性能文档](docs/performance.md#p2p-enabled)。

### NVFP4 v3：最大容量，固定 512-token 输入

固定 512 输入、256 timed decode token（总输出 257）；prefill chunk=1024，其他推理配置不变。
每档丢弃一轮 warmup，再测三轮。这里 decode 使用**首 token 到请求结束的 wall time**，
而上一张表使用 Engine decode 阶段时间。

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1024 | 106.12 ± 0.01 | 1443.78 | 70.04% |
| 2048 | 106.12 ± 0.04 | 1438.44 | 70.04% |
| 4096 | 105.96 ± 0.02 | 1440.81 | 70.04% |
| 8192 | 105.98 ± 0.01 | 1442.96 | 70.04% |
| 16384 | 106.06 ± 0.04 | 1443.37 | 70.04% |
| 32768 | 106.00 ± 0.03 | 1443.77 | 70.04% |
| 65536 | 106.01 ± 0.06 | 1443.42 | 70.04% |

七档容量均精确分配；21 轮输出和 MTP 统计一致，每轮接受 173/247 drafts，83 rounds。
这是只有 512-token 实际占用的容量测试，不能将约 106 tok/s 当成 64K 实际占用的速度。

### 通信与推理回归

10 KiB BF16 allreduce，500 次 host-sync 测量：

| 路径 | 平均延迟 µs | p50 µs | p99 µs |
|---|---:|---:|---:|
| PCIe P2P 开启 | 26.6085 | 25.901 | 43.910 |
| P2P 关闭，verified staging | 47.7961 | 47.181 | 62.424 |

通信平均延迟降低 44.33%，**不等于整段推理快 44.33%**。两路径的精确传输、
不整齐 shape、guard 和 64 连续轮检查通过。

Q4_K_M 与 NVFP4 v3 的真实 MTP、前缀缓存回归通过：Graph/eager 输出、logits、接受状态和
复用前沿一致；两个 probe 的 MTP/plain 32-token 序列一致；各检查 64 个 teacher-forcing
位置，无分歧、worst emitted-logit deficit=0。前缀覆盖重复恢复、追加、前缀变更、
精确前沿、接受轮内提前停止、改写回复后的恢复和恢复后采样。

3274-token 提示的三轮冷/缓存 TTFT 中位数：Q4_K_M 为 3.33492 秒 / 16.8689 毫秒，
NVFP4 为 3.01954 秒 / 14.6475 毫秒。Q4 的两处缓存/全新 prefill 首次分歧对应
teacher-forced logit deficit 都为 0，满足原有 near-tie 判据，不能说所有冷/缓存序列逐位一致。
NVFP4 本次对应探针未报告这些分歧；测试判据没有放宽。

NVFP4 检查采用 Q4 既有 gate 的临时诊断副本，仅改权重身份断言，链接同一 Engine/Ops。
本次未因纯通信配置变化重跑完整 CUDA 数学套件。上述证据不是通用“智商无损”评分。

## 性能：P2P 关闭

这里区分当前同输入控制组和开启 P2P 前的历史结果。原本的 translated `DMA-FQ` IOMMU
domain 阻止 direct P2P，CUDA 采用经过验证的 UVA D2D staging。当前控制组只在诊断进程
使用 CUDA capability-query shim，未更改系统配置；它不是公开 CLI 的 P2P 开关。

### 当前 NVFP4 v3 同输入控制组

与上方 85K P2P 测量使用相同输入、模型、sampling、KV、MTP、图和输出窗口，各测两轮：

| Prefill tok/s | 已提交 Decode tok/s | Wall decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

开启 P2P 的严格同输入增益为 decode +5.0459%、prefill +0.6578%。

### 早期 NVFP4 数据

以下使用重启前语料，不能替代上面的同输入控制组：

| 模型 / 窗口 | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---|---:|---:|---:|
| v2，85K 输入、180K 容量、512 timed 输出 | 1279.44 ± 2.94 | 78.36 ± 0.04 | 79.12% |
| 官方 v3，同一旧 85K 窗口 | 1277.61 ± 3.12 | 78.424 ± 0.0004 | 79.12% |
| 官方 v3，512 输入、2K 容量、512 timed 输出 | 1344.24 ± 33.15 | 98.831 ± 0.027 | — |

官方 v3 还完成了 21-token、100% 接受率的兼容性 smoke，短窗口达到 120.66 tok/s；
这不是稳定长窗口吞吐，也不是 v3 量化本身带来大幅加速的证据。
早期固定 512 输入、256 timed 输出的容量测试如下，每档一轮 warmup 加三轮测量：

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 8192 | 97.65 ± 0.08 | 1369.9 | 70.04% |
| 16384 | 97.67 ± 0.09 | 1370.9 | 70.04% |
| 32768 | 97.76 ± 0.11 | 1370.8 | 70.04% |
| 65536 | 97.55 ± 0.15 | 1369.9 | 70.04% |

### 早期 Q4_K_M 与 LM Studio

Q4_K_M、TP2、INT8 KV、180K 容量、prefill chunk=4096：8192 输入的一次冷 prefill
为 1672.9 tok/s，85000 输入的两轮均值为 1251.44 ± 2.87 tok/s。
85K 输入、MTP3、optimized draft head、CUDA Graph 的两个 128-token decode 窗口测得
**50.68 ± 0.04 tok/s**，MTP 接受率 78.07%。这不是 P2P 开启后的 Q4 decode 测量。

旧容量对照固定 512 输入、256 timed 输出，每档一轮 warmup、三轮测量，greedy、Q8/INT8 KV、
MTP 最大三最小零。NInfer 使用 TP2，LM Studio CUDA 2.33.0 使用自动双卡分配：

| 最大容量 | NInfer Q4_K_M decode tok/s | LM Studio Q4_K_M decode tok/s |
|---:|---:|---:|
| 1024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

同一旧 85K 提示、Q4_K_M、180K 容量的 LM Studio 单次诊断为 35.4977 tok/s，
后端将容量取整到 180224；它不是重复对照。用户先前报告的 45/57 tok/s 缺完整测量元数据，
不当作本轮实测基线，更不能用 Q4_K_M 的 LM Studio 结果证明 NVFP4 的量化质量相同。

### DFlash2 实验与数值限制

独立五层 BF16 DFlash2 实验在 85K 输入、98304 容量下：DFlash3 为 25.19 tok/s、
81.21% 接受率，DFlash7 为 20.52 tok/s、41.56% 接受率且后段出现重复特殊消息。
对应 MTP3 控制组为 52.77 tok/s。该 DFlash 180K 配置不能在本机两张 16 GB 卡上容纳，
不是当前 NVFP4 v3/MTP 推荐路线。

早期 72.28、75.53、78.90 等 NVFP4 成绩来自修复前的 SwiGLU 中间值路径。
独立 FP64 数学 oracle 检查暴露了 FP8/NVFP4 过早写回 BF16 的误差；修复为 FP32 中间值后，
原判据通过且未放宽容差。旧输出会改变，所以这些成绩不是当前合格数值基线。
更快但改变累加顺序的实验也不是交付路线。详见[数值与历史 profiler 说明](docs/performance.md#p2p-disabled)。

## 构建与模型

需要 64 位 Linux、NVIDIA 驱动、CUDA 12.8、CMake 3.28+、C++20、Ninja、pkg-config、
FFmpeg 开发库及 libcurl。依赖版本不足时可用仓库脚本构建私有依赖：

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

构建并发按本机情况调整，保留交互余量，整机 CPU 使用率不要超过约 85%。
模型、原始语料和 profiler 文件是本地前提，不随代码仓库提交；C++ 产品只读取 `.ninfer`。
NVFP4 可使用 [Neroued 官方容器](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)，
本机放在 `/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer`。

需要从本地 LM Studio GGUF 转换 Q4_K_M 时，用含 NumPy 的 Python 3.11 环境，例如本机 `.venv`：

```bash
.venv/bin/python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

GGUF 衍生身份只支持 Text/MTP，保留的 Vision 对象仅用于验证，不代表可以输入图像。

## CLI 和 API 启动

Q4_K_M 使用默认双卡脚本，180K 容量、4096-token chunk、INT8 KV、MTP3：

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "用 Python 实现一个有容量上限的任务队列，并解释测试方法。" \
  --max-new 512 --greedy --no-thinking
```

NVFP4 换模型并使用 **3072-token chunk**；本机 NVFP4、180K 容量时，4096-token chunk 不够显存：

```bash
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
NINFER_V100X2_PREFILL_CHUNK=3072 \
tools/v100/ninfer-v100x2.sh \
  --prompt "用 Python 实现一个有容量上限的任务队列，并解释测试方法。" \
  --max-new 512 --greedy --no-thinking
```

NVFP4、180K、MTP3 API 启动命令：

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-capacity 180000 --kv-dtype int8 --prefill-chunk 3072 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --default-max-tokens 65536 --no-thinking
```

`--default-max-tokens 65536` 是未指定输出长度时的默认值，仍受剩余上下文容量限制。
服务默认复用兼容前缀；`--no-prefix-reuse` 用于强制冷提示对照。
缓存只复用当前前沿或完整 checkpoint，不是无限 RAM KV 存储。
启动时固定能力与显存预算，请求不能临时开启未加载的执行能力。

OpenAI Chat Completions 请求示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"解释 prefill 与 decode 的区别。"}],"max_tokens":512,"temperature":0}'
```

还提供 `/v1/responses`、`/v1/messages`、`/health` 和 `/v1/models`。
流式输出、tool calls、鉴权及状态语义见[HTTP 文档](docs/serving.md)，具体参数以程序 `--help` 为准。

## 致谢与许可

基础来自 [Neroued/ninfer](https://github.com/Neroued/ninfer)、RTX 3060 TP2 路线和
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)。Volta NVFP4/TP2 行为参考并
对照过 [plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo)；prefill 调研参考
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。DFlash2 实验参考
[Inco AI](https://inco.ai/blog/dflash2/) 及其[草稿模型](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)。
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) 已调研但未集成：按查询检索部分历史会改变
完整上下文注意力语义，本配置没有悄悄替换成这种近似方案。

模型来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，NVFP4 的混合 FP8/NVFP4
权重来自 [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)，容器由
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 打包。

本项目采用 [Apache-2.0](LICENSE)；归属声明见 [NOTICE](NOTICE)，第三方依赖保留各自许可。
