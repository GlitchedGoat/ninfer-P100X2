# NInfer PyQt 控制台

`tools/v100/ninfer_gui.py` 是一个本地 PyQt 控制台。它只管理由窗口启动的
`ninfer-serve` 子进程，不修改驱动、IOMMU、GRUB 或系统服务。窗口关闭时会询问是否
停止它自己启动的 API；已经由其他终端启动的 API 不会被杀掉。

## 安装

在有图形桌面的机器上使用 Python 3.10+，安装 PyQt6（已有 PyQt5 也可以）：

```bash
python3 -m venv .venv-gui
.venv-gui/bin/python -m pip install -r tools/v100/requirements-gui.txt
```

如果系统已经提供 PyQt6/PyQt5，可以直接把 `PYTHON` 指向该解释器。不要把 PyQt 放进
推理运行时的 Python 环境；GUI 只启动 C++ `ninfer-serve`。

模型入口

GUI 优先查找仓库旁的 `models/` 软链，因此不会复制或纳入 Git 中的几十 GB artifact。
本机可以这样建立入口（目标目录中已有的文件保持原样）：

```bash
ln -s /Models/ninfer-V100X2 models
```

客户机若 artifact 位于用户目录，则对应使用：

```bash
ln -s /home/z/Models/ninfer models
```

软链目标必须包含所选预设对应的 `.ninfer` 文件；GUI 会在启动前检查文件是否存在，
不会把一个精度的 artifact 当作另一个精度使用。

## 启动

```bash
bash tools/v100/ninfer-gui.sh
```

也可以显式指定仓库、二进制和模型：

```bash
bash tools/v100/ninfer-gui.sh \
  --repo /home/z/ninfer-V100X2 \
  --binary /home/z/ninfer-V100X2/build-v100-tp4/apps/ninfer-serve \
  --artifact /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer
```

客户机若使用独立 artifact 路径，只需在“模型 artifact”框中选择该文件。GUI 会记住
非敏感的路径、端口和推理参数；API key 不写入设置文件，只在当前窗口运行期间使用。

## 预设

- **TP4 · NVFP4 · MTP3 · 180K**：四张 V100，设备 `0,1,2,3`，INT8 KV，chunk 2560，
  优化草稿头。
- **TP4 · Native FP8 · MTP3 · 180K**：四张 V100 的 native block-128 FP8 artifact。
- **TP2 · NVFP4 · DFlash7 · 98K**：已验证的 DFlash7 配置，设备 `0,1`，chunk 1024。
- TP2 NVFP4/Q4_K_M MTP 预设以及“自定义”。

当前产品约束会在启动前阻止 TP4+DFlash、native FP8 非 TP4、设备数量与 TP 不一致、
非法 KV 容量和草稿数等组合。参数框生成的完整命令会显示在窗口中，可复制到终端复现。

## 远程主机

GUI 本身是本地桌面程序。若客户需要控制远端主机，应在远端图形会话中运行它，或使用
SSH X11/远程桌面；不要让 GUI 直接暴露服务端口。API 默认绑定 loopback。也可以只在
GUI 中复制命令，再通过受控 SSH 会话启动。当前四卡验收主机的命令是：

```bash
bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer
```

GUI 的“health”按钮只请求配置的 `/health` 地址，不会停止外部进程。
