#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
NInfer PyQt control console

Usage:
  ninfer-gui.sh [--repo PATH] [--binary PATH] [--artifact PATH]

The launcher selects .venv-gui/bin/python when present, otherwise python3.
The Python interpreter must provide PyQt6 or PyQt5.
EOF
    exit 0
fi
if [[ -n "${PYTHON:-}" ]]; then
    gui_python=${PYTHON}
elif [[ -x "${repo_dir}/.venv-gui/bin/python" ]]; then
    gui_python="${repo_dir}/.venv-gui/bin/python"
else
    gui_python=python3
fi
if ! command -v "${gui_python}" >/dev/null 2>&1 || ! "${gui_python}" -c \
    'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("PyQt6") or importlib.util.find_spec("PyQt5") else 1)' \
    >/dev/null 2>&1; then
    echo "未找到 PyQt6/PyQt5（当前解释器：${gui_python}）。请按 docs/gui.md 安装，或设置 PYTHON。" >&2
    exit 1
fi
exec "${gui_python}" "${script_dir}/ninfer_gui.py" --repo "${repo_dir}" "$@"
