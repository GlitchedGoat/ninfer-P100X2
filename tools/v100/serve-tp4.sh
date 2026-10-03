#!/usr/bin/env bash
set -euo pipefail

# Friend-host Text/MTP deployment. No automatic restart or system configuration
# changes: stop this process before running a four-card benchmark or switching artifacts.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
    printf 'usage: bash tools/v100/serve-tp4.sh /explicit/model.ninfer [server options]\n'
    printf 'Defaults: devices 0,1,2,3; loopback port 6200; capacity 180000; INT8 KV; MTP3; optimized head; max output 65536.\n'
    exit 0
fi
artifact=$1
shift
export LD_LIBRARY_PATH="${repo_dir}/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${repo_dir}/build-v100-tp4/apps/ninfer-serve" "$artifact" \
    --tp 4 --devices 0,1,2,3 --host 127.0.0.1 --port 6200 \
    --max-context 180000 --kv-dtype int8 --prefill-chunk 2560 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 1 --default-max-tokens 65536 \
    "$@"
