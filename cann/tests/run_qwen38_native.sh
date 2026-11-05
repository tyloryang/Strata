#!/usr/bin/env bash
# Native Strata C++/CANN execution on the original two-shard IQ3_XXS model.
# Usage: STRATA_CANN_DEVICE=1 bash cann/tests/run_qwen38_native.sh shard1.gguf [token_ids] [n_predict]
set -e
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
set -u -o pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$#" -lt 1 ]; then
    echo "usage: $0 <00001-of-00002.gguf> [comma-separated token ids, default 785] [n_predict, default 8]" >&2
    exit 2
fi
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"
export STRATA_CANN_DEVICE="${STRATA_CANN_DEVICE:-0}"
export STRATA_BUDGET_GB="${STRATA_BUDGET_GB:-12}"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
cmake -S "$ROOT/cann" -B "$ROOT/build-cann" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build-cann" --target model_seq -j8
exec "$ROOT/build-cann/model_seq" "$1" "${2:-785}" "${3:-8}" 0
