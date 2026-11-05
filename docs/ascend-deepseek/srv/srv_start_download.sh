#!/bin/bash
# Start the 76 GB download in the background so the build can run in parallel.
set -uo pipefail
mkdir -p /data/models /data/logs
LOG=/data/logs/download.log
if pgrep -f "modelscope download" > /dev/null 2>&1; then
    echo "download already running (pid $(pgrep -f 'modelscope download' | head -1))"
    tail -3 "$LOG" 2>/dev/null
    exit 0
fi

command -v modelscope > /dev/null || pip3 install -q -U modelscope 2>&1 | tail -2
echo "modelscope: $(modelscope --version 2>/dev/null | head -1)"

nohup modelscope download \
    --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
    --include 'IQ3_XXS/*' \
    --include 'mmproj-Qwen3.8-Flash-Next-BF16.gguf' \
    --local_dir /data/models/IQ3_XXS \
    > "$LOG" 2>&1 &
echo "started pid $!"
sleep 20
echo "--- first 15 lines of the log ---"
head -15 "$LOG" 2>/dev/null
echo "--- so far ---"
du -sh /data/models/IQ3_XXS 2>/dev/null
