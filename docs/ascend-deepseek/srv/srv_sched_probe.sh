#!/bin/bash
# What the scheduler actually puts on the NPU and what it leaves on the CPU.
# GGML_SCHED_DEBUG=2 prints every split; this shows the shape of that output so we can parse it.
set -uo pipefail
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
export GGML_SCHED_DEBUG=2

/data/build-cann/bin/llama-cli -m /data/models/qwen3-0.6b-f16.gguf -ngl 99 -c 512 -n 8 \
    -p "Hello" < /dev/null > /tmp/sched.log 2>&1
echo "rc=$?  lines=$(wc -l < /tmp/sched.log)"
echo
echo "=== lines that mention a backend or a split ==="
grep -nE 'sched|split|CANN0|CPU' /tmp/sched.log | head -40
