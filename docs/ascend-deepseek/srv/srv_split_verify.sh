#!/bin/bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
M=/data/models/qwen3-0.6b-f16.gguf

echo "########## single device ##########"
timeout 300 /data/build-cann/bin/llama-cli -m "$M" -ngl 99 -c 512 -n 8 --seed 1 -p "Hello" > /tmp/one.log 2>&1
grep -E 'load_tensors|offload|CANN0|CANN1|buffer size' /tmp/one.log | head -8

echo
echo "########## two devices, -ts 0.5,0.5 ##########"
timeout 300 /data/build-cann/bin/llama-cli -m "$M" -ngl 99 --split-mode layer -ts 0.5,0.5 \
    -c 512 -n 8 --seed 1 -p "Hello" > /tmp/two.log 2>&1
grep -E 'load_tensors|offload|CANN0|CANN1|buffer size' /tmp/two.log | head -12

echo
echo "########## device assignment lines ##########"
grep -o 'CAN[N0-9]*' /tmp/two.log | sort | uniq -c
