#!/bin/bash
# Does llama.cpp actually spread layers over both NPUs? A 0.6B fits on one card, so this is the
# only way to check the split path before the 47 GB model depends on it.
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
M=/data/models/qwen3-0.6b-f16.gguf

echo "== before: memory per card =="
npu-smi info 2>/dev/null | grep -o '[0-9]*/ 65536' | head -2

echo
echo "== single card (baseline) =="
timeout 300 /data/build-cann/bin/llama-cli -m "$M" -ngl 99 -c 1024 -n 16 --seed 1 -p "Hello" 2>&1 | tail -3

echo
echo "== two cards, layer split 50/50 =="
timeout 300 /data/build-cann/bin/llama-cli -m "$M" -ngl 99 --split-mode layer -ts 0.5,0.5 \
    -c 1024 -n 16 --seed 1 -p "Hello" 2>&1 | grep -E 'load_tensors|offloaded|assigned|Graph|Prompt|Generation' | head -12

echo
echo "== what the devices reported while it ran =="
timeout 300 /data/build-cann/bin/llama-cli -m "$M" -ngl 99 --split-mode layer -ts 0.5,0.5 \
    -c 1024 -n 400 --seed 1 -p "$(printf 'word%.0s ' {1..200})" > /tmp/split.log 2>&1 &
sleep 25
npu-smi info 2>/dev/null | grep -A6 'Process id'
wait
echo "(generation finished)"
