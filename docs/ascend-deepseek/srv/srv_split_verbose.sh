#!/bin/bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
M=/data/models/qwen3-0.6b-f16.gguf

echo "== timed run with verbose logging, two devices =="
start=$(date +%s)
timeout 420 /data/build-cann/bin/llama-cli -v -m "$M" -ngl 99 --split-mode layer -ts 0.5,0.5 \
    -c 512 -n 4 --seed 1 -p "Hello" > /tmp/verb.log 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc  wall=$((end - start)) s"

echo
echo "== device mentions =="
grep -o 'CANN[0-9]*' /tmp/verb.log | sort | uniq -c
echo
echo "== tensor loading lines =="
grep -i -E 'load_tensors|offloading|buffer size|assigned' /tmp/verb.log | head -12
echo
echo "== timing lines =="
grep -E 'Prompt:|Generation:|load time' /tmp/verb.log | tail -3
