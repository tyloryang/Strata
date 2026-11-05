#!/bin/bash
set -uo pipefail
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"

echo "== what is in the mount =="
ls /mnt/moark-models/Qwen3-0.6B
echo "---"
ls /mnt/moark-models/Qwen3.6-35B-A3B | head -8

OUT=/data/models/qwen3-0.6b-f16.gguf
if [ ! -f "$OUT" ]; then
    echo "== converting 0.6B =="
    cd /data/llama.cpp-master || exit 1
    python3 convert_hf_to_gguf.py /mnt/moark-models/Qwen3-0.6B --outfile "$OUT" --outtype f16 > /data/logs/convert06.log 2>&1
    rc=$?
    echo "convert rc=$rc"
    [ $rc -ne 0 ] && { tail -12 /data/logs/convert06.log; exit $rc; }
fi
ls -la "$OUT"

echo
echo "== NPU run (0.6B) =="
timeout 600 /data/build-cann/bin/llama-cli -m "$OUT" -ngl 99 -c 1024 -n 32 --seed 1 \
    -p "The capital of France is" 2>&1 | tail -20

echo
echo "== download progress =="
tail -1 /data/logs/download.log 2>/dev/null | tr -d '\r'
du -sh /data/models/IQ3_XXS 2>/dev/null
