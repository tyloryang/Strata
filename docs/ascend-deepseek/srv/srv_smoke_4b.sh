#!/bin/bash
# Validate the CANN backend end to end with a model we already have, while the big download runs.
# /mnt/moark-models/Qwen3.5-4B is safetensors, so it is converted to GGUF first.
set -uo pipefail
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"

SRC=/mnt/moark-models/Qwen3.5-4B
OUT=/data/models/qwen35-4b-f16.gguf

echo "== source =="
ls "$SRC" | head -6
du -sh "$SRC"

if [ ! -f "$OUT" ]; then
    echo "== converting =="
    python3 -m pip install -q -U transformers safetensors numpy sentencepiece protobuf 2>&1 | tail -2
    cd /data/llama.cpp-master || exit 1
    python3 convert_hf_to_gguf.py "$SRC" --outfile "$OUT" --outtype f16 > /data/logs/convert.log 2>&1
    rc=$?
    echo "convert rc=$rc"
    [ $rc -ne 0 ] && { tail -15 /data/logs/convert.log; exit $rc; }
fi
ls -la "$OUT"

echo
echo "== run on the NPU =="
timeout 900 /data/build-cann/bin/llama-cli \
    -m "$OUT" -ngl 99 -c 2048 -n 24 --seed 1 -no-cnv \
    -p "The capital of France is" 2>&1 | tail -25
