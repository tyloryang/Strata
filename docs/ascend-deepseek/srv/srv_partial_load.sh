#!/bin/bash
# Does the loader accept this architecture and these metadata keys? It will fail when it reads
# past the end of the partial file, but everything before that tell us the model is understood.
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
F=$(find /data/models/IQ3_XXS -name '*00001-of-00002.gguf*' | head -1)
echo "file: $F ($(stat -c %s "$F") bytes)"
echo
timeout 180 /data/build-cann/bin/llama-cli -m "$F" -ngl 0 -c 128 -n 1 -p "hi" < /dev/null > /tmp/partial_load.log 2>&1
echo "rc=$? (124 = we cut it off, which is fine)"
echo
echo "== what the loader understood =="
grep -iE 'arch|n_layer|n_embd|n_head|expert|qwen|params|vocab|error|failed|not supported|unknown' /tmp/partial_load.log | head -25
echo
echo "== first 25 lines of the log =="
head -25 /tmp/partial_load.log
