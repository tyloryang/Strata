#!/bin/bash
echo "== binaries =="
ls /data/build-cann/bin | grep -E 'llama-cli|llama-server|llama-bench|llama-quantize' || echo "(missing)"
echo
echo "== does it link against CANN? =="
ldd /data/build-cann/bin/llama-cli 2>/dev/null | grep -E 'ascend|nnopbase|opapi' | head -5
echo
echo "== CANN devices seen by llama.cpp =="
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
/data/build-cann/bin/llama-cli --list-devices 2>&1 | head -12
echo
echo "== download progress =="
tail -2 /data/logs/download.log 2>/dev/null
du -sh /data/models/IQ3_XXS 2>/dev/null
find /data/models/IQ3_XXS -name '*.gguf' -printf '%s %p\n' 2>/dev/null | awk '{printf "  %.1f GB  %s\n", $1/1e9, $2}'
df -h /data | tail -1
