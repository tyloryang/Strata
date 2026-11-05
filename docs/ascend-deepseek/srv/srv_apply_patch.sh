#!/bin/bash
# apply the patched ggml-cann sources into the build tree and rebuild
set -uo pipefail
cd /data || exit 1
tar -xzf /data/llamacpp-patch.tar.gz -C /data/llama.cpp-master
echo "extracted"
grep -c 'GGML_OP_TOP_K' /data/llama.cpp-master/ggml/src/ggml-cann/ggml-cann.cpp
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
cmake --build /data/build-cann -j 64 > /data/logs/build2.log 2>&1
echo "build rc=$?"
grep -m 8 -E 'error:' /data/logs/build2.log
ls -la /data/build-cann/bin/test-backend-ops 2>/dev/null || echo "(no test-backend-ops)"
