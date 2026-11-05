#!/bin/bash
# Build llama.cpp with the CANN backend on this x86_64 box.
#
# Two things this box needs that a stock image does not have:
#   * aclnn_fused_infer_attention_score_v2.h - the aclnn headers here are CANN 9.0.0, which predates
#     the fused attention op that current ggml-cann calls, even though libopapi.so already exports
#     it. The header is copied in from the CANN ops-transformer repository.
#   * the driver library directory on LD_LIBRARY_PATH, or the built binary cannot start.
set -euo pipefail

SRC=/data/llama.cpp-master
BUILD=/data/build-cann
CANN=/usr/local/Ascend/cann-9.0.0

echo "arch: $(uname -m)   cores: $(nproc)"

export ASCEND_TOOLKIT_HOME="$CANN"
export ASCEND_HOME_PATH="$CANN"
export LD_LIBRARY_PATH="$CANN/lib64:$CANN/x86_64-linux/lib64"
for d in /usr/local/Ascend/driver/lib64 /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64/driver; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"
done

echo "== driver library =="
if [ -f "$CANN/include/aclnnop/aclnn_fused_infer_attention_score_v2.h" ]; then
    echo "  fused attention header: present"
else
    echo "  fused attention header: MISSING - the build will fail"
    exit 1
fi
find /usr/local/Ascend -maxdepth 5 -name 'libascend_hal.so' 2>/dev/null | head -3

echo "== source =="
if [ ! -d "$SRC" ]; then
    tar -xzf /data/llama.cpp-master.tar.gz
fi
grep -q LLM_ARCH_QWEN4EXP "$SRC/src/llama-arch.cpp" && echo "  qwen4exp: yes" || { echo "  qwen4exp: MISSING"; exit 1; }

echo "== configure =="
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=ON \
    -DLLAMA_CURL=OFF \
    2>&1 | tail -18

echo "== build =="
cmake --build "$BUILD" -j "$(nproc)" > /data/logs/build.log 2>&1
rc=$?
echo "build rc=$rc"
if [ $rc -ne 0 ]; then
    grep -m 10 -E 'error:' /data/logs/build.log
    exit $rc
fi
ls -la "$BUILD/bin" | head -12
