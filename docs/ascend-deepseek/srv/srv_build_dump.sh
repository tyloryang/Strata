#!/bin/bash
set -uo pipefail
cd /data || exit 1
[ -d /data/strata-cann ] && rm -rf /data/strata-cann
mkdir -p /data/strata-src && tar -xzf /data/cann-src.tar.gz -C /data/strata-src
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
cmake -S /data/strata-src/cann -B /data/build-strata -DCMAKE_BUILD_TYPE=Release > /tmp/strata_cmake.log 2>&1
echo "cmake rc=$?"
tail -3 /tmp/strata_cmake.log
cmake --build /data/build-strata --target gguf_dump -j 32 > /tmp/strata_build.log 2>&1
echo "build rc=$?"
grep -m 5 -E 'error:' /tmp/strata_build.log
ls -la /data/build-strata/gguf_dump 2>/dev/null
