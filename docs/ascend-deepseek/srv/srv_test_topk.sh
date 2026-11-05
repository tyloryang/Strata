#!/bin/bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
B=/data/build-cann/bin/test-backend-ops

echo "== does TOP_K now count as supported by CANN? =="
$B -o TOP_K -b CANN0 2>&1 | head -25
echo
echo "== backend list =="
$B 2>&1 | grep -iE 'devices|backends' | head -5
