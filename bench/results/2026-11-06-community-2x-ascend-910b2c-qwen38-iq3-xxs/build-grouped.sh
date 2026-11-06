#!/bin/bash
set -e
cd /data/strata-npu-vector-v4-20261106
source ./env.sh
trap 'echo $? > build-grouped.exit' EXIT
cmake -S cann -B build-grouped -DCMAKE_BUILD_TYPE=Release -DSTRATA_CANN_NPU_ONLY=ON -DSTRATA_CANN_GGML_DIR=/data/llama.cpp-master -DSTRATA_CANN_GGML_LIB_DIR=/data/build-cann/bin
cmake --build build-grouped -j8
cp build-grouped/model_seq model_seq-grouped
