#!/bin/bash
export MODEL_DIR=/data/models
export LD_LIBRARY_PATH=/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
# the script looks for a two-shard name, so point it at a symlink that matches
rm -f /data/models/qwen06-00001-of-00002.gguf
ln -s /data/models/qwen3-0.6b-f16.gguf /data/models/qwen06-00001-of-00002.gguf
bash /data/measure_qwen38.sh 2048 32
