#!/bin/bash
set -e
cd /data/strata-npu-vector-v4-20261106
source ./env.sh
label=${1:?label}; engine=${2:?engine}; deferred=${3:-0}
trap 'echo $? > "$label.exit"' EXIT
export STRATA_CANN_DEVICES=0,1 STRATA_CANN_PACKED_WEIGHTS=1 STRATA_CANN_FUSED_MATVEC=1
export STRATA_CANN_DENSE_CACHE=1 STRATA_CANN_DEFERRED=$deferred STRATA_BUDGET_GB=3
unset STRATA_CANN_DEQUANT_STATS STRATA_CANN_MATVEC_STATS
python3 bench_resident.py --engine "$engine" --model /data/models/IQ3_XXS/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out "$label" --cases resident-cases.json
