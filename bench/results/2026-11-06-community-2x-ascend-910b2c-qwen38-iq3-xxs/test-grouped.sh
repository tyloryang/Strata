#!/bin/bash
set -e
cd /data/strata-npu-vector-v4-20261106
source ./env.sh
trap 'echo $? > test-grouped.exit' EXIT
./build-grouped/grouped_matvec_parity > grouped-parity.log 2>&1
./build-grouped/grouped_matvec_parity --types IQ2_XXS,IQ2_XS,IQ3_XXS,IQ3_S,IQ2_S,Q2_0,IQ4_NL --rows 640 --cols 2560 --selected 10 --input shared --bench --warmup 3 --repeats 15 --deferred > grouped-bench-gate.log 2>&1
./build-grouped/grouped_matvec_parity --types IQ2_XXS,IQ2_XS,IQ3_XXS,IQ3_S,IQ2_S,Q2_0,IQ4_NL --rows 2560 --cols 640 --selected 10 --input per-rank --bench --warmup 3 --repeats 15 --deferred > grouped-bench-down.log 2>&1
export STRATA_NPU_ENGINE=./model_seq-grouped STRATA_CANN_DENSE_CACHE=1 STRATA_CANN_DEFERRED=1 STRATA_CANN_GROUPED_EXPERTS=1
bash ./run-case.sh grouped-r1 1 8
bash ./run-case.sh grouped-r2 1 8
bash ./run-case.sh grouped-r3 1 8
bash ./run-resident.sh grouped-resident ./model_seq-grouped 1
