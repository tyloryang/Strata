#!/bin/bash
# Can we fetch the shard faster with ranged requests than the modelscope CLI does?
BASE=https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/master
F1=IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
echo "== single ranged request, 100 MB =="
timeout 60 curl -sL -r 0-104857599 -o /dev/null -w 'http=%{http_code} size=%{size_download} speed=%{speed_download} B/s\n' \
    "$BASE/$F1"

echo "== head =="
timeout 30 curl -sIL "$BASE/$F1" 2>/dev/null | grep -i -E 'content-length|accept-ranges|location' | head -5

echo "== parallel 4 x 25 MB =="
start=$(date +%s.%N)
for i in 0 1 2 3; do
    s=$((i * 26214400)); e=$((s + 26214399))
    timeout 60 curl -sL -r ${s}-${e} -o /dev/null "$BASE/$F1" &
done
wait
end=$(date +%s.%N)
echo "elapsed $(echo "$end - $start" | bc) s for 100 MB across 4 connections"
