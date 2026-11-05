#!/bin/bash
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
B=/data/build-cann/bin
echo "== flags this build accepts =="
for f in '--load-mode' '--lazy-mode' '--split-mode' '--tensor-split' '--jinja' '--flash-attn' '--no-mmap'; do
  if $B/llama-cli --help 2>&1 | grep -q -- "$f"; then echo "  ok    $f"; else echo "  MISS  $f"; fi
done
echo
echo "== load-mode values =="
$B/llama-cli --help 2>&1 | grep -A3 -- '--load-mode' | head -8
echo
echo "== lazy-mode values =="
$B/llama-cli --help 2>&1 | grep -A4 -- '--lazy-mode' | head -8
echo
echo "== download =="
tail -1 /data/logs/download.log 2>/dev/null | tr -d '\r'
du -sh /data/models/IQ3_XXS 2>/dev/null
df -h /data | tail -1
