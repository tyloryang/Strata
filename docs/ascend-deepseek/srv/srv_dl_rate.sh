#!/bin/bash
echo "== per-file bytes =="
find /data/models/IQ3_XXS -name '*.gguf' -printf '%s %p\n' 2>/dev/null | sort -k2
echo
echo "== rate over 10 s =="
before=$(find /data/models/IQ3_XXS -name '*.gguf' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
sleep 10
after=$(find /data/models/IQ3_XXS -name '*.gguf' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
echo "  $(( (after - before) / 10 / 1024 )) KB/s   now $(( after / 1000000000 )) GB"
echo
echo "== downloader process =="
ps -o pid,etime,pcpu,rss,cmd -p "$(pgrep -f 'modelscope download' | head -1)" 2>/dev/null | tail -2
echo
echo "== incomplete files from modelscope cache =="
du -sh /root/.cache/modelscope 2>/dev/null | head -2
