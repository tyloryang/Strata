#!/bin/bash
if pgrep -f "modelscope download" > /dev/null 2>&1; then
    echo "download: RUNNING (pid $(pgrep -f 'modelscope download' | head -1))"
else
    echo "download: finished or stopped"
fi
echo "--- last log line ---"
tail -c 300 /data/logs/download.log 2>/dev/null | tr -d '\r' | tail -2
echo
echo "--- files ---"
find /data/models/IQ3_XXS -name '*.gguf' -printf '%10.2f GB  %f\n' 2>/dev/null | sort
echo
du -sh /data/models/IQ3_XXS 2>/dev/null
df -h /data | tail -1
