#!/bin/bash
# serve the status page on 127.0.0.1:8188, the port the operator's tunnel forwards
set -uo pipefail
if pgrep -f 'status_page.py' > /dev/null 2>&1; then
    echo "already running (pid $(pgrep -f status_page.py | head -1))"
else
    nohup python3 /data/status_page.py > /data/logs/status_page.log 2>&1 &
    echo "started pid $!"
    sleep 2
fi
echo "--- log ---"
tail -3 /data/logs/status_page.log 2>/dev/null
echo "--- listening ---"
grep -E ':4B38|:1FFC' /proc/net/tcp 2>/dev/null | head -3 || ss -ltnp 2>/dev/null | grep 8188
echo "--- self test ---"
curl -s -m 5 http://127.0.0.1:8188/api/status | head -c 600
echo
echo "--- page head ---"
curl -s -m 5 http://127.0.0.1:8188/ | head -c 200
