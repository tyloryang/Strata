#!/bin/bash
if pgrep -f status_page.py > /dev/null 2>&1; then
    echo "page: running (pid $(pgrep -f status_page.py | head -1))"
else
    echo "page: NOT running, starting"
    nohup python3 /data/status_page.py > /data/logs/status_page.log 2>&1 &
    sleep 2
fi
echo "--- /api/status ---"
curl -s -m 5 http://127.0.0.1:8188/api/status | python3 -c '
import json,sys
d=json.load(sys.stdin)
dl=d["download"]
print("host      ", d["host"])
print("bytes     ", round(dl["bytes"]/1e9,2), "GB of", round(dl["expected"]/1e9,1), "GB  =", dl["percent"], "%")
print("done      ", dl["done"])
print("rate      ", round(dl.get("rate_bps",0)/1e6,1), "MB/s")
print("eta       ", round(dl.get("eta_s",0)/60), "min")
for f in dl["files"]:
    print("   ", f["name"][:52], round(f["bytes"]/1e9,2), "GB", "complete" if f["complete"] else "partial")
print("service   ", "up" if d["service"]["up"] else "down")
for c in d["npus"]:
    print("npu", c["id"], c["name"], c.get("util_pct"), "%", round(c.get("hbm_used_mb",0)), "/", round(c.get("hbm_total_mb",0)), "MB")
'
