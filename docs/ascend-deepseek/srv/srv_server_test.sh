#!/bin/bash
# The final step is llama-server, not llama-cli. Prove that path works on this box first,
# with a small model, so the 47 GB load is not also the first server run.
set -uo pipefail
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH="/data/build-cann/bin:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"

M=/data/models/qwen3-0.6b-f16.gguf
PORT=8081
pkill -f "llama-server.*$PORT" 2>/dev/null
sleep 1

nohup /data/build-cann/bin/llama-server \
    -m "$M" -ngl 99 -c 2048 --host 127.0.0.1 --port $PORT --jinja \
    > /data/logs/server06.log 2>&1 &
echo "server pid $!"

# /health answers 503 "Loading model" while the weights stream in, so wait for a 200 code,
# not merely for the socket to accept
for i in $(seq 1 300); do
    code=$(curl -s -o /tmp/health.json -w '%{http_code}' -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
        echo "health 200 after ${i}s"
        break
    fi
    sleep 1
done
echo "last health code: ${code:-none}" 

echo "--- /health ---"
curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo
echo "--- /v1/models ---"
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo

echo "--- chat turn ---"
time curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话说明什么是张量"}],"max_tokens":64,"temperature":0}' \
  | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("text :", (d["choices"][0]["message"].get("content") or "")[:200])
t=d.get("timings") or {}
for k in ("prompt_n","prompt_ms","prompt_per_second","predicted_n","predicted_per_second"):
    if k in t: print(f"{k:22s} {t[k]}")
print("usage:", d.get("usage"))
'
echo "--- server log tail ---"
tail -5 /data/logs/server06.log
pkill -f "llama-server.*$PORT"
echo "(stopped)"
