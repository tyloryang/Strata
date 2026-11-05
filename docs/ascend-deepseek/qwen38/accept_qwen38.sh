#!/bin/bash
# What "it runs" has to mean before we call this done.
set -uo pipefail
P="${PORT:-8080}"
H="http://127.0.0.1:$P"

echo "== health =="
curl -s -m 10 "$H/health" && echo

echo
echo "== models =="
curl -s -m 10 "$H/v1/models" | head -c 400; echo

echo
echo "== one chat turn (timed) =="
t0=$(date +%s.%N)
curl -s -m 900 "$H/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "用一句话说明什么是混合专家模型"}],
  "max_tokens": 96,
  "temperature": 0
}' | python3 -c '
import json, sys
d = json.load(sys.stdin)
c = d["choices"][0]
msg = c.get("message") or {}
text = msg.get("content") or ""
think = msg.get("reasoning_content") or msg.get("reasoning") or ""
print("finish_reason:", c.get("finish_reason"))
print("content  :", (text or "(empty)")[:300])
if think:
    print("thinking :", think[:300])
print("usage:", d.get("usage"))
t = d.get("timings") or {}
print("timings:", {k: round(v, 2) for k, v in t.items() if isinstance(v, (int, float))})
if not text and think:
    print("NOTE: the reply went to reasoning_content; raise max_tokens to see a final answer")
'
t1=$(date +%s.%N)
echo "wall: $(echo "$t1 - $t0" | bc) s"

echo
echo "== both cards busy? =="
npu-smi info 2>/dev/null | grep -A4 'Process id'
