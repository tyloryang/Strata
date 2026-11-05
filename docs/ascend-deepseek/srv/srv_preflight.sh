#!/bin/bash
echo "== host =="
cat /proc/sys/kernel/hostname 2>/dev/null; uptime | tr -s ' '
echo "== os =="
grep -E '^(PRETTY_NAME|VERSION_ID)' /etc/os-release 2>/dev/null
echo "== cpu/mem =="
nproc; free -g | head -2
echo "== disk =="
df -h 2>/dev/null | grep -vE 'tmpfs|overlay$' | head -12
echo "== npu =="
command -v npu-smi >/dev/null && npu-smi info 2>/dev/null | head -16 || echo "npu-smi MISSING"
echo "== cann =="
ls -d /usr/local/Ascend/cann-* 2>/dev/null || echo "no CANN dir"
for f in /usr/local/Ascend/cann-*/version.cfg; do [ -f "$f" ] && grep -m2 -i version "$f"; done
echo "== driver =="
ls -la /usr/local/Ascend/driver/lib64/libascend_hal.so 2>/dev/null || echo "no libascend_hal.so"
echo "== tools =="
for t in gcc g++ cmake git python3 pip3 ninja make curl; do printf '%-8s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
echo "== versions =="
gcc --version 2>/dev/null | head -1
cmake --version 2>/dev/null | head -1
python3 -V 2>/dev/null
echo "== python pkgs =="
python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)" 2>/dev/null || echo "no torch_npu"
python3 -c "import modelscope; print('modelscope', modelscope.__version__)" 2>/dev/null || echo "no modelscope"
echo "== net =="
for u in https://modelscope.cn https://pypi.org https://github.com https://gitcode.com; do
  if timeout 10 curl -sI "$u" >/dev/null 2>&1; then echo "  ok    $u"; else echo "  BLOCK $u"; fi
done
