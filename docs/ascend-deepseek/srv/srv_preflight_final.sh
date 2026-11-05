#!/bin/bash
echo "== progress =="
find /data/models/IQ3_XXS -name '*.gguf*' -printf '%12s  %p\n' 2>/dev/null
echo
echo "== disk headroom for the rename (same filesystem, so the rename is free) =="
df -h /data | tail -1
echo
echo "== what the scripts will pick up once complete =="
echo "serve  : $(find /data/models/IQ3_XXS -name '*-00001-of-00002.gguf' | head -1)"
echo "measure: $(find /data/models/IQ3_XXS -name '*-00001-of-00002.gguf' ! -name '*.incomplete' | head -1)"
echo
echo "== downloader still alive? =="
pgrep -f 'modelscope download' > /dev/null && echo "  yes" || echo "  no (finished or died)"
