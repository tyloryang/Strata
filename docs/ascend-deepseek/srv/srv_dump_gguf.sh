#!/bin/bash
# The GGUF header and tensor index live at the front of the file, so this works on the partial
# download and tells us the exact tensor names and shapes before the weights arrive.
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
F=$(find /data/models/IQ3_XXS -name '*00001-of-00002.gguf*' | head -1)
echo "file: $F"
ls -la "$F"
echo
/data/build-strata/gguf_dump "$F" > /tmp/dump1.txt 2>&1
echo "rc=$?"
head -60 /tmp/dump1.txt
