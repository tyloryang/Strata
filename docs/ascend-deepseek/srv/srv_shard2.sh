#!/bin/bash
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
F=$(find /data/models/IQ3_XXS -name '*00002-of-00002.gguf*' | head -1)
echo "shard2: $F"
/data/build-strata/gguf_dump "$F" | grep -E '^(file|version|arch|tensors)|split\.|-- types|^[a-z_]+\.' | head -30
echo
echo "=== the tensors shard 1 reported missing ==="
for t in output_norm.weight per_layer_tok_embd.weight ple_key.weight ple_value.weight ple_conv1d.weight ple_norm_key.weight ple_norm_query.weight ple_norm_conv.weight; do
  found=$(/data/build-strata/gguf_dump "$F" "x" 2>/dev/null | grep -c "^$t ")
  if [ "$found" -gt 0 ]; then echo "  shard2 has $t"; else echo "  shard2 lacks $t"; fi
done
echo
echo "=== first 15 tensors of shard 2 ==="
/data/build-strata/gguf_dump "$F" | sed -n '/-- tensors/,$p' | head -18
