#!/bin/bash
# The downloader writes *.gguf.incomplete and renames on completion, so count those too.
total() {
    find /data/models/IQ3_XXS -name '*.gguf' -o -name '*.gguf.incomplete' 2>/dev/null \
        | xargs -r stat -c %s 2>/dev/null | awk '{s+=$1} END {print s+0}'
}
b=$(total)
sleep 20
a=$(total)
rate=$(( (a - b) / 20 / 1024 ))
echo "now: $(( a / 1000000000 )) GB of 75.8 GB"
echo "rate: ${rate} KB/s over 20 s"
[ "$rate" -gt 0 ] && echo "eta: $(( (75800 - a / 1000000) / (rate / 1024) / 60 )) min" || echo "eta: stalled"
echo
find /data/models/IQ3_XXS -name '*.gguf*' -printf '%10s  %f\n' 2>/dev/null
