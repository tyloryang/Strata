#!/bin/bash
total() {
    find /data/models/IQ3_XXS -name '*.gguf' -o -name '*.gguf.incomplete' 2>/dev/null \
        | xargs -r stat -c %s 2>/dev/null | awk '{s+=$1} END {print s+0}'
}
b=$(total)
sleep 120
a=$(total)
delta=$(( a - b ))
echo "delta over 120 s: $(( delta / 1024 / 1024 )) MB"
echo "sustained rate  : $(( delta / 120 / 1024 )) KB/s"
echo "progress        : $(( a / 1000000 )) MB of 75800 MB"
if [ "$delta" -gt 0 ]; then
    echo "eta             : $(( (75800000000 - a) / (delta / 120) / 60 )) min"
else
    echo "eta             : stalled"
fi
