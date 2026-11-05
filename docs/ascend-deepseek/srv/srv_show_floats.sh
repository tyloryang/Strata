#!/bin/bash
echo "== float metadata (the ones that were being lost) =="
sed -n '/-- float metadata/,/-- string metadata/p' /tmp/dump1.txt | head -12
echo
echo "== eps and rope, explicitly =="
grep -E 'epsilon|freq_base|heads_per_ngram|embedding_length_per_layer_input' /tmp/dump1.txt
