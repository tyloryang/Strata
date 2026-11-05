#!/bin/bash
F=/tmp/dump1.txt
echo "== tensor name suffixes (blk.N. removed) =="
grep -E '^blk\.[0-9]+\.' "$F" | sed -E 's/^blk\.[0-9]+\.//' | awk '{print $1}' | sort | uniq -c | sort -rn
echo
echo "== non-blk tensors =="
grep -vE '^blk\.[0-9]+\.' "$F" | grep -E '^[a-z]' | head -30
echo
echo "== one block, all tensors =="
grep -E '^blk\.0\.' "$F"
echo
echo "== block 2 (the PLE layer) =="
grep -E '^blk\.2\.' "$F" | head -20
echo
echo "== block 3 (a full-attention layer) =="
grep -E '^blk\.3\.' "$F"
