#!/bin/bash
H=/usr/local/Ascend/cann-9.0.0/include/aclnnop
echo "############ aclnn_lightning_indexer.h ############"
grep -E 'aclnnStatus|GetWorkspaceSize|^\s+(const|int64_t|bool|char|uint)' "$H/aclnn_lightning_indexer.h" | head -30
echo
echo "############ aclnn_topk.h ############"
grep -E 'aclnnStatus|GetWorkspaceSize|^\s+(const|int64_t|bool)' "$H/aclnn_topk.h" | head -25
echo
echo "############ how many aclnn headers exist in total ############"
ls "$H" | wc -l
echo
echo "############ does libopapi export the two symbols? ############"
for sym in aclnnLightningIndexer aclnnTopk aclnnMoeGatingTopK; do
  if nm -D --defined-only /usr/local/Ascend/cann-9.0.0/lib64/libopapi.so 2>/dev/null | grep -q "$sym"; then
    echo "  exported: $sym"
  else
    echo "  MISSING : $sym"
  fi
done
