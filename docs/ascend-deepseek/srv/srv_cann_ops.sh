#!/bin/bash
H=/usr/local/Ascend/cann-9.0.0/include/aclnnop
echo "== aclnn ops that could stand in for the five missing ones =="
for name in aclnnTopk aclnnTopK aclnnSort aclnnIndexSelect aclnnGather aclnnMaskedFill aclnnMoeGatingTopK \
            aclnnSoftmax aclnnScatter aclnnRepeatInterleave aclnnCumsum aclnnArange; do
  f="$H/${name}.h"
  if [ -f "$f" ]; then echo "  yes  ${name}"; else echo "  no   ${name}"; fi
done
echo
echo "== aclnnTopk signature =="
if [ -f "$H/aclnnTopk.h" ]; then
  grep -A14 'aclnnTopkGetWorkspaceSize' "$H/aclnnTopk.h" | grep -E 'aclnnStatus|GetWorkspaceSize|const|int64_t|bool' | head -16
fi
echo
echo "== aclnnMoeGatingTopK (the MoE top-k) =="
if [ -f "$H/aclnnMoeGatingTopK.h" ]; then
  grep -A12 'aclnnMoeGatingTopKGetWorkspaceSize' "$H/aclnnMoeGatingTopK.h" | head -14
fi
echo
echo "== anything with topk or indexer in the name =="
ls "$H" | grep -i -E 'topk|top_k|indexer|lightning' || echo "  (none beyond the above)"
