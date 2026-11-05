#!/bin/bash
echo "== tree of the model dir =="
find /data/models/IQ3_XXS -maxdepth 2 -printf '%10s  %p\n' 2>/dev/null | sort -k2 | head -20
echo
echo "== anything over 1 GB written in the last hour, anywhere =="
find / -xdev -type f -size +1G -mmin -70 -printf '%10s  %p\n' 2>/dev/null | head -10
echo
echo "== big dirs on / and /data =="
du -sh /tmp /var/tmp /root/.cache 2>/dev/null
du -sh /data/* 2>/dev/null | sort -h | tail -5
echo
echo "== disk pressure =="
df -h / /data | grep -v Filesystem
