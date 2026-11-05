#!/bin/bash
echo "=== one.log (single device) ==="
sed -n '1,40p' /tmp/one.log
echo
echo "=== two.log (two devices) ==="
sed -n '1,40p' /tmp/two.log
