#!/bin/bash
F=IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
echo "== modelscope =="
timeout 40 curl -sL -r 0-104857599 -o /dev/null -w '  speed=%{speed_download} B/s\n' \
    "https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/master/$F"
echo "== hf-mirror =="
timeout 40 curl -sL -r 0-104857599 -o /dev/null -w '  speed=%{speed_download} B/s\n' \
    "https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/$F"
echo "== huggingface =="
timeout 40 curl -sL -r 0-104857599 -o /dev/null -w '  speed=%{speed_download} B/s\n' \
    "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/$F"
