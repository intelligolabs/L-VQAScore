#!/usr/bin/env bash
set -e

PY=python3

#=====================================================================================
ANN="/home/zliu/github/L-VQAScore/examples/annoations.json"     # annotation JSON path
CROPS="./crops"                       # directory to save SAM segmentation images
DEVICE="cuda:0"                     # device
OUT="./vqa_scores.json"               # vqa output json
RESULT="./lvqascore.txt"            # l-vqascore final result
#=====================================================================================

echo "=== L-VQAScore ==="
$PY src/l-vqascore.py \
    --annotation-file "$ANN" \
    --sam-dir "$CROPS" \
    --output-json "$OUT" \
    --device "$DEVICE" \
    --result-file "$RESULT"
