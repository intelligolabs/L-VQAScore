#!/usr/bin/env bash
set -e
PY=python3

#==============================================================================
ANN="..examples/annoations.json"                 # path to JSON annotation file
CROPS="./crops"                        # directory to save SAM-segmented images
DEVICE="cuda:0"                                                        # device
OUT="./vqa_scores.json"                                       # vqa output json
RESULT="./lvqascore.txt"                              # l-vqascore final result
#==============================================================================

echo "=== L-VQAScore ==="
$PY src/l-vqascore.py \
    --annotation-file "$ANN" \
    --sam-dir "$CROPS" \
    --output-json "$OUT" \
    --device "$DEVICE" \
    --result-file "$RESULT"
