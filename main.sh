#!/usr/bin/env bash
set -e

PY=python3

# ---- SETTING --------
ANN="examples/annotations.json"     # annotation JSON path
CROPS="./crops"                       # directory to save crops
DEVICE="cuda:0"                     # device
OUT="./vqa_scores.json"               # vqa output json
RESULT="./lvqascore.txt"            # l-vqascore final result
# ---------------------

echo "=== Step 1: Segment ==="
$PY src/segment.py \
    --annotation-file "$ANN" \
    --output-dir "$CROPS" \
    --device "$DEVICE" 

echo "=== Step 2: L-VQAScore ==="
$PY src/l-vqascore.py \
    --annotation-file "$ANN" \
    --sam-dir "$CROPS" \
    --output-json "$OUT" \
    --device "$DEVICE" \
    --result-file "$RESULT"

echo "Done! Results saved to $OUT and $RESULT"
