#!/usr/bin/env bash
set -e

PY=python3

# ---- EDIT THESE ----
ANN="examples/annotations.json"     # annotation JSON path
CROPS="crops"                       # directory to save crops
DEVICE="cuda:0"                     # device
MODE="blur"                         # crop | blur
OUT="vqa_scores.json"               # output json
# ---------------------

echo "=== Step 1: Segment ==="
$PY src/segment.py \
    --annotation-file "$ANN" \
    --output-dir "$CROPS" \
    --device "$DEVICE" \
    --mode "$MODE"

echo "=== Step 2: L-VQAScore ==="
$PY src/l-vqascore.py \
    --annotation-file "$ANN" \
    --sam-dir "$CROPS" \
    --output-json "$OUT" \
    --device "$DEVICE" \
    --mode "$MODE"

echo "Done! Results saved to $OUT"
