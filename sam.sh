#!/usr/bin/env bash
set -e
PY=python3

#==============================================================================
ANN="examples/annoations.json"                   # path to JSON annotation file
CROPS="../crops"                       # directory to save SAM-segmented images
DEVICE="cuda:0"                                                        # device
#==============================================================================

$PY ../src/segment.py \
    --annotation-file "$ANN" \
    --output-dir "$CROPS" \
    --device "$DEVICE" \
