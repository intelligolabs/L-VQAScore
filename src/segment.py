#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline for item cropping and VQA evaluation.

This script processes a user-provided JSON annotation file describing item-attribute pair in images, to perform automatic item cropping.
It performs the following steps:
1. Parse JSON annotations with item-attribute pairs.
2. Use GroundingDINO + SAM2 to crop each item based on item name.
3. Save segmented images.

"""

import os
import re
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import cv2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError:
    print("Warning: SAM2 or Transformers not found. Please install required packages.")

# -------------------------------------------------------------------------
# Seed
# -------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------------------------------------------------
# Annotation parsing
# -------------------------------------------------------------------------

def load_simple_annotations(json_file):
    """
    Load and process JSON annotations.
    Expected structure:
    [
        {
            "image_id": "001",
            "image_path": "/path/to/image_001.jpg",
            "items": [
                {
                    "item_name": "shirt",
                    "attributes": ["white", "striped"],
                    "questions": ["a white shirt", "a striped shirt"]
                },
                ...
            ]
        },
        ...
    ]
    """
    if not Path(json_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {json_file}")
    
    with open(json_file, "r") as f:
        data = json.load(f)

    image_data = {}
    for record in data:
        image_id = str(record["image_id"])
        image_path = record["image_path"]
        image_data[image_id] = {
            "image_path": image_path,
            "items": record["items"]
        }
    print(f"Loaded {len(image_data)} images from {json_file}")
    return image_data


# -------------------------------------------------------------------------
# SAM + GroundingDINO inference
# -------------------------------------------------------------------------

def crop_image_with_mask(image: Image.Image, mask: np.array, resize=None, pad=False):
    """Crop an image using its binary mask."""
    if image.size[0] != mask.shape[1] or image.size[1] != mask.shape[0]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(image.size, Image.NEAREST)).astype(bool)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    cropped_image_np = np.array(image) * mask_3d
    cropped_image_np = np.clip(cropped_image_np, 0, 255).astype(np.uint8)
    cropped_image = Image.fromarray(cropped_image_np)

    if resize:
        cropped_image = ImageOps.contain(cropped_image, resize, method=Image.LANCZOS)
    if pad:
        cropped_image = ImageOps.pad(cropped_image, resize, color="#fff")
    return cropped_image

def detect_and_crop(image_path, text_prompt, predictor, grounding_model, processor, device):
    """Perform detection and segmentation for a given image and text prompt."""
    image = Image.open(image_path).convert("RGB")
    predictor.set_image(np.array(image))
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, target_sizes=[image.size[::-1]])
    boxes = results[0]["boxes"].cpu().numpy()

    if boxes.shape[0] == 0:
        print(f"No object found for '{text_prompt}' in {image_path}")
        return image, np.zeros((image.size[1], image.size[0]), dtype=bool)

    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
    masks = masks.squeeze()
    if len(masks.shape) == 3:
        mask = np.any(masks, axis=0)
    else:
        mask = masks
    return image, mask

def save_cropped(image, mask, save_path):
    """Save cropped image."""
    cropped = crop_image_with_mask(image, mask)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(save_path)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args):
    device = args.device
    print(f"Running on device: {device}")

    # Load annotations
    annotations = load_simple_annotations(args.annotation_file)

    # Initialize SAM and GroundingDINO
    print("Initializing SAM2 and GroundingDINO...")
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)
    sam_model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    print("Model initialization complete.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id, record in annotations.items():
        image_path = record["image_path"]
        #print(f"\nProcessing image: {image_path}")

        for item in record["items"]:
            item_name = item["item_name"]
            prompt = item_name
            save_path = output_dir / f"{image_id}_{item_name}.png"

            image, mask = detect_and_crop(image_path, prompt, predictor, grounding_model, processor, device)
            save_cropped(image, mask, save_path)
            #print(f"Saved cropped image for '{item_name}' → {save_path}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified SAM + GroundingDINO pipeline for cropping items and scoring.")
    parser.add_argument("--annotation-file", type=str, required=True, help="Path to simplified JSON file.")
    parser.add_argument("--output-dir", type=str, default="./crops", help="Directory to save cropped images.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run SAM and GroundingDINO.")
    parser.add_argument("--sam-checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path.")
    parser.add_argument("--sam-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config path.")
    parser.add_argument("--grounding-model", type=str, default="IDEA-Research/grounding-dino-tiny", help="GroundingDINO model name.")
    args = parser.parse_args()
    main(args)
