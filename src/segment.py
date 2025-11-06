#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline for item cropping and VQA evaluation.

This script processes a user-provided JSON annotation file describing item-attribute pair in images, to perform automatic item cropping.
It performs the following steps:
1. Parse JSON annotations with item-attribute pairs.
2. Use GroundingDINO + SAM2 to blur and crop each item with item name input.
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

def crop_image_with_mask(image: Image.Image, mask: np.array, resize=True):
    """Crop an image using its binary mask. Optional resize."""
    if image.size[0] != mask.shape[1] or image.size[1] != mask.shape[0]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(image.size, Image.NEAREST)).astype(bool)

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    cropped_image_np = np.array(image) * mask_3d
    cropped_image_np = np.clip(cropped_image_np, 0, 255).astype(np.uint8)
    processed_image = Image.fromarray(cropped_image_np)

    if resize:
        orig_w, orig_h = image.size
        processed_image = ImageOps.contain(processed_image, (orig_w, orig_h), method=Image.LANCZOS)
    return processed_image
    
def blur_crop_image_with_mask(image: Image.Image, mask: np.ndarray, resize = True, crop_expansion = 0.1, pad = True, blur_kernel_size = 51, blur_sigma = 25):
    """Given an image and its binary mask, apply background blurring, crop the masked object with optional padding, 
    and resize the cropped output to match the original image dimensions while preserving aspect ratio.
    Note: blurring strength is adjustable, with the default setting recommended from our evaluation."""
    if image.size[0] != mask.shape[1] or image.size[1] != mask.shape[0]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(image.size, Image.NEAREST)).astype(bool)
    image_np = np.array(image)

    blurred_np = cv2.GaussianBlur(image_np, (blur_kernel_size, blur_kernel_size), blur_sigma)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blended_np = np.where(mask_3d, image_np, blurred_np).astype(np.uint8)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        # no foreground → just return blurred background
        cropped_np = blended_np
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Expand bbox
        bw = x_max - x_min
        bh = y_max - y_min
        expand_x = int(bw * crop_expansion)
        expand_y = int(bh * crop_expansion)
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(mask.shape[1], x_max + expand_x)
        y_max = min(mask.shape[0], y_max + expand_y)
        cropped_np = blended_np[y_min:y_max, x_min:x_max]

    processed_image = Image.fromarray(cropped_np)
    if resize:
        orig_w, orig_h = image.size
        processed_image = ImageOps.contain(processed_image, (orig_w, orig_h), method=Image.LANCZOS)
    if pad:
        orig_w, orig_h = image.size
        processed_image = ImageOps.pad(processed_image, (orig_w, orig_h), color="#ffffff")

    return processed_image

def detect_seg(image_path, text_prompt, predictor, grounding_model, processor, device):
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

def save_cropped(image, mask, save_path, mode='blur'):
    """Save cropped image."""
    if mode == 'blur':
        processed_image = blur_crop_image_with_mask(image, mask) 
    elif mode == 'crop':
        processed_image = crop_image_with_mask(image, mask)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'blur' or 'crop'.")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    processed_image.save(save_path)

def main(args):
    set_seed(args.seed)
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

            image, mask = detect_seg(image_path, prompt, predictor, grounding_model, processor, device)
            save_cropped(image, mask, save_path, args.mode)
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
    parser.add_argument("--seed", default=42, help="Set seed")
    parser.add_argument("--mode", type=str, choices=["crop", "blur"], default="blur", help="Segment mode: blur, crop")
    args = parser.parse_args()
    main(args)
