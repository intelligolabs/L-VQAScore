#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline for computing reflected and leaked VQA scores on full or cropped images.

This script processes a user-provided JSON annotation file describing item-attribute pair in images, to perform vqa scoring.
It performs the following steps:
1. Parse JSON annotations with item-attribute pairs.
2. VQA over segmented images.
3. Calculate and return L-VQAScore metrics.

"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import t2v_metrics

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

def compute_stats(scores_reflect, scores_leak, threshold=0.5):
    """Compute precision, recall, f1."""
    scores_reflect = [x for sub in scores_reflect for x in sub]
    scores_leak    = [x for sub in scores_leak for x in sub]
    
    tp = sum(s >= threshold for s in scores_reflect) 
    fp = sum(s >= threshold for s in scores_leak) 
    fn = sum(s < threshold for s in scores_reflect) 

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}

def convert_tensor_to_python(data):
    """Recursively convert torch tensors to Python native types."""
    if isinstance(data, dict):
        return {k: convert_tensor_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_to_python(x) for x in data]
    elif isinstance(data, torch.Tensor):
        return data.item() if data.numel() == 1 else data.tolist()
    else:
        return data

def run_vqa(model, img_path, questions):
    """Run VQA model on given image and list of questions."""
    if not questions:
        return []
    try:
        scores = model(images=[str(img_path)], texts=questions)
        return scores.tolist()[0]
    except Exception as e:
        print(f"⚠️ VQA failed on {img_path}: {e}")
        return [0.0 for _ in questions]

def main(args):
    set_seed(args.seed)
    print("Loading annotations...")
    data = load_simple_annotations(args.annotation_file)

    print("Initializing VQA model...")
    model = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=args.device)

    results, ref_scores_list, leak_scores_list = [], [], []
    for image_id, content in tqdm(data.items(), desc="Processing images"):
        image_path = Path(content["image_path"])
        items = content["items"]
        if not items:
            print(f"⚠️ Skipping image {image_id}: no item.")
            continue
            
        for item in items:
            item_name = item["item_name"]
            attributes = item.get("attributes", [])
            questions = item.get("questions", [])

            if not attributes:
                print(f"⚠️ Skipping item '{item_name}' in image {image_id}: no attributes.")
                continue
            # reflected questions: correct attribute-item pairs
            reflected_questions = [f"a {attr} {item_name}" for attr in attributes]

            # leaked questions: pair this item's attributes with other items in the same image
            leaked_questions = []
            for other in items:
                if other["item_name"] != item_name:
                    for attr in attributes:
                        leaked_questions.append(f"a {attr} {other['item_name']}")
                        
            if not leaked_questions:
                print(f"Skipping item {item_name} in {image_id}: no leaked questions.")
                continue    
            # determine cropped or full image to use
            crop_path = Path(args.sam_dir) / f"{image_id}_{item_name}.png"
            if args.mode == "cropped" and crop_path.exists():
                eval_img = crop_path
            else:
                eval_img = image_path

            # run vqa
            ref_scores = run_vqa(model, eval_img, reflected_questions)
            leak_scores = run_vqa(model, eval_img, leaked_questions)

            ref_scores_list.append(ref_scores)
            leak_scores_list.append(leak_scores)
            results.append({
                "image_id": image_id,
                "item_name": item_name,
                "reflected_questions": reflected_questions,
                "leaked_questions": leaked_questions,
                "reflected_scores": ref_scores,
                "leaked_scores": leak_scores,
                "reflected_avg": float(np.mean(ref_scores)) if ref_scores else 0.0,
                "leaked_avg": float(np.mean(leak_scores)) if leak_scores else 0.0,
                "mode": args.mode
            })

    # stats: cross image
    stats = compute_stats(ref_scores_list, leak_scores_list, args.threshold)
    results = convert_tensor_to_python(results)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"VQA results saved to {args.output_json}")
    print(f"\nL-VQAScore - Overall Precision: {stats['precision']:.3f}, Recall: {stats['recall']:.3f}, F1: {stats['f1']:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reflected and leaked VQA scores")
    parser.add_argument("--annotation-file", type=str, required=True, help="Path to annotation JSON")
    parser.add_argument("--sam-dir", type=str, required=True, help="Path to cropped images (from SAM segmentation)")
    parser.add_argument("--output-json", type=str, default="./vqa_scores.json", help="Output file for VQA results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for Precision, Recall, F1 calculation")
    parser.add_argument("--mode", type=str, choices=["cropped", "full"], default="cropped", help="VQA mode: cropped or full image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for VQA model")
    parser.add_argument("--seed", default=42, help="Set seed")
    args = parser.parse_args()
    main(args)
