#!/usr/bin/env python3
"""MedSAM Standalone Inference CLI.

Performs prompt-driven skin lesion segmentation using trained PEFT MedSAM models
(CG-Adapter, Standard Adapter, LoRA, Decoder-Only).

Usage Examples:
    # 1. Single image with automatic bounding box detection:
    python inference.py --image path/to/sample.jpg --output_dir results/

    # 2. Single image with custom bounding box:
    python inference.py --image path/to/sample.jpg --bbox 120 80 450 380 --model cg_adapter

    # 3. Batch process an entire directory:
    python inference.py --input_dir data/test_samples/ --output_dir batch_results/ --model cg_adapter
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch

from src.eval.inference_utils import (
    compute_lab_contrast_proxy,
    auto_detect_prompt_bbox,
    load_model,
    run_inference,
    create_visual_overlay
)


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM PEFT Standalone Inference CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image (JPEG, PNG)")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images for batch processing")
    parser.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Bounding box prompt. If omitted, automatically estimated.")
    parser.add_argument("--model", type=str, default="cg_adapter",
                        choices=["cg_adapter", "standard_adapter", "lora", "decoder_only"],
                        help="PEFT model architecture (default: cg_adapter)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained PEFT checkpoint (.pth). Defaults to checkpoints/best_<model>_model.pth")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to base medsam_vit_b.pth. Auto-detected in checkpoints/ if omitted.")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Directory to save generated masks, overlays, and reports (default: inference_output)")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument("--no_overlay", action="store_true", help="Disable composite overlay visualization generation")
    parser.add_argument("--no_mask", action="store_true", help="Disable binary mask PNG generation")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save per-sample JSON physics metadata")
    return parser.parse_args()


def process_single_image(
    image_path: str,
    model: torch.nn.Module,
    bbox: Optional[Tuple[int, int, int, int]],
    output_dir: str,
    model_name: str,
    device: torch.device,
    save_overlay: bool = True,
    save_mask: bool = True,
    save_json: bool = True
) -> dict:
    """Process a single image and save specified outputs."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image at {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # Auto-detect bbox if not provided
    if bbox is None:
        used_bbox = auto_detect_prompt_bbox(image_rgb)
        prompt_type = "auto_detected"
    else:
        used_bbox = tuple(bbox)
        prompt_type = "manual"

    # Forward inference
    result = run_inference(model, image_rgb, used_bbox, device=device)
    mask = result["mask"]
    contrast_info = result["contrast_info"]
    contrast_info["prompt_type"] = prompt_type
    contrast_info["image_file"] = os.path.basename(image_path)
    contrast_info["model"] = model_name

    stem = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save binary mask
    if save_mask:
        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        cv2.imwrite(mask_path, mask)
        contrast_info["saved_mask_path"] = mask_path

    # 2. Save visual overlay
    if save_overlay:
        vis_overlay = create_visual_overlay(image_rgb, mask, contrast_info, model_name=model_name)
        overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))
        contrast_info["saved_overlay_path"] = overlay_path

    # 3. Save JSON metadata
    if save_json:
        json_path = os.path.join(output_dir, f"{stem}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(contrast_info, f, indent=2)
        contrast_info["saved_json_path"] = json_path

    return contrast_info


def main():
    args = parse_args()

    if not args.image and not args.input_dir:
        print("[Inference CLI] Error: You must provide either --image or --input_dir.")
        print("Run `python inference.py --help` for full usage documentation.")
        sys.exit(1)

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"============================================================")
    print(f" MedSAM PEFT Inference CLI")
    print(f" Model Architecture: {args.model.upper()}")
    print(f" Execution Device:   {device}")
    print(f"============================================================")

    # Load Model
    model, is_mock = load_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        base_checkpoint_path=args.base_checkpoint,
        device=device
    )
    if is_mock:
        print("[Inference CLI] Notice: Running in lightweight foundation mode (base MedSAM checkpoint not yet downloaded).")
    else:
        print("[Inference CLI] Loaded official MedSAM ViT-B foundation backbone.")

    # Target images list
    if args.image:
        image_files = [args.image]
    else:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_files.sort()
        if not image_files:
            print(f"[Inference CLI] Error: No valid image files found in {args.input_dir}")
            sys.exit(1)
        print(f"[Inference CLI] Found {len(image_files)} images to process in {args.input_dir}")

    # Process
    summary_records = []
    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(img_path)} ...", end=" ", flush=True)
        try:
            info = process_single_image(
                image_path=img_path,
                model=model,
                bbox=args.bbox,
                output_dir=args.output_dir,
                model_name=args.model,
                device=device,
                save_overlay=not args.no_overlay,
                save_mask=not args.no_mask,
                save_json=args.save_json
            )
            print(f"Done! (DeltaE*ab: {info['delta_e']}, Skin: {info['estimated_fst']}, Gamma: {info.get('gamma_factor', 1.0)})")
            summary_records.append({
                "image": info["image_file"],
                "model": info["model"],
                "delta_e": info["delta_e"],
                "gamma_factor": info.get("gamma_factor", 1.0),
                "estimated_fst": info["estimated_fst"],
                "lesion_area_px": info["lesion_area_px"],
                "coverage_pct": info["lesion_coverage_pct"],
                "prompt_type": info["prompt_type"]
            })
        except Exception as e:
            print(f"FAILED ({e})")

    # Batch summary CSV
    if len(summary_records) > 1:
        csv_path = os.path.join(args.output_dir, "batch_summary.csv")
        df = pd.DataFrame(summary_records)
        df.to_csv(csv_path, index=False)
        print(f"\n[Inference CLI] Saved batch summary table: {csv_path}")

    print(f"[Inference CLI] All tasks completed! Outputs stored in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
