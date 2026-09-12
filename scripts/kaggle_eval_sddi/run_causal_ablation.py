"""Causal Signal Ablation Study on sDDI Clinical Partition (E08).

Evaluates the physical necessity of color-contrast gating:
1. Full CG-Adapter (Ours): delta_e = sqrt((dL)^2 + (da)^2 + (db)^2), c_hat = (delta_e - 28.5)/12.0
2. Constant Gate (gamma == 1.0): Gating factor clamped to 1.0
3. Shuffled Proxy: Random permutation of c_hat across test cases
4. Sign-Inverted Proxy: -c_hat (reversing contrast physics)
5. Luminance-Only Proxy: delta_L = |L_core - L_ring|, c_hat = (delta_L - 22.0)/10.0
"""

import os
import sys
import glob
import time
import math
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry

# Reuse the exact classes
class ContrastGatingMLP(nn.Module):
    def __init__(self, in_features: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, c_prompt: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(c_prompt))
        gamma = 2.0 * torch.sigmoid(self.fc2(h))
        return gamma

class BottleneckAdapter(nn.Module):
    def __init__(self, embed_dim: int = 768, bottleneck_rank: int = 16, gated: bool = False):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_rank)
        self.act = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottleneck_rank, embed_dim)
        self.gated = gated
        if gated:
            self.gating = ContrastGatingMLP(in_features=1, hidden_dim=16)
        else:
            self.gating = None
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, c_prompt: Optional[torch.Tensor] = None, force_gamma: Optional[float] = None) -> torch.Tensor:
        residual = self.up_proj(self.act(self.down_proj(x)))
        if force_gamma is not None:
            residual = residual * force_gamma
        elif self.gated and c_prompt is not None:
            gamma = self.gating(c_prompt)
            shape = [gamma.size(0)] + [1] * (x.ndim - 1)
            residual = residual * gamma.view(*shape)
        return x + residual

class PEFTMedSAM(nn.Module):
    def __init__(self, base_medsam: nn.Module, rank: int = 16):
        super().__init__()
        self.base_model = base_medsam
        self.rank = rank
        self.adapters = nn.ModuleList()

        for p in self.base_model.parameters():
            p.requires_grad = False

        for block in self.base_model.image_encoder.blocks:
            adapter = BottleneckAdapter(embed_dim=768, bottleneck_rank=self.rank, gated=True)
            self.adapters.append(adapter)

    def encode_image(self, image_tensor: torch.Tensor, c_prompt: Optional[torch.Tensor] = None, force_gamma: Optional[float] = None) -> torch.Tensor:
        x = self.base_model.image_encoder.patch_embed(image_tensor)
        if self.base_model.image_encoder.pos_embed is not None:
            x = x + self.base_model.image_encoder.pos_embed
        for idx, block in enumerate(self.base_model.image_encoder.blocks):
            x = block(x)
            if idx < len(self.adapters):
                x = self.adapters[idx](x, c_prompt=c_prompt, force_gamma=force_gamma)
        return self.base_model.image_encoder.neck(x.permute(0, 3, 1, 2))

    def decode_prompt_box(self, image_embeddings: torch.Tensor, box_tensor: torch.Tensor) -> torch.Tensor:
        sparse_emb, dense_emb = self.base_model.prompt_encoder(points=None, boxes=box_tensor.unsqueeze(1), masks=None)
        low_res_masks, _ = self.base_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.base_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )
        return low_res_masks

def compute_binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)
    intersection = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    return float(2.0 * intersection / total)

def compute_binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)

def compute_contrasts(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Compute Delta E*ab and Delta L* contrast proxies."""
    H, W = image_rgb.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    mx = int(round(0.50 * bw / 2.0))
    my = int(round(0.50 * bh / 2.0))
    c_xmin = max(0, min(W - 1, xmin + mx))
    c_ymin = max(0, min(H - 1, ymin + my))
    c_xmax = max(c_xmin + 1, min(W, xmax - mx))
    c_ymax = max(c_ymin + 1, min(H, ymax - my))

    ex = int(round(0.25 * bw))
    ey = int(round(0.25 * bh))
    r_xmin = max(0, min(W - 1, xmin - ex))
    r_ymin = max(0, min(H - 1, ymin - ey))
    r_xmax = max(r_xmin + 1, min(W, xmax + ex))
    r_ymax = max(r_ymin + 1, min(H, ymax - ey))

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0

    core_crop = lab[c_ymin:c_ymax, c_xmin:c_xmax]
    if core_crop.size == 0:
        return 28.5, 22.0
    core_mean = core_crop.mean(axis=(0, 1))

    outer_crop = lab[r_ymin:r_ymax, r_xmin:r_xmax]
    ring_mask = np.ones((r_ymax - r_ymin, r_xmax - r_xmin), dtype=bool)
    in_ymin = max(0, ymin - r_ymin)
    in_ymax = min(r_ymax - r_ymin, ymax - r_ymin)
    in_xmin = max(0, xmin - r_xmin)
    in_xmax = min(r_xmax - r_xmin, xmax - r_xmin)
    ring_mask[in_ymin:in_ymax, in_xmin:in_xmax] = False

    ring_pixels = outer_crop[ring_mask]
    if len(ring_pixels) == 0:
        return 28.5, 22.0
    ring_mean = ring_pixels.mean(axis=0)

    delta_e = float(np.sqrt(np.sum((core_mean - ring_mean) ** 2)))
    delta_l = float(abs(core_mean[0] - ring_mean[0]))
    return delta_e, delta_l

def perturb_bounding_box(bbox, img_shape, delta, seed):
    if delta <= 0.0:
        return bbox
    H, W = img_shape
    x_min, y_min, x_max, y_max = bbox
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    rng = np.random.default_rng(seed)
    eps = rng.uniform(-delta, delta, size=4)
    new_xmin = int(round(x_min + eps[0] * w))
    new_ymin = int(round(y_min + eps[1] * h))
    new_xmax = int(round(x_max + eps[2] * w))
    new_ymax = int(round(y_max + eps[3] * h))
    clamped_xmin = max(0, min(W - 2, new_xmin))
    clamped_ymin = max(0, min(H - 2, new_ymin))
    clamped_xmax = max(clamped_xmin + 2, min(W, new_xmax))
    clamped_ymax = max(clamped_ymin + 2, min(H, new_ymax))
    return (clamped_xmin, clamped_ymin, clamped_xmax, clamped_ymax)

def run_ablation_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Causal Ablation Study on {device}...")

    search_roots = ["/kaggle/input", "/kaggle/working", ".", "checkpoints", "/tmp"]
    medsam_ckpt = glob.glob("/kaggle/input/**/medsam_vit_b.pth", recursive=True)[0]
    cg_ckpt = glob.glob("/kaggle/input/**/best_cg_adapter_model.pth", recursive=True)[0]
    ddi_img_dir = os.path.dirname(glob.glob("/kaggle/input/**/000044.png", recursive=True)[0])
    sddi_testing_root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/000006.npy", recursive=True)[0]))

    # Collect samples
    test_samples = []
    for split_name, skin_group, rel in [
        ("test_light", "FST_I_II_Light", "test_light"),
        ("test_med", "FST_III_IV_Medium", "test_med"),
        ("test_dark", "FST_V_VI_Dark", "test_dark"),
    ]:
        for mf in sorted(glob.glob(os.path.join(sddi_testing_root, rel, "*.npy"))):
            img_id = os.path.basename(mf).replace(".npy", "")
            img_p = os.path.join(ddi_img_dir, f"{img_id}.png")
            test_samples.append({
                "image_id": img_id,
                "skin_tone_group": skin_group,
                "mask_path": mf,
                "image_path": img_p
            })
    print(f"Total samples: {len(test_samples)}")

    # Load base SAM and CG-Adapter
    sam = sam_model_registry["vit_b"](checkpoint=medsam_ckpt).to(device)
    model = PEFTMedSAM(sam, rank=16).to(device)
    ckpt = torch.load(cg_ckpt, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    clean_state = {k.replace("base_sam.", "base_model."): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    # Pre-extract data and contrast values
    cached_data = []
    contrast_list = []
    for item in test_samples:
        gt_mask = (np.load(item["mask_path"]) == 1).astype(np.uint8)
        H, W = gt_mask.shape[:2]
        coords = np.argwhere(gt_mask > 0)
        if coords.size == 0:
            continue
        bbox = (int(coords[:, 1].min()), int(coords[:, 0].min()), int(coords[:, 1].max() + 1), int(coords[:, 0].max() + 1))
        bgr = cv2.imread(item["image_path"])
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((H, W, 3), dtype=np.uint8)
        if img_rgb.shape[:2] != (H, W):
            img_rgb = cv2.resize(img_rgb, (W, H))
        resized_img = cv2.resize(img_rgb, (1024, 1024))
        img_tensor = (torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5

        de, dl = compute_contrasts(img_rgb, bbox)
        cached_data.append({
            "image_id": item["image_id"],
            "skin_tone_group": item["skin_tone_group"],
            "gt_binary": gt_mask,
            "img_rgb": img_rgb,
            "img_tensor": img_tensor.unsqueeze(0).to(device),
            "bbox": bbox,
            "H": H, "W": W,
            "delta_e": de,
            "delta_l": dl
        })
        contrast_list.append((de - 28.5) / 12.0)

    # Deterministic shuffle for shuffled ablation
    rng_shuff = np.random.default_rng(42)
    shuffled_contrasts = rng_shuff.permutation(contrast_list).tolist()

    ablation_variants = [
        "Full CG-Adapter (Ours)",
        "Constant Gate (gamma = 1.0)",
        "Shuffled Contrast Proxy",
        "Sign-Inverted Proxy (-c)",
        "Luminance-Only (Delta L*)"
    ]

    all_records = []
    with torch.no_grad():
        for var_name in ablation_variants:
            print(f"\nEvaluating Ablation Variant: {var_name}")
            start_t = time.time()
            for idx, c in enumerate(cached_data):
                H, W = c["H"], c["W"]
                sx, sy = 1024.0 / float(W), 1024.0 / float(H)
                bbox = c["bbox"]
                gt = c["gt_binary"]
                img_t = c["img_tensor"]

                # Prompts: delta = 0.0 (clean) and delta = 0.20 (noisy)
                prompts = [
                    (0.0, 0, bbox),
                    (0.20, 1, perturb_bounding_box(bbox, (H, W), 0.20, 101)),
                    (0.20, 2, perturb_bounding_box(bbox, (H, W), 0.20, 102)),
                    (0.20, 3, perturb_bounding_box(bbox, (H, W), 0.20, 103)),
                ]

                for delta, r_id, p_box in prompts:
                    scaled_box = torch.tensor([[p_box[0] * sx, p_box[1] * sy, p_box[2] * sx, p_box[3] * sy]], dtype=torch.float32, device=device)

                    force_gamma = None
                    c_prompt = None

                    if var_name == "Full CG-Adapter (Ours)":
                        norm_c = (c["delta_e"] - 28.5) / 12.0
                        c_prompt = torch.tensor([[norm_c]], dtype=torch.float32, device=device)
                    elif var_name == "Constant Gate (gamma = 1.0)":
                        force_gamma = 1.0
                    elif var_name == "Shuffled Contrast Proxy":
                        c_prompt = torch.tensor([[shuffled_contrasts[idx]]], dtype=torch.float32, device=device)
                    elif var_name == "Sign-Inverted Proxy (-c)":
                        norm_c = - (c["delta_e"] - 28.5) / 12.0
                        c_prompt = torch.tensor([[norm_c]], dtype=torch.float32, device=device)
                    elif var_name == "Luminance-Only (Delta L*)":
                        norm_l = (c["delta_l"] - 22.0) / 10.0
                        c_prompt = torch.tensor([[norm_l]], dtype=torch.float32, device=device)

                    emb = model.encode_image(img_t, c_prompt=c_prompt, force_gamma=force_gamma)
                    low_res = model.decode_prompt_box(emb, scaled_box)
                    logits = F.interpolate(low_res, size=(H, W), mode="bilinear", align_corners=False)
                    pred = (logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)

                    dice = compute_binary_dice(pred, gt)
                    iou = compute_binary_iou(pred, gt)

                    all_records.append({
                        "ablation_variant": var_name,
                        "image_id": c["image_id"],
                        "skin_tone_group": c["skin_tone_group"],
                        "delta": delta,
                        "realization_id": r_id,
                        "dice": dice,
                        "iou": iou
                    })

            elapsed = time.time() - start_t
            sub_df = pd.DataFrame([r for r in all_records if r["ablation_variant"] == var_name])
            c_d = sub_df[sub_df["delta"] == 0.0]["dice"].mean()
            j_d = sub_df[sub_df["delta"] == 0.20]["dice"].mean()
            print(f"--> {var_name} ({elapsed:.1f}s) | Clean Dice: {c_d:.4f} | Jitter 20% Dice: {j_d:.4f}")

    df_out = pd.DataFrame(all_records)
    df_out.to_csv("causal_ablation_results.csv", index=False)
    print("Saved causal_ablation_results.csv")

if __name__ == "__main__":
    run_ablation_benchmark()
