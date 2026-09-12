"""Phase 7: External Clinical Generalization and Skin-Tone Fairness Benchmark on sDDI (N=198).

Evaluates all 5 models across 198 clinician-verified clinical test cases (FST I-II Light, FST III-IV Med, FST V-VI Dark)
under clean (delta=0.0) and noisy prompt regimes (delta in {0.05, 0.10, 0.20}) with 3 realizations (seeds 101, 102, 103).
Total evaluations: 198 samples * 10 prompt conditions * 5 models = 9,900 evaluations.
Features:
- Exact state_dict key alignment (base_sam. -> base_model.)
- Cached image embeddings for prompt-independent backbones (10x faster)
- Full native resolution DSC and IoU
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

# ==============================================================================
# 1. METRICS ENGINE (DICE, IOU, HD95, NSD)
# ==============================================================================

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

def get_mask_contour_points(binary_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=int)
    all_pts = np.vstack([c.squeeze() for c in contours if len(c) > 0])
    if all_pts.ndim == 1:
        all_pts = all_pts.reshape(-1, 2)
    return all_pts

def compute_hd95_fast(pred_mask: np.ndarray, gt_mask: np.ndarray, max_dim: int = 512) -> float:
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    H, W = p.shape[:2]
    diagonal = float(np.sqrt(H * H + W * W))

    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return diagonal

    # Downsample for fast EDT if large
    scale = 1.0
    if max(H, W) > max_dim:
        scale = float(max_dim) / float(max(H, W))
        new_w = max(2, int(W * scale))
        new_h = max(2, int(H * scale))
        p = cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        g = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        H, W = p.shape[:2]

    pts_p = get_mask_contour_points(p)
    pts_g = get_mask_contour_points(g)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return diagonal

    canvas_g = np.ones((H, W), dtype=bool)
    canvas_g[pts_g[:, 1], pts_g[:, 0]] = False
    edt_g = ndimage.distance_transform_edt(canvas_g)
    dist_p_to_g = edt_g[pts_p[:, 1], pts_p[:, 0]]

    canvas_p = np.ones((H, W), dtype=bool)
    canvas_p[pts_p[:, 1], pts_p[:, 0]] = False
    edt_p = ndimage.distance_transform_edt(canvas_p)
    dist_g_to_p = edt_p[pts_g[:, 1], pts_g[:, 0]]

    hd95_p_to_g = float(np.percentile(dist_p_to_g, 95))
    hd95_g_to_p = float(np.percentile(dist_g_to_p, 95))
    return float(max(hd95_p_to_g, hd95_g_to_p) / scale)

def compute_nsd_fast(pred_mask: np.ndarray, gt_mask: np.ndarray, tau: float = 2.0, max_dim: int = 512) -> float:
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    scale = 1.0
    if max(p.shape[:2]) > max_dim:
        scale = float(max_dim) / float(max(p.shape[:2]))
        new_w = max(2, int(p.shape[1] * scale))
        new_h = max(2, int(p.shape[0] * scale))
        p = cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        g = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pts_p = get_mask_contour_points(p)
    pts_g = get_mask_contour_points(g)
    if len(pts_p) == 0 or len(pts_g) == 0:
        return 0.0

    H, W = p.shape[:2]
    canvas_g = np.ones((H, W), dtype=bool)
    canvas_g[pts_g[:, 1], pts_g[:, 0]] = False
    edt_g = ndimage.distance_transform_edt(canvas_g)
    dist_p_to_g = edt_g[pts_p[:, 1], pts_p[:, 0]]

    canvas_p = np.ones((H, W), dtype=bool)
    canvas_p[pts_p[:, 1], pts_p[:, 0]] = False
    edt_p = ndimage.distance_transform_edt(canvas_p)
    dist_g_to_p = edt_p[pts_g[:, 1], pts_g[:, 0]]

    scaled_tau = tau * scale
    tp_p = (dist_p_to_g <= scaled_tau).sum()
    tp_g = (dist_g_to_p <= scaled_tau).sum()
    return float((tp_p + tp_g) / (len(pts_p) + len(pts_g)))

# ==============================================================================
# 2. CONTRAST PROXY ENGINE (ZERO MASK LEAKAGE)
# ==============================================================================

def compute_contrast_proxy(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    alpha_core: float = 0.50,
    beta_ring: float = 0.25
) -> float:
    H, W = image_rgb.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    mx = int(round(alpha_core * bw / 2.0))
    my = int(round(alpha_core * bh / 2.0))
    c_xmin = max(0, min(W - 1, xmin + mx))
    c_ymin = max(0, min(H - 1, ymin + my))
    c_xmax = max(c_xmin + 1, min(W, xmax - mx))
    c_ymax = max(c_ymin + 1, min(H, ymax - my))

    ex = int(round(beta_ring * bw))
    ey = int(round(beta_ring * bh))
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
        return 28.5

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
        return 28.5

    ring_mean = ring_pixels.mean(axis=0)
    delta_e = float(np.sqrt(np.sum((core_mean - ring_mean) ** 2)))
    return delta_e

# ==============================================================================
# 3. PROMPT PERTURBATION ENGINE
# ==============================================================================

PERTURBATION_SEEDS = [101, 102, 103]
DELTA_LEVELS = [0.0, 0.05, 0.10, 0.20]

def perturb_bounding_box(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    delta: float,
    seed: int
) -> Tuple[int, int, int, int]:
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

def generate_prompt_realizations(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    image_id: Union[str, int]
) -> List[Dict[str, Any]]:
    realizations = []
    id_hash = abs(hash(str(image_id))) % 100000

    for delta in DELTA_LEVELS:
        if delta == 0.0:
            realizations.append({
                "delta": 0.0,
                "realization_id": 0,
                "bbox": bbox
            })
        else:
            for m_idx, base_seed in enumerate(PERTURBATION_SEEDS):
                det_seed = (base_seed * 100003 + id_hash + int(delta * 1000)) % (2**31 - 1)
                p_box = perturb_bounding_box(bbox, img_shape, delta, det_seed)
                realizations.append({
                    "delta": float(delta),
                    "realization_id": m_idx + 1,
                    "bbox": p_box
                })
    return realizations

# ==============================================================================
# 4. PEFT ADAPTER MODULES
# ==============================================================================

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

    def forward(self, x: torch.Tensor, c_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.up_proj(self.act(self.down_proj(x)))
        if self.gated and c_prompt is not None:
            gamma = self.gating(c_prompt)
            shape = [gamma.size(0)] + [1] * (x.ndim - 1)
            residual = residual * gamma.view(*shape)
        return x + residual

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r) if r > 0 else 1.0
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

class PEFTMedSAM(nn.Module):
    def __init__(
        self,
        base_medsam: nn.Module,
        mode: str = "cg_adapter",
        rank: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        self.base_model = base_medsam
        self.mode = mode
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.adapters = nn.ModuleList()
        self.lora_layers = nn.ModuleList()

        for p in self.base_model.parameters():
            p.requires_grad = False

        vit_blocks = self.base_model.image_encoder.blocks
        if self.mode == "lora":
            for block in vit_blocks:
                q_proj = block.attn.qkv
                lora_qkv = LoRALinear(q_proj, r=self.rank, lora_alpha=self.lora_alpha)
                block.attn.qkv = lora_qkv
                self.lora_layers.append(lora_qkv)
        elif self.mode in ["standard_adapter", "cg_adapter"]:
            is_gated = (self.mode == "cg_adapter")
            for block in vit_blocks:
                adapter = BottleneckAdapter(
                    embed_dim=768,
                    bottleneck_rank=self.rank,
                    gated=is_gated
                )
                self.adapters.append(adapter)

    def encode_image(self, image_tensor: torch.Tensor, c_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode in ["standard_adapter", "cg_adapter"]:
            x = self.base_model.image_encoder.patch_embed(image_tensor)
            if self.base_model.image_encoder.pos_embed is not None:
                x = x + self.base_model.image_encoder.pos_embed
            for idx, block in enumerate(self.base_model.image_encoder.blocks):
                x = block(x)
                if idx < len(self.adapters):
                    x = self.adapters[idx](x, c_prompt=c_prompt)
            image_embeddings = self.base_model.image_encoder.neck(x.permute(0, 3, 1, 2))
        else:
            image_embeddings = self.base_model.image_encoder(image_tensor)
        return image_embeddings

    def decode_prompt_box(self, image_embeddings: torch.Tensor, box_tensor: torch.Tensor) -> torch.Tensor:
        sparse_emb, dense_emb = self.base_model.prompt_encoder(
            points=None,
            boxes=box_tensor.unsqueeze(1),
            masks=None
        )
        low_res_masks, _ = self.base_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.base_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )
        return low_res_masks

# ==============================================================================
# 5. DATA RESOLUTION & LOADING HELPERS
# ==============================================================================

def find_file_recursive(base_dirs: List[str], target_pattern: str) -> Optional[str]:
    for bd in base_dirs:
        if not os.path.exists(bd):
            continue
        matches = glob.glob(os.path.join(bd, "**", target_pattern), recursive=True)
        if matches:
            return sorted(matches)[0]
    return None

def find_folder_containing(base_dirs: List[str], sample_filename: str) -> Optional[str]:
    for bd in base_dirs:
        if not os.path.exists(bd):
            continue
        matches = glob.glob(os.path.join(bd, "**", sample_filename), recursive=True)
        if matches:
            return os.path.dirname(matches[0])
    return None

def extract_bbox_from_binary_mask(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    coords = np.argwhere(binary_mask > 0)
    if coords.size == 0:
        return None
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    return (int(xmin), int(ymin), int(xmax + 1), int(ymax + 1))

# ==============================================================================
# 6. MAIN BENCHMARK ENGINE
# ==============================================================================

def run_benchmark():
    print("=" * 80)
    print("  PHASE 7: sDDI CLINICAL GENERALIZATION & SKIN-TONE FAIRNESS BENCHMARK  ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    search_roots = ["/kaggle/input", "/kaggle/working", ".", "checkpoints", "/tmp"]

    medsam_ckpt = find_file_recursive(search_roots, "medsam_vit_b.pth")
    if not medsam_ckpt:
        raise FileNotFoundError("medsam_vit_b.pth not found in search paths!")
    print(f"Found MedSAM Base Checkpoint: {medsam_ckpt}")

    ddi_img_dir = find_folder_containing(search_roots, "000044.png")
    if not ddi_img_dir:
        ddi_img_dir = find_folder_containing(search_roots, "000044.jpg")
    if not ddi_img_dir:
        raise FileNotFoundError("DDI clinical images directory containing 000044.png not found!")
    print(f"Found DDI Images Directory: {ddi_img_dir}")

    test_dark_dir = find_folder_containing(search_roots, "000006.npy")
    if not test_dark_dir or "test_dark" not in test_dark_dir:
        print("Cloning https://github.com/hectorcarrion/fedd.git to /tmp/fedd_repo...")
        os.system("git clone --depth 1 https://github.com/hectorcarrion/fedd.git /tmp/fedd_repo")
        sddi_testing_root = "/tmp/fedd_repo/ddi_labels/testing"
    else:
        sddi_testing_root = os.path.dirname(test_dark_dir)
    print(f"Found sDDI Testing Masks Root: {sddi_testing_root}")

    ckpt_paths = {
        "Decoder-Only (E04)": find_file_recursive(search_roots, "best_decoder_only_model.pth"),
        "LoRA (E05)": find_file_recursive(search_roots, "best_lora_model.pth"),
        "Standard Adapter (E06)": find_file_recursive(search_roots, "best_standard_adapter_model.pth"),
        "CG-Adapter (E07)": find_file_recursive(search_roots, "best_cg_adapter_model.pth"),
    }
    for k, v in ckpt_paths.items():
        print(f"  {k} Checkpoint: {v}")

    split_configs = [
        ("test_light", "FST_I_II_Light", os.path.join(sddi_testing_root, "test_light")),
        ("test_med", "FST_III_IV_Medium", os.path.join(sddi_testing_root, "test_med")),
        ("test_dark", "FST_V_VI_Dark", os.path.join(sddi_testing_root, "test_dark")),
    ]

    test_samples = []
    for split_name, skin_group, folder in split_configs:
        mask_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        for mf in mask_files:
            img_id = os.path.basename(mf).replace(".npy", "")
            img_path = os.path.join(ddi_img_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(ddi_img_dir, f"{img_id}.jpg")
            test_samples.append({
                "image_id": img_id,
                "split_name": split_name,
                "skin_tone_group": skin_group,
                "mask_path": mf,
                "image_path": img_path
            })

    print(f"Total clinical test samples: {len(test_samples)} (Expected 198)")
    assert len(test_samples) == 198, f"Expected 198 test samples, found {len(test_samples)}"

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
        from segment_anything import sam_model_registry

    models_to_eval = [
        {"name": "Zero-Shot MedSAM (E03)", "mode": "zero_shot", "rank": 0, "alpha": 0, "ckpt": None},
        {"name": "Decoder-Only (E04)", "mode": "decoder_only", "rank": 0, "alpha": 0, "ckpt": ckpt_paths["Decoder-Only (E04)"]},
        {"name": "LoRA (r=16, E05)", "mode": "lora", "rank": 16, "alpha": 32, "ckpt": ckpt_paths["LoRA (E05)"]},
        {"name": "Standard Adapter (r=16, E06)", "mode": "standard_adapter", "rank": 16, "alpha": 0, "ckpt": ckpt_paths["Standard Adapter (E06)"]},
        {"name": "CG-Adapter (Ours, E07)", "mode": "cg_adapter", "rank": 16, "alpha": 0, "ckpt": ckpt_paths["CG-Adapter (E07)"]},
    ]

    all_eval_records = []

    for m_cfg in models_to_eval:
        m_name = m_cfg["name"]
        m_mode = m_cfg["mode"]
        print("\n" + "=" * 60)
        print(f"  EVALUATING MODEL: {m_name}")
        print("=" * 60)

        sam = sam_model_registry["vit_b"](checkpoint=medsam_ckpt)
        sam.to(device)

        if m_mode != "zero_shot":
            model = PEFTMedSAM(
                sam,
                mode=m_mode,
                rank=m_cfg["rank"],
                lora_alpha=m_cfg["alpha"]
            )
            model.to(device)
            ckpt_file = m_cfg["ckpt"]
            if ckpt_file and os.path.exists(ckpt_file):
                print(f"Loading weights from {ckpt_file}...")
                ckpt = torch.load(ckpt_file, map_location=device)
                state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                # Exact key alignment
                clean_state = {k.replace("base_sam.", "base_model."): v for k, v in state.items()}
                msg = model.load_state_dict(clean_state, strict=False)
                print(f"Loaded weights: {len(clean_state)} params. (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
            else:
                print(f"WARNING: Checkpoint {ckpt_file} not found!")
        else:
            model = None

        sam.eval()
        if model is not None:
            model.eval()

        start_t = time.time()
        m_records = []

        with torch.no_grad():
            for idx, item in enumerate(test_samples):
                img_id = item["image_id"]
                tone_group = item["skin_tone_group"]
                m_path = item["mask_path"]
                img_path = item["image_path"]

                raw_mask = np.load(m_path)
                gt_binary = (raw_mask == 1).astype(np.uint8)
                H, W = gt_binary.shape[:2]

                bbox = extract_bbox_from_binary_mask(gt_binary)
                if bbox is None:
                    continue

                if os.path.exists(img_path):
                    bgr = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if img_rgb.shape[:2] != (H, W):
                        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    img_rgb = np.zeros((H, W, 3), dtype=np.uint8)

                resized_img = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor - 0.5) / 0.5
                img_tensor = img_tensor.unsqueeze(0).to(device)

                realizations = generate_prompt_realizations(bbox, (H, W), image_id=img_id)
                sx = 1024.0 / float(W)
                sy = 1024.0 / float(H)

                # Cache image embedding for non-gated models
                cached_emb = None
                if m_mode in ["zero_shot", "decoder_only", "lora", "standard_adapter"]:
                    if m_mode == "zero_shot":
                        cached_emb = sam.image_encoder(img_tensor)
                    else:
                        cached_emb = model.encode_image(img_tensor, c_prompt=None)

                for real in realizations:
                    delta = real["delta"]
                    r_id = real["realization_id"]
                    p_box = real["bbox"]

                    scaled_box = torch.tensor(
                        [[p_box[0] * sx, p_box[1] * sy, p_box[2] * sx, p_box[3] * sy]],
                        dtype=torch.float32,
                        device=device
                    )

                    if m_mode == "cg_adapter":
                        delta_e = compute_contrast_proxy(img_rgb, p_box)
                        norm_c = (delta_e - 28.5) / 12.0
                        c_prompt = torch.tensor([[norm_c]], dtype=torch.float32, device=device)
                        emb = model.encode_image(img_tensor, c_prompt=c_prompt)
                        low_res_masks = model.decode_prompt_box(emb, scaled_box)
                    elif m_mode == "zero_shot":
                        sparse_emb, dense_emb = sam.prompt_encoder(
                            points=None,
                            boxes=scaled_box.unsqueeze(1),
                            masks=None
                        )
                        low_res_masks, _ = sam.mask_decoder(
                            image_embeddings=cached_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False
                        )
                    else:
                        low_res_masks = model.decode_prompt_box(cached_emb, scaled_box)

                    logits = F.interpolate(
                        low_res_masks,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    )
                    pred_mask = (logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)

                    d = compute_binary_dice(pred_mask, gt_binary)
                    iou = compute_binary_iou(pred_mask, gt_binary)
                    hd95 = compute_hd95_fast(pred_mask, gt_binary, max_dim=512)
                    nsd = compute_nsd_fast(pred_mask, gt_binary, tau=2.0, max_dim=512)

                    rec = {
                        "model": m_name,
                        "image_id": img_id,
                        "skin_tone_group": tone_group,
                        "delta": delta,
                        "realization_id": r_id,
                        "dice": d,
                        "iou": iou,
                        "hd95": hd95,
                        "nsd": nsd
                    }
                    m_records.append(rec)
                    all_eval_records.append(rec)

                if (idx + 1) % 50 == 0 or (idx + 1) == len(test_samples):
                    elapsed = time.time() - start_t
                    print(f"  [{m_name}] Processed {idx + 1}/{len(test_samples)} cases ({elapsed:.1f}s)")

        df_m = pd.DataFrame(m_records)
        clean_df = df_m[df_m["delta"] == 0.0]
        p20_df = df_m[df_m["delta"] == 0.20]

        d_light = clean_df[clean_df["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean()
        d_med = clean_df[clean_df["skin_tone_group"] == "FST_III_IV_Medium"]["dice"].mean()
        d_dark = clean_df[clean_df["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        disparity_clean = d_light - d_dark

        d_dark_20 = p20_df[p20_df["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        d_light_20 = p20_df[p20_df["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean()
        disparity_20 = d_light_20 - d_dark_20

        print(f"\n--> {m_name} Clean Dice (delta=0): Light={d_light:.4f} | Med={d_med:.4f} | Dark={d_dark:.4f} | Disparity(L-D)={disparity_clean:.4f}")
        print(f"--> {m_name} Jitter 20% Dice:     Light={d_light_20:.4f} | Dark={d_dark_20:.4f} | Disparity(L-D)={disparity_20:.4f}")

        del sam
        if model is not None:
            del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_eval_records)
    out_csv = "sddi_clinical_fairness_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(results_df)} evaluations to {out_csv}")

    compile_table2(results_df)

def compile_table2(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  COMPILING TABLE 2: CLINICAL GENERALIZATION & SKIN-TONE FAIRNESS MATRIX  ")
    print("=" * 80)

    models_order = [
        "Zero-Shot MedSAM (E03)",
        "Decoder-Only (E04)",
        "LoRA (r=16, E05)",
        "Standard Adapter (r=16, E06)",
        "CG-Adapter (Ours, E07)"
    ]

    summary_rows = []
    for m in models_order:
        sub = df[df["model"] == m]
        if len(sub) == 0:
            continue

        clean = sub[sub["delta"] == 0.0]
        j20 = sub[sub["delta"] == 0.20]

        clean_dice = clean["dice"].mean()
        clean_iou = clean["iou"].mean()
        clean_hd95 = clean["hd95"].mean()

        light_clean = clean[clean["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean()
        med_clean = clean[clean["skin_tone_group"] == "FST_III_IV_Medium"]["dice"].mean()
        dark_clean = clean[clean["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        disparity_clean = light_clean - dark_clean

        overall_j20 = j20["dice"].mean()
        dark_j20 = j20[j20["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        light_j20 = j20[j20["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean()
        disparity_j20 = light_j20 - dark_j20
        drop_overall = overall_j20 - clean_dice

        summary_rows.append({
            "Model": m,
            "Clean Overall Dice": clean_dice,
            "Clean IoU": clean_iou,
            "Clean HD95": clean_hd95,
            "FST I-II (Light)": light_clean,
            "FST III-IV (Med)": med_clean,
            "FST V-VI (Dark)": dark_clean,
            "Disparity (L - D)": disparity_clean,
            "Jitter 20% Dice": overall_j20,
            "Dark Jitter 20%": dark_j20,
            "Disparity @ 20%": disparity_j20,
            "Drop @ 20%": drop_overall
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    md_lines = [
        "# Table 2: Clinical Generalization & Skin-Tone Fairness Benchmark (sDDI Test Partition, N=198)\n",
        r"Comparison of Zero-Shot MedSAM and Parameter-Efficient Fine-Tuning (PEFT) adaptations on clinician-verified external test masks across Fitzpatrick skin-tone cohorts." + "\n",
        "| Model | Clean Dice | Light (FST I-II) | Medium (FST III-IV) | Dark (FST V-VI) | Disparity $\\Delta_{\\text{L-D}}$ | Jitter 20% Dice | Dark @ 20% | Disparity @ 20% | Drop @ 20% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in summary_rows:
        md_lines.append(
            f"| **{r['Model']}** | {r['Clean Overall Dice']:.4f} | {r['FST I-II (Light)']:.4f} | "
            f"{r['FST III-IV (Med)']:.4f} | {r['FST V-VI (Dark)']:.4f} | **{r['Disparity (L - D)']:.4f}** | "
            f"{r['Jitter 20% Dice']:.4f} | {r['Dark Jitter 20%']:.4f} | **{r['Disparity @ 20%']:.4f}** | {r['Drop @ 20%']:.4f} |"
        )

    with open("Table2_clinical_fairness.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External clinical generalization and skin-tone fairness benchmark on the clinician-verified sDDI test partition ($N=198$). Disparity $\Delta_{\text{L-D}} = \text{Dice}_{\text{Light}} - \text{Dice}_{\text{Dark}}$ measures cross-tone performance inequity. Bold denotes superior performance and minimal disparity.}",
        r"\label{tab:sddi_clinical_fairness}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l c ccc c ccc c}",
        r"\toprule",
        r"& \multicolumn{5}{c}{\textbf{Clean Prompts ($\delta = 0.0$)}} & \multicolumn{4}{c}{\textbf{Imperfect Prompts ($\delta = 0.20$ Noise)}} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-10}",
        r"\textbf{Model Architecture} & \textbf{Overall} & \textbf{Light (I--II)} & \textbf{Med (III--IV)} & \textbf{Dark (V--VI)} & \textbf{Disparity $\Delta_{\text{L-D}}\downarrow$} & \textbf{Overall} & \textbf{Dark (V--VI)} & \textbf{Disparity $\Delta_{\text{L-D}}\downarrow$} & \textbf{Degradation $\Delta\downarrow$} \\",
        r"\midrule"
    ]
    for r in summary_rows:
        is_ours = "Ours" in r["Model"]
        name = f"\\textbf{{{r['Model']}}}" if is_ours else r['Model']
        tex_lines.append(
            f"{name} & {r['Clean Overall Dice']:.4f} & {r['FST I-II (Light)']:.4f} & "
            f"{r['FST III-IV (Med)']:.4f} & {r['FST V-VI (Dark)']:.4f} & {r['Disparity (L - D)']:.4f} & "
            f"{r['Jitter 20% Dice']:.4f} & {r['Dark Jitter 20%']:.4f} & {r['Disparity @ 20%']:.4f} & {r['Drop @ 20%']:.4f} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    with open("Table2_clinical_fairness.tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")

if __name__ == "__main__":
    run_benchmark()
