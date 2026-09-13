"""Core Inference and Visual Analysis Utilities for MedSAM PEFT Models.

Shared by the standalone CLI (inference.py) and the Web Demo API (web_demo/app.py).
"""

import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.peft_medsam import PEFTMedSAM
from src.models.medsam_wrapper import MedSAMWrapper, MockMedSAMModel

# Canonical normalization constants from ISIC 2018 training distribution
CANONICAL_MU = 28.5
CANONICAL_SIGMA = 12.0


def compute_lab_contrast_proxy(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    alpha: float = 0.50,
    beta: float = 0.25
) -> Dict[str, Any]:
    """Compute zero-leakage CIE L*a*b* optical color distance between eroded core and background ring.

    Args:
        image_rgb: (H, W, 3) uint8 numpy array in RGB format.
        bbox: (x1, y1, x2, y2) bounding box coordinates.
        alpha: fractional erosion factor for core proxy (default: 0.50).
        beta: fractional expansion factor for perilesional ring (default: 0.25).

    Returns:
        dict containing delta_e, c_normalized, estimated_fst, core_lab, ring_lab, and geometry.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(x1, x2)), min(orig_w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(orig_h, max(y1, y2))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # 1. Eroded Core Proxy (C_core)
    cw_erode = int(round(alpha * w / 2.0))
    ch_erode = int(round(alpha * h / 2.0))
    core_x1 = min(x2 - 1, x1 + cw_erode)
    core_y1 = min(y2 - 1, y1 + ch_erode)
    core_x2 = max(core_x1 + 1, x2 - cw_erode)
    core_y2 = max(core_y1 + 1, y2 - ch_erode)

    # 2. Perilesional Outer Ring (R_ring)
    rw_expand = int(round(beta * w))
    rh_expand = int(round(beta * h))
    outer_x1 = max(0, x1 - rw_expand)
    outer_y1 = max(0, y1 - rh_expand)
    outer_x2 = min(orig_w, x2 + rw_expand)
    outer_y2 = min(orig_h, y2 + rh_expand)

    # Convert RGB to CIE L*a*b*
    lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Standardize OpenCV Lab to canonical CIE ranges: L* in [0, 100], a* in [-128, 127], b* in [-128, 127]
    L = lab_img[:, :, 0] * (100.0 / 255.0)
    A = lab_img[:, :, 1] - 128.0
    B = lab_img[:, :, 2] - 128.0

    # Extract Core Centroid
    core_L = float(np.mean(L[core_y1:core_y2, core_x1:core_x2]))
    core_A = float(np.mean(A[core_y1:core_y2, core_x1:core_x2]))
    core_B = float(np.mean(B[core_y1:core_y2, core_x1:core_x2]))

    # Extract Ring Mask: inside outer rectangle but strictly outside prompt bbox
    ring_mask = np.zeros((orig_h, orig_w), dtype=bool)
    ring_mask[outer_y1:outer_y2, outer_x1:outer_x2] = True
    ring_mask[y1:y2, x1:x2] = False

    if np.sum(ring_mask) > 0:
        ring_L = float(np.mean(L[ring_mask]))
        ring_A = float(np.mean(A[ring_mask]))
        ring_B = float(np.mean(B[ring_mask]))
    else:
        ring_L, ring_A, ring_B = 50.0, 0.0, 0.0

    # Euclidean color distance in L*a*b* space
    delta_e = float(np.sqrt((core_L - ring_L) ** 2 + (core_A - ring_A) ** 2 + (core_B - ring_B) ** 2))
    c_normalized = float((delta_e - CANONICAL_MU) / CANONICAL_SIGMA)

    # Individual Typology Angle (ITA) and Fitzpatrick category estimation from perilesional skin
    # ITA = arctan((L* - 50) / b*) * 180 / pi
    if abs(ring_B) < 1e-4:
        ita = 0.0
    else:
        ita = float(np.arctan((ring_L - 50.0) / (ring_B + 1e-6)) * (180.0 / np.pi))

    if ring_L >= 62.0 or ita > 41.0:
        estimated_fst = "Light (FST I–II)"
    elif ring_L >= 48.0 or ita > 10.0:
        estimated_fst = "Medium (FST III–IV)"
    else:
        estimated_fst = "Dark (FST V–VI)"

    return {
        "delta_e": round(delta_e, 2),
        "c_normalized": round(c_normalized, 4),
        "estimated_fst": estimated_fst,
        "ring_luminance": round(ring_L, 2),
        "ita_degrees": round(ita, 2),
        "core_lab": [round(core_L, 2), round(core_A, 2), round(core_B, 2)],
        "ring_lab": [round(ring_L, 2), round(ring_A, 2), round(ring_B, 2)],
        "geometry": {
            "prompt_bbox": [x1, y1, x2, y2],
            "core_bbox": [core_x1, core_y1, core_x2, core_y2],
            "outer_bbox": [outer_x1, outer_y1, outer_x2, outer_y2]
        }
    }


def auto_detect_prompt_bbox(image_rgb: np.ndarray) -> Tuple[int, int, int, int]:
    """Automated lesion bounding box estimation based on color saliency and Otsu thresholding.

    Args:
        image_rgb: (H, W, 3) uint8 numpy array.

    Returns:
        (x1, y1, x2, y2) bounding box.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Invert so dark lesion appears as bright foreground
    inv = cv2.bitwise_not(blurred)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to seal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Pick the largest contour reasonably close to center
        center_x, center_y = orig_w / 2.0, orig_h / 2.0
        best_cnt = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            dist_to_center = np.sqrt(((x + w / 2.0) - center_x) ** 2 + ((y + h / 2.0) - center_y) ** 2)
            score = area / (1.0 + 0.005 * dist_to_center)
            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is not None:
            x, y, w, h = cv2.boundingRect(best_cnt)
            # Add 12% padding for clinical margin
            pad_x = int(round(0.12 * w))
            pad_y = int(round(0.12 * h))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(orig_w, x + w + pad_x)
            y2 = min(orig_h, y + h + pad_y)
            return (x1, y1, x2, y2)

    # Fallback to center 60% crop if contour detection yields nothing
    x1 = int(round(0.20 * orig_w))
    y1 = int(round(0.20 * orig_h))
    x2 = int(round(0.80 * orig_w))
    y2 = int(round(0.80 * orig_h))
    return (x1, y1, x2, y2)


def _ensure_segment_anything_compatibility():
    """Ensure segment_anything can be imported cleanly across CPU/GPU environments."""
    import sys
    import types
    from PIL import Image

    if "torchvision.ops" not in sys.modules or "torchvision.transforms.functional" not in sys.modules:
        try:
            import torchvision
            import torchvision.ops
        except Exception:
            tv = types.ModuleType("torchvision")
            tv_tf = types.ModuleType("torchvision.transforms")
            tv_tff = types.ModuleType("torchvision.transforms.functional")
            tv_ops = types.ModuleType("torchvision.ops")
            tv_ops_boxes = types.ModuleType("torchvision.ops.boxes")

            def resize(img, size, interpolation=None, max_size=None, antialias=None):
                if isinstance(img, Image.Image):
                    return img.resize((size[1], size[0]) if isinstance(size, (tuple, list)) else (size, size))
                elif isinstance(img, torch.Tensor):
                    return F.interpolate(img.unsqueeze(0) if img.dim() == 3 else img, size=size, mode="bilinear", align_corners=False).squeeze(0)
                return img

            def to_pil_image(pic, mode=None):
                if isinstance(pic, torch.Tensor):
                    return Image.fromarray(pic.byte().cpu().numpy())
                return Image.fromarray(pic)

            def batched_nms(boxes, scores, idxs, iou_threshold):
                return torch.arange(len(boxes))

            def box_area(boxes):
                return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            tv_tff.resize = resize
            tv_tff.to_pil_image = to_pil_image
            tv_ops_boxes.batched_nms = batched_nms
            tv_ops_boxes.box_area = box_area
            tv_ops.boxes = tv_ops_boxes
            tv_tf.functional = tv_tff
            tv.transforms = tv_tf
            tv.ops = tv_ops
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.transforms"] = tv_tf
            sys.modules["torchvision.transforms.functional"] = tv_tff
            sys.modules["torchvision.ops"] = tv_ops
            sys.modules["torchvision.ops.boxes"] = tv_ops_boxes


def load_model(
    model_name: str = "cg_adapter",
    checkpoint_path: Optional[str] = None,
    base_checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, bool]:
    """Load PEFT MedSAM model with base checkpoint and trained weights.

    Returns:
        (model, is_mock_backbone)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_segment_anything_compatibility()

    base_medsam = None
    is_mock = False

    # Check for base MedSAM weights
    possible_base = [
        base_checkpoint_path,
        "checkpoints/medsam_vit_b.pth",
        "work_dir/MedSAM/medsam_vit_b.pth",
        "medsam_vit_b.pth"
    ]
    resolved_base = None
    for p in possible_base:
        if p and os.path.isfile(p):
            resolved_base = p
            break

    if resolved_base is not None:
        try:
            from segment_anything.build_sam import sam_model_registry
            base_medsam = sam_model_registry["vit_b"]()
            state_dict = torch.load(resolved_base, map_location=device)
            base_medsam.load_state_dict(state_dict)
            base_medsam.to(device)
            is_mock = False
        except Exception as e:
            print(f"[ModelLoader] Warning: Unable to load base MedSAM ({e}). Using mock foundation.")
            is_mock = True
    else:
        is_mock = True

    model = PEFTMedSAM(base_medsam=base_medsam, mode=model_name, device=device)
    model.to(device)

    # Resolve trained checkpoint path
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/best_{model_name}_model.pth"

    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Prefix normalization
        adapted_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("base_sam.", "base_model.")
            adapted_dict[new_k] = v

        try:
            missing, unexpected = model.load_state_dict(adapted_dict, strict=False)
            print(f"[ModelLoader] Successfully loaded checkpoint: {checkpoint_path} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        except Exception as e:
            print(f"[ModelLoader] Checkpoint key mismatch ({e}). Proceeding in native mode.")
    else:
        print(f"[ModelLoader] Notice: Checkpoint {checkpoint_path} not found. Running with initial weights.")

    model.eval()
    return model, is_mock


def run_inference(
    model: nn.Module,
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Execute forward inference pass and return predicted mask and physics metrics."""
    if device is None:
        device = next(model.parameters()).device

    orig_h, orig_w = image_rgb.shape[:2]
    # Compute contrast physics
    contrast_info = compute_lab_contrast_proxy(image_rgb, bbox)

    # Preprocessing to 1024x1024
    target_size = 1024
    resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = ((img_tensor - 0.5) / 0.5).to(device)

    # Scale bbox to 1024x1024
    scale_x = target_size / float(orig_w)
    scale_y = target_size / float(orig_h)
    x1, y1, x2, y2 = bbox
    scaled_box = torch.tensor([[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]], dtype=torch.float32, device=device)

    # Contrast tensor for CG-Adapter
    c_tensor = torch.tensor([[contrast_info["c_normalized"]]], dtype=torch.float32, device=device)

    with torch.no_grad():
        if hasattr(model, "mode") and model.mode == "cg_adapter":
            logits = model(img_tensor, scaled_box, c_prompt=c_tensor)
            # Calculate gate factor gamma
            if hasattr(model, "mock_adapter") and hasattr(model.mock_adapter, "gating_mlp"):
                gamma_val = float(model.mock_adapter.gating_mlp(c_tensor).squeeze().cpu().item())
            elif hasattr(model, "adapters") and len(model.adapters) > 0 and hasattr(model.adapters[0], "gating_mlp"):
                gamma_val = float(model.adapters[0].gating_mlp(c_tensor).squeeze().cpu().item())
            else:
                gamma_val = 1.0
        else:
            logits = model(img_tensor, scaled_box)
            gamma_val = 1.0

        # Upsample logits to native resolution
        upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(upsampled).squeeze().cpu().numpy()
        binary_mask = (probs >= 0.50).astype(np.uint8) * 255

    contrast_info["gamma_factor"] = round(gamma_val, 4)
    contrast_info["lesion_area_px"] = int(np.sum(binary_mask > 0))
    contrast_info["lesion_coverage_pct"] = round(float(np.sum(binary_mask > 0)) / (orig_h * orig_w) * 100.0, 2)

    return {
        "mask": binary_mask,
        "probability_map": probs,
        "contrast_info": contrast_info
    }


def create_visual_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    contrast_info: Dict[str, Any],
    model_name: str = "CG-Adapter"
) -> np.ndarray:
    """Create publication-grade composite overlay with bounding box, contour, and HUD metrics."""
    orig_h, orig_w = image_rgb.shape[:2]
    vis = image_rgb.copy()

    # 1. Translucent mask fill (Cyan overlay)
    mask_bool = binary_mask > 0
    overlay_color = np.array([0, 220, 255], dtype=np.uint8)  # Cyan in RGB
    vis[mask_bool] = (0.55 * vis[mask_bool] + 0.45 * overlay_color).astype(np.uint8)

    # 2. Smooth contour boundary (Bright Emerald Green)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 128), 2, cv2.LINE_AA)

    # 3. Draw Prompt Bounding Box (Amber Yellow)
    x1, y1, x2, y2 = contrast_info["geometry"]["prompt_bbox"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

    # 4. Optional subtle indicators for eroded core and background ring
    cx1, cy1, cx2, cy2 = contrast_info["geometry"]["core_bbox"]
    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (80, 160, 255), 1, cv2.LINE_AA)

    # 5. Top-left HUD badge
    hud_h, hud_w = 95, min(orig_w - 20, 360)
    hud_bg = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
    # Blend semi-transparent dark HUD card
    vis[10:10+hud_h, 10:10+hud_w] = (0.25 * vis[10:10+hud_h, 10:10+hud_w] + 0.75 * hud_bg).astype(np.uint8)
    cv2.rectangle(vis, (10, 10), (10+hud_w, 10+hud_h), (80, 80, 80), 1, cv2.LINE_AA)

    # Text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"Model: {model_name.upper()}", (20, 32), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    delta_e = contrast_info["delta_e"]
    gamma_str = f" | Gamma: {contrast_info.get('gamma_factor', 1.0):.2f}" if "gamma_factor" in contrast_info else ""
    cv2.putText(vis, f"DeltaE*ab: {delta_e:.1f}{gamma_str}", (20, 56), font, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    fst_str = contrast_info["estimated_fst"]
    cv2.putText(vis, f"Skin: {fst_str}", (20, 80), font, 0.50, (180, 255, 180), 1, cv2.LINE_AA)

    return vis
