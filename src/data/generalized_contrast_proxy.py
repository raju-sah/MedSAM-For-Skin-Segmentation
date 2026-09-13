"""Generalized Zero-Leakage Contrast Proxy Extraction for Multi-Modal Medical Vision.

Extends the contrast-gating formulation beyond skin lesion dermatology to:
1. Cutaneous Lesions (Dermoscopy / Clinical Photography): CIE L*a*b* Color Distance (Delta-E*ab)
2. Colorectal Endoscopy (Polyps vs Mucosa): Mucosal Hemoglobin Absorption Ratio (Delta-H)
3. Diagnostic Ultrasound (Breast / Thyroid Lesions): Acoustic Impedance Echogenicity Contrast (C_US)

Zero-Leakage Guarantee:
All proxies are computed strictly using the clinician's prompt bounding box and its local
surrounding context ring WITHOUT accessing the ground-truth segmentation mask.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2


def extract_prompt_regions(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ring_margin_ratio: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract inner target region and outer context ring from prompt bounding box.
    
    Args:
        image: HxW or HxWxC uint8 or float32 image
        bbox: (xmin, ymin, xmax, ymax) coordinates
        ring_margin_ratio: Fractional expansion for the surrounding background ring
        
    Returns:
        inner_pixels: Pixels inside the target bounding box
        outer_pixels: Pixels in the expanded ring excluding the inner box
    """
    H, W = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(W - 2, int(xmin)))
    ymin = max(0, min(H - 2, int(ymin)))
    xmax = max(xmin + 1, min(W, int(xmax)))
    ymax = max(ymin + 1, min(H, int(ymax)))

    bw = xmax - xmin
    bh = ymax - ymin
    mx = max(2, int(bw * ring_margin_ratio))
    my = max(2, int(bh * ring_margin_ratio))

    ring_xmin = max(0, xmin - mx)
    ring_ymin = max(0, ymin - my)
    ring_xmax = min(W, xmax + mx)
    ring_ymax = min(H, ymax + my)

    inner_crop = image[ymin:ymax, xmin:xmax]
    outer_crop = image[ring_ymin:ring_ymax, ring_xmin:ring_xmax]

    # Create mask for outer ring excluding inner box
    ring_mask = np.ones((ring_ymax - ring_ymin, ring_xmax - ring_xmin), dtype=bool)
    sub_ymin = ymin - ring_ymin
    sub_ymax = ymax - ring_ymin
    sub_xmin = xmin - ring_xmin
    sub_xmax = xmax - ring_xmin
    ring_mask[sub_ymin:sub_ymax, sub_xmin:sub_xmax] = False

    inner_pixels = inner_crop.reshape(-1, inner_crop.shape[-1]) if image.ndim == 3 else inner_crop.flatten()
    outer_pixels = outer_crop[ring_mask]

    # Fallback if outer ring is degenerate
    if len(outer_pixels) == 0:
        outer_pixels = inner_pixels

    return inner_pixels, outer_pixels


def compute_dermatology_contrast(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> float:
    """Compute perceptual CIE L*a*b* Euclidean color distance (Delta E*ab).
    
    Formula: Delta E*ab = sqrt((L1 - L2)^2 + (a1 - a2)^2 + (b1 - b2)^2)
    Domain: Cutaneous Lesions / Skin Photography / Dermoscopy
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Dermatology contrast requires 3-channel RGB image.")

    img_uint8 = np.clip(image_rgb, 0, 255).astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    inner_lab, outer_lab = extract_prompt_regions(lab, bbox)
    mean_inner = np.mean(inner_lab, axis=0)
    mean_outer = np.mean(outer_lab, axis=0)

    delta_e = float(np.linalg.norm(mean_inner - mean_outer))
    return float(np.clip(delta_e, 0.0, 100.0))


def compute_endoscopy_contrast(
    image_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    eps: float = 1e-5
) -> float:
    """Compute mucosal hemoglobin absorption contrast for colonoscopy polyps.
    
    In white-light endoscopy and narrow-band imaging (NBI), adenomatous polyps
    exhibit localized microvascular proliferation with differential red/green chroma.
    
    Formula:
        H_ratio = R / (G + B + eps)
        Delta H = 100 * |H_inner - H_outer| / (H_outer + eps)
    Domain: Colorectal Endoscopy (Polyps vs. Healthy Mucosa)
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Endoscopy contrast requires 3-channel RGB image.")

    img = image_rgb.astype(np.float32)
    inner_rgb, outer_rgb = extract_prompt_regions(img, bbox)

    # Compute hemoglobin chromatic ratio R / (G + B)
    r_in, g_in, b_in = inner_rgb[:, 0], inner_rgb[:, 1], inner_rgb[:, 2]
    r_out, g_out, b_out = outer_rgb[:, 0], outer_rgb[:, 1], outer_rgb[:, 2]

    h_inner = np.mean(r_in / (g_in + b_in + eps))
    h_outer = np.mean(r_out / (g_out + b_out + eps))

    delta_h = 50.0 * abs(h_inner - h_outer) / (h_outer + eps)
    return float(np.clip(delta_h, 0.0, 100.0))


def compute_ultrasound_contrast(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    eps: float = 1e-5
) -> float:
    """Compute acoustic impedance echogenicity contrast ratio for ultrasound.
    
    In diagnostic ultrasound (B-mode breast and thyroid scans), solid lesions
    are characterized by hypoechogenicity relative to adjacent fibroglandular parenchyma.
    
    Formula:
        C_US = 100 * |mu_inner - mu_outer| / (sqrt(sigma^2_inner + sigma^2_outer) + eps)
    Domain: B-Mode Breast / Thyroid Ultrasound
    """
    # Convert to single-channel luminance if RGB
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0].astype(np.float32)
    else:
        gray = image.astype(np.float32)

    inner_gray, outer_gray = extract_prompt_regions(gray, bbox)

    mu_in = np.mean(inner_gray)
    mu_out = np.mean(outer_gray)
    var_in = np.var(inner_gray)
    var_out = np.var(outer_gray)

    c_us = 25.0 * abs(mu_in - mu_out) / (np.sqrt(var_in + var_out) + eps)
    return float(np.clip(c_us, 0.0, 100.0))


def compute_generalized_contrast_proxy(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    modality: str = "dermatology"
) -> float:
    """Unified entrypoint for zero-leakage contrast proxy extraction across modalities.
    
    Args:
        image: RGB or grayscale image array
        bbox: (xmin, ymin, xmax, ymax) clinician prompt box
        modality: One of 'dermatology', 'endoscopy', 'ultrasound'
        
    Returns:
        Scalar contrast proxy value normalized to [0.0, 100.0]
    """
    modality = modality.lower().strip()
    if modality in ("dermatology", "skin", "dermoscopy"):
        return compute_dermatology_contrast(image, bbox)
    elif modality in ("endoscopy", "polyp", "colonoscopy"):
        return compute_endoscopy_contrast(image, bbox)
    elif modality in ("ultrasound", "breast_us", "thyroid_us", "busi"):
        return compute_ultrasound_contrast(image, bbox)
    else:
        raise ValueError(
            f"Unsupported modality '{modality}'. Must be 'dermatology', 'endoscopy', or 'ultrasound'."
        )
