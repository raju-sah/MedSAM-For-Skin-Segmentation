"""Prompt-conditioned contrast proxy extraction and validation module.

This module extracts localized lesion-to-skin pigment contrast strictly from the
input RGB image and bounding-box prompt, with zero access to ground-truth masks.
It also provides audit-only functions for empirical validation in E02.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any
import numpy as np
import cv2


@dataclass
class ContrastResult:
    """Stores inference-time contrast features and numerical diagnostics."""
    delta_ita: float
    delta_l: float
    delta_e_ab: float
    delta_a: float
    delta_b: float
    core_median_ita: float
    skin_median_ita: float
    core_median_l: float
    skin_median_l: float
    core_median_a: float
    skin_median_a: float
    core_median_b: float
    skin_median_b: float
    core_valid_pixels: int
    core_filtered_pixels: int
    skin_valid_pixels: int
    skin_filtered_pixels: int
    b_star_singular_pixels: int
    is_valid: bool
    warning_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_lesion_core_proxy(
    bbox: Tuple[int, int, int, int],
    erosion_alpha: float = 0.50,
    img_shape: Optional[Tuple[int, int]] = None
) -> Tuple[int, int, int, int]:
    """Calculate the eroded lesion-core proxy bounding box.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        erosion_alpha: Fraction of box dimensions retained (default 0.50).
        img_shape: (H, W) for boundary clipping if provided.

    Returns:
        (core_xmin, core_ymin, core_xmax, core_ymax)
    """
    x_min, y_min, x_max, y_max = bbox
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)

    margin_x = int(round((1.0 - erosion_alpha) * w / 2.0))
    margin_y = int(round((1.0 - erosion_alpha) * h / 2.0))

    core_xmin = x_min + margin_x
    core_ymin = y_min + margin_y
    core_xmax = x_max - margin_x
    core_ymax = y_max - margin_y

    # Guarantee at least 1 pixel extent if original box is valid
    if core_xmax <= core_xmin:
        core_xmin = x_min + w // 4
        core_xmax = max(core_xmin + 1, x_max - w // 4)
    if core_ymax <= core_ymin:
        core_ymin = y_min + h // 4
        core_ymax = max(core_ymin + 1, y_max - h // 4)

    if img_shape is not None:
        H, W = img_shape
        core_xmin = max(0, min(W - 1, core_xmin))
        core_ymin = max(0, min(H - 1, core_ymin))
        core_xmax = max(core_xmin + 1, min(W, core_xmax))
        core_ymax = max(core_ymin + 1, min(H, core_ymax))

    return (int(core_xmin), int(core_ymin), int(core_xmax), int(core_ymax))


def get_perilesional_background_proxy_mask(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    margin_beta: float = 0.25
) -> np.ndarray:
    """Generate a binary mask of the perilesional background annular halo.

    The mask is 1 for pixels in the expanded outer margin strictly outside the
    prompt bounding box, and 0 elsewhere.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        img_shape: (H, W)
        margin_beta: Relative expansion fraction beyond box width/height.

    Returns:
        Boolean numpy array of shape (H, W) where True indicates proxy skin.
    """
    H, W = img_shape
    x_min, y_min, x_max, y_max = bbox
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)

    pad_x = int(round(margin_beta * w))
    pad_y = int(round(margin_beta * h))

    outer_xmin = max(0, x_min - pad_x)
    outer_ymin = max(0, y_min - pad_y)
    outer_xmax = min(W, x_max + pad_x)
    outer_ymax = min(H, y_max + pad_y)

    mask = np.zeros((H, W), dtype=bool)
    mask[outer_ymin:outer_ymax, outer_xmin:outer_xmax] = True
    # Exclude the interior bounding box completely
    mask[max(0, y_min):min(H, y_max), max(0, x_min):min(W, x_max)] = False

    return mask


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to standard float32 CIE L*a*b*."""
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    float_rgb = image_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(float_rgb, cv2.COLOR_RGB2LAB)
    return lab


def calculate_ita(
    l_star: np.ndarray,
    b_star: np.ndarray,
    b_star_epsilon: float = 1e-4,
    min_l: float = 10.0,
    max_l: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Calculate Individual Typology Angle (ITA) with strict numerical safeguards.

    ITA = arctan((L* - 50) / b*) * (180 / pi)

    Safeguards:
    - Filters extreme shadows (L* < min_l) and specular flash (L* > max_l).
    - When |b*| < b_star_epsilon:
        ITA = +90.0 if L* >= 50 else -90.0
    - Tracks and returns counts of filtered and singular pixels without silent replacement.

    Returns:
        ita_values: 1D array of valid ITA values in degrees.
        valid_mask: Boolean mask indicating valid pixels.
        singular_count: Number of pixels where |b*| < epsilon.
    """
    valid_l = (l_star >= min_l) & (l_star <= max_l) & np.isfinite(l_star) & np.isfinite(b_star)

    l_sub = l_star[valid_l]
    b_sub = b_star[valid_l]

    if len(l_sub) == 0:
        return np.array([], dtype=np.float32), valid_l, 0

    singular = np.abs(b_sub) < b_star_epsilon
    singular_count = int(np.sum(singular))

    ita = np.zeros_like(l_sub, dtype=np.float32)

    # Handle standard non-singular case
    non_sing = ~singular
    ita[non_sing] = np.arctan2(l_sub[non_sing] - 50.0, b_sub[non_sing]) * (180.0 / np.pi)

    # Handle singular case with explicit branch-cut values
    if singular_count > 0:
        ita[singular] = np.where(l_sub[singular] >= 50.0, 90.0, -90.0)

    return ita, valid_l, singular_count


def compute_contrast_proxy(
    image_rgb: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    erosion_alpha: float = 0.50,
    margin_beta: float = 0.25,
    min_luminance: float = 10.0,
    max_luminance: float = 95.0,
    b_star_epsilon: float = 1e-4
) -> ContrastResult:
    """Inference-time prompt-conditioned contrast extractor (Zero-Leakage API).

    Computes Delta-ITA, Delta-L*, and Delta-E*ab strictly from the RGB image and
    the prompt bounding box. Has zero access to ground-truth segmentation masks.

    Args:
        image_rgb: (H, W, 3) uint8 image.
        bounding_box: (x_min, y_min, x_max, y_max)
        erosion_alpha: Core proxy erosion fraction (default 0.50).
        margin_beta: Outer ring expansion fraction (default 0.25).
        min_luminance: L* shadow filter threshold.
        max_luminance: L* specularity filter threshold.
        b_star_epsilon: Singularity threshold for b*.

    Returns:
        ContrastResult dataclass with contrast metrics and numerical diagnostics.
    """
    H, W, C = image_rgb.shape
    x_min, y_min, x_max, y_max = bounding_box

    # Degenerate bounding box check
    if x_max <= x_min or y_max <= y_min or x_min >= W or y_min >= H:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=0, core_filtered_pixels=0,
            skin_valid_pixels=0, skin_filtered_pixels=0,
            b_star_singular_pixels=0, is_valid=False,
            warning_message="Degenerate or out-of-bounds bounding box"
        )

    # 1. Geometric Proxies
    core_box = get_lesion_core_proxy(bounding_box, erosion_alpha=erosion_alpha, img_shape=(H, W))
    c_xmin, c_ymin, c_xmax, c_ymax = core_box
    skin_mask = get_perilesional_background_proxy_mask(bounding_box, (H, W), margin_beta=margin_beta)

    # 2. CIE L*a*b* conversion
    lab = rgb_to_lab(image_rgb)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Core pixel values
    core_L = L[c_ymin:c_ymax, c_xmin:c_xmax].flatten()
    core_A = A[c_ymin:c_ymax, c_xmin:c_xmax].flatten()
    core_B = B[c_ymin:c_ymax, c_xmin:c_xmax].flatten()

    # Skin proxy pixel values
    skin_L = L[skin_mask].flatten()
    skin_A = A[skin_mask].flatten()
    skin_B = B[skin_mask].flatten()

    total_core = len(core_L)
    total_skin = len(skin_L)

    if total_core == 0 or total_skin == 0:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=0, core_filtered_pixels=total_core,
            skin_valid_pixels=0, skin_filtered_pixels=total_skin,
            b_star_singular_pixels=0, is_valid=False,
            warning_message="Empty core or skin proxy region"
        )

    # 3. ITA calculation with safeguards
    core_ita, core_valid_mask, core_sing = calculate_ita(
        core_L, core_B, b_star_epsilon=b_star_epsilon, min_l=min_luminance, max_l=max_luminance
    )
    skin_ita, skin_valid_mask, skin_sing = calculate_ita(
        skin_L, skin_B, b_star_epsilon=b_star_epsilon, min_l=min_luminance, max_l=max_luminance
    )

    core_valid_count = len(core_ita)
    skin_valid_count = len(skin_ita)
    core_filtered_count = total_core - core_valid_count
    skin_filtered_count = total_skin - skin_valid_count

    if core_valid_count < 5 or skin_valid_count < 5:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=core_valid_count, core_filtered_pixels=core_filtered_count,
            skin_valid_pixels=skin_valid_count, skin_filtered_pixels=skin_filtered_count,
            b_star_singular_pixels=core_sing + skin_sing, is_valid=False,
            warning_message="Insufficient valid pixels after luminance filtering"
        )

    # 4. Compute median statistics
    core_med_ita = float(np.median(core_ita))
    skin_med_ita = float(np.median(skin_ita))

    core_med_l = float(np.median(core_L[core_valid_mask]))
    skin_med_l = float(np.median(skin_L[skin_valid_mask]))

    core_med_a = float(np.median(core_A[core_valid_mask]))
    skin_med_a = float(np.median(skin_A[skin_valid_mask]))

    core_med_b = float(np.median(core_B[core_valid_mask]))
    skin_med_b = float(np.median(skin_B[skin_valid_mask]))

    # Contrast candidate 1: Delta-ITA
    delta_ita = skin_med_ita - core_med_ita

    # Contrast candidate 2: Delta-L*
    delta_l = skin_med_l - core_med_l

    # Additional components
    delta_a = skin_med_a - core_med_a
    delta_b = skin_med_b - core_med_b

    # Contrast candidate 3: Delta-E*ab
    dl = delta_l
    da = delta_a
    db = delta_b
    delta_e_ab = float(np.sqrt(dl * dl + da * da + db * db))

    return ContrastResult(
        delta_ita=delta_ita,
        delta_l=delta_l,
        delta_e_ab=delta_e_ab,
        delta_a=delta_a,
        delta_b=delta_b,
        core_median_ita=core_med_ita,
        skin_median_ita=skin_med_ita,
        core_median_l=core_med_l,
        skin_median_l=skin_med_l,
        core_median_a=core_med_a,
        skin_median_a=skin_med_a,
        core_median_b=core_med_b,
        skin_median_b=skin_med_b,
        core_valid_pixels=core_valid_count,
        core_filtered_pixels=core_filtered_count,
        skin_valid_pixels=skin_valid_count,
        skin_filtered_pixels=skin_filtered_count,
        b_star_singular_pixels=core_sing + skin_sing,
        is_valid=True,
        warning_message=""
    )


def audit_proxy_composition(
    gt_mask: np.ndarray,
    core_box: Tuple[int, int, int, int],
    skin_mask: np.ndarray,
    is_sddi: bool = False
) -> Dict[str, float]:
    """Audit-only composition analysis using ground-truth mask (E02 validation).

    IMPORTANT: This function is strictly for pre-training validation and audit.
    It must NEVER be called in model training or inference.

    Args:
        gt_mask: 2D integer array of ground-truth mask.
                 For binary masks (ISIC): 0 = background, >0 = lesion.
                 For sDDI multi-class: 0=bg, 1=lesion, 2=marker, 3=ruler, 4=skin.
        core_box: (c_xmin, c_ymin, c_xmax, c_ymax)
        skin_mask: Boolean mask of shape (H, W) for outer proxy.
        is_sddi: Boolean indicating whether multi-class sDDI encoding applies.

    Returns:
        Dictionary with detailed purity and composition percentages.
    """
    c_xmin, c_ymin, c_xmax, c_ymax = core_box
    core_gt = gt_mask[c_ymin:c_ymax, c_xmin:c_xmax].flatten()
    skin_gt = gt_mask[skin_mask].flatten()

    if len(core_gt) == 0 or len(skin_gt) == 0:
        return {
            "core_lesion_purity": 0.0,
            "skin_percentage": 0.0,
            "other_background_percentage": 0.0,
            "marker_contamination": 0.0,
            "ruler_contamination": 0.0,
            "lesion_spillover_into_skin": 0.0,
            "invalid_empty_fraction": 1.0,
            "skin_proxy_purity": 0.0
        }

    if is_sddi:
        # Class 1 = lesion
        core_lesion_purity = float(np.mean(core_gt == 1))
        # Exact semantic composition of background proxy
        skin_pct = float(np.mean(skin_gt == 4))
        other_bg_pct = float(np.mean(skin_gt == 0))
        marker_contam = float(np.mean(skin_gt == 2))
        ruler_contam = float(np.mean(skin_gt == 3))
        lesion_spill = float(np.mean(skin_gt == 1))
        # Total valid background proxy (skin + generic background)
        skin_proxy_purity = skin_pct + other_bg_pct
    else:
        # Binary mask (ISIC): >0 is lesion, 0 is background skin
        core_lesion_purity = float(np.mean(core_gt > 0))
        skin_pct = float(np.mean(skin_gt == 0))
        other_bg_pct = 0.0
        marker_contam = 0.0
        ruler_contam = 0.0
        lesion_spill = float(np.mean(skin_gt > 0))
        skin_proxy_purity = skin_pct

    return {
        "core_lesion_purity": core_lesion_purity,
        "skin_percentage": skin_pct,
        "other_background_percentage": other_bg_pct,
        "marker_contamination": marker_contam,
        "ruler_contamination": ruler_contam,
        "lesion_spillover_into_skin": lesion_spill,
        "invalid_empty_fraction": 0.0,
        "skin_proxy_purity": skin_proxy_purity
    }


def compute_point_contrast_proxy(
    image_rgb: np.ndarray,
    foreground_point: Tuple[int, int],
    background_point: Optional[Tuple[int, int]] = None,
    estimated_radius: int = 25,
    min_luminance: float = 10.0,
    max_luminance: float = 95.0,
    b_star_epsilon: float = 0.01
) -> ContrastResult:
    """Extract zero-leakage color contrast proxy from a point prompt.

    Args:
        image_rgb: RGB image as uint8 array (H, W, 3).
        foreground_point: (x, y) coordinates of lesion foreground click.
        background_point: Optional (x, y) coordinates of perilesional skin click.
        estimated_radius: Radius in pixels used to construct local sampling regions.
        min_luminance: Lower threshold for L* shadow filtering.
        max_luminance: Upper threshold for L* specularity filtering.
        b_star_epsilon: Singularity threshold for b*.

    Returns:
        ContrastResult dataclass containing CIE Lab delta_e_ab and diagnostics.
    """
    H, W, _ = image_rgb.shape
    xf, yf = int(round(foreground_point[0])), int(round(foreground_point[1]))

    # Bounds check
    if xf < 0 or xf >= W or yf < 0 or yf >= H:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=0, core_filtered_pixels=0,
            skin_valid_pixels=0, skin_filtered_pixels=0,
            b_star_singular_pixels=0, is_valid=False,
            warning_message="Foreground point out of image bounds"
        )

    r_core = max(3, int(round(0.4 * estimated_radius)))

    # Coordinate grid
    yy, xx = np.ogrid[:H, :W]
    dist_sq_fg = (xx - xf) ** 2 + (yy - yf) ** 2
    core_mask = dist_sq_fg <= (r_core ** 2)

    if background_point is not None:
        xb, yb = int(round(background_point[0])), int(round(background_point[1]))
        if 0 <= xb < W and 0 <= yb < H:
            r_bg = max(3, int(round(0.4 * estimated_radius)))
            dist_sq_bg = (xx - xb) ** 2 + (yy - yb) ** 2
            skin_mask = dist_sq_bg <= (r_bg ** 2)
        else:
            skin_mask = np.zeros((H, W), dtype=bool)
    else:
        # Annular ring around foreground point
        r_inner = int(round(1.5 * estimated_radius))
        r_outer = int(round(2.5 * estimated_radius))
        skin_mask = (dist_sq_fg >= (r_inner ** 2)) & (dist_sq_fg <= (r_outer ** 2))

    lab = rgb_to_lab(image_rgb)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    core_L = L[core_mask].flatten()
    core_A = A[core_mask].flatten()
    core_B = B[core_mask].flatten()

    skin_L = L[skin_mask].flatten()
    skin_A = A[skin_mask].flatten()
    skin_B = B[skin_mask].flatten()

    total_core = len(core_L)
    total_skin = len(skin_L)

    if total_core == 0 or total_skin == 0:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=0, core_filtered_pixels=total_core,
            skin_valid_pixels=0, skin_filtered_pixels=total_skin,
            b_star_singular_pixels=0, is_valid=False,
            warning_message="Empty core or skin sampling region"
        )

    core_ita, core_valid_mask, core_sing = calculate_ita(
        core_L, core_B, b_star_epsilon=b_star_epsilon, min_l=min_luminance, max_l=max_luminance
    )
    skin_ita, skin_valid_mask, skin_sing = calculate_ita(
        skin_L, skin_B, b_star_epsilon=b_star_epsilon, min_l=min_luminance, max_l=max_luminance
    )

    core_valid_count = len(core_ita)
    skin_valid_count = len(skin_ita)

    if core_valid_count < 3 or skin_valid_count < 3:
        return ContrastResult(
            delta_ita=0.0, delta_l=0.0, delta_e_ab=0.0, delta_a=0.0, delta_b=0.0,
            core_median_ita=0.0, skin_median_ita=0.0,
            core_median_l=0.0, skin_median_l=0.0,
            core_median_a=0.0, skin_median_a=0.0,
            core_median_b=0.0, skin_median_b=0.0,
            core_valid_pixels=core_valid_count, core_filtered_pixels=total_core - core_valid_count,
            skin_valid_pixels=skin_valid_count, skin_filtered_pixels=total_skin - skin_valid_count,
            b_star_singular_pixels=core_sing + skin_sing, is_valid=False,
            warning_message="Insufficient valid pixels after luminance filtering"
        )

    core_med_ita = float(np.median(core_ita))
    skin_med_ita = float(np.median(skin_ita))

    core_med_l = float(np.median(core_L[core_valid_mask]))
    skin_med_l = float(np.median(skin_L[skin_valid_mask]))

    core_med_a = float(np.median(core_A[core_valid_mask]))
    skin_med_a = float(np.median(skin_A[skin_valid_mask]))

    core_med_b = float(np.median(core_B[core_valid_mask]))
    skin_med_b = float(np.median(skin_B[skin_valid_mask]))

    delta_ita = abs(core_med_ita - skin_med_ita)
    delta_l = abs(core_med_l - skin_med_l)
    delta_a = abs(core_med_a - skin_med_a)
    delta_b = abs(core_med_b - skin_med_b)
    delta_e_ab = float(np.sqrt((core_med_l - skin_med_l) ** 2 + (core_med_a - skin_med_a) ** 2 + (core_med_b - skin_med_b) ** 2))

    return ContrastResult(
        delta_ita=delta_ita, delta_l=delta_l, delta_e_ab=delta_e_ab,
        delta_a=delta_a, delta_b=delta_b,
        core_median_ita=core_med_ita, skin_median_ita=skin_med_ita,
        core_median_l=core_med_l, skin_median_l=skin_med_l,
        core_median_a=core_med_a, skin_median_a=skin_med_a,
        core_median_b=core_med_b, skin_median_b=skin_med_b,
        core_valid_pixels=core_valid_count,
        core_filtered_pixels=total_core - core_valid_count,
        skin_valid_pixels=skin_valid_count,
        skin_filtered_pixels=total_skin - skin_valid_count,
        b_star_singular_pixels=core_sing + skin_sing,
        is_valid=True
    )
