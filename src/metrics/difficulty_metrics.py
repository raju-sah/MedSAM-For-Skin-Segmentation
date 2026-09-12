"""Independent segmentation difficulty and boundary complexity metrics.

Provides mathematical formulations of segmentation difficulty derived independently
from image gradients and ground-truth morphological contours:
1. Boundary Transition Ambiguity (Scharr gradient strength across boundary band)
2. Boundary Morphological Complexity (Circularity / Isoperimetric distortion)
3. Lesion-to-Skin Ground-Truth Contrast
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def compute_boundary_transition_gradient(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    band_radius: int = 5
) -> float:
    """Compute the mean image gradient magnitude across the ground-truth boundary band.

    A lower boundary gradient corresponds directly to diffuse, low-contrast,
    ambiguous boundaries that cause segmentation failure.

    Args:
        image_rgb: (H, W, 3) uint8 image.
        binary_mask: (H, W) uint8 binary mask (0 = bg, >0 = lesion).
        band_radius: Morphological dilation/erosion kernel radius.

    Returns:
        Mean gradient magnitude across the boundary transition band.
    """
    H, W = binary_mask.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))

    # Binary boundary band = Dilated mask XOR Eroded mask
    dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel)
    boundary_band = (dilated > 0) ^ (eroded > 0)

    if np.sum(boundary_band) == 0:
        return 0.0

    # Convert to grayscale float
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Scharr gradient for optimal rotational symmetry
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    mean_grad = float(np.mean(grad_mag[boundary_band]))
    return mean_grad


def compute_shape_complexity(binary_mask: np.ndarray) -> Dict[str, float]:
    """Compute morphological boundary complexity and circularity distortion.

    Circularity = 4 * pi * Area / (Perimeter^2)
    Difficulty_shape = 1.0 - Circularity (higher means more irregular/spiculated)

    Args:
        binary_mask: (H, W) uint8 binary mask.

    Returns:
        Dictionary with circularity and shape complexity difficulty score.
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"area": 0.0, "perimeter": 0.0, "circularity": 0.0, "shape_complexity": 1.0}

    # Use primary contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, closed=True))

    if perimeter <= 0 or area <= 0:
        return {"area": area, "perimeter": perimeter, "circularity": 0.0, "shape_complexity": 1.0}

    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    circularity = min(1.0, max(0.0, circularity))
    shape_complexity = 1.0 - circularity

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "shape_complexity": shape_complexity
    }


def compute_contour_tortuosity(binary_mask: np.ndarray) -> float:
    """Compute normalized contour tortuosity (Class A: independent of pixel values).

    Measures angular direction variance along the boundary contour.
    Irregular, notched, or dendritic boundaries exhibit high tortuosity.
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.5
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 10:
        return 0.5

    pts = largest_contour.squeeze()
    if pts.ndim != 2 or len(pts) < 10:
        return 0.5

    # Compute tangent vectors
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    # Angular changes
    angle_diffs = np.abs(np.diff(angles))
    # Wrap around [-pi, pi]
    angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
    tortuosity = float(np.mean(angle_diffs)) / np.pi
    return min(1.0, max(0.0, tortuosity))


def compute_independent_difficulty(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive difficulty metrics with explicit independence classification.

    Class A: Purely morphological/geometric (derived strictly from GT mask, zero pixel contrast input).
    Class B: Partially overlapping (contrast derived from true mask).
    Class C: Directly pixel-coupled (gradient across boundary is mathematically proportional to contrast).

    Returns:
        Dictionary containing Class A, B, and C difficulty metrics.
    """
    # Class C: Pixel-coupled transition gradient
    bg_grad = compute_boundary_transition_gradient(image_rgb, binary_mask)
    norm_grad = min(1.0, bg_grad / 400.0)
    boundary_ambiguity_coupled = float(1.0 - norm_grad)

    # Class A: Purely morphological complexity (zero pixel intensity input)
    shape_dict = compute_shape_complexity(binary_mask)
    tortuosity = compute_contour_tortuosity(binary_mask)
    shape_comp = float(shape_dict["shape_complexity"])
    area = float(shape_dict["area"])
    H, W = binary_mask.shape[:2]
    total_area = float(H * W)
    area_frac = area / total_area if total_area > 0 else 0.0
    scale_diff = float(1.0 - min(1.0, area_frac * 20.0))  # smaller lesions -> higher difficulty

    # Composite Class A Difficulty (Purely Geometric)
    morphological_difficulty_class_a = float(0.50 * shape_comp + 0.30 * tortuosity + 0.20 * scale_diff)

    return {
        # Class A: Purely independent morphological difficulty
        "morphological_difficulty_class_a": round(morphological_difficulty_class_a, 4),
        "shape_complexity": round(shape_comp, 4),
        "contour_tortuosity": round(tortuosity, 4),
        "scale_difficulty": round(scale_diff, 4),
        # Class C: Coupled boundary gradient ambiguity
        "boundary_gradient": round(bg_grad, 4),
        "boundary_ambiguity_class_c": round(boundary_ambiguity_coupled, 4),
        "lesion_area_pixels": int(area),
        "lesion_perimeter_pixels": round(shape_dict["perimeter"], 2)
    }
