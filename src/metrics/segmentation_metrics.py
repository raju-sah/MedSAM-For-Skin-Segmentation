"""Standardized segmentation and fairness evaluation metrics.

Provides mathematical implementations for:
1. Dice Similarity Coefficient (DSC)
2. Intersection over Union (IoU / Jaccard Index)
3. 95th Percentile Hausdorff Distance (HD95)
4. Normalized Surface Distance (NSD / Surface Dice)
5. Stratified subgroup fairness and prompt disparity metrics
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import scipy.ndimage as ndimage
import cv2


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    r"""Calculate Dice Similarity Coefficient between two binary masks.

    DSC = 2 * |P \cap G| / (|P| + |G|)
    """
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()

    if total == 0:
        return 1.0  # Both empty -> perfect agreement
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    return float(2.0 * intersection / total)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    r"""Calculate Intersection over Union (Jaccard Index).

    IoU = |P \cap G| / |P \cup G|
    """
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def get_mask_contour_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extract boundary contour coordinate points from binary mask."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=int)
    all_pts = np.vstack([c.squeeze() for c in contours if len(c) > 0])
    if all_pts.ndim == 1:
        all_pts = all_pts.reshape(-1, 2)
    return all_pts


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    max_penalty: Optional[float] = None
) -> float:
    """Calculate 95th percentile Hausdorff Distance (HD95) in pixel units.

    Uses Euclidean Distance Transform (EDT) for high precision and efficiency.
    """
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    H, W = p.shape[:2]
    diagonal = float(np.sqrt(H * H + W * W))
    default_penalty = max_penalty if max_penalty is not None else diagonal

    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return default_penalty

    # Extract contours
    pts_p = get_mask_contour_points(p)
    pts_g = get_mask_contour_points(g)

    if len(pts_p) == 0 or len(pts_g) == 0:
        return default_penalty

    # Compute Euclidean Distance Transform from ground truth contour
    canvas_g = np.ones((H, W), dtype=bool)
    canvas_g[pts_g[:, 1], pts_g[:, 0]] = False
    edt_g = ndimage.distance_transform_edt(canvas_g)
    dist_p_to_g = edt_g[pts_p[:, 1], pts_p[:, 0]]

    # Compute Euclidean Distance Transform from prediction contour
    canvas_p = np.ones((H, W), dtype=bool)
    canvas_p[pts_p[:, 1], pts_p[:, 0]] = False
    edt_p = ndimage.distance_transform_edt(canvas_p)
    dist_g_to_p = edt_p[pts_g[:, 1], pts_g[:, 0]]

    combined_distances = np.concatenate([dist_p_to_g, dist_g_to_p])
    return float(np.percentile(combined_distances, 95))


def compute_nsd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tau: float = 2.0
) -> float:
    """Calculate Normalized Surface Distance (NSD / Surface Dice) at tolerance tau.

    Measures the fraction of contour points within distance tau of the other surface.
    """
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    H, W = p.shape[:2]
    pts_p = get_mask_contour_points(p)
    pts_g = get_mask_contour_points(g)

    if len(pts_p) == 0 or len(pts_g) == 0:
        return 0.0

    canvas_g = np.ones((H, W), dtype=bool)
    canvas_g[pts_g[:, 1], pts_g[:, 0]] = False
    edt_g = ndimage.distance_transform_edt(canvas_g)
    dist_p_to_g = edt_g[pts_p[:, 1], pts_p[:, 0]]

    canvas_p = np.ones((H, W), dtype=bool)
    canvas_p[pts_p[:, 1], pts_p[:, 0]] = False
    edt_p = ndimage.distance_transform_edt(canvas_p)
    dist_g_to_p = edt_p[pts_g[:, 1], pts_g[:, 0]]

    n_p_valid = np.sum(dist_p_to_g <= tau)
    n_g_valid = np.sum(dist_g_to_p <= tau)

    total_pts = len(pts_p) + len(pts_g)
    return float((n_p_valid + n_g_valid) / total_pts) if total_pts > 0 else 0.0


def evaluate_segmentation_pair(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tau: float = 2.0
) -> Dict[str, float]:
    """Compute full suite of standard segmentation metrics for a single prediction/target pair."""
    return {
        "dice": round(compute_dice(pred_mask, gt_mask), 4),
        "iou": round(compute_iou(pred_mask, gt_mask), 4),
        "hd95": round(compute_hd95(pred_mask, gt_mask), 4),
        "nsd": round(compute_nsd(pred_mask, gt_mask, tau=tau), 4)
    }
