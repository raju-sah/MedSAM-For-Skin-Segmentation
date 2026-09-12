"""Normalized multi-realization prompt perturbation module.

Implements the normalized bounding-box prompt perturbation protocol from Section 5:
- Perturbation magnitude scaled by box dimensions (w, h) across noise levels delta in {0.0, 0.05, 0.10, 0.20}.
- For delta > 0, generates M fixed perturbation realizations per image using stored reproducible seeds.
- Enforces strict image boundary clipping and non-degenerate box guarantees.
"""

from typing import List, Tuple, Union, Dict, Any
import numpy as np


PERTURBATION_SEEDS = [101, 102, 103]
DEFAULT_DELTA_LEVELS = [0.0, 0.05, 0.10, 0.20]


def perturb_bounding_box(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    delta: float,
    seed: int
) -> Tuple[int, int, int, int]:
    """Apply normalized perturbation to a single bounding box.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        img_shape: (H, W)
        delta: Perturbation magnitude fraction in [0.0, 1.0]
        seed: Random seed for deterministic perturbation

    Returns:
        (x_min_pert, y_min_pert, x_max_pert, y_max_pert)
    """
    if delta <= 0.0:
        return bbox

    H, W = img_shape
    x_min, y_min, x_max, y_max = bbox
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)

    rng = np.random.default_rng(seed)
    eps = rng.uniform(-delta, delta, size=4)

    # Shift corners
    new_xmin = int(round(x_min + eps[0] * w))
    new_ymin = int(round(y_min + eps[1] * h))
    new_xmax = int(round(x_max + eps[2] * w))
    new_ymax = int(round(y_max + eps[3] * h))

    # Strict clipping within image boundaries
    clamped_xmin = max(0, min(W - 2, new_xmin))
    clamped_ymin = max(0, min(H - 2, new_ymin))
    clamped_xmax = max(clamped_xmin + 2, min(W, new_xmax))
    clamped_ymax = max(clamped_ymin + 2, min(H, new_ymax))

    return (clamped_xmin, clamped_ymin, clamped_xmax, clamped_ymax)


def generate_prompt_realizations(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    image_id: Union[str, int],
    delta_levels: List[float] = DEFAULT_DELTA_LEVELS,
    realization_seeds: List[int] = PERTURBATION_SEEDS
) -> List[Dict[str, Any]]:
    """Generate all standard prompt realizations for a given image and bounding box.

    For delta = 0.0: 1 realization (clean prompt, realization_id=0).
    For delta > 0.0: len(realization_seeds) realizations (e.g. M=3).

    Returns:
        List of dicts with keys:
        - delta: float
        - realization_id: int
        - seed: int
        - bbox: (x_min, y_min, x_max, y_max)
    """
    realizations = []

    # Numeric hash of image_id for seed differentiation
    id_hash = abs(hash(str(image_id))) % 100000

    for delta in delta_levels:
        if delta == 0.0:
            realizations.append({
                "delta": 0.0,
                "realization_id": 0,
                "seed": 0,
                "bbox": bbox
            })
        else:
            for m_idx, base_seed in enumerate(realization_seeds):
                # Combined deterministic seed
                det_seed = (base_seed * 100003 + id_hash + int(delta * 1000)) % (2**31 - 1)
                pert_box = perturb_bounding_box(bbox, img_shape, delta, det_seed)
                realizations.append({
                    "delta": float(delta),
                    "realization_id": m_idx + 1,
                    "seed": det_seed,
                    "bbox": pert_box
                })

    return realizations
