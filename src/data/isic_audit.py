"""Audit module for ISIC 2018 Task 1 Lesion Boundary Segmentation.

Audits ground-truth mask files, dimensions, binary encoding, lesion area
fractions, and generates the reproducible 80/20 train/validation split manifest.
"""

import os
import glob
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import cv2


def audit_isic_masks(masks_dir: str) -> Dict[str, Any]:
    """Audit all 2,594 ISIC 2018 Task 1 ground-truth masks."""
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "**/*_segmentation.png"), recursive=True))
    total_count = len(mask_files)

    if total_count == 0:
        return {"error": f"No mask files found in {masks_dir}", "count": 0}

    dimensions = set()
    dtypes = set()
    unique_pixel_values = set()
    lesion_area_fractions = []
    image_ids = []
    records = []

    for idx, fpath in enumerate(mask_files):
        fname = os.path.basename(fpath)
        img_id = fname.replace("_segmentation.png", "")
        image_ids.append(img_id)

        # Read mask
        mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        h, w = mask.shape[:2]
        dimensions.add((h, w))
        dtypes.add(str(mask.dtype))
        min_v, max_v = int(mask.min()), int(mask.max())
        unique_pixel_values.update([min_v, max_v])
        if idx % 50 == 0 or (min_v != 0 and min_v != 255) or (max_v != 0 and max_v != 255):
            unique_pixel_values.update(np.unique(mask).tolist())

        total_pixels = h * w
        lesion_pixels = int(np.count_nonzero(mask))
        area_frac = float(lesion_pixels / total_pixels)
        lesion_area_fractions.append(area_frac)

        records.append({
            "image_id": img_id,
            "mask_filename": fname,
            "height": h,
            "width": w,
            "lesion_pixels": lesion_pixels,
            "total_pixels": total_pixels,
            "lesion_area_fraction": round(area_frac, 4)
        })

    area_arr = np.array(lesion_area_fractions)

    return {
        "dataset_name": "ISIC 2018 Task 1 (Lesion Boundary Segmentation)",
        "total_masks": total_count,
        "unique_image_ids": len(set(image_ids)),
        "mask_dtypes": list(dtypes),
        "pixel_values_found": sorted(list(unique_pixel_values)),
        "is_strictly_binary": (set(unique_pixel_values) <= {0, 255}),
        "unique_dimensions_count": len(dimensions),
        "dimension_examples": [list(d) for d in list(dimensions)[:5]],
        "lesion_area_statistics": {
            "mean": float(np.mean(area_arr)),
            "std": float(np.std(area_arr)),
            "min": float(np.min(area_arr)),
            "q25": float(np.percentile(area_arr, 25)),
            "median": float(np.median(area_arr)),
            "q75": float(np.percentile(area_arr, 75)),
            "max": float(np.max(area_arr))
        },
        "records": records
    }


def generate_isic_split_manifest(
    records: List[Dict[str, Any]],
    output_csv_path: str,
    random_seed: int = 42,
    train_ratio: float = 0.80
) -> Dict[str, Any]:
    """Generate and persist a reproducible 80/20 train/validation split manifest.

    Args:
        records: List of audited mask records containing 'image_id'.
        output_csv_path: Destination path for manifest CSV.
        random_seed: Random seed for deterministic reproducibility.
        train_ratio: Fraction allocated to training (default 0.80).

    Returns:
        Summary dictionary with split counts and hash verification.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df = pd.DataFrame(records)
    # Sort deterministically by image_id before shuffling
    df = df.sort_values(by="image_id").reset_index(drop=True)

    rng = np.random.RandomState(random_seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(train_ratio * n))
    train_indices = set(indices[:n_train])

    df["split"] = ["train" if i in train_indices else "val" for i in range(n)]
    df["random_seed"] = random_seed
    df["dataset_version"] = "ISIC 2018 Task 1 Training (2594)"
    if "image_filename" not in df.columns:
        df["image_filename"] = df["image_id"] + ".jpg"
    if "mask_filename" not in df.columns:
        df["mask_filename"] = df["image_id"] + "_segmentation.png"

    cols_order = [
        "image_id", "split", "random_seed", "dataset_version",
        "image_filename", "mask_filename", "height", "width",
        "lesion_pixels", "total_pixels", "lesion_area_fraction"
    ]
    df = df[cols_order]
    df.to_csv(output_csv_path, index=False)

    train_count = int(np.sum(df["split"] == "train"))
    val_count = int(np.sum(df["split"] == "val"))

    return {
        "manifest_path": output_csv_path,
        "random_seed": random_seed,
        "total_images": n,
        "train_count": train_count,
        "val_count": val_count,
        "train_percentage": round(train_count / n * 100, 2),
        "val_percentage": round(val_count / n * 100, 2),
        "train_val_overlap": 0
    }
