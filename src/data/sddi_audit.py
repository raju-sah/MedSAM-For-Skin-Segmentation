"""Audit module for sDDI (Segmentation masks for Diverse Dermatology Images).

Analyzes mask files, class distributions, lesion area statistics, artifact
contamination, and split overlap for the sDDI benchmark.
"""

import os
import glob
from typing import Dict, Any, List, Set, Tuple
import numpy as np


CLASS_MAP = {
    0: "background",
    1: "lesion",
    2: "marker",
    3: "ruler",
    4: "skin"
}


def audit_sddi_split(folder_path: str) -> Dict[str, Any]:
    """Audit a single sDDI folder containing .npy mask files."""
    if not os.path.exists(folder_path):
        return {"error": f"Folder does not exist: {folder_path}", "count": 0}

    files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    count = len(files)

    if count == 0:
        return {"count": 0, "files": []}

    shapes = set()
    dtypes = set()
    unique_classes_found = set()

    lesion_area_fractions = []
    marker_present_count = 0
    ruler_present_count = 0

    file_summaries = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            arr = np.load(fpath)
            shapes.add(arr.shape)
            dtypes.add(str(arr.dtype))
            u = np.unique(arr)
            unique_classes_found.update(u.tolist())

            total_pixels = arr.size
            lesion_pixels = int(np.sum(arr == 1))
            marker_pixels = int(np.sum(arr == 2))
            ruler_pixels = int(np.sum(arr == 3))
            skin_pixels = int(np.sum(arr == 4))
            bg_pixels = int(np.sum(arr == 0))

            lesion_frac = float(lesion_pixels / total_pixels)
            lesion_area_fractions.append(lesion_frac)

            if marker_pixels > 0:
                marker_present_count += 1
            if ruler_pixels > 0:
                ruler_present_count += 1

            file_summaries.append({
                "filename": fname,
                "shape": list(arr.shape),
                "lesion_area_fraction": round(lesion_frac, 4),
                "has_marker": marker_pixels > 0,
                "has_ruler": ruler_pixels > 0,
                "marker_pixels": marker_pixels,
                "ruler_pixels": ruler_pixels,
                "skin_pixels": skin_pixels,
                "bg_pixels": bg_pixels
            })
        except Exception as e:
            file_summaries.append({"filename": fname, "error": str(e)})

    area_arr = np.array(lesion_area_fractions)

    return {
        "count": count,
        "shapes": [list(s) for s in shapes],
        "dtypes": list(dtypes),
        "classes_found": sorted(list(unique_classes_found)),
        "class_meanings": {int(c): CLASS_MAP.get(int(c), "unknown") for c in unique_classes_found},
        "marker_frequency": marker_present_count / count if count > 0 else 0.0,
        "ruler_frequency": ruler_present_count / count if count > 0 else 0.0,
        "lesion_area_fraction": {
            "mean": float(np.mean(area_arr)) if len(area_arr) > 0 else 0.0,
            "std": float(np.std(area_arr)) if len(area_arr) > 0 else 0.0,
            "min": float(np.min(area_arr)) if len(area_arr) > 0 else 0.0,
            "median": float(np.median(area_arr)) if len(area_arr) > 0 else 0.0,
            "max": float(np.max(area_arr)) if len(area_arr) > 0 else 0.0
        },
        "files": file_summaries
    }


def audit_sddi_repository(sddi_root: str) -> Dict[str, Any]:
    """Audit the complete sDDI dataset structure and verify split non-overlap."""
    splits_to_check = {
        "test_light": os.path.join(sddi_root, "testing/test_light"),
        "test_med": os.path.join(sddi_root, "testing/test_med"),
        "test_dark": os.path.join(sddi_root, "testing/test_dark"),
        "training_10_percent_train": os.path.join(sddi_root, "training/10%/train"),
        "training_10_percent_val": os.path.join(sddi_root, "training/10%/val"),
    }

    split_results = {}
    file_sets: Dict[str, Set[str]] = {}

    for split_name, split_path in splits_to_check.items():
        res = audit_sddi_split(split_path)
        split_results[split_name] = res
        file_sets[split_name] = {item["filename"] for item in res.get("files", []) if "filename" in item}

    # Verify split disjointness
    test_light_files = file_sets.get("test_light", set())
    test_med_files = file_sets.get("test_med", set())
    test_dark_files = file_sets.get("test_dark", set())
    all_test_files = test_light_files | test_med_files | test_dark_files

    train_10_files = file_sets.get("training_10_percent_train", set())
    val_10_files = file_sets.get("training_10_percent_val", set())
    adaptation_files = train_10_files | val_10_files

    overlap_light_med = len(test_light_files & test_med_files)
    overlap_light_dark = len(test_light_files & test_dark_files)
    overlap_med_dark = len(test_med_files & test_dark_files)
    overlap_adaptation_test = len(adaptation_files & all_test_files)

    leakage_check = {
        "test_internal_disjoint": (overlap_light_med == 0 and overlap_light_dark == 0 and overlap_med_dark == 0),
        "adaptation_test_disjoint": (overlap_adaptation_test == 0),
        "overlap_adaptation_test_count": overlap_adaptation_test,
        "total_test_masks": len(all_test_files),
        "total_adaptation_train_masks": len(train_10_files),
        "total_adaptation_val_masks": len(val_10_files)
    }

    return {
        "dataset_name": "sDDI (Diverse Dermatology Images - Segmentation Masks)",
        "source": "Carrion et al., MICCAI 2023 (hectorcarrion/fedd)",
        "license_masks": "MIT License",
        "license_images": "DDI Stanford Research Use Agreement (ddi-dataset.github.io)",
        "image_availability_note": "Masks available locally; raw RGB clinical images require signed DDI agreement",
        "leakage_verification": leakage_check,
        "splits": split_results
    }
