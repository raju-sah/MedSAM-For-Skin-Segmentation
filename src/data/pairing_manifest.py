"""Pairing manifest generator and integrity verification for sDDI.

Verifies exact pairing between:
- DDI clinical RGB images
- sDDI/FEDD segmentation masks
- Skin-tone groups (light: FST I-II, med: FST III-IV, dark: FST V-VI)
- Split assignments (test_light=59, test_med=80, test_dark=59, 10%/train=60, 10%/val=15)
- Confirms zero overlap between external test set and adaptation train/val.
- Explicitly excludes training/10%/test to prevent data contamination.
"""

import os
import glob
from typing import Dict, Any, List, Set, Tuple
import pandas as pd


TONE_GROUP_MAP = {
    "test_light": "FST_I_II_Light",
    "test_med": "FST_III_IV_Medium",
    "test_dark": "FST_V_VI_Dark"
}


def build_sddi_pairing_manifest(
    sddi_labels_root: str,
    images_dir: str = "",
    output_csv: str = "manifests/sddi_pairing_manifest.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build and verify the exact pairing manifest for sDDI benchmark.

    Args:
        sddi_labels_root: Root directory of sDDI masks (e.g. /tmp/fedd_repo/ddi_labels).
        images_dir: Optional path to raw clinical images (e.g. folder containing 000001.png).
        output_csv: Path to save the manifest CSV.

    Returns:
        DataFrame manifest and validation summary dictionary.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    records = []

    # 1. External Test Benchmark (198 images)
    test_splits = {
        "test_light": ("testing/test_light", "FST_I_II_Light", "test"),
        "test_med": ("testing/test_med", "FST_III_IV_Medium", "test"),
        "test_dark": ("testing/test_dark", "FST_V_VI_Dark", "test"),
    }

    test_file_sets: Dict[str, Set[str]] = {}

    for split_key, (rel_path, tone_group, split_role) in test_splits.items():
        folder = os.path.join(sddi_labels_root, rel_path)
        mask_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        test_file_sets[split_key] = set()

        for m_path in mask_files:
            fname = os.path.basename(m_path)
            img_id = fname.replace(".npy", "")
            test_file_sets[split_key].add(img_id)

            # Check potential image path
            img_path = ""
            img_exists = False
            if images_dir and os.path.exists(images_dir):
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                    cand = os.path.join(images_dir, img_id + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        img_exists = True
                        break

            records.append({
                "image_id": img_id,
                "split_name": split_key,
                "split_role": split_role,
                "skin_tone_group": tone_group,
                "mask_path": m_path,
                "image_path": img_path,
                "image_available": img_exists
            })

    # 2. Adaptation Subsets (10% Official Balanced Split)
    adaptation_splits = {
        "10%_train": ("training/10%/train", "Adaptation_Balanced", "adapt_train"),
        "10%_val": ("training/10%/val", "Adaptation_Balanced", "adapt_val"),
    }

    adapt_file_sets: Dict[str, Set[str]] = {}

    for split_key, (rel_path, tone_group, split_role) in adaptation_splits.items():
        folder = os.path.join(sddi_labels_root, rel_path)
        mask_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        adapt_file_sets[split_key] = set()

        for m_path in mask_files:
            fname = os.path.basename(m_path)
            img_id = fname.replace(".npy", "")
            adapt_file_sets[split_key].add(img_id)

            img_path = ""
            img_exists = False
            if images_dir and os.path.exists(images_dir):
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                    cand = os.path.join(images_dir, img_id + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        img_exists = True
                        break

            records.append({
                "image_id": img_id,
                "split_name": split_key,
                "split_role": split_role,
                "skin_tone_group": tone_group,
                "mask_path": m_path,
                "image_path": img_path,
                "image_available": img_exists
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    # 3. Verification & Disjointness Check
    all_test = test_file_sets["test_light"] | test_file_sets["test_med"] | test_file_sets["test_dark"]
    all_adapt = adapt_file_sets["10%_train"] | adapt_file_sets["10%_val"]
    leakage_count = len(all_test & all_adapt)

    summary = {
        "manifest_path": output_csv,
        "total_manifest_records": len(df),
        "test_counts": {
            "test_light": len(test_file_sets["test_light"]),
            "test_med": len(test_file_sets["test_med"]),
            "test_dark": len(test_file_sets["test_dark"]),
            "total_external_test": len(all_test)
        },
        "adaptation_counts": {
            "10%_train": len(adapt_file_sets["10%_train"]),
            "10%_val": len(adapt_file_sets["10%_val"]),
            "total_adaptation": len(all_adapt)
        },
        "leakage_verification": {
            "test_light_expected": 59,
            "test_med_expected": 80,
            "test_dark_expected": 59,
            "test_total_expected": 198,
            "adapt_train_expected": 60,
            "adapt_val_expected": 15,
            "counts_match_protocol": (
                len(test_file_sets["test_light"]) == 59 and
                len(test_file_sets["test_med"]) == 80 and
                len(test_file_sets["test_dark"]) == 59 and
                len(adapt_file_sets["10%_train"]) == 60 and
                len(adapt_file_sets["10%_val"]) == 15
            ),
            "leakage_overlap_count": leakage_count,
            "is_strictly_disjoint": (leakage_count == 0),
            "training_10_percent_test_excluded": True
        }
    }

    return df, summary
