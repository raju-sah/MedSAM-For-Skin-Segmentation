#!/usr/bin/env python3
"""Checkpoint Integrity Verification Utility for MedSAM PEFT Models.

Verifies local model weights against official SHA-256 checksums.

Usage:
    python scripts/verify_checkpoints.py
    python scripts/verify_checkpoints.py --checkpoints_dir custom/path/
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

# Official SHA-256 Checksums
OFFICIAL_CHECKSUMS = {
    "best_cg_adapter_model.pth": "a09fcf2808dc101c87ea255b79a9830b4cb4c1e2e7ad1bd36e9c6756cbf73b38",
    "best_standard_adapter_model.pth": "8e8e4a04acc7aa05236b34ffceb1f1168d7d48e86b15c232117e41bcca28ead3",
    "best_lora_model.pth": "e8c107d3c34e8122931909adcbb05410be42c1d79665fa61a735c6c7c82ea2ac",
    "best_decoder_only_model.pth": "695430f33f9772da50972ed3a363671f0625a5081804336264be2d02f4ad896f",
    "medsam_vit_b.pth": "34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74"
}


def get_file_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="MedSAM Checkpoint Integrity Verifier")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Directory containing .pth checkpoints (default: checkpoints)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    print("============================================================")
    print(f" Verifying Checkpoints in: {ckpt_dir.resolve()}")
    print("============================================================")

    all_passed = True
    missing_count = 0
    passed_count = 0

    for filename, expected_hash in OFFICIAL_CHECKSUMS.items():
        file_path = ckpt_dir / filename
        if not file_path.is_file():
            print(f"[MISSING]  {filename:<32} (File not found)")
            if filename != "medsam_vit_b.pth":
                all_passed = False
            missing_count += 1
            continue

        actual_hash = get_file_sha256(str(file_path))
        if actual_hash.lower() == expected_hash.lower():
            print(f"[PASSED]   {filename:<32} (SHA-256 match)")
            passed_count += 1
        else:
            print(f"[MISMATCH] {filename:<32}")
            print(f"  Expected: {expected_hash}")
            print(f"  Actual:   {actual_hash}")
            all_passed = False

    print("------------------------------------------------------------")
    if all_passed and missing_count == 0:
        print(f"All {passed_count} checkpoints verified successfully! Integrity 100% intact.")
        sys.exit(0)
    elif all_passed and missing_count > 0:
        print(f"Notice: {passed_count} checkpoints passed, but {missing_count} optional files were missing.")
        sys.exit(0)
    else:
        print("Verification FAILED! One or more checkpoints are corrupted or mismatched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
