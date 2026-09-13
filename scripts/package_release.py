#!/usr/bin/env python3
"""Automated Release Packaging Script for CG-MedSAM.

Generates:
1. release_assets/cg_medsam_peft_weights_v1.0.tar.gz (All 4 trained PEFT checkpoints)
2. release_assets/RELEASE_MANIFEST.json (Full checksum, architecture, and parameter manifest)
3. release_assets/RELEASE_NOTES.md (Ready for GitHub Releases publication)
"""

import os
import sys
import json
import hashlib
import tarfile
import subprocess
from datetime import datetime
from pathlib import Path


def get_file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints"
    release_dir = root_dir / "release_assets"
    os.makedirs(release_dir, exist_ok=True)

    print("============================================================")
    print(" CG-MedSAM Release Packaging Engine")
    print("============================================================")

    # Checkpoint specifications
    ckpts = [
        {
            "name": "best_cg_adapter_model.pth",
            "model_type": "cg_adapter",
            "role": "Proposed Model (Flagship)",
            "trainable_params": 4363824,
            "trainable_pct": 4.64,
            "description": "Contrast-Gated Bottleneck Adapter with zero-leakage CIE Lab modulation"
        },
        {
            "name": "best_standard_adapter_model.pth",
            "model_type": "standard_adapter",
            "role": "Comparative Baseline (Parameter-Matched)",
            "trainable_params": 4362660,
            "trainable_pct": 4.64,
            "description": "Bottleneck Adapter with static factor gamma=1.0"
        },
        {
            "name": "best_lora_model.pth",
            "model_type": "lora",
            "role": "Comparative Baseline",
            "trainable_params": 4648164,
            "trainable_pct": 4.93,
            "description": "Low-Rank Adaptation (r=16, alpha=32) on ViT attention projections"
        },
        {
            "name": "best_decoder_only_model.pth",
            "model_type": "decoder_only",
            "role": "Comparative Baseline",
            "trainable_params": 4058340,
            "trainable_pct": 4.33,
            "description": "Mask Decoder fine-tuning with frozen ViT-B image encoder"
        }
    ]

    manifest_entries = []
    files_to_tar = []

    for item in ckpts:
        p = checkpoints_dir / item["name"]
        if not p.is_file():
            print(f"[Packaging] Error: Required checkpoint {p} does not exist!")
            sys.exit(1)

        size_bytes = p.stat().st_size
        sha = get_file_sha256(str(p))
        print(f"Verified: {item['name']} ({size_bytes / (1024*1024):.2f} MB) -> {sha[:12]}...")

        item["filename"] = item["name"]
        item["size_bytes"] = size_bytes
        item["size_mb"] = round(size_bytes / (1024 * 1024), 2)
        item["sha256"] = sha
        manifest_entries.append(item)
        files_to_tar.append(p)

    # 1. Create Archive
    tar_name = "cg_medsam_peft_weights_v1.0.tar.gz"
    tar_path = release_dir / tar_name
    print(f"\n[Packaging] Creating tarball archive: {tar_name} ...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files_to_tar:
            tar.add(f, arcname=f.name)

    archive_size = tar_path.stat().st_size
    archive_sha = get_file_sha256(str(tar_path))
    print(f"[Packaging] Created {tar_name} ({archive_size / (1024*1024):.2f} MB)")
    print(f"Archive SHA-256: {archive_sha}")

    # Git Commit Hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root_dir).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    # 2. Generate RELEASE_MANIFEST.json
    manifest_data = {
        "release_version": "1.0.0",
        "release_tag": "v1.0.0",
        "release_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_commit": git_hash,
        "archive": {
            "filename": tar_name,
            "size_bytes": archive_size,
            "size_mb": round(archive_size / (1024 * 1024), 2),
            "sha256": archive_sha
        },
        "checkpoints": manifest_entries
    }

    manifest_path = release_dir / "RELEASE_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[Packaging] Saved release manifest: {manifest_path}")

    # 3. Generate RELEASE_NOTES.md
    notes_content = f"""# CG-MedSAM Release v1.0.0

Official checkpoint release for **CG-MedSAM**: Contrast-Gated Parameter-Efficient MedSAM for Skin-Tone-Robust Lesion Segmentation.

## Release Assets
- **PEFT Model Archive:** `{tar_name}` ({round(archive_size / (1024*1024), 2)} MB)
- **Archive SHA-256:** `{archive_sha}`
- **Git Commit:** `{git_hash}`

## Model Checksums (SHA-256)
| Model | Role | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `best_cg_adapter_model.pth` | **Proposed (Ours)** | 16.72 MB | `{manifest_entries[0]['sha256']}` |
| `best_standard_adapter_model.pth` | Baseline Adapter | 16.70 MB | `{manifest_entries[1]['sha256']}` |
| `best_lora_model.pth` | LoRA Baseline | 101.15 MB | `{manifest_entries[2]['sha256']}` |
| `best_decoder_only_model.pth` | Decoder Baseline | 15.52 MB | `{manifest_entries[3]['sha256']}` |

## Verification Command
```bash
python scripts/verify_checkpoints.py
```
"""
    notes_path = release_dir / "RELEASE_NOTES.md"
    with open(notes_path, "w") as f:
        f.write(notes_content)
    print(f"[Packaging] Saved GitHub release notes: {notes_path}")
    print("\n[Packaging] All release assets generated successfully!\n")


if __name__ == "__main__":
    main()
