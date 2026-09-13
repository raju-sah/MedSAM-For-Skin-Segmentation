#!/usr/bin/env python3
"""Automated Hugging Face Model Hub and Space Deployment Script.

Deploys:
1. Model Hub: raju-ai/CG-MedSAM (MODEL_CARD.md, weights, checksums)
2. Space: raju-ai/CG-MedSAM-Demo (Interactive Clinical Segmentation Web Demo)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def deploy_model_hub(api: HfApi, repo_id: str = "raju-ai/CG-MedSAM"):
    """Deploy weights and model card to Hugging Face Model Hub."""
    print(f"\n[HuggingFace] Ensuring model repository exists: {repo_id} ...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints"

    # 1. Upload Model Card as README.md
    model_card_path = root_dir / "MODEL_CARD.md"
    if model_card_path.is_file():
        print("[HuggingFace] Uploading MODEL_CARD.md as README.md ...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )

    # 2. Upload Checksums
    checksum_path = checkpoints_dir / "checksums.sha256"
    if checksum_path.is_file():
        print("[HuggingFace] Uploading checksums.sha256 ...")
        api.upload_file(
            path_or_fileobj=str(checksum_path),
            path_in_repo="checksums.sha256",
            repo_id=repo_id,
            repo_type="model"
        )

    # 3. Upload Trained PEFT Checkpoints
    ckpts = [
        "best_cg_adapter_model.pth",
        "best_standard_adapter_model.pth",
        "best_lora_model.pth",
        "best_decoder_only_model.pth"
    ]
    for ckpt_name in ckpts:
        p = checkpoints_dir / ckpt_name
        if p.is_file():
            print(f"[HuggingFace] Uploading checkpoint: {ckpt_name} ({p.stat().st_size / (1024*1024):.2f} MB) ...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=ckpt_name,
                repo_id=repo_id,
                repo_type="model"
            )

    # 4. Upload Inference Script
    inf_path = root_dir / "inference.py"
    if inf_path.is_file():
        api.upload_file(
            path_or_fileobj=str(inf_path),
            path_in_repo="inference.py",
            repo_id=repo_id,
            repo_type="model"
        )

    print(f"[HuggingFace] Successfully deployed to Model Hub: https://huggingface.co/{repo_id}")


def main():
    api = HfApi()
    who = api.whoami()
    username = who["name"]
    print("============================================================")
    print(f" Hugging Face Deployment Engine")
    print(f" Authenticated User: {username}")
    print("============================================================")

    model_repo = f"{username}/CG-MedSAM"
    deploy_model_hub(api, repo_id=model_repo)


if __name__ == "__main__":
    main()
