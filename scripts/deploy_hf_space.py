#!/usr/bin/env python3
"""Deploy Interactive Web Demo to Hugging Face Static Space (raju-ai/CG-MedSAM-Showcase)."""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "raju-ai/CG-MedSAM-Showcase"
LIVE_BACKEND_URL = "https://cliff-knowing-howard-replication.trycloudflare.com"

def prepare_space_bundle(bundle_dir: Path) -> Path:
    """Prepare static files for Hugging Face Space."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(__file__).resolve().parent.parent
    web_demo_dir = root_dir / "web_demo"
    static_dir = web_demo_dir / "static"
    samples_dir = web_demo_dir / "samples"

    # 1. README.md with YAML Space Frontmatter
    readme_content = f"""---
title: CG-MedSAM Interactive Clinical Demo
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
---

# CG-MedSAM: Contrast-Gated Skin Lesion Segmentation

Interactive clinical demonstration of **CG-MedSAM**: Contrast-gated parameter-efficient foundation model adaptation for skin-tone-robust lesion segmentation across Fitzpatrick skin types.

### Live Architecture
- **Interactive Frontend:** Hosted as a static high-performance client in this Hugging Face Space.
- **Inference Engine:** Powered by CG-MedSAM PyTorch weights modulated by optical contrast proxy ($\\Delta E^*_{{ab}}$).
- **Default Cloud Endpoint:** `{LIVE_BACKEND_URL}`

### Resources
- **Model Card & Checkpoints:** [raju-ai/CG-MedSAM](https://huggingface.co/raju-ai/CG-MedSAM)
- **Academic Research Showcase:** [https://raju-sah.github.io/MedSAM-For-Skin-Segmentation/](https://raju-sah.github.io/MedSAM-For-Skin-Segmentation/)
- **GitHub Repository:** [raju-sah/MedSAM-For-Skin-Segmentation](https://github.com/raju-sah/MedSAM-For-Skin-Segmentation)
"""
    with open(bundle_dir / "README.md", "w") as f:
        f.write(readme_content)

    # 2. index.html (Adjust /static/ references to relative)
    with open(static_dir / "index.html", "r") as f:
        html = f.read()
    html = html.replace('href="/static/styles.css"', 'href="styles.css"')
    html = html.replace('src="/static/app.js"', 'src="app.js"')
    with open(bundle_dir / "index.html", "w") as f:
        f.write(html)

    # 3. styles.css
    shutil.copy2(static_dir / "styles.css", bundle_dir / "styles.css")

    # 4. app.js with dynamic API base URL detection
    with open(static_dir / "app.js", "r") as f:
        js = f.read()

    # Prepend dynamic API_BASE routing to app.js
    api_prefix = f"""// Dynamic API Base URL for Hugging Face Space deployment
const API_BASE = (window.location.hostname.includes('huggingface.co') || window.location.hostname.includes('hf.space') || window.location.protocol === 'file:')
  ? '{LIVE_BACKEND_URL}'
  : '';

"""
    # Replace relative /api/ calls with API_BASE + /api/
    js_mod = js.replace("'/api/", "`${API_BASE}/api/")
    js_mod = js_mod.replace('"/api/', '`${API_BASE}/api/')
    js_mod = js_mod.replace("src=\"/api/", "src=\"' + API_BASE + '/api/")

    with open(bundle_dir / "app.js", "w") as f:
        f.write(api_prefix + js_mod)

    # 5. Copy samples directory
    space_samples = bundle_dir / "samples"
    space_samples.mkdir(exist_ok=True)
    for sample_file in samples_dir.glob("*.*"):
        shutil.copy2(sample_file, space_samples / sample_file.name)

    print(f"[+] Successfully staged Space assets in {bundle_dir}")
    return bundle_dir


def deploy_to_huggingface():
    """Deploy prepared static space to Hugging Face Hub."""
    print("================================================================")
    print(f" Deploying to Hugging Face Space: {SPACE_REPO_ID}")
    print("================================================================")

    api = HfApi()
    create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="static", exist_ok=True)

    root_dir = Path(__file__).resolve().parent.parent
    bundle_dir = root_dir / "scratch" / "hf_space_bundle"
    prepare_space_bundle(bundle_dir)

    print(f"[+] Uploading files to {SPACE_REPO_ID} ...")
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=SPACE_REPO_ID,
        repo_type="space"
    )

    space_url = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
    print(f"\n[✓] Space deployment successful!")
    print(f"    🌐 View live on Hugging Face: {space_url}")


if __name__ == "__main__":
    deploy_to_huggingface()
