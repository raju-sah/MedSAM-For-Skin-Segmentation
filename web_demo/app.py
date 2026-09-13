"""FastAPI Backend for MedSAM PEFT Interactive Skin Lesion Segmentation Web Demo.

Provides REST APIs for:
- Model inference across CG-Adapter, Standard Adapter, LoRA, and Decoder-Only models.
- Real-time CIE L*a*b* optical contrast proxy extraction and Fitzpatrick estimation.
- Automatic prompt bounding box proposals.
- Sample gallery streaming and mask exports.
"""

import os
import io
import base64
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.eval.inference_utils import (
    compute_lab_contrast_proxy,
    auto_detect_prompt_bbox,
    load_model,
    run_inference,
    create_visual_overlay
)

app = FastAPI(
    title="CG-MedSAM Skin Lesion Segmentation Demo",
    description="Contrast-Gated Parameter-Efficient MedSAM for Skin-Tone-Robust Segmentation",
    version="1.0.0"
)

# CORS middleware for local frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Cache
MODEL_CACHE: Dict[str, Any] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def get_or_load_model(model_name: str = "cg_adapter"):
    """Retrieve model from cache or initialize."""
    if model_name not in MODEL_CACHE:
        model, is_mock = load_model(model_name=model_name, device=DEVICE)
        MODEL_CACHE[model_name] = {"model": model, "is_mock": is_mock}
    return MODEL_CACHE[model_name]["model"], MODEL_CACHE[model_name]["is_mock"]


@app.get("/api/status")
async def get_system_status():
    """Return hardware, CUDA, and model status."""
    base_exists = os.path.isfile("checkpoints/medsam_vit_b.pth")
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "base_medsam_checkpoint_present": base_exists,
        "loaded_models": list(MODEL_CACHE.keys())
    }


@app.post("/api/auto_prompt")
async def auto_prompt(file: UploadFile = File(...)):
    """Compute automated salient bounding box proposal for an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox = auto_detect_prompt_bbox(img_rgb)
    contrast_info = compute_lab_contrast_proxy(img_rgb, bbox)

    return {
        "bbox": list(bbox),
        "width": img_rgb.shape[1],
        "height": img_rgb.shape[0],
        "contrast_info": contrast_info
    }


@app.post("/api/segment")
async def segment_lesion(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    model_name: str = Form("cg_adapter")
):
    """Run prompt-guided segmentation and return mask, overlay, and telemetry."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Validate bounding box
    bbox = (
        max(0, min(orig_w - 1, x1)),
        max(0, min(orig_h - 1, y1)),
        max(1, min(orig_w, x2)),
        max(1, min(orig_h, y2))
    )

    # Load model
    model, is_mock = get_or_load_model(model_name)

    # Forward Pass
    result = run_inference(model, img_rgb, bbox, device=DEVICE)
    mask = result["mask"]
    contrast_info = result["contrast_info"]

    # Generate composite overlay
    overlay_rgb = create_visual_overlay(img_rgb, mask, contrast_info, model_name=model_name)

    # Encode mask and overlay to PNG base64
    _, mask_buffer = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(mask_buffer).decode("utf-8")

    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    _, overlay_buffer = cv2.imencode(".png", overlay_bgr)
    overlay_b64 = base64.b64encode(overlay_buffer).decode("utf-8")

    return {
        "status": "success",
        "model": model_name,
        "is_mock_foundation": is_mock,
        "mask_base64": f"data:image/png;base64,{mask_b64}",
        "overlay_base64": f"data:image/png;base64,{overlay_b64}",
        "dimensions": {"width": orig_w, "height": orig_h},
        "contrast_info": contrast_info
    }


@app.get("/api/samples")
async def get_curated_samples():
    """Return metadata for curated multi-tone demonstration samples."""
    samples = [
        {
            "id": "dark_fst_v",
            "name": "Dark Skin Lesion (Fitzpatrick V–VI)",
            "description": "Low optical contrast lesion on melanin-rich skin. Demonstrates CG-Adapter dynamic gate expansion.",
            "skin_type": "Dark (FST V–VI)",
            "diagnosis": "Melanocytic Lesion",
            "delta_e_expected": 14.2,
            "filename": "sample_dark_fst_v.jpg",
            "suggested_bbox": [95, 75, 410, 390]
        },
        {
            "id": "medium_fst_iv",
            "name": "Medium Skin Lesion (Fitzpatrick III–IV)",
            "description": "Balanced pigment contrast typical of Mediterranean / Latin American clinical photography.",
            "skin_type": "Medium (FST III–IV)",
            "diagnosis": "Dysplastic Nevus",
            "delta_e_expected": 28.7,
            "filename": "sample_medium_fst_iv.jpg",
            "suggested_bbox": [110, 85, 400, 375]
        },
        {
            "id": "light_fst_ii",
            "name": "Light Skin Lesion (Fitzpatrick I–II)",
            "description": "High optical contrast erythematous lesion with clear chromatic boundary.",
            "skin_type": "Light (FST I–II)",
            "diagnosis": "Superficial Spreading Lesion",
            "delta_e_expected": 44.8,
            "filename": "sample_light_fst_ii.jpg",
            "suggested_bbox": [125, 95, 385, 355]
        }
    ]
    return {"samples": samples}


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    """Serve sample image from samples directory."""
    sample_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.isfile(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(sample_path, media_type="image/jpeg")


# Mount static assets directory
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve SPA index page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>CG-MedSAM Demo UI Initializing...</h1>")
