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


@app.post("/api/compare")
async def compare_models(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    """Run dual inference with CG-Adapter and Standard Adapter to generate a side-by-side comparison."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    bbox = (
        max(0, min(orig_w - 1, x1)),
        max(0, min(orig_h - 1, y1)),
        max(1, min(orig_w, x2)),
        max(1, min(orig_h, y2))
    )

    # Load both models
    model_cg, _ = get_or_load_model("cg_adapter")
    model_std, _ = get_or_load_model("standard_adapter")

    # Run inference on both
    res_cg = run_inference(model_cg, img_rgb, bbox, device=DEVICE)
    res_std = run_inference(model_std, img_rgb, bbox, device=DEVICE)

    mask_cg = res_cg["mask"]
    mask_std = res_std["mask"]

    # Compute agreement Dice between the two predictions
    intersection = np.sum((mask_cg > 0) & (mask_std > 0))
    union_sum = np.sum(mask_cg > 0) + np.sum(mask_std > 0)
    agreement_dice = round(float(2.0 * intersection / max(1, union_sum)), 4)

    # Comparative RGB visual overlay
    # Agreement: Green tint [34, 197, 94]
    # CG-Adapter unique: Crimson tint [239, 68, 68]
    # Standard Adapter unique: Amber tint [245, 158, 11]
    overlay = img_rgb.copy().astype(np.float32)
    cg_pos = (mask_cg > 0)
    std_pos = (mask_std > 0)

    both = cg_pos & std_pos
    cg_only = cg_pos & (~std_pos)
    std_only = std_pos & (~cg_pos)

    overlay[both] = 0.45 * overlay[both] + 0.55 * np.array([34, 197, 94], dtype=np.float32)
    overlay[cg_only] = 0.45 * overlay[cg_only] + 0.55 * np.array([239, 68, 68], dtype=np.float32)
    overlay[std_only] = 0.45 * overlay[std_only] + 0.55 * np.array([245, 158, 11], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Add legend to top of comparison image
    cv2.putText(overlay, f"Agreement Dice: {agreement_dice:.3f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "Green: Agree | Red: CG Only | Amber: Std Only", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    _, mask_cg_buf = cv2.imencode(".png", mask_cg)
    _, mask_std_buf = cv2.imencode(".png", mask_std)
    _, overlay_comp_buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "status": "success",
        "agreement_dice": agreement_dice,
        "cg_area_px": int(np.sum(cg_pos)),
        "std_area_px": int(np.sum(std_pos)),
        "agreement_area_px": int(np.sum(both)),
        "contrast_info": res_cg["contrast_info"],
        "mask_cg_base64": f"data:image/png;base64,{base64.b64encode(mask_cg_buf).decode('utf-8')}",
        "mask_std_base64": f"data:image/png;base64,{base64.b64encode(mask_std_buf).decode('utf-8')}",
        "overlay_comparison_base64": f"data:image/png;base64,{base64.b64encode(overlay_comp_buf).decode('utf-8')}"
    }


@app.post("/api/segment_point")
async def segment_lesion_point(
    file: UploadFile = File(...),
    fg_x: int = Form(...),
    fg_y: int = Form(...),
    bg_x: Optional[int] = Form(None),
    bg_y: Optional[int] = Form(None),
    radius: int = Form(30),
    model_name: str = Form("cg_adapter")
):
    """Run point-prompt guided segmentation and return mask, overlay, and point contrast physics."""
    from src.data.contrast_proxy import compute_point_contrast_proxy

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    fg_point = (max(0, min(orig_w - 1, fg_x)), max(0, min(orig_h - 1, fg_y)))
    bg_point = (max(0, min(orig_w - 1, bg_x)), max(0, min(orig_h - 1, bg_y))) if (bg_x is not None and bg_y is not None) else None

    # Compute point contrast proxy
    contrast_res = compute_point_contrast_proxy(img_rgb, fg_point, bg_point, estimated_radius=radius)

    # Construct prompt bounding box centered on point for MedSAM backbone
    r = max(15, radius)
    bbox = (
        max(0, fg_point[0] - r),
        max(0, fg_point[1] - r),
        min(orig_w, fg_point[0] + r),
        min(orig_h, fg_point[1] + r)
    )

    model, is_mock = get_or_load_model(model_name)
    result = run_inference(model, img_rgb, bbox, device=DEVICE)
    mask = result["mask"]
    contrast_info = result["contrast_info"]
    contrast_info["point_delta_e"] = round(float(contrast_res.delta_e_ab), 2)
    contrast_info["prompt_mode"] = "point_click"

    overlay_rgb = create_visual_overlay(img_rgb, mask, contrast_info, model_name=f"{model_name} (Point)")
    # Draw point markers on overlay
    cv2.circle(overlay_rgb, fg_point, 5, (0, 255, 0), -1)
    cv2.circle(overlay_rgb, fg_point, 7, (255, 255, 255), 2)
    if bg_point is not None:
        cv2.circle(overlay_rgb, bg_point, 5, (255, 0, 0), -1)
        cv2.circle(overlay_rgb, bg_point, 7, (255, 255, 255), 2)

    _, mask_buffer = cv2.imencode(".png", mask)
    _, overlay_buffer = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    return {
        "status": "success",
        "model": model_name,
        "is_mock_foundation": is_mock,
        "mask_base64": f"data:image/png;base64,{base64.b64encode(mask_buffer).decode('utf-8')}",
        "overlay_base64": f"data:image/png;base64,{base64.b64encode(overlay_buffer).decode('utf-8')}",
        "dimensions": {"width": orig_w, "height": orig_h},
        "contrast_info": contrast_info,
        "foreground_point": fg_point,
        "background_point": bg_point
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
