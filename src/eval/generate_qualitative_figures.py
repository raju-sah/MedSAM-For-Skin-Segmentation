"""Generate Publication Qualitative Visualization Panels (E12).

Produces multi-tone comparative panels contrasting:
1. Input Clinical Photograph + Prompt Bounding Box
2. Clinician Ground Truth Annotation
3. Zero-Shot MedSAM (E03)
4. Standard Adapter (E06)
5. CG-Adapter (Ours, E07)
Across Fitzpatrick Skin Types: FST I-II (Light), FST III-IV (Medium), FST V-VI (Dark).
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Blend a colored binary mask over an RGB image with boundary contour."""
    overlay = image_rgb.copy().astype(np.float32)
    m = (mask > 0)
    if np.any(m):
        c_arr = np.array(color, dtype=np.float32)
        overlay[m] = overlay[m] * (1.0 - alpha) + c_arr * alpha
        # Draw contour boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def render_comparison_figure(
    cases_data: List[Dict[str, Any]],
    output_path: str = "reports/figure_qualitative_comparisons.png"
):
    """Render a 3-row x 5-column publication figure.
    
    Rows: [Light (FST I-II), Medium (FST III-IV), Dark (FST V-VI)]
    Cols: [Input + Box, Ground Truth, Zero-Shot, Standard Adapter, CG-Adapter (Ours)]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_rows = len(cases_data)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 4 * n_rows), dpi=300)
    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    col_titles = [
        "(a) Input & Bounding Box",
        "(b) Ground Truth",
        "(c) Zero-Shot MedSAM",
        "(d) Standard Adapter",
        "(e) CG-Adapter (Ours)"
    ]

    for r_idx, c_data in enumerate(cases_data):
        img = c_data["image_rgb"]
        bbox = c_data["bbox"]  # (xmin, ymin, xmax, ymax)
        gt_mask = c_data["gt_mask"]
        zs_mask = c_data["zs_mask"]
        std_mask = c_data["std_mask"]
        cg_mask = c_data["cg_mask"]
        tone_label = c_data.get("tone_label", f"Case {r_idx+1}")

        # 1. Input + Box
        ax0 = axes[r_idx, 0]
        ax0.imshow(img)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), w, h,
            linewidth=2.2, edgecolor="#FFD700", facecolor="none", linestyle="--"
        )
        ax0.add_patch(rect)
        ax0.set_ylabel(tone_label, fontsize=12, fontweight="bold", labelpad=8)

        # 2. GT
        ax1 = axes[r_idx, 1]
        ov_gt = draw_overlay(img, gt_mask, color=(34, 197, 94), alpha=0.5)
        ax1.imshow(ov_gt)

        # 3. Zero-Shot
        ax2 = axes[r_idx, 2]
        ov_zs = draw_overlay(img, zs_mask, color=(239, 68, 68), alpha=0.5)
        ax2.imshow(ov_zs)
        d_zs = c_data.get("zs_dice", 0.0)
        ax2.text(0.04, 0.06, f"DSC: {d_zs:.3f}", transform=ax2.transAxes,
                 color="white", fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75))

        # 4. Standard Adapter
        ax3 = axes[r_idx, 3]
        ov_std = draw_overlay(img, std_mask, color=(245, 158, 11), alpha=0.5)
        ax3.imshow(ov_std)
        d_std = c_data.get("std_dice", 0.0)
        ax3.text(0.04, 0.06, f"DSC: {d_std:.3f}", transform=ax3.transAxes,
                 color="white", fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75))

        # 5. CG-Adapter
        ax4 = axes[r_idx, 4]
        ov_cg = draw_overlay(img, cg_mask, color=(59, 130, 246), alpha=0.5)
        ax4.imshow(ov_cg)
        d_cg = c_data.get("cg_dice", 0.0)
        ax4.text(0.04, 0.06, f"DSC: {d_cg:.3f}", transform=ax4.transAxes,
                 color="#A7F3D0", fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#064E3B", alpha=0.85))

        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered qualitative panel figure to: {output_path}")

if __name__ == "__main__":
    print("Qualitative visualization module loaded successfully.")
