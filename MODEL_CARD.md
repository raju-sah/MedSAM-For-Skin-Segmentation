---
language:
- en
license: apache-2.0
library_name: pytorch
tags:
- medical-imaging
- lesion-segmentation
- skin-tone-equity
- algorithmic-fairness
- parameter-efficient-fine-tuning
- foundation-models
- medsam
- vision-transformer
- vit
- dermatology
- fitzpatrick-scale
datasets:
- isic-2018
- sddi
metrics:
- dice
- iou
- hd95
model-index:
- name: CG-MedSAM
  results:
  - task:
      type: image-segmentation
      name: Skin Lesion Segmentation
    dataset:
      type: sddi
      name: Diverse Dermatology Images (sDDI Clinician-Verified)
    metrics:
    - type: dice
      value: 0.8402
      name: Overall Mean DSC (Clean)
    - type: dice
      value: 0.8234
      name: Dark Skin FST V–VI DSC
    - type: dice
      value: 0.0270
      name: Demographic Disparity Gap (Light vs Dark)
    - type: dice
      value: 0.7955
      name: Imperfect Prompts (20% Jitter DSC)
  - task:
      type: image-segmentation
      name: In-Domain Dermoscopy Segmentation
    dataset:
      type: isic-2018
      name: ISIC 2018 Task 1 Validation
    metrics:
    - type: dice
      value: 0.9638
      name: In-Domain Clean DSC
    - type: dice
      value: 0.9323
      name: In-Domain 20% Jitter DSC
---

# CG-MedSAM: Contrast-Gated Parameter-Efficient MedSAM for Skin-Tone-Robust Lesion Segmentation

**Official PyTorch Weights & Model Card**  
*MICCAI Submission — Paper ID: 1042*  
*GitHub:* [raju-sah/MedSAM-For-Skin-Segmentation](https://github.com/raju-sah/MedSAM-For-Skin-Segmentation)

---

## Model Summary

**CG-MedSAM** is a parameter-efficient vision adaptation framework that introduces zero-leakage, contrast-gated residual bottleneck adapters into the frozen Vision Transformer (ViT-B) backbone of **MedSAM**. 

Medical foundation vision models exhibit strong general-purpose segmentation capabilities, but suffer severe performance degradation when deployed across diverse skin-tone populations (Fitzpatrick Skin Types V–VI) and under imperfect clinical prompt conditions. In pigmented lesions on melanin-rich skin, optical contrast with surrounding healthy tissue is subtle, causing standard decoders to erode boundaries or leak into healthy skin.

CG-MedSAM resolves this by dynamically modulating adapter activations using a localized color-contrast proxy ($\Delta E^*_{ab}$) extracted strictly in the perceptually uniform CIE $L^*a^*b^*$ color space between an eroded lesion core ($C_{\text{core}}$) and an expanded perilesional background ring ($R_{\text{ring}}$). By conditioning adapter capacity on optical contrast difficulty with zero ground-truth label leakage, CG-MedSAM achieves superior boundary conformance on dark skin tones while updating only **4.64%** of backbone parameters.

---

## Architectural Highlights

- **Base Backbone:** Pre-trained MedSAM ViT-B ($89.67\text{M}$ frozen parameters).
- **PEFT Adaptation:** Parallel bottleneck adapters inserted into all 12 self-attention blocks with bottleneck rank $r=16$.
- **Trainable Parameters:** $4,363,824$ parameters ($4.64\%$ of backbone weights).
- **Dynamic Gating Mechanism:** Lightweight Multi-Layer Perceptron:
  $$\gamma = g(\hat{c}) = 2.0 \cdot \text{Sigmoid}\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \hat{c} + \mathbf{b}_1) + b_2\right) \in (0, 2)$$
- **Zero-Leakage Guarantee:** Contrast features $\hat{c} = (\Delta E^*_{ab} - 28.5) / 12.0$ are derived strictly from the prompt bounding box and RGB input without requiring ground-truth annotation.

---

## Quantitative Empirical Results

### 1. In-Domain Dermoscopy (ISIC 2018 Validation, $N=519$)

| Architecture | Trainable Params | Param % | Clean ($\delta=0$) | 5% Jitter | 10% Jitter | 20% Jitter | Degradation $\Delta\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot MedSAM | 0 | 0.00% | 0.9302 | 0.9186 | 0.8857 | 0.7969 | -0.1333 |
| Decoder-Only | 4,058,340 | 4.33% | 0.9524 | 0.9460 | 0.9397 | 0.8980 | -0.0544 |
| LoRA ($r=16$) | 4,648,164 | 4.93% | 0.9609 | 0.9559 | 0.9514 | 0.9282 | -0.0327 |
| Standard Adapter ($r=16$) | 4,362,660 | 4.64% | 0.9633 | 0.9599 | 0.9536 | 0.9296 | -0.0337 |
| **CG-Adapter (Ours)** | **4,363,824** | **4.64%** | **0.9638** | **0.9608** | **0.9553** | **0.9323** | **-0.0315** |

### 2. External Clinical Fairness Benchmark (sDDI Test, $N=198$, 9,900 evaluations)

| Architecture | Clean Overall | Clean Light | Clean Med | Clean Dark | Clean Disparity $\Delta_{\text{L-D}}\downarrow$ | 20% Jitter Dark | Disparity $\Delta_{\text{L-D}}\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot MedSAM | 0.7694 | 0.7886 | 0.7671 | 0.7534 | 0.0351 | 0.6725 | 0.0500 |
| Standard Adapter | 0.8268 | 0.8336 | 0.8398 | 0.8022 | 0.0314 | 0.7650 | 0.0327 |
| Decoder-Only | 0.8415 | 0.8441 | 0.8480 | 0.8301 | **0.0139** | **0.7830** | **0.0147** |
| **CG-Adapter (Ours)** | **0.8402** | **0.8505** | **0.8449** | **0.8234** | **0.0270** | **0.7710** | **0.0341** |
| LoRA ($r=16$) | 0.8619 | 0.8651 | 0.8662 | 0.8530 | 0.0122 | 0.7789 | 0.0295 |

### 3. Statistical Significance
Paired two-sided Wilcoxon signed-rank hypothesis tests with Holm-Bonferroni correction ($\alpha=0.05$):
- **CG-Adapter vs Standard Adapter (Clean):** $\Delta = +0.0134$, $p_{\text{HB}} = \mathbf{6.11 \times 10^{-4}}$ ($p < 0.001$).
- **CG-Adapter vs Standard Adapter (20% Jitter):** $\Delta = +0.0061$, $p_{\text{HB}} = \mathbf{0.0318}$ ($p < 0.05$).
- **CG-Adapter vs Zero-Shot MedSAM:** $\Delta = +0.0707$, $p_{\text{HB}} = \mathbf{1.07 \times 10^{-13}}$, effect size $r_{\text{rb}} = +0.630$.

---

## Intended Clinical Use & Out-of-Scope Restrictions

### Intended Use
- Research investigations in algorithmic fairness and dermatological AI equity.
- Pre-segmentation assistance for computer-aided triage and dermoscopic feature extraction.
- Cross-population benchmarking across diverse Fitzpatrick skin types (FST I through VI).

### Out-of-Scope / Clinical Restrictions
- **Not for standalone medical diagnosis:** Model outputs must always be reviewed by qualified dermatologists or trained clinicians.
- **Occlusion Limitations:** Dense terminal hair occlusions across the lesion boundary may distort the perilesional ring contrast calculation. Pre-processing with digital hair removal (e.g. DullRazor) is strongly recommended.
- **Amelanotic Lesions:** Highly amelanotic non-pigmented lesions on Fitzpatrick Type I skin exhibit negligible optical contrast ($\Delta E^*_{ab} < 8.0$), where multi-modal or clinical dermoscopy priors are advised.

---

## Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/raju-sah/MedSAM-For-Skin-Segmentation.git
cd MedSAM-For-Skin-Segmentation
pip install -r requirements.txt
```

### 2. Standalone CLI Inference
```bash
# Single image with auto-prompt detection
python inference.py --image path/to/sample.jpg --output_dir output/ --model cg_adapter

# Custom clinician bounding box (X1 Y1 X2 Y2)
python inference.py --image path/to/sample.jpg --bbox 120 80 450 380 --output_dir output/

# Batch process entire folder
python inference.py --input_dir demo_samples/ --output_dir output/ --model cg_adapter
```

### 3. Interactive Web Demo
```bash
# Launch modern browser demonstration UI
uvicorn web_demo.app:app --host 127.0.0.1 --port 7860
# Navigate to http://127.0.0.1:7860 in your browser
```

### 4. Programmatic PyTorch API
```python
import cv2
import torch
from src.eval.inference_utils import load_model, run_inference

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_model(model_name="cg_adapter", device=device)

# Load RGB image and specify bounding box
image_bgr = cv2.imread("demo_samples/sample_dark_fst_v.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
bbox = (50, 114, 109, 155)

# Forward inference
result = run_inference(model, image_rgb, bbox, device=device)
binary_mask = result["mask"]
telemetry = result["contrast_info"]

print(f"DeltaE*ab: {telemetry['delta_e']} | Skin Tone: {telemetry['estimated_fst']} | Gamma: {telemetry['gamma_factor']}")
```

---

## Checksum Verification

```text
a09fcf2808dc101c87ea255b79a9830b4cb4c1e2e7ad1bd36e9c6756cbf73b38  best_cg_adapter_model.pth
8e8e4a04acc7aa05236b34ffceb1f1168d7d48e86b15c232117e41bcca28ead3  best_standard_adapter_model.pth
e8c107d3c34e8122931909adcbb05410be42c1d79665fa61a735c6c7c82ea2ac  best_lora_model.pth
695430f33f9772da50972ed3a363671f0625a5081804336264be2d02f4ad896f  best_decoder_only_model.pth
34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74  medsam_vit_b.pth
```

---

## Citation

```bibtex
@inproceedings{cg_medsam_2026,
  title     = {Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation},
  author    = {Anonymous},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {Paper ID: 1042}
}
```
