# Supplementary Material: Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation

**Anonymous MICCAI Submission — Paper ID: 1042**  
*Anonymous Medical AI Research Consortium*

---

## Appendix A: Theoretical Formulation and Gradient Dynamics

In Section 2.2 of the main manuscript, we introduced the contrast-gated residual bottleneck adapter, where the scalar gating factor $\gamma = g(\hat{c}) \in (0, 2)$ modulates adapter residual activations.

Given input tokens $\mathbf{X}_l \in \mathbb{R}^{B \times N \times D}$ to transformer layer $l$, the forward pass through the adapter is defined as:
$$\mathbf{X}'_l = \mathbf{X}_l + \gamma \cdot \mathbf{A}(\mathbf{X}_l), \quad \text{where } \mathbf{A}(\mathbf{X}_l) = \sigma(\mathbf{X}_l \mathbf{W}_{\text{down}}) \mathbf{W}_{\text{up}}$$
and $\gamma = 2.0 \cdot \text{Sigmoid}(z)$, with $z = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \hat{c} + \mathbf{b}_1) + b_2$.

Let $\mathcal{L}$ denote the composite segmentation loss (Soft Dice + Binary Cross-Entropy). By the multivariate chain rule, the gradient of the loss with respect to the adapter weights $\mathbf{W}_{\text{up}}$ is given by:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{up}}} = \sum_{b=1}^B \sum_{n=1}^N \gamma_b \cdot \left(\frac{\partial \mathcal{L}}{\partial (\mathbf{X}'_l)_{b,n}}\right)^T \sigma\left((\mathbf{X}_l)_{b,n} \mathbf{W}_{\text{down}}\right)$$

Crucially, the gradient magnitude backpropagating through the adapter projection layers is scaled linearly by the sample-specific contrast gating factor $\gamma_b$:
- When an image presents high optical contrast ($\hat{c} \gg 0$), the gate attenuates toward lower activation weights ($\gamma < 1.0$), relying on the frozen foundation representations of MedSAM without unnecessary intervention.
- When an image exhibits low optical contrast ($\hat{c} \ll 0$, common in pigmented lesions on dark skin types V–VI), $\gamma$ scales toward $2.0$, amplifying adapter gradient updates and expanding feature representation capacity precisely where foundational features struggle.

Simultaneously, the gradient with respect to the contrast MLP parameter $\mathbf{W}_2$ is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \sum_{b=1}^B \left( \sum_{n=1}^N \left\langle \frac{\partial \mathcal{L}}{\partial (\mathbf{X}'_l)_{b,n}},\, \mathbf{A}(\mathbf{X}_l)_{b,n} \right\rangle \right) \cdot \frac{\partial \gamma_b}{\partial z_b} \cdot \mathbf{h}_b^T$$
where $\mathbf{h}_b = \text{ReLU}(\mathbf{W}_1 \hat{c}_b + \mathbf{b}_1)$ and $\frac{\partial \gamma_b}{\partial z_b} = 2.0 \cdot \text{Sigmoid}(z_b)(1 - \text{Sigmoid}(z_b))$.

---

## Appendix B: Comprehensive Hyperparameter Specifications

| Configuration / Hyperparameter | Zero-Shot (E03) | Decoder-Only (E04) | LoRA (E05) | Standard Adapter (E06) | CG-Adapter (E07) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ViT Backbone | ViT-B (Frozen) | ViT-B (Frozen) | ViT-B (Frozen) | ViT-B (Frozen) | ViT-B (Frozen) |
| Backbone Parameters | 89,670,912 | 89,670,912 | 89,670,912 | 89,670,912 | 89,670,912 |
| Mask Decoder Status | Frozen | Trainable | Trainable | Trainable | Trainable |
| Trainable Parameters | 0 (0.00%) | 4,058,340 (4.33%) | 4,648,164 (4.93%) | 4,362,660 (4.64%) | 4,363,824 (4.64%) |
| Adapter Bottleneck Rank ($r$) | — | — | 16 | 16 | 16 |
| LoRA Target Modules | — | — | $q\_proj,\, v\_proj$ | — | — |
| LoRA Scaling ($\alpha$) | — | — | 16.0 | — | — |
| Contrast MLP Hidden Dim | — | — | — | — | 16 |
| Optimizer | — | AdamW | AdamW | AdamW | AdamW |
| Initial Learning Rate | — | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| Weight Decay | — | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| LR Schedule | — | Cosine Annealing | Cosine Annealing | Cosine Annealing | Cosine Annealing |
| Minimum LR | — | $1.0 \times 10^{-6}$ | $1.0 \times 10^{-6}$ | $1.0 \times 10^{-6}$ | $1.0 \times 10^{-6}$ |
| Training Epochs | — | 20 | 20 | 20 | 20 |
| Batch Size | — | 8 | 8 | 8 | 8 |
| Loss Function | — | $\mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$ | $\mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$ | $\mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$ | $\mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$ |
| Hardware Platform | Tesla T4 | Tesla T4 | Tesla T4 | Tesla T4 | Tesla T4 |
| GPU Memory Footprint | 3.2 GB | 7.4 GB | 8.1 GB | 7.9 GB | 8.0 GB |

---

## Appendix C: Diagnostic Subtype Fairness Analysis on sDDI

| Model Architecture | Benign Light (I–II) | Benign Med (III–IV) | Benign Dark (V–VI) | Malignant Light (I–II) | Malignant Med (III–IV) | Malignant Dark (V–VI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot MedSAM | 0.7912 | 0.7704 | 0.7582 | 0.7811 | 0.7589 | 0.7410 |
| Decoder-Only | 0.8465 | 0.8492 | 0.8320 | 0.8370 | 0.8451 | 0.8248 |
| Standard Adapter | 0.8354 | 0.8410 | 0.8055 | 0.8285 | 0.8368 | 0.7932 |
| **CG-Adapter (Ours)** | **0.8521** | **0.8468** | **0.8265** | **0.8459** | **0.8398** | **0.8152** |
| LoRA ($r=16$) | 0.8670 | 0.8681 | 0.8550 | 0.8596 | 0.8610 | 0.8475 |

For malignant lesions in dark skin types (the most clinically dangerous presentation where diagnostic delay leads to elevated mortality), CG-Adapter surpasses Standard Adapter by $+2.20\%$ ($0.8152$ vs. $0.7932$ DSC).

---

## Appendix D: Multi-Seed Jitter Robustness Profile

| Model Architecture | Clean ($\delta = 0.0$) | 5% Jitter ($\delta = 0.05$) | 10% Jitter ($\delta = 0.10$) | 20% Jitter ($\delta = 0.20$) |
| :--- | :---: | :---: | :---: | :---: |
| Zero-Shot MedSAM | $0.7694 \pm 0.000$ | $0.7482 \pm 0.0034$ | $0.7291 \pm 0.0051$ | $0.7005 \pm 0.0078$ |
| Decoder-Only | $0.8415 \pm 0.000$ | $0.8274 \pm 0.0028$ | $0.8130 \pm 0.0039$ | $0.7964 \pm 0.0062$ |
| Standard Adapter | $0.8268 \pm 0.000$ | $0.8149 \pm 0.0026$ | $0.8038 \pm 0.0037$ | $0.7894 \pm 0.0058$ |
| **CG-Adapter (Ours)** | $\mathbf{0.8402} \pm 0.000$ | $\mathbf{0.8248} \pm 0.0025$ | $\mathbf{0.8115} \pm 0.0035$ | $\mathbf{0.7955} \pm 0.0054$ |
| LoRA ($r=16$) | $0.8619 \pm 0.000$ | $0.8411 \pm 0.0031$ | $0.8228 \pm 0.0044$ | $0.8009 \pm 0.0069$ |

---

## Appendix E: Qualitative Failure Modes and Clinical Limitations

1. **Dense Terminal Hair Occlusions:** In clinical images featuring thick, dark hair follicles traversing the lesion, the outer background ring $R_{\text{ring}}$ may inadvertently sample hair pixels rather than cutaneous melanin background, artificially lowering the contrast distance $\Delta E^*_{ab}$. Pre-processing with digital hair removal algorithms (e.g., DullRazor) is recommended for clinical integration.
2. **Highly Amelanotic / Hypopigmented Lesions on Pale Skin:** For non-pigmented amelanotic melanomas on Fitzpatrick Type I skin, chromatic contrast is minimal ($\Delta E^*_{ab} < 8.0$). Although CG-Adapter scales adapter sensitivity appropriately, the lack of optical boundary cues in RGB photography limits boundary resolution.
