# MICCAI 2026 & Springer LNCS Double-Blind Submission Checklist

**Paper Title:** Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation  
**Submission Venue:** International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2026)  
**Assigned Paper ID:** 1042  
**Target Format:** Springer Lecture Notes in Computer Science (LNCS)  
**Anonymity Policy:** Strict Double-Blind Review  

---

## 1. Double-Blind Anonymization Compliance Audit

| Item | Requirement | Verification Status | Notes / Location |
| :---: | :--- | :---: | :--- |
| **1.1** | **Author Names & Affiliations Omitted** | **PASS** | LaTeX header: `\author{Anonymous MICCAI Submission \and Paper ID: 1042}`, `\institute{Anonymous Medical AI Research Consortium}` in both `paper/main.tex` and `paper/supplementary.tex`. |
| **1.2** | **Running Author Header Anonymized** | **PASS** | `\authorrunning{Anonymous et al.}` verified. |
| **1.3** | **Self-References in Third Person** | **PASS** | Any prior works cited as third-party (e.g., "Sah et al. demonstrated..." rather than "In our previous work..."). |
| **1.4** | **No Institutional Repository URLs** | **PASS** | Manuscript contains no links to personal GitHub accounts or identifying server IPs during review phase. General anonymous release repository reference used. |
| **1.5** | **Figure Metadata & Watermark Audit** | **PASS** | EXIF / author metadata stripped from all embedded figures (`figure_qualitative_comparisons.png`, `figure1_qualitative_panel.png`). |
| **1.6** | **Grant & Funding Acknowledgments** | **PASS** | Excluded from the review manuscript; reserved for camera-ready version upon acceptance. |

---

## 2. Formatting & Page Limit Specifications

| Item | Requirement | Actual Value | Compliance Status |
| :---: | :--- | :---: | :---: |
| **2.1** | **Document Class** | `llncs.cls` (Springer LNCS) | **PASS** |
| **2.2** | **Main Body Page Limit** | $\le 8$ pages (excluding references) | **PASS** (Paper text comfortably within LNCS 8-page budget) |
| **2.3** | **References Page Limit** | Up to 2 pages | **PASS** (1.5 pages BibTeX references) |
| **2.4** | **Supplementary Material** | Separate PDF, no strict limit | **PASS** (`supplementary.tex` compiled as standalone companion document) |
| **2.5** | **Font Size & Margins** | Standard LNCS geometry (10pt on 12pt baseline) | **PASS** (Untouched standard geometry) |
| **2.6** | **Equation Numbering** | Sequentially numbered $(1), (2), \dots$ | **PASS** (Contrast proxy, gating factor, and loss formulation numbered) |

---

## 3. Scientific & Methodological Rigor

| Item | Validation Step | Benchmark Value | Location |
| :---: | :--- | :---: | :--- |
| **3.1** | **Zero-Leakage Guarantee** | Optical proxy $\Delta E^*_{ab}$ derived strictly from prompt box without GT mask access | Verified in `test_contrast_proxy.py` and Section 2.1 |
| **3.2** | **Evaluation Protocol** | Multi-seed prompt noise ($\delta \in \{0.00, 0.05, 0.10, 0.20\}$) across 3 seeds | Table 1 & Table 2 (ISIC 2018 $N=519$, sDDI $N=198$) |
| **3.3** | **Statistical Significance** | Non-parametric Wilcoxon signed-rank test with Holm-Bonferroni correction | Table 4 ($p < 0.001$ vs. Standard Adapter and Zero-Shot) |
| **3.4** | **Causal Ablation** | Permuted proxy and frozen gate controls isolating contrast signal contribution | Table 3 ($p < 10^{-7}$ for proxy shuffling) |
| **3.5** | **Reproducibility** | Full SHA-256 integrity checksums for all adapter weights | Manifest in `checkpoints/checksums.sha256` |

---

## 4. Figures & Visual Assets Quality Check

- **Color Accessibility:** All qualitative segmentation overlays utilize high-contrast, colorblind-safe palettes (Cyan for Ground Truth, Crimson for Prediction).
- **Resolution:** Rendered at 300 DPI vector / raster density.
- **Aspect Ratio:** Fits neatly within 122mm text column width without margin clipping.

---

## 5. Submission Archive Verification

- **Overleaf Import Bundle:** `paper/overleaf_miccai_submission.tar.gz` verified containing:
  - `main.tex`
  - `supplementary.tex`
  - `references.bib`
  - `llncs.cls` & `splncs04.bst`
  - `figure_qualitative_comparisons.png`
- Single-click import tested and ready for submission portal upload.
