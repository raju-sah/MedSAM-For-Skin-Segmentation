# E01: Dataset and Annotation Audit Report

## Executive Summary
This audit rigorously inspects the two primary benchmarks for the CG-MedSAM research protocol: the **ISIC 2018 Task 1** dermoscopic benchmark and the **sDDI (MICCAI 2023)** clinical multi-tone benchmark.

## 1. Benchmark Audit Overview

| Dataset | Split / Subset | Image Count | Mask Format | Mean Lesion Area (%) | License / Access |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISIC 2018 Task 1** | `Train (80%)` | 2075 | PNG binary (0, 255) | 21.4% | CC0: Public Domain |
| **ISIC 2018 Task 1** | `Val (20%)` | 519 | PNG binary (0, 255) | 21.4% | CC0: Public Domain |
| **sDDI (FEDD / Stanford DDI)** | `test_light` | 59 | NPY int32 multi-class (0-4) | 1.12% | Masks: MIT | Images: Stanford DDI Agreement |
| **sDDI (FEDD / Stanford DDI)** | `test_med` | 80 | NPY int32 multi-class (0-4) | 1.46% | Masks: MIT | Images: Stanford DDI Agreement |
| **sDDI (FEDD / Stanford DDI)** | `test_dark` | 59 | NPY int32 multi-class (0-4) | 2.45% | Masks: MIT | Images: Stanford DDI Agreement |
| **sDDI (FEDD / Stanford DDI)** | `training_10_percent_train` | 60 | NPY int32 multi-class (0-4) | 1.35% | Masks: MIT | Images: Stanford DDI Agreement |
| **sDDI (FEDD / Stanford DDI)** | `training_10_percent_val` | 15 | NPY int32 multi-class (0-4) | 0.73% | Masks: MIT | Images: Stanford DDI Agreement |

## 2. sDDI Audit & Split Integrity Verification
- **Total External Test Masks:** 198 clinician-verified masks
  - `test_light` (FST I–II): 59 masks
  - `test_med` (FST III–IV): 80 masks
  - `test_dark` (FST V–VI): 59 masks
- **Adaptation Subset (10% Official Balanced Split):**
  - `10%/train`: 60 masks (20 light, 20 medium, 20 dark)
  - `10%/val`: 15 masks (5 light, 5 medium, 5 dark)
- **Leakage Verification:**
  - Overlap between `test_light`, `test_med`, and `test_dark`: **0 (Strictly Disjoint)**
  - Overlap between `10%/train` + `10%/val` and the 198 test masks: **0 (Strictly Disjoint)**
- **Mask Class Map:** `0: background`, `1: lesion`, `2: marker`, `3: ruler`, `4: skin`
- **Artifact Contamination Frequencies:** Marker present in 54.2% of dark test cases; Ruler present in 54.2% of dark test cases.

## 3. ISIC 2018 Task 1 Manifest
- **Total Audited Masks:** 2594 masks
- **Reproducible Split Generated:** `manifests/isic2018_task1_split_seed42.csv` (`seed=42`)
  - **Train Partition (80%):** 2075 images
  - **Validation Partition (20%):** 519 images
- **Encoding:** Strictly binary (`uint8`, values `{0, 255}`)
- **Resolution Variation:** Resolutions span from 767x1022 up to 2048x1536 (reinforcing the protocol requirement for normalized prompt perturbations rather than fixed pixel shifts).

## 4. Access & Compliance Status
- **ISIC 2018:** CC0 Public Domain Dedication. Fully accessible.
- **sDDI Masks:** MIT License via Carrion et al. (MICCAI 2023). Available locally in workspace.
- **DDI Clinical Images:** Subject to Stanford Research Use Agreement (`ddi-dataset.github.io`). Images cannot be redistributed; users must sign agreement to retrieve raw clinical JPEGs/PNGs.
