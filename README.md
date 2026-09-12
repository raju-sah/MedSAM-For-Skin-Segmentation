# MedSAM-For-Skin-Segmentation

## Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation (CG-MedSAM)

This repository contains the official implementation, validation suite, and reproducible benchmarks for **CG-MedSAM**: a parameter-efficient adaptation framework designed to improve skin lesion segmentation robustness across diverse Fitzpatrick skin-tone groups under both clean and perturbed prompt conditions.

---

### 1. Research Question & Hypothesis
- **Core Question:** Can contrast-conditioned parameter-efficient adaptation of MedSAM maintain reliable skin-lesion segmentation across different skin-tone groups while updating substantially fewer parameters than full fine-tuning?
- **Hypothesis:** Localized prompt-conditioned CIE Lab color distance ($\Delta E^*_{ab}$), extracted strictly from the prompt bounding box with zero ground-truth mask leakage, provides a strong conditioning signal to modulate ViT bottleneck adapters, reducing segmentation degradation on low-contrast lesions and dark skin-tone cohorts (Fitzpatrick V–VI).

---

### 2. Key Contributions & Repository Structure

```
├── configs/
│   └── audit_config.yaml                  # Dataset paths and validation configuration
├── manifests/
│   ├── isic2018_task1_split_seed42.csv     # Reproducible train (2,075) / val (519) split
│   └── sddi_pairing_manifest.csv          # External clinical fairness benchmark (198 test)
├── reports/
│   ├── E02_DECISION.md                    # Formal contrast proxy validation gate decision
│   ├── e02_full_results.csv               # Empirical proxy purity across 198 external masks
│   ├── e03_zero_shot_report.md            # Zero-shot MedSAM baseline performance report
│   └── e03_zero_shot_results.csv          # In-domain perturbation evaluation (5,190 records)
├── scripts/
│   ├── run_e01_audit.py                   # ISIC and sDDI data integrity audits
│   ├── run_e02_full_validation.py         # OLS, bootstrap, and purity audit pipeline
│   ├── kaggle_runner/                     # Kaggle Dual-T4 GPU execution harness (E03 Zero-Shot)
│   └── kaggle_peft_runner/                # Kaggle GPU runner for PEFT training (E04–E08)
├── src/
│   ├── data/
│   │   ├── contrast_proxy.py              # Zero-leakage CIE Lab Delta-E*ab proxy extraction
│   │   ├── perturbation.py                # Multi-realization normalized prompt perturbation engine
│   │   ├── isic_audit.py                  # ISIC 2018 audit utilities
│   │   └── sddi_audit.py                  # sDDI clinical audit utilities
│   ├── eval/
│   │   └── zero_shot_eval.py              # Zero-shot baseline evaluation orchestration
│   ├── metrics/
│   │   └── segmentation_metrics.py        # Dice, IoU, HD95, and NSD (tau=2.0) metrics
│   ├── models/
│   │   ├── adapters.py                    # LoRALinear, BottleneckAdapter, ContrastGatingMLP
│   │   ├── medsam_wrapper.py              # MedSAM ViT-B preprocessing & wrapper
│   │   └── peft_medsam.py                 # Unified PEFT MedSAM model supporting all modes
│   └── training/
│       ├── dataset.py                     # ISIC 2018 dynamic prompt jitter PyTorch dataset
│       ├── loss.py                        # Combined Soft Dice + Binary Cross-Entropy loss
│       └── trainer.py                     # Mixed-precision PEFT adaptation training loop
└── tests/
    ├── test_audit_pipeline.py             # Data split and manifest reproducibility tests
    ├── test_contrast_proxy.py             # Zero-leakage and numerical safety tests
    ├── test_e03_pipeline.py               # Perturbation and metric unit tests
    ├── test_peft_models.py                # Adapter identity and parameter budget tests
    └── test_training_pipeline.py          # Loss differentiability and trainer loop tests
```

---

### 3. Experimental Roadmap

- **E01: Dataset Audits & Leakage Proof:** Rigorous audit of 2,594 ISIC 2018 masks and 198 clinician-verified sDDI masks.
- **E02: Contrast Proxy Validation Gate:** Empirical evaluation of lesion-core purity (98.93% mean purity, 0.00% background spillover) and robust statistical comparison establishing $\Delta E^*_{ab}$ over luminance-only contrast ($\Delta L^*$).
- **E03: Zero-Shot MedSAM Baseline:** Evaluated on Nvidia Tesla T4 GPU (5,190 prompt evaluations). Achieved **0.9302 Dice** on clean prompts and documented the -13.33% degradation down to **0.7969 Dice** under 20% box jitter.
- **E04–E08: Parameter-Efficient Adaptation:**
  - `E04`: Decoder-only baseline
  - `E05`: LoRA ($r=16, \alpha=32$)
  - `E06`: Standard Bottleneck Adapter ($r=16$, constant $\gamma = 1.0$)
  - `E07`: Proposed Contrast-Gated Adapter ($\gamma = g(\Delta E^*_{ab})$)
  - `E08`: Causal signal ablation (constant, shuffled, sign-flipped, $\Delta L^*$, $\Delta\text{ITA}$)

---

### 4. Running Tests
Run the automated test suite (23 unit tests):
```bash
python3 -m unittest discover tests -v
```
