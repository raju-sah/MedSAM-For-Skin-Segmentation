# Table 5: Cross-Domain Multi-Modal Generalization Benchmark
> [!NOTE]
> Evaluates the generalized zero-leakage contrast proxy across three clinical imaging domains:
> Cutaneous skin lesions (CIE $L^*a^*b^*$), Colorectal polyps (hemoglobin chroma $\Delta\mathcal{H}$), and Breast ultrasound (acoustic impedance $\mathcal{C}_{\text{US}}$).

| Modality | Target Contrast Subgroup | Mean Contrast | Zero-Shot MedSAM | Standard Adapter | CG-Adapter (Ours) | Gain vs Standard | Total Gain vs Zero-Shot |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Cutaneous Lesions (sDDI) | High Contrast (FST I–II) | 32.4 | 0.7886 | 0.8336 | **0.8505** | **+1.69%** | **+6.19%** |
| Cutaneous Lesions (sDDI) | Medium Contrast (FST III–IV) | 24.1 | 0.7671 | 0.8398 | **0.8449** | **+0.51%** | **+7.78%** |
| Cutaneous Lesions (sDDI) | Low Contrast (FST V–VI) | 14.8 | 0.7534 | 0.8022 | **0.8234** | **+2.12%** | **+7.00%** |
| Colorectal Polyps (Endoscopy) | High Contrast (Pedunculated) | 41.6 | 0.8120 | 0.8640 | **0.8710** | **+0.70%** | **+5.90%** |
| Colorectal Polyps (Endoscopy) | Low Contrast (Flat / Sessile) | 11.2 | 0.7045 | 0.7830 | **0.8085** | **+2.55%** | **+10.40%** |
| Breast Ultrasound (BUSI) | High Contrast (Circumscribed) | 38.5 | 0.7950 | 0.8420 | **0.8530** | **+1.10%** | **+5.80%** |
| Breast Ultrasound (BUSI) | Low Contrast (Infiltrating / Ill-Defined) | 9.4 | 0.6815 | 0.7590 | **0.7865** | **+2.75%** | **+10.50%** |

### Key Takeaways:
1. **Consistent Advantage on Low-Contrast Boundaries:** Across all three modalities, CG-Adapter achieves its largest performance margins on the lowest contrast cohorts (+2.12% on Dark Skin, +2.55% on Flat Polyps, and +2.75% on Infiltrating Ultrasound).
2. **Identical Parameter Budget:** All gains are achieved without increasing parameter counts over standard bottleneck adapters (4.64% trainable parameters).
3. **Modality-Agnostic Physics Conditioning:** Validates that contrast-conditioned modulation is a generalizable principle across medical imaging modalities.
