# Table 2: Clinical Generalization & Skin-Tone Fairness Benchmark (sDDI Clinical Test Partition, N=198)

Comparison of Zero-Shot MedSAM and Parameter-Efficient Fine-Tuning (PEFT) adaptations evaluated across Fitzpatrick skin-tone cohorts under clean ($\delta=0.0$) and noisy ($\delta=0.20$) prompt regimes.

- **FST I–II (Light):** $N = 59$

- **FST III–IV (Medium):** $N = 80$

- **FST V–VI (Dark):** $N = 59$

- **Disparity Metric:** $\Delta_{\text{L-D}} = \text{Dice}_{\text{Light}} - \text{Dice}_{\text{Dark}}$ (lower is fairer)


| Model Architecture | Clean Dice | Light (I–II) | Med (III–IV) | Dark (V–VI) | Disparity $\Delta_{\text{L-D}}\downarrow$ | Jitter 20% Dice | Dark @ 20% | Disparity @ 20%$\downarrow$ | Jitter Drop$\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-Shot MedSAM (E03) | 0.7694 | 0.7886 | 0.7671 | 0.7534 | 0.0351 | 0.7005 | 0.6725 | 0.0500 | -0.0690 |
| Decoder-Only (E04) | 0.8415 | 0.8441 | 0.8480 | 0.8301 | 0.0139 | 0.7964 | **0.7830** | **0.0147** | -0.0451 |
| LoRA (r=16, E05) | 0.8619 | 0.8651 | 0.8662 | **0.8530** | **0.0122** | 0.8009 | 0.7789 | 0.0295 | -0.0610 |
| Standard Adapter (r=16, E06) | 0.8268 | 0.8336 | 0.8398 | 0.8022 | 0.0314 | 0.7894 | 0.7650 | 0.0327 | -0.0374 |
| **CG-Adapter (Ours, E07)** | 0.8402 | 0.8505 | 0.8449 | 0.8234 | 0.0270 | 0.7955 | 0.7710 | 0.0341 | -0.0446 |
