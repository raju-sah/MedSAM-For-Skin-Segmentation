# Master Comparative PEFT Benchmark Report (Table 1)

Evaluation of parameter-efficient adaptation strategies on ISIC 2018 dermoscopy benchmark.

| Model Architecture | Trainable Params | Param % | Clean Dice ($\delta=0$) | 5% Jitter ($\delta=0.05$) | 10% Jitter ($\delta=0.10$) | 20% Jitter ($\delta=0.20$) | Drop at 20% ($\Delta\text{Dice}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-Shot MedSAM (E03) | 0 | 0.00% | 0.9302 | 0.9186 | 0.8857 | 0.7969 | -0.1333 |
| Decoder-Only (E04) | 4,058,340 | 4.33% | 0.9524 | 0.9460 | 0.9397 | 0.8980 | -0.0544 |
| LoRA (r=16, E05) | 4,648,164 | 4.93% | 0.9609 | 0.9559 | 0.9514 | 0.9282 | -0.0327 |
| Standard Adapter (r=16, E06) | 4,362,660 | 4.64% | 0.9633 | 0.9599 | 0.9536 | 0.9296 | -0.0337 |
| **CG-Adapter (Ours, E07)** | 4,363,824 | 4.64% | 0.9638 | 0.9608 | 0.9553 | 0.9323 | -0.0315 |


## Key Empirical Insights:
1. **Direct Contrast Gating Benefit (E07 vs. E06):** Comparing CG-Adapter against the identical ungated Standard Bottleneck Adapter isolates the exact contribution of prompt-conditioned Delta-E*ab gating.
2. **Prompt Noise Invariance:** Quantifies how effectively contrast-conditioned adaptation preserves segmentation boundaries under severe (20%) prompt perturbations.
