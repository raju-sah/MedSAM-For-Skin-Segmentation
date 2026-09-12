# Table 1: In-Domain Dermoscopic Segmentation and Prompt Noise Robustness (ISIC 2018)

Comparison of parameter-efficient adaptation strategies under clean ($\delta=0.0$) and perturbed bounding-box prompts ($\delta \in \{0.05, 0.10, 0.20\}$).

| Model Architecture | Trainable Params | Param % | Clean Dice ($\delta=0$) | Jitter 5% ($\delta=0.05$) | Jitter 10% ($\delta=0.10$) | Jitter 20% ($\delta=0.20$) | Drop at 20% ($\Delta\text{Dice}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-Shot MedSAM (E03) | 0 (Frozen) | 0.00% | 0.9302 | 0.9186 | 0.8857 | 0.7969 | -0.1333 |
| Decoder-Only (E04) | 4,058,340 | 4.33% | 0.9524 | 0.9460 | 0.9397 | 0.8980 | -0.0544 |
| LoRA (r=16, E05) | 4,648,164 | 4.93% | 0.9609 | 0.9559 | 0.9514 | 0.9282 | -0.0327 |
| Standard Adapter (r=16, E06) | 4,362,660 | 4.64% | 0.9633 | 0.9599 | 0.9536 | 0.9296 | -0.0337 |
| **CG-Adapter (Ours, E07)** | 4,363,824 | 4.64% | 0.9638 | 0.9608 | 0.9553 | 0.9323 | -0.0315 |


### Scientific Observations:
1. **Contrast-Conditioned Superiority:** CG-Adapter preserves higher boundary fidelity across all perturbation noise regimes compared to both LoRA and the ungated Standard Adapter.
2. **Isolation of Gating Mechanism:** The performance delta between Standard Adapter (E06) and CG-Adapter (E07) directly quantifies the value of prompt-conditioned CIE Lab Delta-E*ab modulation.
