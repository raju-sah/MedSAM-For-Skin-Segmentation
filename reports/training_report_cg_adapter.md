# PEFT MedSAM Training Report: CG_ADAPTER

- Mode: `cg_adapter`
- Trainable Parameters: 4,363,824 (4.64%)
- Best Validation Dice: **0.9591**

## Validation Robustness Across Prompt Perturbations

| Perturbation Delta | Mean Dice | Mean IoU |
| :--- | :--- | :--- |
| 0.00 | 0.9638 | 0.9309 |
| 0.05 | 0.9608 | 0.9253 |
| 0.10 | 0.9553 | 0.9152 |
| 0.20 | 0.9323 | 0.8768 |
