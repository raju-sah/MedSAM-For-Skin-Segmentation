# E03: Zero-Shot MedSAM Baseline Evaluation Report (Kaggle GPU Dual-T4)

Evaluated 519 images across 10 prompt conditions (5190 evaluations total).

## In-Domain Dermoscopy Performance Across Prompt Noise

| Prompt Noise $\delta$ | Mean Dice | Median Dice | Std Dev | Mean IoU | Mean HD95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\delta = 0.00$ | 0.9302 | 0.9387 | 0.0442 | 0.8724 | 56.13 |
| $\delta = 0.05$ | 0.9186 | 0.9285 | 0.0513 | 0.8531 | 65.56 |
| $\delta = 0.10$ | 0.8857 | 0.9012 | 0.0733 | 0.8013 | 93.22 |
| $\delta = 0.20$ | 0.7969 | 0.8213 | 0.1250 | 0.6777 | 156.77 |
