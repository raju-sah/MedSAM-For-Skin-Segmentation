# Table 4: Rigorous Statistical Significance & Hypothesis Testing Matrix (E11)

Two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni multiple testing correction ($\alpha = 0.05$) and non-parametric bootstrap 95% Confidence Intervals comparing **CG-Adapter (Ours)** against all baselines across the clinician-verified sDDI test benchmark.

| Evaluation Regime | Comparison Baseline | N Pairs | Mean Difference $\Delta$ | 95% Bootstrap CI | Wilcoxon $W$ | Raw $p$-value | Holm-Bonferroni $p_{\text{adj}}$ | Significant ($\alpha=0.05$)? | Effect Size ($r_{\text{rb}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean ($\delta=0.0$) | **Zero-Shot MedSAM (E03)** | 198 | +0.0707 | [0.0532, 0.0898] | 3646 | 1.53e-14 | **1.07e-13** | **Yes ($p < 0.05$)** | +0.630 |
| Clean ($\delta=0.0$) | **Decoder-Only (E04)** | 198 | +-0.0014 | [-0.0109, 0.0074] | 9463 | 7.19e-01 | **1.0000** | No | +0.030 |
| Clean ($\delta=0.0$) | **LoRA (r=16, E05)** | 198 | +-0.0218 | [-0.0329, -0.0117] | 6577 | 5.02e-05 | **3.01e-04** | **Yes ($p < 0.05$)** | +-0.332 |
| Clean ($\delta=0.0$) | **Standard Adapter (r=16, E06)** | 198 | +0.0134 | [0.0054, 0.0218] | 6749 | 1.22e-04 | **6.11e-04** | **Yes ($p < 0.05$)** | +0.315 |
| Jitter 20% ($\delta=0.20$) | **Zero-Shot MedSAM (E03)** | 594 | +0.0950 | [0.0839, 0.1054] | 27021 | 1.19e-48 | **9.55e-48** | **Yes ($p < 0.05$)** | +0.694 |
| Jitter 20% ($\delta=0.20$) | **Decoder-Only (E04)** | 594 | +-0.0009 | [-0.0064, 0.0040] | 86578 | 7.22e-01 | **1.0000** | No | +0.017 |
| Jitter 20% ($\delta=0.20$) | **LoRA (r=16, E05)** | 594 | +-0.0054 | [-0.0119, 0.0005] | 82258 | 1.45e-01 | **0.4347** | No | +-0.069 |
| Jitter 20% ($\delta=0.20$) | **Standard Adapter (r=16, E06)** | 594 | +0.0061 | [0.0017, 0.0109] | 76983 | 7.95e-03 | **0.0318** | **Yes ($p < 0.05$)** | +0.126 |
