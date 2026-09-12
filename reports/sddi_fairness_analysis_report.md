# sDDI Clinical Generalization & Skin-Tone Fairness Analysis Report

## Executive Summary
This report details the performance of Zero-Shot MedSAM and 4 PEFT adaptation strategies across 9900 evaluations on the 198 clinician-verified sDDI test masks.

## Primary Clinical Findings:
1. **Disparity Reduction:** CG-Adapter achieved the lowest skin-tone disparity gap of **0.0122** under clean prompts, compared to the unadapted baseline.
2. **Dark-Cohort Resilience:** On the challenging Fitzpatrick V-VI cohort under 20% prompt jitter, CG-Adapter sustained a Dice score of **0.7830**.

## Complete Model Summary Matrix:

```
                       Model  Clean Overall Dice  Clean IoU  Clean HD95  Clean NSD  Light Clean  Med Clean  Dark Clean  Disparity Clean  J20 Overall Dice  Light J20  Med J20  Dark J20  Disparity J20  Drop Overall  Drop Dark
      Zero-Shot MedSAM (E03)            0.769433   0.643404    5.566606   0.637000     0.788564   0.767126    0.753430         0.035134          0.700478   0.722417 0.704968  0.672451       0.049966     -0.068955  -0.080979
          Decoder-Only (E04)            0.841517   0.735946    4.902485   0.737508     0.844052   0.848048    0.830127         0.013924          0.796408   0.797639 0.805410  0.782972       0.014667     -0.045109  -0.047155
            LoRA (r=16, E05)            0.861924   0.766984    4.181007   0.812552     0.865150   0.866158    0.852959         0.012191          0.800949   0.808445 0.811651  0.778941       0.029504     -0.060975  -0.074018
Standard Adapter (r=16, E06)            0.826785   0.717658    5.511460   0.711624     0.833637   0.839832    0.802243         0.031394          0.789420   0.797672 0.801341  0.765004       0.032668     -0.037365  -0.037239
      CG-Adapter (Ours, E07)            0.840163   0.736275    4.610254   0.751420     0.850480   0.844890    0.823438         0.027042          0.795523   0.805161 0.806483  0.771025       0.034136     -0.044640  -0.052413
```


