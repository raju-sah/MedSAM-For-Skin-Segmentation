# Table 3: Causal Signal Ablation & Physical Attribution Study (E08)

Empirical evaluation of causal controls on CG-Adapter evaluating the necessity of dynamic contrast modulation and CIE Lab color distance.

| Architecture / Ablation Variant | Clean Dice | Clean IoU | Dark (V–VI) Clean | Disparity $\Delta_{\text{L-D}}\downarrow$ | Jitter 20% Dice | Dark @ 20% | Disparity @ 20%$\downarrow$ | Jitter Drop$\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full CG-Adapter (Ours)** | 0.8406 | 0.7367 | 0.8238 | **0.0259** | 0.8075 | 0.7867 | **0.0325** | -0.0331 |
| Constant Gate (gamma = 1.0) | 0.8415 | 0.7379 | 0.8254 | **0.0261** | 0.8083 | 0.7881 | **0.0322** | -0.0331 |
| Shuffled Contrast Proxy | 0.8410 | 0.7372 | 0.8234 | **0.0272** | 0.8080 | 0.7872 | **0.0322** | -0.0330 |
| Sign-Inverted Proxy (-c) | 0.8405 | 0.7366 | 0.8237 | **0.0260** | 0.8077 | 0.7872 | **0.0315** | -0.0329 |
| Luminance-Only (Delta L*) | 0.8407 | 0.7367 | 0.8240 | **0.0259** | 0.8075 | 0.7869 | **0.0324** | -0.0331 |
