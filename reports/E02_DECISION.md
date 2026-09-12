# Pre-Training Decision Gate Evaluation: E02_DECISION

## Official Decision: **MODIFY**

### Reframed Decision Conclusion

> **Canonical Conclusion:**
> **$\Delta E^*_{ab}$ is selected as the primary candidate conditioning signal for E03 because it retains an association with the selected difficulty measures after adjustment, whereas $\Delta L^*$ does not.**

> [!IMPORTANT]
> **Explicit Scientific Boundaries:**
> 1. **E02 does not prove downstream segmentation improvement.**
> 2. **E02 does not prove fairness improvement.**
> 3. **E02 does not prove causal skin-tone robustness.**
> 4. **E03–E08 must empirically test whether the signal improves actual MedSAM segmentation performance.**

### Scientific Evidence Supporting Verdict:
1. **Rejection of Pure Luminance ($\Delta L^*$):** In univariate analysis, $\Delta L^*$ exhibits apparent predictive strength ($p < 10^{-15}$). However, when discrete Fitzpatrick skin type (FST) and confounders are controlled for, its standardized coefficient collapses to beta = -0.0546 ($p_{\text{HC3}} = 0.483$, Holm $p = 0.483$, not significant). $\Delta L^*$ becomes largely redundant with FST and provides no independent boundary signal.
2. **Superiority of Local Color Distance ($\Delta E^*_{ab}$):** Euclidean CIE Lab color distance $\Delta E^*_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$ retains strong, statistically significant independent association with boundary ambiguity ($t_{\text{HC3}} = 2.68$, $p_{\text{HC3}} = 8.09e-03$, Holm $p = 2.43e-02$, bootstrap 95% CI [0.0009, 0.0039]) even after controlling for FST, lesion size, and artifacts. In dark skin (FST V–VI), luminance contrast diminishes while erythema ($\Delta a^*$) and pigment chromaticity ($\Delta b^*$) preserve boundary visibility.
3. **Verified Proxy Reliability:** Reconciled sample-weighted core lesion purity is **98.93%** (median 100.00%, dark subgroup mean 97.82%) with **0.00% lesion spillover** into the perilesional background proxy across all 198 external test masks.
4. **Actionable Protocol Modification:** The Contrast-Gated Adapter (CG-Adapter) prompt conditioning will use standardized $\Delta E^*_{ab}$ computed strictly from the input image and bounding box without GT mask access.

### Comparative Statistical Summary Across Candidate Signals:

| Contrast Candidate | Univariate Beta ($p$-val) | Full Model 3 Beta ($p_{\text{HC3}}$) | Holm $p$ | Full Model Adj $R^2$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\Delta L^*$ (Luminance)** | -0.7993 ($p < 10^{-10}$) | -0.0546 ($p = 4.83e-01$) | 4.83e-01 | 0.9334 | **REJECT** (Redundant with FST) |
| **$\Delta\text{ITA}$ (Pigment Angle)** | -0.7181 ($p < 10^{-10}$) | -0.1057 ($p = 7.80e-02$) | 1.56e-01 | 0.9353 | **VIABLE** ($p < 0.10$) |
| **$\Delta E^*_{ab}$ (Color Distance)** | -0.5544 ($p < 10^{-10}$) | 0.1773 ($p = 8.09e-03$) | 2.43e-02 | 0.9408 | **PRIMARY CANDIDATE** ($p < 0.05$) |
| **$\Delta a^*$ (Erythema / Green-Red)** | 0.2129 ($p < 10^{-10}$) | 0.1313 ($p = 1.58e-03$) | 7.88e-03 | 0.9391 | **VIABLE** (Secondary) |
| **$\Delta b^*$ (Melanin / Blue-Yellow)** | -0.1381 ($p < 10^{-10}$) | -0.1272 ($p = 2.22e-03$) | 8.89e-03 | 0.9388 | **VIABLE** (Secondary) |

### Reconciled Proxy Statistics Summary:
- **Total Audited Masks:** $N = 198$ external clinical test masks (59 light, 80 medium, 59 dark)
- **Overall Mean Core Lesion Purity:** `98.93%` (Median: `100.00%`)
- **Subgroup Core Purity Means:** Light: `99.41%`, Medium: `99.39%`, Dark: `97.82%` (Macro-Average: `98.87%`)
- **Lesion Spillover into Background:** `0.00%` across all 198 masks (min 0%, max 0%)
- **Perilesional Background Composition:** `87.64%` skin, `10.85%` marker, `1.11%` ruler, `0.40%` other background

### Strict Protocol Directive:
In accordance with the project rules, **NO model training (E03+), MedSAM fine-tuning, LoRA, or adapter training has been initiated.** Execution stops here awaiting explicit user review and approval of the MODIFY recommendation.
