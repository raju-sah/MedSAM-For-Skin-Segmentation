# E02: Lesion-Core & Perilesional Background Proxy Composition Report

## 1. Executive Summary & Statistical Reconciliation

This report presents the canonical evaluation of spatial proxy purity and composition across all **198 clinician-verified external test masks** (59 Light [FST I–II], 80 Medium [FST III–IV], 59 Dark [FST V–VI]) in the sDDI benchmark.

### Reconciliation of Previously Reported Statistics
An audit of previous reports identified a discrepancy between:
- `e02_proxy_purity_report.md` (which reported a core purity mean of **98.87%**)
- `E02_DECISION.md` (which reported a core purity mean of **97.82%**)

**Root Cause Analysis:**
1. **Different Aggregation Methods:** The value `98.87%` was the unweighted macro-average across the three skin-tone subgroup means:
   $$\text{Macro Mean} = \frac{99.41\% + 99.39\% + 97.82\%}{3} = 98.873\%$$
2. **Mislabeling of Minimum Subgroup Mean:** The value `97.82%` was the mean core purity of the **Dark skin subgroup (FST V–VI)** alone (`core_purity_min`), which was mistakenly written as the 'overall mean across all skin tones' in `E02_DECISION.md`.
3. **Canonical Recomputed Statistics:** All reports now report both the **exact sample-weighted overall mean across all 198 masks (98.93%)**, the **overall median (100.00%)**, and the explicit subgroup statistics.

## 2. Lesion-Core Proxy Purity ($\Omega_{\text{core}}$, $\alpha=0.50$)

| Skin Tone Group | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FST_I_II_Light** | 59 | 99.41% | 100.00% | 2.02% | 0.00% | 95.12% | 87.63% | 100.00% |
| **FST_III_IV_Medium** | 80 | 99.39% | 100.00% | 2.60% | 0.00% | 98.85% | 85.27% | 100.00% |
| **FST_V_VI_Dark** | 59 | 97.82% | 100.00% | 11.82% | 0.00% | 96.29% | 11.96% | 100.00% |
| **Overall (Canonical)** | **198** | **98.93%** | **100.00%** | **6.79%** | **0.00%** | **95.06%** | **11.96%** | **100.00%** |

## 3. Full Perilesional Background Proxy Composition ($\Omega_{\text{skin}}$, $\beta=0.25$)

> [!NOTE]
> **Terminology Clarification:** This region is strictly designated as the **perilesional background proxy**, not 'healthy skin'. As documented below, while cutaneous tissue comprises ~87.6% of the annular halo, surgical marker ink accounts for ~10.9% and ruler markings account for ~1.1%.

### Composition Metric: Cutaneous Tissue (Skin %)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 85.55% | 89.03% | 18.27% | 21.56% | 45.19% | 12.20% | 100.00% |
| FST_III_IV_Medium | 80 | 90.00% | 95.45% | 13.33% | 15.87% | 65.96% | 36.10% | 100.00% |
| FST_V_VI_Dark | 59 | 86.53% | 94.25% | 20.33% | 16.77% | 36.54% | 0.00% | 100.00% |
| **Overall** | **198** | **87.64%** | **94.57%** | **17.27%** | **18.68%** | **47.81%** | **0.00%** | **100.00%** |

### Composition Metric: Lesion Spillover (%)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| FST_III_IV_Medium | 80 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| FST_V_VI_Dark | 59 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Overall** | **198** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |

### Composition Metric: Surgical Marker Contamination (%)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 13.01% | 5.57% | 18.08% | 18.89% | 0.00% | 0.00% | 87.80% |
| FST_III_IV_Medium | 80 | 8.51% | 0.00% | 13.13% | 13.18% | 0.00% | 0.00% | 63.90% |
| FST_V_VI_Dark | 59 | 11.88% | 0.77% | 20.12% | 15.39% | 0.00% | 0.00% | 100.00% |
| **Overall** | **198** | **10.85%** | **1.12%** | **17.08%** | **16.05%** | **0.00%** | **0.00%** | **100.00%** |

### Composition Metric: Ruler Contamination (%)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 1.45% | 0.00% | 5.32% | 0.00% | 0.00% | 0.00% | 29.34% |
| FST_III_IV_Medium | 80 | 1.05% | 0.00% | 3.73% | 0.00% | 0.00% | 0.00% | 25.78% |
| FST_V_VI_Dark | 59 | 0.86% | 0.00% | 2.99% | 0.00% | 0.00% | 0.00% | 14.11% |
| **Overall** | **198** | **1.11%** | **0.00%** | **4.10%** | **0.00%** | **0.00%** | **0.00%** | **29.34%** |

### Composition Metric: Other / Generic Background (%)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| FST_III_IV_Medium | 80 | 0.44% | 0.00% | 3.77% | 0.00% | 0.00% | 0.00% | 33.96% |
| FST_V_VI_Dark | 59 | 0.73% | 0.00% | 5.12% | 0.00% | 0.00% | 0.00% | 39.61% |
| **Overall** | **198** | **0.40%** | **0.00%** | **3.69%** | **0.00%** | **0.00%** | **0.00%** | **39.61%** |

### Composition Metric: Invalid / Empty Region (%)

| Stratum | N | Mean | Median | Std Dev | IQR | 5th Pct | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FST_I_II_Light | 59 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| FST_III_IV_Medium | 80 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| FST_V_VI_Dark | 59 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Overall** | **198** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |

## 4. Failure Analysis & Boundary Cases

1. **Lesion Spillover:** Across all 198 masks, the outer annular halo exhibits **0.00% lesion spillover** (min=0%, max=0%). The outer margin expansion $\beta=0.25$ strictly isolates perilesional tissue without encroaching on lesion tissue.
2. **Core Purity Outlier:** A single dark-skin lesion (image_id 446) exhibited core purity of 11.96% due to an extremely narrow, crescentic morphology where erosion clipped into healthy skin. However, 95% of dark-skin lesions exceed 96.29% core purity (median 100.00%).
3. **Surgical Marker Impact:** Marker contamination is prevalent in clinical dermatology photographs (mean 10.85%, max 100% in local rings). This motivates the artifact sensitivity analysis in `reports/e02_contrast_analysis.md`.
