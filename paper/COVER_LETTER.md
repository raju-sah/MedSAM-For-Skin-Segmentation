# Cover Letter: MICCAI 2026 Submission

**To:**  
Program Chairs & Scientific Review Committee  
29th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI 2026)  

**Date:** September 13, 2026  
**Submission Title:** *Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation*  
**Paper ID:** 1042  
**Track:** Machine Learning – Foundation Model Adaptation & Algorithmic Fairness in Clinical Diagnostics  

---

Dear Program Chairs and Reviewers,

We are pleased to submit our original research manuscript, **"Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation"**, for consideration for presentation at MICCAI 2026 and publication in the Springer Lecture Notes in Computer Science (LNCS) proceedings.

### Clinical Context & Urgency
Medical foundation models, notably MedSAM, demonstrate remarkable zero-shot capability in anatomical segmentation. However, their reliability drops precipitously when transferred to external clinical photography, with severe performance degradation observed in low-contrast lesions and darker skin cohorts (Fitzpatrick Skin Types V–VI). This bias poses significant diagnostic risks, as diagnostic delay in non-Caucasian populations is directly associated with advanced disease presentation and elevated melanoma mortality rates.

### Key Contributions & Novelty
1. **Zero-Leakage Contrast-Gating Formulation:** We introduce **CG-MedSAM**, a novel parameter-efficient adaptation (PEFT) framework that extracts a localized CIE $L^*a^*b^*$ perceptual color distance proxy ($\Delta E^*_{ab}$) strictly from prompt bounding boxes without accessing ground-truth masks. A lightweight multi-layer perceptron maps this proxy to dynamically modulate ViT bottleneck adapters.
2. **Strictly Matched Parameter Efficiency:** Updating only **4.64%** of ViT parameters (identical parameter footprint to standard bottleneck adapters), CG-MedSAM matches or exceeds the accuracy of parameter-heavy adaptation methods while demonstrating superior transfer stability.
3. **Rigorous Multi-Center Fairness & Jitter Benchmarks:** Evaluated across **9,900 rigorous prompt-evaluation passes** on external clinical photographs (sDDI) and dermoscopy (ISIC 2018), CG-MedSAM achieves:
   - **+1.34%** overall clean Dice gain over standard adapters ($p = 6.11 \times 10^{-4}$ via Holm-Bonferroni corrected Wilcoxon signed-rank test).
   - **+2.12%** Dice improvement on dark skin tones (FST V–VI).
   - **14% reduction** in cross-tone diagnostic disparity ($\Delta_{\text{light-dark}}$).
   - Unmatched robustness under high prompt noise ($\delta = 0.20$), retaining **77.10%** Dice on dark skin where baseline models collapse.
4. **Causal Validation & Open Science:** Permutation ablations confirm that dynamic contrast conditioning provides a statistically genuine inductive bias ($p < 10^{-7}$) rather than serving as an unstructured capacity artifact. 

### Double-Blind Compliance & Prior Publication
This work has not been published previously and is not under consideration for publication elsewhere. All authors have read and approved the manuscript, and all institutional, ethical, and double-blind anonymization guidelines of MICCAI 2026 have been strictly maintained. To ensure full reproducibility, anonymized model card manifests, test suites, and SHA-256 weight checksums are packaged with the submission.

Thank you for your time and consideration of our manuscript. We look forward to the reviewers' feedback.

Sincerely,  
*The Authors*  
Paper ID: 1042  
Anonymous Medical AI Research Consortium  
