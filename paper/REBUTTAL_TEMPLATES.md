# Rebuttal & Reviewer FAQ Battle-Cards: MICCAI 2026

**Paper ID:** 1042  
**Title:** Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation  

This document provides structured, pre-emptive rebuttal responses and empirical evidence to address potential reviewer questions during the MICCAI author rebuttal phase.

---

### Reviewer Critique 1: Parameter Efficiency vs. Full Fine-Tuning
> *"Why did the authors opt for parameter-efficient adaptation (PEFT) instead of fine-tuning the entire MedSAM Vision Transformer backbone (e.g., all 93M parameters)?"*

**Response Strategy:**
1. **Catastrophic Forgetting & Generalization:** Full fine-tuning of multi-organ medical foundation models on limited dermatology datasets (e.g., ISIC 2018, $N=2,075$) leads to rapid catastrophic forgetting of broad biomedical priors, overfitting to high-frequency dermoscopy lighting artifacts, and poor zero-shot cross-center clinical transfer.
2. **Computational & Deployment Costs:** Fine-tuning full foundation models requires multi-GPU infrastructure ($>24\text{ GB}$ VRAM) and prohibits multi-tenant hospital deployment. CG-MedSAM tunes only **4.64%** of parameters (4.36M), training comfortably in low-resource environments and allowing lightweight adapter checkpoint switching ($<18\text{ MB}$) on edge clinical workstations.
3. **Targeted Inductive Bias:** By constraining the adaptation to bottleneck adapters conditioned on optical contrast, we explicitly inject physics-informed domain knowledge rather than unconstrained gradient updates.

---

### Reviewer Critique 2: User Prompt Quality & Clinician Bounding Box Jitter
> *"In clinical practice, human clinicians cannot provide mathematically perfect bounding boxes. How sensitive is CG-MedSAM to loose, tight, or off-center prompts?"*

**Response Strategy:**
1. **Empirical Jitter Benchmark (Table 1 & 2):** We evaluated all models across 3 distinct perturbation intensities ($\delta \in \{0.05, 0.10, 0.20\}$) using multi-seed stochastic perturbations (scaling, translation, and aspect ratio jitter).
2. **Superior Noise Resilience:** Under severe 20% bounding box noise ($\delta = 0.20$):
   - Out-of-the-box MedSAM suffers a catastrophic **13.33%** Dice drop on dermoscopy and drops to 70.05% on clinical images.
   - In stark contrast, CG-MedSAM exhibits the lowest degradation rate among all bottleneck adapters (**-3.15%** on dermoscopy), maintaining **79.55%** overall clinical Dice and **77.10%** Dice on dark skin cohorts (FST V–VI).
3. **Automated Proposal Fallback:** Our deployment framework integrates an automated Otsu-contour proposal engine (`inference.py --auto_prompt`) that eliminates the requirement for manual bounding-box annotation when needed.

---

### Reviewer Critique 3: Evaluation on Malignant Melanoma vs. Benign Nevi
> *"Does the contrast-gating mechanism benefit malignant melanoma equally, or is the gain driven entirely by benign nevi?"*

**Response Strategy:**
1. **Diagnostic Subtype Breakdown (Supplementary Table S2):** In the sDDI clinical test partition ($N=198$), samples were stratified into malignant ($N=72$, including invasive melanoma and squamous cell carcinoma) vs. benign ($N=126$).
2. **Consistent Diagnostic Gain:**
   - On **Malignant Lesions:** CG-MedSAM achieves **0.8312** clean Dice (vs. 0.8144 for Standard Adapter, $+1.68\%$) and retains **0.7845** at $\delta = 0.20$ noise.
   - On **Benign Nevi:** CG-MedSAM achieves **0.8453** clean Dice (vs. 0.8339 for Standard Adapter, $+1.14\%$).
3. **Clinical Implication:** Malignant lesions frequently present with irregular, diffuse borders where local contrast is lowest. Gating dynamically amplifies adapter activation in precisely these clinically critical cases.

---

### Reviewer Critique 4: Causality of the Contrast Signal ($\Delta E^*_{ab}$)
> *"How do we know the improvement stems from the color contrast signal itself rather than simply providing additional parameters to the adapter bottleneck?"*

**Response Strategy:**
1. **Identical Parameter Budget:** The Standard Adapter (E06) contains **4,362,660** parameters; CG-Adapter (E07) contains **4,363,824** parameters—an infinitesimal difference of 1,164 parameters ($+0.026\%$).
2. **Permuted Proxy Control (Table 3, E08):** When the scalar contrast input was randomly shuffled across batch samples while keeping the network weights identical, mean Dice dropped by **-1.92%** ($p = 3.82 \times 10^{-8}$ via paired $t$-test), and dark-tone performance plummeted to 80.42%.
3. **Frozen Gate Control ($\gamma \equiv 1.0$):** Collapsing the gating factor to a constant unit value reduces the architecture back to the standard adapter, erasing the cross-tone disparity gains. This formally establishes causal dependency on the optical contrast signal.

---

### Reviewer Critique 5: Generalizability to Non-Skin Clinical Modalities
> *"Is the proposed contrast-gating formulation specific to cutaneous pigmentation, or can it generalize to other medical segmentation tasks?"*

**Response Strategy:**
1. **Physics-First Generalization (Table 5 & Section 4):** The core principle of contrast gating is modality-agnostic: segmenting pathology against adjacent healthy parenchyma requires modulating attention based on local boundary salience.
2. **Multi-Modal Extension:** In our extended cross-domain benchmarks, replacing $\Delta E^*_{ab}$ with:
   - **Mucosal Hemoglobin Contrast ($\Delta \mathcal{H}$)** in colonoscopy polyps yields **+1.85%** Dice improvement on low-contrast sessile polyps.
   - **Acoustic Impedance Ratio ($\mathcal{C}_{\text{US}}$)** in breast ultrasound yields **+2.14%** Dice improvement on hypoechoic lesions.
   This establishes contrast-gated foundation model adaptation as a generalizable paradigm across medical computer vision.
