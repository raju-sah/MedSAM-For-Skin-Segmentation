# arXiv Preprint Submission Guide (cs.CV / eess.IV)

This guide provides instructions for posting the unblinded preprint of CG-MedSAM to arXiv under **Computer Vision (cs.CV)** and **Image and Video Processing (eess.IV)**.

---

## 1. Timing & Anonymity Policy
- **MICCAI Dual Submission / Preprint Policy:** MICCAI allows preprints on arXiv prior to or during submission provided the submitted conference version remains strictly double-blind (i.e. does not cite the arXiv preprint or identify the authors in the MICCAI submission portal).
- **Recommended Timing:** Submit to arXiv immediately after the conference submission deadline passes or upon camera-ready acceptance.

---

## 2. Unblinding the Manuscript

Prior to compiling for arXiv, update `paper/main.tex` and `paper/supplementary.tex`:

1. **Replace Anonymous Author Block:**
```latex
% Replace:
\author{Anonymous MICCAI Submission \and Paper ID: 1042}
\authorrunning{Anonymous et al.}
\institute{Anonymous Medical AI Research Consortium}

% With your official author metadata:
\author{First Author\inst{1} \and Second Author\inst{2} \and Senior Author\inst{1}}
\authorrunning{F. Author et al.}
\institute{Department of Computer Science / Artificial Intelligence\\
\email{author@university.edu}}
```

2. **Restore Code & Checkpoint Links:**
```latex
\url{https://github.com/raju-sah/MedSAM-For-Skin-Segmentation}
\url{https://huggingface.co/raju-ai/CG-MedSAM}
```

3. **Restore Acknowledgments & Grants:**
Uncomment any funding acknowledgments in the `\subsubsection*{Acknowledgments}` block.

---

## 3. Preparing the Source Archive for arXiv

arXiv requires source files (`.tex`, `.bbl`, figures) rather than just a pre-compiled PDF:

1. **Generate `.bbl` File:**
Compile the LaTeX document locally or on Overleaf. Download the resulting `main.bbl` file.
2. **Bundle Required Assets:**
```bash
tar -czvf arxiv_submission.tar.gz \
    main.tex \
    supplementary.tex \
    main.bbl \
    llncs.cls \
    splncs04.bst \
    figure_qualitative_comparisons.png
```
3. **arXiv Upload Settings:**
- **Primary Category:** `cs.CV` (Computer Vision and Pattern Recognition)
- **Secondary Category:** `eess.IV` (Image and Video Processing)
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Title:** *Contrast-Gated Parameter-Efficient Adaptation of MedSAM for Skin-Tone-Robust Lesion Segmentation*
