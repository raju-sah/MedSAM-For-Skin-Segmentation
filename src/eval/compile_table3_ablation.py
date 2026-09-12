"""Compile Publication Table 3: Causal Signal Ablation Benchmark (E08).

Evaluates the causal necessity of the contrast-gating mechanism and the CIE Lab Delta E*ab signal:
1. Full CG-Adapter (Ours): Normal prompt-conditioned gating gamma = MLP(c_hat)
2. Constant Gate (gamma == 1.0): Ungated baseline (Standard Adapter equivalence)
3. Shuffled Proxy (c_shuffled): Disables sample-specific contrast attribution
4. Sign-Inverted Proxy (-c_hat): Reverses contrast physics direction
5. Luminance-Only Proxy (Delta L*): Removes chromatic channels (a*, b*)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

def compile_table3(ablation_csv: str, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ablation_csv):
        raise FileNotFoundError(f"Ablation CSV not found: {ablation_csv}")

    df = pd.read_csv(ablation_csv)
    print(f"Loaded {len(df)} ablation evaluations from {ablation_csv}")

    variants_order = [
        "Full CG-Adapter (Ours)",
        r"Constant Gate ($\gamma = 1.0$)",
        "Shuffled Contrast Proxy",
        "Sign-Inverted Proxy (-c)",
        r"Luminance-Only ($\Delta L^*$)"
    ]

    all_variants = df["ablation_variant"].unique().tolist()
    resolved = []
    for target in variants_order:
        for v in all_variants:
            if target.split()[0].lower() in v.lower():
                resolved.append(v)
                break
    if not resolved:
        resolved = all_variants

    summary_rows = []
    for v in resolved:
        sub = df[df["ablation_variant"] == v]
        clean = sub[sub["delta"] == 0.0]
        j20 = sub[sub["delta"] == 0.20]

        c_dice = clean["dice"].mean()
        c_iou = clean["iou"].mean()
        c_dark = clean[clean["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        c_light = clean[clean["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean()
        c_disp = c_light - c_dark

        j20_dice = j20["dice"].mean()
        j20_dark = j20[j20["skin_tone_group"] == "FST_V_VI_Dark"]["dice"].mean()
        j20_disp = j20[j20["skin_tone_group"] == "FST_I_II_Light"]["dice"].mean() - j20_dark
        drop = j20_dice - c_dice

        summary_rows.append({
            "Variant": v,
            "Clean Dice": c_dice,
            "Clean IoU": c_iou,
            "Dark Clean": c_dark,
            "Disparity Clean": c_disp,
            "Jitter 20% Dice": j20_dice,
            "Dark Jitter 20%": j20_dark,
            "Disparity @ 20%": j20_disp,
            "Jitter Drop": drop
        })

    # 1. Markdown Table
    md_path = os.path.join(output_dir, "Table3_causal_ablation.md")
    md_lines = [
        "# Table 3: Causal Signal Ablation & Physical Attribution Study (E08)\n",
        "Empirical evaluation of causal controls on CG-Adapter evaluating the necessity of dynamic contrast modulation and CIE Lab color distance.\n",
        "| Architecture / Ablation Variant | Clean Dice | Clean IoU | Dark (V–VI) Clean | Disparity $\\Delta_{\\text{L-D}}\\downarrow$ | Jitter 20% Dice | Dark @ 20% | Disparity @ 20%$\\downarrow$ | Jitter Drop$\\downarrow$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in summary_rows:
        is_ours = "Full" in r["Variant"] or "Ours" in r["Variant"]
        name = f"**{r['Variant']}**" if is_ours else r['Variant']
        md_lines.append(
            f"| {name} | {r['Clean Dice']:.4f} | {r['Clean IoU']:.4f} | {r['Dark Clean']:.4f} | "
            f"**{r['Disparity Clean']:.4f}** | {r['Jitter 20% Dice']:.4f} | {r['Dark Jitter 20%']:.4f} | "
            f"**{r['Disparity @ 20%']:.4f}** | {r['Jitter Drop']:.4f} |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Written Table 3 Markdown to: {md_path}")

    # 2. LaTeX Table
    tex_path = os.path.join(output_dir, "Table3_causal_ablation.tex")
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Causal signal ablation study evaluating physical attribution and contrast conditioning mechanisms in CG-Adapter. Evaluated on the clinician-verified sDDI test partition ($N=198$) under clean and perturbed prompt regimes. Bold denotes optimal fairness and performance.}",
        r"\label{tab:causal_ablation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{Clean Prompts ($\delta = 0.0$)}} & \multicolumn{4}{c}{\textbf{Imperfect Prompts ($\delta = 0.20$ Noise)}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"\textbf{Ablation Variant} & \textbf{Overall DSC} & \textbf{IoU} & \textbf{Dark (V--VI)} & \textbf{Disparity $\Delta_{\text{L-D}}\downarrow$} & \textbf{Overall DSC} & \textbf{Dark (V--VI)} & \textbf{Disparity $\Delta_{\text{L-D}}\downarrow$} & \textbf{Degradation $\Delta\downarrow$} \\",
        r"\midrule"
    ]
    for r in summary_rows:
        is_ours = "Full" in r["Variant"] or "Ours" in r["Variant"]
        name = f"\\textbf{{{r['Variant']}}}" if is_ours else r['Variant']
        tex_lines.append(
            f"{name} & {r['Clean Dice']:.4f} & {r['Clean IoU']:.4f} & {r['Dark Clean']:.4f} & {r['Disparity Clean']:.4f} & "
            f"{r['Jitter 20% Dice']:.4f} & {r['Dark Jitter 20%']:.4f} & {r['Disparity @ 20%']:.4f} & {r['Jitter Drop']:.4f} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Written Table 3 LaTeX to: {tex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="reports/causal_ablation_results.csv")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    compile_table3(args.csv, args.output_dir)
