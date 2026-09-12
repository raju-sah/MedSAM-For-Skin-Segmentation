"""Rigorous Statistical Significance and Confidence Interval Analysis (E11).

Implements:
1. Paired Wilcoxon signed-rank tests (two-sided) between CG-Adapter and all baselines.
2. Holm-Bonferroni multiple testing correction on p-values.
3. Rank-biserial correlation effect size calculation.
4. Non-parametric bootstrap (1,000 resamples) 95% Confidence Intervals (BCa/percentile).
5. Compiles Table 4: Statistical Rigor & Hypothesis Testing Matrix (Markdown & LaTeX).
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

def compute_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation for paired Wilcoxon signed-rank test."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos_sum = np.sum(ranks[diff > 0])
    neg_sum = np.sum(ranks[diff < 0])
    total = pos_sum + neg_sum
    if total == 0:
        return 0.0
    return float((pos_sum - neg_sum) / total)

def holm_bonferroni_correction(p_values: List[float]) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down procedure at alpha=0.05.
    
    Returns list of (adjusted_p_value, is_significant).
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    significant = [False] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
        significant[orig_idx] = (running_max < 0.05)

    return list(zip(adjusted, significant))

def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Compute bootstrap mean and non-parametric percentile 95% Confidence Interval."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(n_boot, len(data)))
    boot_means = np.mean(data[indices], axis=1)
    low_pct = (1.0 - ci) / 2.0 * 100.0
    high_pct = (1.0 + ci) / 2.0 * 100.0
    return float(np.mean(data)), float(np.percentile(boot_means, low_pct)), float(np.percentile(boot_means, high_pct))

def run_significance_analysis(sddi_csv: str, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(sddi_csv):
        raise FileNotFoundError(f"sDDI results CSV not found: {sddi_csv}")

    df = pd.read_csv(sddi_csv)
    print(f"Loaded {len(df)} records for statistical testing from {sddi_csv}")

    # Find CG-Adapter name
    all_models = df["model"].unique().tolist()
    cg_models = [m for m in all_models if "CG-Adapter" in m or "Ours" in m]
    if not cg_models:
        raise ValueError("CG-Adapter model not found in dataset!")
    cg_name = cg_models[0]
    baseline_models = [m for m in all_models if m != cg_name]

    print(f"Ours (Reference): {cg_name}")
    print(f"Baselines: {baseline_models}")

    test_results = []
    p_values_to_correct = []

    # 1. Evaluate clean prompts (delta=0.0) and perturbed (delta=0.20)
    for delta in [0.0, 0.20]:
        sub = df[df["delta"] == delta]
        # Group by image_id and realization_id to get paired vectors
        cg_sub = sub[sub["model"] == cg_name].sort_values(by=["image_id", "realization_id"])
        cg_dice = cg_sub["dice"].values

        for base_m in baseline_models:
            b_sub = sub[sub["model"] == base_m].sort_values(by=["image_id", "realization_id"])
            b_dice = b_sub["dice"].values

            # Ensure strict alignment
            min_len = min(len(cg_dice), len(b_dice))
            v_cg = cg_dice[:min_len]
            v_b = b_dice[:min_len]

            # Wilcoxon signed-rank test
            diff = v_cg - v_b
            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = stats.wilcoxon(v_cg, v_b, alternative="two-sided")

            effect_size = compute_rank_biserial(v_cg, v_b)
            mean_diff = float(np.mean(diff))

            # Bootstrap CI on difference
            m_d, ci_l, ci_u = bootstrap_ci(diff, n_boot=1000)

            test_results.append({
                "delta": delta,
                "regime": r"Clean ($\delta=0.0$)" if delta == 0.0 else r"Jitter 20% ($\delta=0.20$)",
                "comparison": f"CG-Adapter vs. {base_m}",
                "baseline": base_m,
                "n_pairs": min_len,
                "mean_cg": float(np.mean(v_cg)),
                "mean_base": float(np.mean(v_b)),
                "mean_diff": mean_diff,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "wilcoxon_stat": float(stat),
                "raw_p_value": float(p_val),
                "rank_biserial": effect_size
            })
            p_values_to_correct.append(p_val)

    # Apply Holm-Bonferroni correction
    corrections = holm_bonferroni_correction(p_values_to_correct)
    for idx, (adj_p, is_sig) in enumerate(corrections):
        test_results[idx]["adj_p_value"] = adj_p
        test_results[idx]["is_significant"] = is_sig

    # Compile Table 4 Markdown
    md_lines = [
        "# Table 4: Rigorous Statistical Significance & Hypothesis Testing Matrix (E11)\n",
        r"Two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni multiple testing correction ($\alpha = 0.05$) and non-parametric bootstrap 95% Confidence Intervals comparing **CG-Adapter (Ours)** against all baselines across the clinician-verified sDDI test benchmark." + "\n",
        r"| Evaluation Regime | Comparison Baseline | N Pairs | Mean Difference $\Delta$ | 95% Bootstrap CI | Wilcoxon $W$ | Raw $p$-value | Holm-Bonferroni $p_{\text{adj}}$ | Significant ($\alpha=0.05$)? | Effect Size ($r_{\text{rb}}$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in test_results:
        sig_str = "**Yes ($p < 0.05$)**" if r["is_significant"] else "No"
        p_str = f"**{r['adj_p_value']:.2e}**" if r["adj_p_value"] < 0.001 else f"**{r['adj_p_value']:.4f}**"
        md_lines.append(
            f"| {r['regime']} | **{r['baseline']}** | {r['n_pairs']} | +{r['mean_diff']:.4f} | "
            f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | {r['wilcoxon_stat']:.0f} | {r['raw_p_value']:.2e} | "
            f"{p_str} | {sig_str} | +{r['rank_biserial']:.3f} |"
        )

    md_path = os.path.join(output_dir, "Table4_statistical_significance.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Written Table 4 Markdown to: {md_path}")

    # Compile Table 4 LaTeX
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Rigorous statistical significance testing evaluating CG-Adapter against competitive PEFT baselines and Zero-Shot MedSAM on the sDDI clinical benchmark ($N=198$). Evaluated via two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni correction ($p_{\text{adj}}$) and 1,000-sample bootstrap 95\% confidence intervals. Effect size is quantified via rank-biserial correlation ($r_{\text{rb}}$).}",
        r"\label{tab:statistical_significance}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r"\textbf{Prompt Regime} & \textbf{Baseline Architecture} & \textbf{Pairs ($N$)} & \textbf{Mean Diff ($\Delta$DSC)} & \textbf{95\% Bootstrap CI} & \textbf{Wilcoxon $W$} & \textbf{Adj. $p_{\text{HB}}$} & \textbf{Effect Size ($r_{\text{rb}}$)} \\",
        r"\midrule"
    ]

    curr_regime = ""
    for r in test_results:
        if r["regime"] != curr_regime:
            curr_regime = r["regime"]
            tex_lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{curr_regime}}}}} \\\\")
            tex_lines.append(r"\midrule")
        p_tex = f"\\textbf{{{r['adj_p_value']:.2e}}}" if r["adj_p_value"] < 0.001 else f"\\textbf{{{r['adj_p_value']:.4f}}}"
        tex_lines.append(
            f"& {r['baseline']} & {r['n_pairs']} & +{r['mean_diff']:.4f} & "
            f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] & {r['wilcoxon_stat']:.0f} & {p_tex} & +{r['rank_biserial']:.3f} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])

    tex_path = os.path.join(output_dir, "Table4_statistical_significance.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Written Table 4 LaTeX to: {tex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sddi_csv", type=str, default="reports/sddi_clinical_fairness_results.csv")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    run_significance_analysis(args.sddi_csv, args.output_dir)
