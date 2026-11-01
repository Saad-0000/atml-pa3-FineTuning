"""
Task 3 Plot Generator
Generates publication-quality figures P01 through P16 as specified in the assignment.
Saves both PNG and PDF versions with labeled axes, units, legends, and clear markings.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_plot_style():
    """Apply consistent styling across all figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "grid.alpha": 0.5,
    })


def save_fig(fig: plt.Figure, output_dir: Path, fig_id: str):
    """Save figure in both PNG and PDF format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{fig_id}.png"
    pdf_path = output_dir / f"{fig_id}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# P01 & P02: Accuracy and Oracle Coverage vs Candidate Budget N
# ==============================================================================

def plot_p01_math_accuracy_oracle(df_t04: pd.DataFrame, output_dir: Path, df_t05: Optional[pd.DataFrame] = None):
    """
    P01: MATH accuracy and oracle-coverage curves vs candidate budget N.
    Shows FirstAns, SC, Highest L, Highest K, K+RWC, and Oracle.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    sub = df_t04[df_t04["dataset"] == "MATH"].sort_values("budget_n")
    if sub.empty:
        # Placeholder if unrun
        ax.text(0.5, 0.5, "P01: MATH Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P01_math_accuracy_oracle")
        return

    # Plot Oracle Coverage first
    oracle_df = sub.drop_duplicates(subset=["budget_n"])
    ax.plot(oracle_df["budget_n"], oracle_df["oracle_coverage"] * 100, label="Oracle Coverage", 
            color="black", linestyle="--", marker="s", linewidth=2.0)
    
    # Method lines
    methods = [
        ("first_ans", "FirstAns", "#7f7f7f", "^"),
        ("self_consistency", "Self-Consistency (SC)", "#1f77b4", "o"),
        ("highest_mean_logp", "Highest Mean LogP (L)", "#2ca02c", "v"),
        ("highest_self_certainty", "Highest Self-Certainty (K)", "#ff7f0e", "D"),
        ("rwc_self_certainty", "Self-Certainty + RWC", "#d62728", "P"),
    ]
    
    for m_id, m_label, color, marker in methods:
        m_sub = sub[sub["method"] == m_id]
        if not m_sub.empty:
            ax.plot(m_sub["budget_n"], m_sub["accuracy"] * 100, label=m_label,
                    color=color, marker=marker)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Candidate Budget N")
    ax.set_ylabel("Accuracy / Coverage (%)")
    ax.set_title("P01: MATH Test Subset — Accuracy and Oracle Coverage vs. Budget N")
    ax.legend(loc="lower right", frameon=True)
    ax.set_ylim(bottom=0)
    save_fig(fig, output_dir, "P01_math_accuracy_oracle")


def plot_p02_mathvista_accuracy_oracle(df_t04: pd.DataFrame, output_dir: Path):
    """
    P02: MathVista accuracy and oracle-coverage curves vs candidate budget N.
    Includes vision-only and VisConf methods.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    sub = df_t04[df_t04["dataset"] == "MathVista"].sort_values("budget_n")
    if sub.empty:
        ax.text(0.5, 0.5, "P02: MathVista Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P02_mathvista_accuracy_oracle")
        return

    oracle_df = sub.drop_duplicates(subset=["budget_n"])
    ax.plot(oracle_df["budget_n"], oracle_df["oracle_coverage"] * 100, label="Oracle Coverage", 
            color="black", linestyle="--", marker="s", linewidth=2.0)
    
    methods = [
        ("first_ans", "FirstAns", "#7f7f7f", "^"),
        ("self_consistency", "Self-Consistency (SC)", "#1f77b4", "o"),
        ("highest_mean_logp", "Highest Mean LogP (L)", "#2ca02c", "v"),
        ("highest_self_certainty", "Highest Self-Certainty (K)", "#ff7f0e", "D"),
        ("rwc_self_certainty", "Self-Certainty + RWC", "#d62728", "P"),
        ("highest_vision_only", "Vision-only Highest (V)", "#9467bd", "x"),
        ("rwc_vision_only", "Vision-only RWC (V)", "#8c564b", "+"),
        ("highest_visconf", "VisConf Highest (Z)", "#e377c2", "*"),
        ("rwc_visconf", "VisConf + RWC (Z)", "#17becf", "h"),
    ]
    
    for m_id, m_label, color, marker in methods:
        m_sub = sub[sub["method"] == m_id]
        if not m_sub.empty:
            ax.plot(m_sub["budget_n"], m_sub["accuracy"] * 100, label=m_label,
                    color=color, marker=marker)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Candidate Budget N")
    ax.set_ylabel("Accuracy / Coverage (%)")
    ax.set_title("P02: MathVista testmini — Accuracy and Oracle Coverage vs. Budget N")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_ylim(bottom=0)
    save_fig(fig, output_dir, "P02_mathvista_accuracy_oracle")


# ==============================================================================
# P03: Answer Diversity vs Budget N
# ==============================================================================

def plot_p03_answer_diversity(df_t03: pd.DataFrame, output_dir: Path):
    """
    P03: Mean distinct normalized answer groups vs N for both datasets.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if df_t03.empty:
        ax.text(0.5, 0.5, "P03: Diversity Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P03_answer_diversity")
        return

    for ds, color, marker in [("MATH", "#1f77b4", "o"), ("MathVista", "#ff7f0e", "s")]:
        sub = df_t03[df_t03["dataset"] == ds].sort_values("budget_n")
        if not sub.empty:
            ax.plot(sub["budget_n"], sub["mean_distinct_answer_groups"], label=ds,
                    color=color, marker=marker, linewidth=2.0)
            
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Candidate Budget N")
    ax.set_ylabel("Mean Distinct Answer Groups")
    ax.set_title("P03: Answer Diversity vs. Candidate Budget N")
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, output_dir, "P03_answer_diversity")


# ==============================================================================
# P04: Sampling vs Selection Failure Decomposition
# ==============================================================================

def plot_p04_failure_decomposition(df_t04: pd.DataFrame, output_dir: Path):
    """
    P04: Sampling failure (no correct candidate) vs selection failure
    (correct candidate available but missed) across budgets.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    
    if df_t04.empty:
        ax1.text(0.5, 0.5, "P04: Failure Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P04_failure_decomposition")
        return

    for ax, ds, title in [(ax1, "MATH", "MATH Test"), (ax2, "MathVista", "MathVista testmini")]:
        sub = df_t04[(df_t04["dataset"] == ds) & (df_t04["method"] == "self_consistency")].sort_values("budget_n")
        if not sub.empty:
            budgets = sub["budget_n"].values
            samp_fail = sub["sampling_failure_rate"].values * 100
            sel_fail = sub["selection_failure_rate"].values * 100
            
            x = np.arange(len(budgets))
            width = 0.35
            
            ax.bar(x - width/2, samp_fail, width, label="Sampling Failure (No Correct Rollout)", color="#d62728", alpha=0.85)
            ax.bar(x + width/2, sel_fail, width, label="Selection Failure (Correct Available)", color="#ff7f0e", alpha=0.85)
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in budgets])
            ax.set_xlabel("Candidate Budget N")
            ax.set_title(f"{title} (Self-Consistency)")
            ax.set_ylabel("Failure Rate (%)")
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            ax.set_ylim(0, 100)

    fig.suptitle("P04: Sampling Failure vs. Selection Failure Decomposition", fontsize=13)
    save_fig(fig, output_dir, "P04_failure_decomposition")


# ==============================================================================
# P05 & P06: Offline Accuracy vs Generated Token Cost Trade-off
# ==============================================================================

def plot_p05_math_cost_tradeoff(df_t04: pd.DataFrame, output_dir: Path):
    """
    P05: Accuracy vs actual generated token cost for MATH.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    sub = df_t04[df_t04["dataset"] == "MATH"]
    if sub.empty:
        ax.text(0.5, 0.5, "P05: Cost Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P05_math_cost_tradeoff")
        return

    methods = [
        ("first_ans", "FirstAns", "#7f7f7f", "^"),
        ("self_consistency", "Self-Consistency (SC)", "#1f77b4", "o"),
        ("highest_mean_logp", "Highest Mean LogP (L)", "#2ca02c", "v"),
        ("highest_self_certainty", "Highest Self-Certainty (K)", "#ff7f0e", "D"),
        ("rwc_self_certainty", "Self-Certainty + RWC", "#d62728", "P"),
    ]
    
    for m_id, m_label, color, marker in methods:
        m_sub = sub[sub["method"] == m_id].sort_values("budget_n")
        if not m_sub.empty:
            ax.plot(m_sub["mean_token_cost"], m_sub["accuracy"] * 100, label=m_label,
                    color=color, marker=marker, linewidth=1.5)
            # Annotate budget points
            for _, row in m_sub.iterrows():
                ax.annotate(f"N={int(row['budget_n'])}", 
                            (row["mean_token_cost"], row["accuracy"] * 100),
                            textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.set_xlabel("Mean Generated Token Cost per Question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("P05: MATH Test Subset — Accuracy vs. Generated Token Cost")
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, output_dir, "P05_math_cost_tradeoff")


def plot_p06_mathvista_cost_tradeoff(df_t04: pd.DataFrame, output_dir: Path):
    """
    P06: Accuracy vs actual generated token cost for MathVista.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    sub = df_t04[df_t04["dataset"] == "MathVista"]
    if sub.empty:
        ax.text(0.5, 0.5, "P06: Cost Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P06_mathvista_cost_tradeoff")
        return

    methods = [
        ("first_ans", "FirstAns", "#7f7f7f", "^"),
        ("self_consistency", "Self-Consistency (SC)", "#1f77b4", "o"),
        ("highest_self_certainty", "Highest K", "#ff7f0e", "D"),
        ("rwc_self_certainty", "K + RWC", "#d62728", "P"),
        ("highest_vision_only", "Vision-only Highest", "#9467bd", "x"),
        ("highest_visconf", "VisConf Highest", "#e377c2", "*"),
        ("rwc_visconf", "VisConf + RWC", "#17becf", "h"),
    ]
    
    for m_id, m_label, color, marker in methods:
        m_sub = sub[sub["method"] == m_id].sort_values("budget_n")
        if not m_sub.empty:
            ax.plot(m_sub["mean_token_cost"], m_sub["accuracy"] * 100, label=m_label,
                    color=color, marker=marker, linewidth=1.5)

    ax.set_xlabel("Mean Generated Token Cost per Question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("P06: MathVista testmini — Accuracy vs. Generated Token Cost")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    save_fig(fig, output_dir, "P06_mathvista_cost_tradeoff")


# ==============================================================================
# P07: Length and Score Quartile Correctness Analysis
# ==============================================================================

def plot_p07_quartile_analysis(df_t06: pd.DataFrame, output_dir: Path):
    """
    P07: Top vs Bottom quartile within-question correctness difference
    for length and quality scores, including shortest-rollout control.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    if df_t06.empty:
        ax1.text(0.5, 0.5, "P07: Quartile Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P07_quartile_analysis")
        return

    for ax, ds, title in [(ax1, "MATH", "MATH Test"), (ax2, "MathVista", "MathVista testmini")]:
        sub = df_t06[df_t06["dataset"] == ds]
        if not sub.empty:
            signals = sub["signal_name"].values
            diffs = sub["mean_top_minus_bottom_diff"].values * 100
            colors = ["#1f77b4" if d >= 0 else "#d62728" for d in diffs]
            
            y_pos = np.arange(len(signals))
            ax.barh(y_pos, diffs, color=colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(signals)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Top − Bottom Quartile Accuracy Difference (pp)")
            ax.set_title(title)

    fig.suptitle("P07: Within-Question Quartile Correctness Differences (N=32)", fontsize=13)
    save_fig(fig, output_dir, "P07_quartile_analysis")


# ==============================================================================
# P08: MathVista Confidence–Vision Relationship
# ==============================================================================

def plot_p08_confidence_vision(df_t07: pd.DataFrame, rollout_corrs: Optional[List[float]], output_dir: Path):
    """
    P08: Tokenwise calibrated confidence vs visual engagement relationship.
    Shows the distribution of rollout-level Pearson correlation coefficients.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if rollout_corrs is None or len(rollout_corrs) == 0:
        ax.text(0.5, 0.5, "P08: Correlation Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P08_confidence_vision")
        return

    corrs = [c for c in rollout_corrs if not np.isnan(c)]
    ax.hist(corrs, bins=30, color="#17becf", edgecolor="black", alpha=0.75)
    mean_corr = np.mean(corrs) if corrs else 0.0
    ax.axvline(mean_corr, color="#d62728", linestyle="--", linewidth=2.0, label=f"Mean r = {mean_corr:.3f}")
    
    ax.set_xlabel("Within-Rollout Pearson Correlation r(c, v)")
    ax.set_ylabel("Rollout Count")
    ax.set_title("P08: MathVista — Tokenwise Confidence vs. Visual Engagement Correlation")
    ax.legend(loc="upper right", frameon=True)
    save_fig(fig, output_dir, "P08_confidence_vision")


# ==============================================================================
# P09: Tokenwise vs Trajectory-Level Fusion
# ==============================================================================

def plot_p09_fusion_comparison(df_t08: pd.DataFrame, cov_diffs: Optional[List[float]], output_dir: Path):
    """
    P09: Z_token vs Z, covariance/difference distribution, and accuracy comparison.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if cov_diffs is not None and len(cov_diffs) > 0:
        ax1.hist(cov_diffs, bins=30, color="#9467bd", edgecolor="black", alpha=0.75)
        ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_xlabel(r"$Z_{token} - Z = \mathrm{Cov}(c, v)$")
        ax1.set_ylabel("Rollout Count")
        ax1.set_title("Fusion Difference Distribution")
    else:
        ax1.text(0.5, 0.5, "Covariance Data Not Available", ha="center", va="center")

    if not df_t08.empty:
        # Comparison of accuracy
        methods = df_t08["method"].values
        accs = df_t08["accuracy"].values * 100
        x = np.arange(len(methods))
        ax2.bar(x, accs, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][:len(methods)], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Selection Accuracy under Fusion Variants (N=32)")
        ax2.set_ylim(0, 100)
    else:
        ax2.text(0.5, 0.5, "Accuracy Data Not Available", ha="center", va="center")

    fig.suptitle("P09: Tokenwise vs. Trajectory-Level Fusion (MathVista, N=32)", fontsize=13)
    save_fig(fig, output_dir, "P09_fusion_comparison")


# ==============================================================================
# P10 & P11: Live Strategy Comparisons (50 MATH questions)
# ==============================================================================

def plot_p10_live_accuracy_tokens(df_t10: pd.DataFrame, output_dir: Path):
    """
    P10: Live strategy accuracy vs actual output tokens/cost.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    if df_t10.empty:
        ax.text(0.5, 0.5, "P10: Live Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P10_live_accuracy_tokens")
        return

    strategies = df_t10["strategy"].values
    accs = df_t10["accuracy"].values * 100
    mean_tokens = df_t10["mean_output_tokens"].values
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, strat in enumerate(strategies):
        ax.scatter(mean_tokens[i], accs[i], color=colors[i % len(colors)], s=120, label=strat, zorder=3)
        ax.annotate(strat, (mean_tokens[i], accs[i]), textcoords="offset points", xytext=(8, -3), fontsize=9)

    ax.set_xlabel("Mean Output Tokens per Question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("P10: Live Generation Strategies — Accuracy vs. Output Tokens (50 MATH)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, output_dir, "P10_live_accuracy_tokens")


def plot_p11_live_runtime_breakdown(df_t10: pd.DataFrame, output_dir: Path):
    """
    P11: Actual input tokens, output tokens, and runtime across strategies.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if df_t10.empty:
        ax1.text(0.5, 0.5, "P11: Live Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P11_live_runtime_breakdown")
        return

    strats = df_t10["strategy"].values
    x = np.arange(len(strats))
    width = 0.35
    
    in_tok = df_t10["mean_input_tokens"].values
    out_tok = df_t10["mean_output_tokens"].values
    
    ax1.bar(x - width/2, in_tok, width, label="Input Tokens", color="#7f7f7f", alpha=0.85)
    ax1.bar(x + width/2, out_tok, width, label="Output Tokens", color="#1f77b4", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strats, rotation=15, ha="right")
    ax1.set_ylabel("Token Count")
    ax1.set_title("Mean Tokens per Question")
    ax1.legend(loc="upper left", frameon=True)

    runtimes = df_t10["mean_runtime_sec"].values
    ax2.bar(x, runtimes, color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strats, rotation=15, ha="right")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Mean Generation Time (Excl. Model Load)")

    fig.suptitle("P11: Live Strategy Token and Runtime Breakdown", fontsize=13)
    save_fig(fig, output_dir, "P11_live_runtime_breakdown")


# ==============================================================================
# P12: Refinement Transition Matrix Visualization
# ==============================================================================

def plot_p12_refinement_transitions(df_t11: pd.DataFrame, output_dir: Path):
    """
    P12: Wrong->correct, correct->wrong, correct->correct, wrong->wrong counts.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if df_t11.empty:
        ax.text(0.5, 0.5, "P12: Transition Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P12_refinement_transitions")
        return

    categories = ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]
    labels = ["Wrong → Correct (Gain)", "Correct → Wrong (Degradation)", 
              "Correct → Correct (Retained)", "Wrong → Wrong (Persistent)"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    
    counts = []
    for cat in categories:
        val = df_t11[df_t11["transition"] == cat]["count"].values
        counts.append(val[0] if len(val) > 0 else 0)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Question Count (Out of 50)")
    ax.set_title("P12: Sequential Refinement — Initial to Revised Transitions")
    
    for i, c in enumerate(counts):
        ax.text(c + 0.5, i, f"{c} ({c/50*100:.1f}%)", va="center", fontsize=9)

    save_fig(fig, output_dir, "P12_refinement_transitions")


# ==============================================================================
# P13: Adaptive Development Threshold Selection Grid
# ==============================================================================

def plot_p13_adaptive_dev_grid(df_t12: pd.DataFrame, output_dir: Path):
    """
    P13: Development accuracy vs mean token cost across tau and gamma.
    Marks fixed-32 reference, -1 pp feasibility boundary, and selected setting.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    if df_t12.empty:
        ax1.text(0.5, 0.5, "P13: Dev Grid Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P13_adaptive_dev_grid")
        return

    for ax, ds, title in [(ax1, "MATH", "MATH Dev (200 questions)"), (ax2, "MathVista", "MathVista Dev (200 questions)")]:
        sub = df_t12[df_t12["dataset"] == ds]
        if sub.empty:
            continue
            
        fixed_32_acc = sub[sub["policy_family"] == "fixed_32"]["accuracy"].values
        f32 = fixed_32_acc[0] if len(fixed_32_acc) > 0 else 0.5
        boundary = f32 - 0.01

        ax.axhline(f32 * 100, color="black", linestyle="--", label=f"Fixed-32 SC ({f32*100:.1f}%)")
        ax.axhline(boundary * 100, color="red", linestyle=":", label=f"Feasibility Limit (-1 pp, {boundary*100:.1f}%)")

        # Scatter points for agreement
        ag_sub = sub[sub["policy_family"] == "agreement"]
        if not ag_sub.empty:
            ax.scatter(ag_sub["mean_token_cost"], ag_sub["accuracy"] * 100, 
                       c=ag_sub["tau"], cmap="Blues", s=60, marker="o", label="Agreement (varying τ)", edgecolors="black")

        # Scatter points for quality-aware
        qa_sub = sub[sub["policy_family"] == "quality_aware"]
        if not qa_sub.empty:
            ax.scatter(qa_sub["mean_token_cost"], qa_sub["accuracy"] * 100,
                       c=qa_sub["gamma"], cmap="Oranges", s=60, marker="s", label="Quality-Aware (varying γ)", edgecolors="black")

        # Highlight selected
        sel_sub = sub[sub["is_selected"] == True]
        for _, srow in sel_sub.iterrows():
            ax.scatter(srow["mean_token_cost"], srow["accuracy"] * 100,
                       color="gold", s=150, marker="*", edgecolors="black", linewidth=1.5,
                       label=f"Selected: {srow['policy_family']} (τ={srow['tau']})")

        ax.set_xlabel("Mean Generated Token Cost per Question")
        ax.set_ylabel("Dev Accuracy (%)")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=True, fontsize=8)

    fig.suptitle("P13: Adaptive Development Policy Selection Grid (τ, γ)", fontsize=13)
    save_fig(fig, output_dir, "P13_adaptive_dev_grid")


# ==============================================================================
# P14: Adaptive Held-out Accuracy vs Cost Trade-off
# ==============================================================================

def plot_p14_adaptive_heldout_comparison(df_t13: pd.DataFrame, df_t04: pd.DataFrame, output_dir: Path):
    """
    P14: Frozen adaptive settings vs fixed-N SC (N in 1,2,4,8,16,32) on held-out.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    if df_t13.empty:
        ax1.text(0.5, 0.5, "P14: Held-out Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P14_adaptive_heldout_comparison")
        return

    for ax, ds, title in [(ax1, "MATH", "MATH Held-out (800)"), (ax2, "MathVista", "MathVista Held-out (800)")]:
        # Fixed SC curve
        sc_sub = df_t04[(df_t04["dataset"] == ds) & (df_t04["method"] == "self_consistency")].sort_values("budget_n")
        if not sc_sub.empty:
            ax.plot(sc_sub["mean_token_cost"], sc_sub["accuracy"] * 100, 
                    label="Fixed-N SC", color="#1f77b4", marker="o", linewidth=1.8)
            for _, r in sc_sub.iterrows():
                ax.annotate(f"N={int(r['budget_n'])}", (r["mean_token_cost"], r["accuracy"] * 100),
                            textcoords="offset points", xytext=(0, 5), fontsize=8)

        # Adaptive points
        ad_sub = df_t13[df_t13["dataset"] == ds]
        colors = {"agreement": "#2ca02c", "quality_aware": "#d62728"}
        markers = {"agreement": "P", "quality_aware": "D"}
        for _, arow in ad_sub.iterrows():
            pol = arow["policy_family"]
            ax.scatter(arow["mean_token_cost"], arow["accuracy"] * 100,
                       color=colors.get(pol, "black"), marker=markers.get(pol, "o"), s=120,
                       label=f"Adaptive: {pol.replace('_', ' ').title()}", zorder=5)

        ax.set_xlabel("Mean Token Cost")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("P14: Adaptive Policies vs. Fixed-Budget Self-Consistency (Held-out 800)", fontsize=13)
    save_fig(fig, output_dir, "P14_adaptive_heldout_comparison")


# ==============================================================================
# P15: Adaptive Stopping-Count Distributions
# ==============================================================================

def plot_p15_stopping_distributions(df_t13: pd.DataFrame, output_dir: Path):
    """
    P15: Distribution of stopping points (4, 8, 16, 32 candidates) for held-out data.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    if df_t13.empty:
        ax1.text(0.5, 0.5, "P15: Stop Distribution Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P15_stopping_distributions")
        return

    stages = [4, 8, 16, 32]
    x = np.arange(len(stages))
    width = 0.35

    for ax, ds, title in [(ax1, "MATH", "MATH Held-out"), (ax2, "MathVista", "MathVista Held-out")]:
        sub = df_t13[df_t13["dataset"] == ds]
        ag = sub[sub["policy_family"] == "agreement"]
        qa = sub[sub["policy_family"] == "quality_aware"]

        ag_rates = [ag[f"frac_stopped_at_{s}"].values[0] * 100 if f"frac_stopped_at_{s}" in ag.columns and not ag.empty else 0 for s in stages]
        qa_rates = [qa[f"frac_stopped_at_{s}"].values[0] * 100 if f"frac_stopped_at_{s}" in qa.columns and not qa.empty else 0 for s in stages]

        ax.bar(x - width/2, ag_rates, width, label="Agreement Policy", color="#2ca02c", alpha=0.85)
        ax.bar(x + width/2, qa_rates, width, label="Quality-Aware Policy", color="#d62728", alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={s}" for s in stages])
        ax.set_xlabel("Stopping Checkpoint (Rollouts Consumed)")
        ax.set_ylabel("Questions Stopped (%)")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.set_ylim(0, 100)

    fig.suptitle("P15: Adaptive Inference — Stopping Count Distributions", fontsize=13)
    save_fig(fig, output_dir, "P15_stopping_distributions")


# ==============================================================================
# P16: Premature-Stop Failure Analysis
# ==============================================================================

def plot_p16_premature_stops(df_t14: pd.DataFrame, output_dir: Path):
    """
    P16: Counts/rates of incorrect early stops and cases recovered by fixed-32 SC.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if df_t14.empty:
        ax.text(0.5, 0.5, "P16: Premature Stop Data Not Available", ha="center", va="center")
        save_fig(fig, output_dir, "P16_premature_stops")
        return

    checkpoints = [4, 8, 16]
    x = np.arange(len(checkpoints))
    width = 0.35

    premature_counts = [df_t14[df_t14["stop_stage"] == s]["premature_stop_count"].sum() if "stop_stage" in df_t14.columns else 0 for s in checkpoints]
    recovered_counts = [df_t14[df_t14["stop_stage"] == s]["recovered_by_f32_count"].sum() if "stop_stage" in df_t14.columns else 0 for s in checkpoints]

    ax.bar(x - width/2, premature_counts, width, label="Incorrect Early Stops", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, recovered_counts, width, label="Recovered by Fixed-32 SC", color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in checkpoints])
    ax.set_xlabel("Early Stopping Stage")
    ax.set_ylabel("Question Count")
    ax.set_title("P16: Premature Early Stops and Recovery by Full Fixed-32 SC")
    ax.legend(loc="upper right", frameon=True)
    save_fig(fig, output_dir, "P16_premature_stops")
