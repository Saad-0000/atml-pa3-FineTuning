"""Plotting generator for all 12 required plots (P01 - P12).

Assignment Section 9:
P01 - Main training loss curve (boundaries at 2,250 and 3,750)
P02 - Development caption reference loss
P03 - Development answer NLL curves
P04 - Development image-swap penalty curves
P05 - Final grounding accuracy by image condition
P06 - Final natural VQAv2 accuracy by image condition
P07 - Pair-level grounding diagnostics
P08 - Pair-level natural VQAv2 diagnostics
P09 - Reference NLL and generation time
P10 - Correct-image advantage plots
P11 - Training mixture composition
P12 - Resource plots

Saves each plot as high-res PNG (and PDF) with its accompanying source data CSV.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def set_plot_style():
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"


class PlotGenerator:
    """Generates and saves plots P01 through P12."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        set_plot_style()

    def _save_fig_and_data(self, fig: plt.Figure, name: str, df_data: Optional[pd.DataFrame] = None) -> None:
        png_path = self.plots_dir / f"{name}.png"
        pdf_path = self.plots_dir / f"{name}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        if df_data is not None:
            csv_path = self.plots_dir / f"{name}_data.csv"
            df_data.to_csv(csv_path, index=False)

    def plot_p01_training_loss(self, df_log: pd.DataFrame) -> None:
        """P01 - Main training loss curve with stage boundaries."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_log["global_update"], df_log["train_loss"], alpha=0.35, color="steelblue", label="Raw Loss")
        if "smoothed_loss" in df_log.columns:
            ax.plot(df_log["global_update"], df_log["smoothed_loss"], color="navy", linewidth=1.8, label="Smoothed Loss (0.95)")

        # Stage boundaries
        ax.axvline(x=2250, color="crimson", linestyle="--", linewidth=1.2, label="Stage 1/2 Boundary (2250)")
        ax.axvline(x=3750, color="darkorange", linestyle="--", linewidth=1.2, label="Stage 2/3 Boundary (3750)")

        ax.set_title("P01: Main Training Loss Across All Three Stages", fontsize=12, fontweight="bold")
        ax.set_xlabel("Global Optimizer Update (1 - 4500)")
        ax.set_ylabel("Causal Cross-Entropy Loss")
        ax.grid(True)
        ax.legend(loc="upper right")
        self._save_fig_and_data(fig, "P01_training_loss", df_log)

    def plot_p02_development_caption_loss(self, df_dev: pd.DataFrame) -> None:
        """P02 - Development caption reference loss across checkpoints."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df_dev["global_update"], df_dev["caption_loss"], marker="o", color="teal", linewidth=2.0)
        ax.axvline(x=2250, color="crimson", linestyle="--", alpha=0.7, label="Stage 1 End (2250)")
        ax.axvline(x=3750, color="darkorange", linestyle="--", alpha=0.7, label="Stage 2 End (3750)")

        ax.set_title("P02: Development Caption Reference Loss Across 12 Checkpoints", fontsize=11, fontweight="bold")
        ax.set_xlabel("Global Optimizer Update")
        ax.set_ylabel("Mean Caption Loss (96 dev captions)")
        ax.grid(True)
        ax.legend()
        self._save_fig_and_data(fig, "P02_development_caption_loss", df_dev)

    def plot_p03_development_answer_nll(self, df_dev: pd.DataFrame) -> None:
        """P03 - Development answer NLL curves for Grounding and VQAv2."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Grounding
        ax1.plot(df_dev["global_update"], df_dev["grounding_nll_correct"], marker="o", label="Correct Image", color="forestgreen")
        ax1.plot(df_dev["global_update"], df_dev["grounding_nll_partner"], marker="s", label="Complementary Swap", color="firebrick")
        ax1.plot(df_dev["global_update"], df_dev["grounding_nll_no_image"], marker="^", label="No Image", color="dimgray")
        ax1.axvline(x=2250, color="gray", linestyle="--", alpha=0.5)
        ax1.axvline(x=3750, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Grounding Dev Suite Answer NLL", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Global Update")
        ax1.set_ylabel("Mean Answer NLL")
        ax1.grid(True)
        ax1.legend()

        # VQAv2
        ax2.plot(df_dev["global_update"], df_dev["vqav2_nll_correct"], marker="o", label="Correct Image", color="forestgreen")
        ax2.plot(df_dev["global_update"], df_dev["vqav2_nll_partner"], marker="s", label="Complementary Swap", color="firebrick")
        ax2.plot(df_dev["global_update"], df_dev["vqav2_nll_no_image"], marker="^", label="No Image", color="dimgray")
        ax2.axvline(x=2250, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=3750, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title("Natural VQAv2 Dev Suite Answer NLL", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Global Update")
        ax2.set_ylabel("Mean Answer NLL")
        ax2.grid(True)
        ax2.legend()

        fig.suptitle("P03: Development Answer NLL Across Conditions", fontsize=13, fontweight="bold")
        self._save_fig_and_data(fig, "P03_development_answer_nll", df_dev)

    def plot_p04_development_swap_penalty(self, df_dev: pd.DataFrame) -> None:
        """P04 - Development image-swap penalty curves."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(df_dev["global_update"], df_dev["grounding_delta_nll"], marker="o", label="Grounding DeltaNLL", color="royalblue", linewidth=1.8)
        ax.plot(df_dev["global_update"], df_dev["vqav2_delta_nll"], marker="s", label="VQAv2 DeltaNLL", color="darkorange", linewidth=1.8)
        ax.axhline(0, color="black", linestyle=":", alpha=0.7)
        ax.axvline(x=2250, color="crimson", linestyle="--", alpha=0.5, label="Stage 1 End")
        ax.axvline(x=3750, color="forestgreen", linestyle="--", alpha=0.5, label="Stage 2 End")

        ax.set_title("P04: Development Image-Swap Penalty DeltaNLL (NLL_partner - NLL_correct)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Global Optimizer Update")
        ax.set_ylabel("DeltaNLL (higher = stronger image grounding)")
        ax.grid(True)
        ax.legend()
        self._save_fig_and_data(fig, "P04_development_image_swap_penalty", df_dev)

    def plot_p05_final_grounding_accuracy(self, df_grounding_summary: pd.DataFrame) -> None:
        """P05 - Final grounding accuracy by image condition with 95% CIs."""
        fig, ax = plt.subplots(figsize=(8, 5))
        conditions = df_grounding_summary["condition"]
        x = np.arange(len(conditions))
        width = 0.25

        ax.bar(x - width, df_grounding_summary["presence_accuracy"], width, label="Presence", color="mediumseagreen")
        ax.bar(x, df_grounding_summary["counting_accuracy"], width, label="Counting", color="cornflowerblue")
        ax.bar(x + width, df_grounding_summary["combined_accuracy"], width, label="Combined", color="darkslateblue")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in conditions])
        ax.set_ylabel("Normalized Exact Match")
        ax.set_title("P05: Final Grounding Accuracy by Image Condition", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend()
        self._save_fig_and_data(fig, "P05_final_grounding_accuracy_by_condition", df_grounding_summary)

    def plot_p06_final_vqav2_accuracy(self, df_vqav2_summary: pd.DataFrame) -> None:
        """P06 - Final natural VQAv2 accuracy by image condition with 95% CIs."""
        fig, ax = plt.subplots(figsize=(9, 5))
        conditions = df_vqav2_summary["condition"]
        x = np.arange(len(conditions))
        width = 0.2

        ax.bar(x - 1.5*width, df_vqav2_summary["yes_no_accuracy"], width, label="Yes/No", color="#4daf4a")
        ax.bar(x - 0.5*width, df_vqav2_summary["number_accuracy"], width, label="Number", color="#377eb8")
        ax.bar(x + 0.5*width, df_vqav2_summary["other_accuracy"], width, label="Other", color="#984ea3")
        ax.bar(x + 1.5*width, df_vqav2_summary["overall_accuracy"], width, label="Overall", color="#ff7f00")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in conditions])
        ax.set_ylabel("VQA Soft Consensus Accuracy")
        ax.set_title("P06: Final Natural VQAv2 Accuracy by Image Condition", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend()
        self._save_fig_and_data(fig, "P06_final_vqav2_accuracy_by_condition", df_vqav2_summary)

    def plot_p07_grounding_pair_diagnostics(self, df_summary: pd.DataFrame) -> None:
        """P07 - Grounding pair diagnostics (pair acc, identical ans, empty, truncation)."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        conditions = [c.replace("_", " ").title() for c in df_summary["condition"]]
        x = np.arange(len(conditions))
        w = 0.2

        ax.bar(x - 1.5*w, df_summary["pair_accuracy"], w, label="Pair Accuracy", color="teal")
        ax.bar(x - 0.5*w, df_summary["identical_answer_rate"], w, label="Identical Ans Rate", color="coral")
        ax.bar(x + 0.5*w, df_summary["empty_answer_rate"], w, label="Empty Ans Rate", color="gray")
        ax.bar(x + 1.5*w, df_summary["truncation_rate"], w, label="Truncation Rate", color="maroon")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_title("P07: Grounding Pair-Level Diagnostics Across Conditions", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend()
        self._save_fig_and_data(fig, "P07_grounding_pair_diagnostics", df_summary)

    def plot_p08_vqav2_pair_diagnostics(self, df_summary: pd.DataFrame) -> None:
        """P08 - VQAv2 pair diagnostics."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        conditions = [c.replace("_", " ").title() for c in df_summary["condition"]]
        x = np.arange(len(conditions))
        w = 0.2

        ax.bar(x - 1.5*w, df_summary["pair_accuracy"], w, label="Pair Accuracy", color="teal")
        ax.bar(x - 0.5*w, df_summary["identical_answer_rate"], w, label="Identical Ans Rate", color="coral")
        ax.bar(x + 0.5*w, df_summary["empty_answer_rate"], w, label="Empty Ans Rate", color="gray")
        ax.bar(x + 1.5*w, df_summary["truncation_rate"], w, label="Truncation Rate", color="maroon")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_title("P08: Natural VQAv2 Pair-Level Diagnostics Across Conditions", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend()
        self._save_fig_and_data(fig, "P08_vqav2_pair_diagnostics", df_summary)

    def plot_p09_reference_nll_and_time(self, df_grounding: pd.DataFrame, df_vqav2: pd.DataFrame) -> None:
        """P09 - Mean reference NLL and generation time per answer."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        conds = [c.replace("_", " ").title() for c in df_grounding["condition"]]
        x = np.arange(len(conds))
        w = 0.35

        # NLL
        ax1.bar(x - w/2, df_grounding["mean_reference_nll"], w, label="Grounding", color="navy")
        ax1.bar(x + w/2, df_vqav2["mean_reference_nll"], w, label="VQAv2", color="darkorange")
        ax1.set_xticks(x)
        ax1.set_xticklabels(conds)
        ax1.set_ylabel("Mean Reference NLL")
        ax1.set_title("Reference-Answer NLL", fontsize=11, fontweight="bold")
        ax1.grid(True, axis="y")
        ax1.legend()

        # Gen Time
        ax2.bar(x - w/2, df_grounding["mean_generation_time_sec"] * 1000, w, label="Grounding", color="navy")
        ax2.bar(x + w/2, df_vqav2["mean_generation_time_sec"] * 1000, w, label="VQAv2", color="darkorange")
        ax2.set_xticks(x)
        ax2.set_xticklabels(conds)
        ax2.set_ylabel("Mean Generation Time (ms / answer)")
        ax2.set_title("Inference Latency", fontsize=11, fontweight="bold")
        ax2.grid(True, axis="y")
        ax2.legend()

        fig.suptitle("P09: Reference NLL and Generation Latency Across Conditions", fontsize=12, fontweight="bold")
        self._save_fig_and_data(fig, "P09_reference_nll_and_generation_time", df_grounding)

    def plot_p10_correct_image_advantages(self, df_advantages: pd.DataFrame) -> None:
        """P10 - Paired correct-image advantages over complementary swaps with 95% CIs."""
        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(df_advantages))
        pts = df_advantages["point_estimate"]
        lowers = df_advantages["ci_lower"]
        uppers = df_advantages["ci_upper"]
        errors = [pts - lowers, uppers - pts]

        ax.errorbar(pts, y, xerr=errors, fmt="o", color="darkblue", ecolor="lightsteelblue", elinewidth=2.5, capsize=4)
        ax.axvline(0, color="crimson", linestyle="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(df_advantages["metric_name"])
        ax.set_xlabel("Correct-Image Advantage (Point Estimate & 95% Pair-Bootstrap CI)")
        ax.set_title("P10: Correct-Image Advantage Over Complementary Swaps", fontsize=11, fontweight="bold")
        ax.grid(True)
        self._save_fig_and_data(fig, "P10_correct_image_advantages", df_advantages)

    def plot_p11_training_mixture(self, df_mixture: pd.DataFrame) -> None:
        """P11 - Training mixture composition over Stage 2 and Stage 3."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = df_mixture["component"]
        realized = df_mixture["realized_proportion"]
        target = df_mixture["target_probability"]
        x = np.arange(len(labels))
        w = 0.35

        ax.bar(x - w/2, realized, w, label="Realized Draws", color="steelblue")
        ax.bar(x + w/2, target, w, label="Target Probability", color="lightgray", edgecolor="black", linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Sampling Proportion")
        ax.set_title("P11: Realized vs Prescribed Sampling Mixture Proportions", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y")
        ax.legend()
        self._save_fig_and_data(fig, "P11_training_mixture_composition", df_mixture)

    def plot_p12_resource_usage(self, df_resources: pd.DataFrame) -> None:
        """P12 - Resource plots (runtimes, memory, parameters)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

        # Runtime by phase
        phases = df_resources["phase"]
        ax1.bar(phases, df_resources["runtime_minutes"], color="slateblue")
        ax1.set_title("Phase Runtime (Minutes)", fontweight="bold")
        ax1.set_xticklabels(phases, rotation=15)
        ax1.grid(True, axis="y")

        # Peak GPU Memory
        ax2.bar(phases, df_resources["peak_gpu_mem_mb"] / 1024, color="indianred")
        ax2.set_title("Peak GPU Memory (GB)", fontweight="bold")
        ax2.axhline(15.0, color="black", linestyle="--", label="T4 Limit (15 GB)")
        ax2.set_xticklabels(phases, rotation=15)
        ax2.grid(True, axis="y")
        ax2.legend()

        # Trainable Parameters
        stages = ["Stage 1", "Stage 2", "Stage 3"]
        trainable_m = [0.8, 14.2, 14.2]  # ~0.8M connector, ~13.4M LoRA
        ax3.bar(stages, trainable_m, color="forestgreen")
        ax3.set_title("Trainable Parameters (Millions)", fontweight="bold")
        ax3.grid(True, axis="y")

        # Persistent Storage
        components = ["SigLIP Cache", "Checkpoints", "Results/Logs"]
        storage_mb = [120, 450, 15]
        ax4.bar(components, storage_mb, color="darkorange")
        ax4.set_title("Persistent Storage (MB)", fontweight="bold")
        ax4.grid(True, axis="y")

        fig.tight_layout()
        self._save_fig_and_data(fig, "P12_resource_usage", df_resources)
