"""Metric aggregators and table data formatting for Task 2.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from task2.src.metrics.vqa_scorer import (
    compute_vqa_soft_score,
    compute_exact_match,
    compute_pair_metrics,
)
from task2.src.metrics.bootstrap import (
    compute_pair_bootstrap_ci,
    compute_paired_advantage_bootstrap_ci,
)


def aggregate_grounding_condition_metrics(
    df_condition: pd.DataFrame,
    n_bootstrap: int = 4000,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Aggregate all required metrics for a single image condition on the grounding suite."""
    # Ensure presence and counting flags
    df = df_condition.copy()
    if "is_presence" not in df.columns:
        df["is_presence"] = df["pair_type"] == "presence"
    if "is_counting" not in df.columns:
        df["is_counting"] = df["pair_type"] == "counting"

    df["exact_match"] = [
        compute_exact_match(p, r) for p, r in zip(df["prediction"], df["reference"])
    ]
    df["is_empty"] = df["prediction"].apply(lambda s: len(str(s).strip()) == 0)
    df["is_truncated"] = df.get("is_truncated", False)

    # Subsets
    df_pres = df[df["is_presence"]]
    df_count = df[df["is_counting"]]

    presence_acc = float(df_pres["exact_match"].mean()) if len(df_pres) > 0 else 0.0
    counting_acc = float(df_count["exact_match"].mean()) if len(df_count) > 0 else 0.0
    combined_acc = float(df["exact_match"].mean())

    # Pair metrics
    pair_ids = df["pair_id"].unique()
    pair_correct_flags = []
    identical_flags = []

    for pid in pair_ids:
        pair_rows = df[df["pair_id"] == pid]
        if len(pair_rows) == 2:
            p1, p2 = pair_rows["prediction"].iloc[0], pair_rows["prediction"].iloc[1]
            r1, r2 = pair_rows["reference"].iloc[0], pair_rows["reference"].iloc[1]
            pc, id_flag = compute_pair_metrics(p1, p2, r1, r2, is_vqa=False)
            pair_correct_flags.append(pc)
            identical_flags.append(id_flag)

    pair_accuracy = float(np.mean(pair_correct_flags)) if pair_correct_flags else 0.0
    identical_answer_rate = float(np.mean(identical_flags)) if identical_flags else 0.0
    empty_answer_rate = float(df["is_empty"].mean())
    truncation_rate = float(df["is_truncated"].mean()) if "is_truncated" in df.columns else 0.0
    mean_reference_nll = float(df["reference_nll"].mean()) if "reference_nll" in df.columns else float("nan")
    mean_gen_time = float(df["generation_time_sec"].mean()) if "generation_time_sec" in df.columns else float("nan")

    # Bootstrap 95% CI for combined accuracy
    def boot_metric(pair_idx_array: np.ndarray) -> float:
        sampled_pids = pair_ids[pair_idx_array]
        sampled_rows = df[df["pair_id"].isin(sampled_pids)]
        return float(sampled_rows["exact_match"].mean())

    pt, ci_low, ci_high, _ = compute_pair_bootstrap_ci(
        pair_ids=pair_ids,
        metric_fn=boot_metric,
        n_resamples=n_bootstrap,
        seed=seed,
    )

    return {
        "presence_accuracy": presence_acc,
        "counting_accuracy": counting_acc,
        "combined_accuracy": combined_acc,
        "combined_accuracy_ci": (ci_low, ci_high),
        "pair_accuracy": pair_accuracy,
        "identical_answer_rate": identical_answer_rate,
        "empty_answer_rate": empty_answer_rate,
        "truncation_rate": truncation_rate,
        "mean_reference_nll": mean_reference_nll,
        "mean_generation_time_sec": mean_gen_time,
        "n_examples": len(df),
        "n_pairs": len(pair_ids),
    }


def aggregate_vqav2_condition_metrics(
    df_condition: pd.DataFrame,
    n_bootstrap: int = 4000,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Aggregate all required metrics for a single image condition on the natural VQAv2 suite."""
    df = df_condition.copy()

    # Calculate soft scores
    soft_scores = []
    for _, row in df.iterrows():
        human_ans = row.get("human_answers", [row["reference"]])
        if isinstance(human_ans, str):
            import ast
            try:
                human_ans = ast.literal_eval(human_ans)
            except Exception:
                human_ans = [human_ans]
        score = compute_vqa_soft_score(row["prediction"], human_ans)
        soft_scores.append(score)

    df["vqa_score"] = soft_scores
    df["is_empty"] = df["prediction"].apply(lambda s: len(str(s).strip()) == 0)
    df["is_truncated"] = df.get("is_truncated", False)

    # By answer type
    yn_df = df[df["answer_type"] == "yes/no"]
    num_df = df[df["answer_type"] == "number"]
    other_df = df[df["answer_type"] == "other"]

    yn_acc = float(yn_df["vqa_score"].mean()) if len(yn_df) > 0 else 0.0
    num_acc = float(num_df["vqa_score"].mean()) if len(num_df) > 0 else 0.0
    other_acc = float(other_df["vqa_score"].mean()) if len(other_df) > 0 else 0.0
    overall_acc = float(df["vqa_score"].mean())

    # Pair metrics
    pair_ids = df["pair_id"].unique()
    pair_correct_flags = []
    identical_flags = []

    for pid in pair_ids:
        prows = df[df["pair_id"] == pid]
        if len(prows) == 2:
            r1, r2 = prows.iloc[0], prows.iloc[1]
            h1 = r1.get("human_answers", [r1["reference"]])
            h2 = r2.get("human_answers", [r2["reference"]])
            pc, id_flag = compute_pair_metrics(
                r1["prediction"], r2["prediction"],
                r1["reference"], r2["reference"],
                human_answers1=h1, human_answers2=h2,
                is_vqa=True,
            )
            pair_correct_flags.append(pc)
            identical_flags.append(id_flag)

    pair_accuracy = float(np.mean(pair_correct_flags)) if pair_correct_flags else 0.0
    identical_answer_rate = float(np.mean(identical_flags)) if identical_flags else 0.0
    empty_answer_rate = float(df["is_empty"].mean())
    truncation_rate = float(df["is_truncated"].mean()) if "is_truncated" in df.columns else 0.0
    mean_reference_nll = float(df["reference_nll"].mean()) if "reference_nll" in df.columns else float("nan")
    mean_gen_time = float(df["generation_time_sec"].mean()) if "generation_time_sec" in df.columns else float("nan")

    # Bootstrap 95% CI for overall accuracy
    def boot_metric(pair_idx_array: np.ndarray) -> float:
        sampled_pids = pair_ids[pair_idx_array]
        sampled_rows = df[df["pair_id"].isin(sampled_pids)]
        return float(sampled_rows["vqa_score"].mean())

    pt, ci_low, ci_high, _ = compute_pair_bootstrap_ci(
        pair_ids=pair_ids,
        metric_fn=boot_metric,
        n_resamples=n_bootstrap,
        seed=seed,
    )

    return {
        "yes_no_accuracy": yn_acc,
        "number_accuracy": num_acc,
        "other_accuracy": other_acc,
        "overall_accuracy": overall_acc,
        "overall_accuracy_ci": (ci_low, ci_high),
        "pair_accuracy": pair_accuracy,
        "identical_answer_rate": identical_answer_rate,
        "empty_answer_rate": empty_answer_rate,
        "truncation_rate": truncation_rate,
        "mean_reference_nll": mean_reference_nll,
        "mean_generation_time_sec": mean_gen_time,
        "n_examples": len(df),
        "n_pairs": len(pair_ids),
    }
