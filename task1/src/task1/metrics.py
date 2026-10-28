"""Metrics computation, controlled comparisons, and percentage point delta calculations."""

from typing import Any, Dict, List, Optional
import pandas as pd


def compute_percentage_point_delta(score_condition: float, score_baseline: float) -> float:
    """
    Computes change in percentage points: 100 * (score_condition - score_baseline).
    """
    return 100.0 * (score_condition - score_baseline)


def build_benchmark_summary_table(
    condition_results: Dict[str, Dict[str, float]],
    baseline_id: str = "baseline",
    benchmarks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds the primary results table for all conditions x benchmarks.
    Displays primary score and percentage-point delta versus baseline.
    """
    if benchmarks is None:
        benchmarks = ["gsm8k", "hellaswag", "boolq", "arc_easy"]

    base_scores = condition_results.get(baseline_id, {})

    rows = []
    for cond_id, b_scores in condition_results.items():
        row = {"condition": cond_id}
        for b in benchmarks:
            score = b_scores.get(b, float("nan"))
            base_s = base_scores.get(b, float("nan"))
            delta = compute_percentage_point_delta(score, base_s) if cond_id != baseline_id else 0.0

            row[f"{b}_score"] = score
            row[f"{b}_delta_pp"] = delta
        rows.append(row)

    return pd.DataFrame(rows)


def build_controlled_comparisons_table(
    condition_results: Dict[str, Dict[str, float]],
    benchmarks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes the mandatory controlled comparisons:
    1. LoRA-4 vs LoRA-16 (adapter rank)
    2. Full FT vs Full FT + L2-SP (explicit regularization)
    3. LoRA-16 vs OPLoRA-16 (orthogonal protection)
    4. Full FT vs LoRA-16 (adaptation vs retention)
    5. Baseline -> Midpoint -> Final Full FT (progression)
    """
    if benchmarks is None:
        benchmarks = ["gsm8k", "hellaswag", "boolq", "arc_easy"]

    comparisons = [
        ("LoRA Rank (LoRA-16 vs LoRA-4)", "lora16_final", "lora4_final"),
        ("L2-SP Regularization (L2-SP vs Full FT)", "l2sp_final", "full_ft_final"),
        ("Orthogonal Protection (OPLoRA-16 vs LoRA-16)", "oplora16_final", "lora16_final"),
        ("Adaptation vs Retention (Full FT vs LoRA-16)", "full_ft_final", "lora16_final"),
        ("Progression (Full FT Midpoint vs Baseline)", "full_ft_midpoint", "baseline"),
        ("Progression (Full FT Final vs Midpoint)", "full_ft_final", "full_ft_midpoint"),
    ]

    rows = []
    for comp_name, cond_a, cond_b in comparisons:
        scores_a = condition_results.get(cond_a, {})
        scores_b = condition_results.get(cond_b, {})

        row: Dict[str, Any] = {
            "comparison": comp_name,
            "condition_A": cond_a,
            "condition_B": cond_b,
        }

        for b in benchmarks:
            sa = scores_a.get(b, float("nan"))
            sb = scores_b.get(b, float("nan"))
            diff_pp = compute_percentage_point_delta(sa, sb)
            row[f"{b}_diff_pp (A - B)"] = diff_pp

        rows.append(row)

    return pd.DataFrame(rows)

