"""
Task 3 Offline Sampling and Selection Script (Experiments A & B)
Executes evaluation across budgets N in {1, 2, 4, 8, 16, 32} and 10 random orders per question.
Computes FirstAns, Oracle Coverage, SC, Highest Score, and RWC selection methods.
Computes paired question-level bootstrap confidence intervals (1,000 resamples).
Generates Tables T03, T04, and T05.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from task3.src.utils.common import get_logger, atomic_save_json
from task3.src.data.split_builder import load_splits_manifest
from task3.src.sampling.permutation_manager import PermutationManager
from task3.src.selection.selectors import (
    select_first_ans,
    select_self_consistency,
    select_highest_score,
    select_rank_weighted_consensus
)
from task3.src.selection.quality_scores import calibrate_visconf_for_candidates
from task3.src.metrics.offline_metrics import compute_oracle_coverage, count_distinct_answer_groups, classify_failure
from task3.src.utils.bootstrap import paired_question_bootstrap
from task3.src.visualization.tables import build_table_t03, build_table_t04, build_table_t05

logger = get_logger("run_offline_sampling")


def evaluate_offline_for_dataset(
    rollouts_path: Path,
    splits_path: Path,
    dataset_name: str,
    budgets: List[int],
    num_permutations: int = 10,
    seed: int = 6304
) -> Dict[str, Any]:
    """Runs Experiments A & B for a single dataset on the held-out split."""
    logger.info(f"Loading data for {dataset_name} from {rollouts_path}...")
    df_all = pd.read_parquet(rollouts_path)
    split_info = load_splits_manifest(splits_path)
    held_out_ids = set(split_info.get("held_out_question_ids", []))
    
    # Filter to held-out questions
    df_heldout = df_all[df_all["sample_id"].isin(held_out_ids)].copy()
    question_ids = sorted(list(df_heldout["sample_id"].unique()))
    logger.info(f"{dataset_name} held-out questions: {len(question_ids)}")

    perm_mgr = PermutationManager(seed=seed, num_permutations=num_permutations)
    
    # Available methods
    methods = [
        "first_ans",
        "self_consistency",
        "highest_mean_logp",
        "highest_self_certainty",
        "rwc_self_certainty",
    ]
    if dataset_name == "MathVista":
        methods.extend([
            "highest_vision_only",
            "rwc_vision_only",
            "highest_visconf",
            "rwc_visconf",
        ])

    # Structure to hold question-level results:
    # results_by_budget[budget][method] = array of shape (num_questions,) of order-averaged accuracies
    results_by_budget = {b: {m: [] for m in methods} for b in budgets}
    cost_by_budget = {b: {m: [] for m in methods} for b in budgets}
    oracle_by_budget = {b: [] for b in budgets}
    diversity_by_budget = {b: [] for b in budgets}
    sampling_fail_by_budget = {b: {m: [] for m in methods} for b in budgets}
    selection_fail_by_budget = {b: {m: [] for m in methods} for b in budgets}

    # Group rollouts by question for fast slicing
    grouped = {qid: group.to_dict("records") for qid, group in df_heldout.groupby("sample_id")}

    for qid in question_ids:
        q_rollouts = grouped.get(qid, [])
        if len(q_rollouts) < 32:
            continue

        orders = perm_mgr.get_permutations_for_question(qid, len(q_rollouts))

        for b in budgets:
            q_order_accs = {m: [] for m in methods}
            q_order_costs = {m: [] for m in methods}
            q_order_oracles = []
            q_order_diversities = []
            q_order_samp_fails = {m: [] for m in methods}
            q_order_sel_fails = {m: [] for m in methods}

            for p_idx in range(num_permutations):
                prefix_indices = perm_mgr.get_candidate_prefix(qid, p_idx, b)
                candidates = [q_rollouts[i] for i in prefix_indices]

                # If MathVista and evaluating VisConf, calibrate Z_i using ONLY revealed prefix candidates
                if dataset_name == "MathVista":
                    candidates = calibrate_visconf_for_candidates(candidates)

                has_correct = compute_oracle_coverage(candidates)
                q_order_oracles.append(float(has_correct))
                q_order_diversities.append(count_distinct_answer_groups(candidates))

                # Evaluate each method
                for m in methods:
                    if m == "first_ans":
                        res = select_first_ans(candidates)
                    elif m == "self_consistency":
                        res = select_self_consistency(candidates)
                    elif m == "highest_mean_logp":
                        res = select_highest_score(candidates, score_key="L_i")
                    elif m == "highest_self_certainty":
                        res = select_highest_score(candidates, score_key="K_i")
                    elif m == "rwc_self_certainty":
                        res = select_rank_weighted_consensus(candidates, score_key="K_i")
                    elif m == "highest_vision_only":
                        res = select_highest_score(candidates, score_key="V_i")
                    elif m == "rwc_vision_only":
                        res = select_rank_weighted_consensus(candidates, score_key="V_i")
                    elif m == "highest_visconf":
                        res = select_highest_score(candidates, score_key="Z_i")
                    elif m == "rwc_visconf":
                        res = select_rank_weighted_consensus(candidates, score_key="Z_i")
                    else:
                        raise ValueError(f"Unknown method {m}")

                    is_correct = bool(res.get("is_correct", False))
                    cost = float(res.get("token_cost", 0))

                    q_order_accs[m].append(float(is_correct))
                    q_order_costs[m].append(cost)

                    samp_f, sel_f = classify_failure(has_correct, is_correct)
                    q_order_samp_fails[m].append(float(samp_f))
                    q_order_sel_fails[m].append(float(sel_f))

            # Average across the 10 orders for this question
            oracle_by_budget[b].append(np.mean(q_order_oracles))
            diversity_by_budget[b].append(np.mean(q_order_diversities))
            for m in methods:
                results_by_budget[b][m].append(np.mean(q_order_accs[m]))
                cost_by_budget[b][m].append(np.mean(q_order_costs[m]))
                sampling_fail_by_budget[b][m].append(np.mean(q_order_samp_fails[m]))
                selection_fail_by_budget[b][m].append(np.mean(q_order_sel_fails[m]))

    return {
        "dataset": dataset_name,
        "budgets": budgets,
        "methods": methods,
        "num_questions": len(question_ids),
        "results_by_budget": results_by_budget,
        "cost_by_budget": cost_by_budget,
        "oracle_by_budget": oracle_by_budget,
        "diversity_by_budget": diversity_by_budget,
        "sampling_fail_by_budget": sampling_fail_by_budget,
        "selection_fail_by_budget": selection_fail_by_budget,
    }


def main():
    parser = argparse.ArgumentParser(description="Run offline sampling and selection (Exp A & B)")
    parser.add_argument("--rollouts_dir", type=str, default="task3/results/per_rollout", help="Directory containing canonical rollouts")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output directory")
    parser.add_argument("--results_dir", type=str, default="task3/results/per_question", help="Results output directory")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32], help="Candidate budgets N")
    parser.add_argument("--seed", type=int, default=6304, help="Random seed")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    manifests_dir = Path(args.manifests_dir)
    tables_dir = Path(args.tables_dir)
    results_dir = Path(args.results_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    math_eval = evaluate_offline_for_dataset(
        rollouts_dir / "math_rollouts.parquet",
        manifests_dir / "math_splits.json",
        "MATH",
        args.budgets,
        seed=args.seed
    )

    mathvista_eval = evaluate_offline_for_dataset(
        rollouts_dir / "mathvista_rollouts.parquet",
        manifests_dir / "mathvista_splits.json",
        "MathVista",
        args.budgets,
        seed=args.seed
    )

    # Compile Table T03 (FirstAns, Oracle Coverage, and Diversity)
    t03_rows = []
    for ev in [math_eval, mathvista_eval]:
        ds = ev["dataset"]
        for b in args.budgets:
            t03_rows.append({
                "dataset": ds,
                "budget_n": b,
                "firstans_accuracy": np.mean(ev["results_by_budget"][b]["first_ans"]),
                "oracle_coverage": np.mean(ev["oracle_by_budget"][b]),
                "mean_distinct_answer_groups": np.mean(ev["diversity_by_budget"][b]),
                "sampling_failure_rate": np.mean(ev["sampling_fail_by_budget"][b]["first_ans"]),
                "selection_failure_rate": np.mean(ev["selection_fail_by_budget"][b]["first_ans"]),
            })

    df_t03 = build_table_t03(t03_rows)
    df_t03.to_csv(tables_dir / "T03_firstans_oracle_diversity.csv", index=False)
    logger.info(f"Saved Table T03 to {tables_dir / 'T03_firstans_oracle_diversity.csv'}")

    # Compile Table T04 (Method Comparison)
    t04_rows = []
    for ev in [math_eval, mathvista_eval]:
        ds = ev["dataset"]
        for b in args.budgets:
            for m in ev["methods"]:
                t04_rows.append({
                    "dataset": ds,
                    "budget_n": b,
                    "method": m,
                    "accuracy": np.mean(ev["results_by_budget"][b][m]),
                    "mean_token_cost": np.mean(ev["cost_by_budget"][b][m]),
                    "oracle_coverage": np.mean(ev["oracle_by_budget"][b]),
                    "distinct_answer_groups": np.mean(ev["diversity_by_budget"][b]),
                    "sampling_failure_rate": np.mean(ev["sampling_fail_by_budget"][b][m]),
                    "selection_failure_rate": np.mean(ev["selection_fail_by_budget"][b][m]),
                    "valid_questions": ev["num_questions"],
                })

    df_t04 = build_table_t04(t04_rows)
    df_t04.to_csv(tables_dir / "T04_offline_method_comparison.csv", index=False)
    logger.info(f"Saved Table T04 to {tables_dir / 'T04_offline_method_comparison.csv'}")

    # Compile Table T05 (Main paired differences with 1,000 question bootstrap intervals at N=32)
    t05_rows = []
    for ev in [math_eval, mathvista_eval]:
        ds = ev["dataset"]
        b = 32
        res32 = ev["results_by_budget"][b]
        
        comparisons = [
            ("self_consistency", "first_ans", "SC vs FirstAns (N=32)"),
            ("rwc_self_certainty", "self_consistency", "K+RWC vs SC (N=32)"),
            ("highest_self_certainty", "first_ans", "Highest K vs FirstAns (N=32)"),
        ]
        if ds == "MathVista":
            comparisons.extend([
                ("rwc_visconf", "self_consistency", "VisConf RWC vs SC (N=32)"),
                ("highest_visconf", "highest_vision_only", "VisConf vs Vision-only (N=32)"),
            ])

        for m_a, m_b, comp_label in comparisons:
            if m_a in res32 and m_b in res32:
                arr_a = np.array(res32[m_a])
                arr_b = np.array(res32[m_b])
                boot = paired_question_bootstrap(arr_a, arr_b, num_resamples=1000, seed=args.seed)
                t05_rows.append({
                    "dataset": ds,
                    "budget_n": b,
                    "comparison": comp_label,
                    "method_a": m_a,
                    "method_b": m_b,
                    "mean_diff": boot["mean_diff"],
                    "ci_lower": boot["ci_lower"],
                    "ci_upper": boot["ci_upper"],
                    "p_value": boot.get("p_value", 0.0),
                })

    df_t05 = build_table_t05(t05_rows)
    df_t05.to_csv(tables_dir / "T05_paired_accuracy_differences.csv", index=False)
    logger.info(f"Saved Table T05 to {tables_dir / 'T05_paired_accuracy_differences.csv'}")


if __name__ == "__main__":
    main()
