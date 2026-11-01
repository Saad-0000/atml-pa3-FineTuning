"""
Task 3 Adaptive Held-Out Evaluation Script
Evaluates frozen agreement and quality-aware policies on the 800-question held-out split.
Computes accuracy, token costs, rollouts consumed, stopping distributions, and premature stop recovery.
Generates Tables T13 and T14.
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
from task3.src.adaptive.controller import AgreementController, QualityAwareController
from task3.src.selection.selectors import select_self_consistency
from task3.src.visualization.tables import build_table_t13, build_table_t14

logger = get_logger("run_adaptive_test")


def evaluate_policy_on_heldout(
    df_heldout: pd.DataFrame,
    dataset_name: str,
    controller,
    perm_mgr: PermutationManager,
    num_permutations: int = 10
) -> Dict[str, Any]:
    """Evaluates an adaptive controller on the held-out split across 10 orders."""
    grouped = {qid: group.to_dict("records") for qid, group in df_heldout.groupby("sample_id")}
    question_ids = sorted(list(grouped.keys()))

    q_accs = []
    q_costs = []
    q_rollouts = []
    stop_counts = {4: 0, 8: 0, 16: 0, 32: 0}
    
    premature_stops = {4: 0, 8: 0, 16: 0}
    recovered_by_f32 = {4: 0, 8: 0, 16: 0}
    total_evals = 0

    for qid in question_ids:
        rollouts = grouped[qid]
        if len(rollouts) < 32:
            continue

        orders = perm_mgr.get_permutations_for_question(qid, len(rollouts))
        
        q_order_accs = []
        q_order_costs = []
        q_order_rollouts = []

        for p_idx in range(num_permutations):
            total_evals += 1
            perm_indices = orders[p_idx]
            ordered_candidates = [rollouts[i] for i in perm_indices]

            # Fixed-32 SC outcome for reference
            f32_res = select_self_consistency(ordered_candidates)
            f32_correct = bool(f32_res.get("is_correct", False))

            # Run controller
            dec = controller.evaluate_rollout_sequence(ordered_candidates)
            is_correct = bool(dec.get("is_correct", False))
            cost = float(dec.get("token_cost", 0))
            consumed = int(dec.get("candidates_consumed", 32))

            q_order_accs.append(float(is_correct))
            q_order_costs.append(cost)
            q_order_rollouts.append(consumed)
            stop_counts[consumed] = stop_counts.get(consumed, 0) + 1

            # Retrospective check on premature stops
            if consumed in [4, 8, 16] and not is_correct:
                premature_stops[consumed] += 1
                if f32_correct:
                    recovered_by_f32[consumed] += 1

        q_accs.append(np.mean(q_order_accs))
        q_costs.append(np.mean(q_order_costs))
        q_rollouts.append(np.mean(q_order_rollouts))

    return {
        "dataset": dataset_name,
        "policy_family": controller.policy_family if hasattr(controller, "policy_family") else "adaptive",
        "accuracy": float(np.mean(q_accs)),
        "mean_token_cost": float(np.mean(q_costs)),
        "mean_rollouts_consumed": float(np.mean(q_rollouts)),
        "frac_stopped_at_4": stop_counts.get(4, 0) / max(total_evals, 1),
        "frac_stopped_at_8": stop_counts.get(8, 0) / max(total_evals, 1),
        "frac_stopped_at_16": stop_counts.get(16, 0) / max(total_evals, 1),
        "frac_stopped_at_32": stop_counts.get(32, 0) / max(total_evals, 1),
        "premature_stops": premature_stops,
        "recovered_by_f32": recovered_by_f32,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen adaptive policies on held-out split")
    parser.add_argument("--rollouts_dir", type=str, default="task3/results/per_rollout", help="Rollouts directory")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables directory")
    parser.add_argument("--results_dir", type=str, default="task3/results/adaptive", help="Results directory")
    parser.add_argument("--seed", type=int, default=6304, help="Random seed")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    manifests_dir = Path(args.manifests_dir)
    tables_dir = Path(args.tables_dir)
    results_dir = Path(args.results_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    math_splits = load_splits_manifest(manifests_dir / "math_splits.json")
    mathvista_splits = load_splits_manifest(manifests_dir / "mathvista_splits.json")

    math_heldout_ids = set(math_splits.get("held_out_question_ids", []))
    mathvista_heldout_ids = set(mathvista_splits.get("held_out_question_ids", []))

    df_math = pd.read_parquet(rollouts_dir / "math_rollouts.parquet")
    df_math_heldout = df_math[df_math["sample_id"].isin(math_heldout_ids)].copy()

    df_mathvista = pd.read_parquet(rollouts_dir / "mathvista_rollouts.parquet")
    df_mathvista_heldout = df_mathvista[df_mathvista["sample_id"].isin(mathvista_heldout_ids)].copy()

    with open(manifests_dir / "adaptive_selected_policies.json", "r", encoding="utf-8") as f:
        selected_policies = json.load(f)

    perm_mgr = PermutationManager(seed=args.seed)

    t13_records = []
    t14_records = []

    for ds, df_h in [("MATH", df_math_heldout), ("MathVista", df_mathvista_heldout)]:
        ds_pol = selected_policies.get(ds, {})
        ag_setting = ds_pol.get("agreement", {"tau": 0.75})
        qa_setting = ds_pol.get("quality_aware", {"tau": 0.75, "gamma": 0.25})

        ag_ctrl = AgreementController(tau=ag_setting.get("tau", 0.75))
        ag_ctrl.policy_family = "agreement"
        ag_eval = evaluate_policy_on_heldout(df_h, ds, ag_ctrl, perm_mgr)
        t13_records.append(ag_eval)

        q_key = "K_i" if ds == "MATH" else "Z_i"
        qa_ctrl = QualityAwareController(
            tau=qa_setting.get("tau", 0.75),
            gamma=qa_setting.get("gamma", 0.25),
            quality_score_key=q_key,
            dataset_name=ds
        )
        qa_ctrl.policy_family = "quality_aware"
        qa_eval = evaluate_policy_on_heldout(df_h, ds, qa_ctrl, perm_mgr)
        t13_records.append(qa_eval)

        # Premature stops for T14
        for pol_eval in [ag_eval, qa_eval]:
            p_fam = pol_eval["policy_family"]
            for st in [4, 8, 16]:
                t14_records.append({
                    "dataset": ds,
                    "policy_family": p_fam,
                    "stop_stage": st,
                    "premature_stop_count": pol_eval["premature_stops"].get(st, 0),
                    "recovered_by_f32_count": pol_eval["recovered_by_f32"].get(st, 0),
                    "recovery_rate": round(pol_eval["recovered_by_f32"].get(st, 0) / max(pol_eval["premature_stops"].get(st, 1), 1), 3),
                })

    df_t13 = build_table_t13(t13_records)
    df_t13.to_csv(tables_dir / "T13_adaptive_heldout_results.csv", index=False)
    logger.info(f"Saved Table T13 to {tables_dir / 'T13_adaptive_heldout_results.csv'}")

    df_t14 = build_table_t14(t14_records)
    df_t14.to_csv(tables_dir / "T14_premature_stop_recovery.csv", index=False)
    logger.info(f"Saved Table T14 to {tables_dir / 'T14_premature_stop_recovery.csv'}")

    atomic_save_json({"t13": t13_records, "t14": t14_records}, results_dir / "heldout_adaptive_results.json")


if __name__ == "__main__":
    main()
