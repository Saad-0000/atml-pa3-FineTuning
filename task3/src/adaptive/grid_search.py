"""
Grid search and threshold selection for adaptive stopping on the development split.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from task3.src.adaptive.controller import AgreementController, QualityAwareController
from task3.src.sampling.permutation_manager import PermutationManager
from task3.src.scoring.normalization import grade_answer
from task3.src.selection.selectors import select_self_consistency, get_cand_length
from task3.src.utils.common import setup_logger

logger = setup_logger("task3.adaptive.grid_search")


def run_development_grid_search(
    df_dev: pd.DataFrame,
    dataset_name: str,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Runs full grid search on development split (200 questions) for Agreement and Quality-Aware controllers."""
    grouped = {qid: group.to_dict("records") for qid, group in df_dev.groupby("sample_id")}
    qids = sorted(list(grouped.keys()))
    perm_mgr = PermutationManager(seed=seed, num_permutations=10)

    # 1. Evaluate fixed-32 SC on dev split as reference
    fixed_32_accs = []
    fixed_32_costs = []
    for qid in qids:
        cands = grouped[qid]
        if len(cands) < 32:
            continue
        sc_res = select_self_consistency(cands[:32], dataset=dataset_name)
        fixed_32_accs.append(float(sc_res.get("is_correct", False)))
        fixed_32_costs.append(float(sc_res.get("token_cost", 0)))

    f32_acc = float(np.mean(fixed_32_accs)) if fixed_32_accs else 0.5
    f32_cost = float(np.mean(fixed_32_costs)) if fixed_32_costs else 3200.0

    tau_grid = [0.5, 0.625, 0.75, 0.875, 1.0]
    gamma_grid = [0.0, 0.1, 0.25, 0.5]
    feasibility_boundary = f32_acc - 0.01

    grid_results = []

    # Agreement grid
    ag_candidates = []
    for tau in tau_grid:
        ctrl = AgreementController(tau=tau, dataset=dataset_name)
        q_accs = []
        q_costs = []
        q_rolls = []

        for qid in qids:
            cands = grouped[qid]
            orders = perm_mgr.get_permutations_for_question(qid, len(cands))
            ord_accs = []
            ord_costs = []
            ord_rolls = []
            for p_idx in range(len(orders)):
                perm = orders[p_idx]
                ordered = [cands[i] for i in perm]
                dec = ctrl.evaluate_rollout_sequence(ordered)
                ord_accs.append(float(dec.get("is_correct", False)))
                ord_costs.append(float(dec.get("token_cost", 0)))
                ord_rolls.append(float(dec.get("candidates_consumed", 32)))

            q_accs.append(np.mean(ord_accs))
            q_costs.append(np.mean(ord_costs))
            q_rolls.append(np.mean(ord_rolls))

        mean_acc = float(np.mean(q_accs))
        mean_cost = float(np.mean(q_costs))
        mean_rolls = float(np.mean(q_rolls))
        is_feasible = mean_acc >= feasibility_boundary

        row = {
            "dataset": dataset_name,
            "policy_family": "agreement",
            "tau": tau,
            "gamma": 0.0,
            "accuracy": mean_acc,
            "mean_token_cost": mean_cost,
            "mean_rollouts": mean_rolls,
            "is_feasible": is_feasible,
            "is_selected": False,
        }
        grid_results.append(row)
        ag_candidates.append(row)

    # Select best agreement
    feasible_ag = [r for r in ag_candidates if r["is_feasible"]]
    if feasible_ag:
        feasible_ag.sort(key=lambda r: (r["mean_token_cost"], -r["accuracy"], r["tau"]))
        best_ag = feasible_ag[0]
    else:
        best_ag = ag_candidates[-1]
    best_ag["is_selected"] = True

    # Quality-aware grid
    qa_candidates = []
    q_key = "K_i" if dataset_name.upper() == "MATH" else "Z_i"
    for tau in tau_grid:
        for gamma in gamma_grid:
            ctrl = QualityAwareController(tau=tau, gamma=gamma, quality_score_key=q_key, dataset_name=dataset_name)
            q_accs = []
            q_costs = []
            q_rolls = []

            for qid in qids:
                cands = grouped[qid]
                orders = perm_mgr.get_permutations_for_question(qid, len(cands))
                ord_accs = []
                ord_costs = []
                ord_rolls = []
                for p_idx in range(len(orders)):
                    perm = orders[p_idx]
                    ordered = [cands[i] for i in perm]
                    dec = ctrl.evaluate_rollout_sequence(ordered)
                    ord_accs.append(float(dec.get("is_correct", False)))
                    ord_costs.append(float(dec.get("token_cost", 0)))
                    ord_rolls.append(float(dec.get("candidates_consumed", 32)))

                q_accs.append(np.mean(ord_accs))
                q_costs.append(np.mean(ord_costs))
                q_rolls.append(np.mean(ord_rolls))

            mean_acc = float(np.mean(q_accs))
            mean_cost = float(np.mean(q_costs))
            mean_rolls = float(np.mean(q_rolls))
            is_feasible = mean_acc >= feasibility_boundary

            row = {
                "dataset": dataset_name,
                "policy_family": "quality_aware",
                "tau": tau,
                "gamma": gamma,
                "accuracy": mean_acc,
                "mean_token_cost": mean_cost,
                "mean_rollouts": mean_rolls,
                "is_feasible": is_feasible,
                "is_selected": False,
            }
            grid_results.append(row)
            qa_candidates.append(row)

    feasible_qa = [r for r in qa_candidates if r["is_feasible"]]
    if feasible_qa:
        feasible_qa.sort(key=lambda r: (r["mean_token_cost"], -r["accuracy"], r["tau"], r["gamma"]))
        best_qa = feasible_qa[0]
    else:
        best_qa = qa_candidates[-1]
    best_qa["is_selected"] = True

    # Reference fixed-32 row
    grid_results.append({
        "dataset": dataset_name,
        "policy_family": "fixed_32",
        "tau": 1.0,
        "gamma": 1.0,
        "accuracy": f32_acc,
        "mean_token_cost": f32_cost,
        "mean_rollouts": 32.0,
        "is_feasible": True,
        "is_selected": False,
    })

    return {
        "dataset": dataset_name,
        "fixed_32_dev_accuracy": f32_acc,
        "grid_results": grid_results,
        "selected_policies": {
            "agreement": {"tau": best_ag["tau"]},
            "quality_aware": {"tau": best_qa["tau"], "gamma": best_qa["gamma"]},
        }
    }
