"""
Within-question quartile correctness analysis and shortest-rollout control.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_within_question_quartiles(
    candidates: List[Dict[str, Any]],
    score_key: str,
    correct_labels: List[bool],
) -> Dict[str, float]:
    """Computes top and bottom quartile correctness within a single question (N=32).

    Top quartile: 8 highest scored candidates.
    Bottom quartile: 8 lowest scored candidates.
    """
    n = len(candidates)
    if n < 4:
        return {"top_acc": 0.0, "bottom_acc": 0.0, "diff": 0.0}

    q_size = max(1, n // 4)
    scores = np.array([float(c.get(score_key, 0.0)) for c in candidates])
    # Descending sort
    sorted_indices = np.argsort(-scores)

    top_indices = sorted_indices[:q_size]
    bottom_indices = sorted_indices[-q_size:]

    top_acc = float(np.mean([correct_labels[i] for i in top_indices]))
    bottom_acc = float(np.mean([correct_labels[i] for i in bottom_indices]))
    diff = top_acc - bottom_acc

    return {
        "top_acc": top_acc,
        "bottom_acc": bottom_acc,
        "diff": diff,
        "q_size": q_size,
    }


def select_shortest_rollout(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shortest-rollout control: picks the candidate with the fewest generated tokens.

    Ties broken by earliest candidate in current order.
    """
    lengths = [cand.get("token_length", cand.get("length", len(cand.get("token_ids", [])))) for cand in candidates]
    min_len = min(lengths)
    best_idx = lengths.index(min_len)

    selected = candidates[best_idx]
    return {
        "selected_candidate_idx": best_idx,
        "selected_answer": selected.get("normalized_answer", selected.get("extracted_answer", "")),
        "is_correct": selected.get("is_correct", False),
        "min_length": min_len,
        "consumed_candidates": len(candidates),
        "consumed_token_cost": sum(lengths),
    }


def compute_quartile_analysis_for_dataset(
    df,
    dataset_name: str,
    signals: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Averages top vs bottom quartile differences across all questions at N=32."""
    grouped = {qid: group.to_dict("records") for qid, group in df.groupby("sample_id")}
    results = []

    for sig_name, sig_key in signals:
        top_accs = []
        bot_accs = []
        diffs = []

        for qid, cands in grouped.items():
            if len(cands) < 4:
                continue
            labels = [bool(c.get("is_correct", False)) for c in cands]
            q_res = compute_within_question_quartiles(cands, sig_key, labels)
            top_accs.append(q_res["top_acc"])
            bot_accs.append(q_res["bottom_acc"])
            diffs.append(q_res["diff"])

        results.append({
            "dataset": dataset_name,
            "signal_name": sig_name,
            "mean_top_quartile_acc": float(np.mean(top_accs)) if top_accs else 0.0,
            "mean_bottom_quartile_acc": float(np.mean(bot_accs)) if bot_accs else 0.0,
            "mean_top_minus_bottom_diff": float(np.mean(diffs)) if diffs else 0.0,
            "valid_questions": len(diffs),
        })

    # Add shortest rollout control
    shortest_accs = []
    for qid, cands in grouped.items():
        if cands:
            s_res = select_shortest_rollout(cands)
            shortest_accs.append(float(s_res["is_correct"]))

    results.append({
        "dataset": dataset_name,
        "signal_name": "shortest_rollout_control",
        "mean_top_quartile_acc": float(np.mean(shortest_accs)) if shortest_accs else 0.0,
        "mean_bottom_quartile_acc": 0.0,
        "mean_top_minus_bottom_diff": float(np.mean(shortest_accs)) if shortest_accs else 0.0,
        "valid_questions": len(shortest_accs),
    })

    return results
