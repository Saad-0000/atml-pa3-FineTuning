"""
Selection algorithms: FirstAns, Self-Consistency, Highest-Score, and Rank-Weighted Consensus (RWC).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import rankdata

from task3.src.scoring.normalization import NO_ANSWER, are_answers_equivalent, normalize_answer, grade_correctness


def get_cand_answer(cand: Dict[str, Any]) -> str:
    """Extracts answer string from normalized_answer or extracted_answer."""
    if "normalized_answer" in cand and cand["normalized_answer"] is not None:
        return str(cand["normalized_answer"])
    return str(cand.get("extracted_answer", NO_ANSWER))


def get_cand_length(cand: Dict[str, Any]) -> int:
    """Extracts token length from token_length, length, or token_ids."""
    if "token_length" in cand:
        return int(cand["token_length"])
    if "length" in cand:
        return int(cand["length"])
    if "token_ids" in cand and isinstance(cand["token_ids"], list):
        return len(cand["token_ids"])
    return 100


def group_candidates_by_equivalence(
    candidates: List[Dict[str, Any]],
    dataset: str = "math",
) -> List[List[int]]:
    """Groups candidate indices into equivalence groups without consulting gold answers."""
    groups: List[List[int]] = []

    for i, cand in enumerate(candidates):
        ans_i = get_cand_answer(cand)
        matched_group = None
        for grp in groups:
            rep_cand = candidates[grp[0]]
            ans_rep = get_cand_answer(rep_cand)
            if are_answers_equivalent(ans_i, ans_rep, dataset=dataset):
                matched_group = grp
                break

        if matched_group is not None:
            matched_group.append(i)
        else:
            groups.append([i])

    return groups


def select_first_ans(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """FirstAns: Returns the first candidate in the given order."""
    first = candidates[0]
    ans = get_cand_answer(first)
    cost = get_cand_length(first)
    res = {
        "selected_candidate_idx": 0,
        "selected_answer": ans,
        "selected_rollout_id": first.get("rollout_id", 0),
        "consumed_candidates": 1,
        "consumed_token_cost": cost,
        "token_cost": cost,
    }
    if "is_correct" in first:
        res["is_correct"] = bool(first["is_correct"])
    return res


def select_self_consistency(
    candidates: List[Dict[str, Any]],
    dataset: str = "math",
) -> Dict[str, Any]:
    """Self-Consistency: Equal vote per candidate, selects majority answer group.

    Ties are resolved deterministically by earliest supporting rollout in the current order.
    """
    n = len(candidates)
    total_cost = sum(get_cand_length(cand) for cand in candidates)
    if n == 1:
        c = candidates[0]
        ans = get_cand_answer(c)
        res = {
            "selected_candidate_idx": 0,
            "selected_answer": ans,
            "selected_rollout_id": c.get("rollout_id", 0),
            "vote_counts": {0: 1},
            "winner_group_size": 1,
            "support_fraction": 1.0,
            "is_unique_leader": True,
            "consumed_candidates": 1,
            "consumed_token_cost": total_cost,
            "token_cost": total_cost,
        }
        if "is_correct" in c:
            res["is_correct"] = bool(c["is_correct"])
        return res

    groups = group_candidates_by_equivalence(candidates, dataset=dataset)
    groups_sorted = sorted(groups, key=lambda g: (-len(g), g[0]))

    winner_grp = groups_sorted[0]
    winner_idx = winner_grp[0]
    winner_cand = candidates[winner_idx]
    winner_ans = get_cand_answer(winner_cand)
    support_fraction = len(winner_grp) / float(n)

    is_unique_leader = True
    if len(groups_sorted) > 1 and len(groups_sorted[1]) == len(winner_grp):
        is_unique_leader = False

    res = {
        "selected_candidate_idx": winner_idx,
        "selected_rollout_id": winner_cand.get("rollout_id", winner_idx),
        "selected_answer": winner_ans,
        "groups": groups,
        "winner_group": winner_grp,
        "winner_group_size": len(winner_grp),
        "support_fraction": support_fraction,
        "is_unique_leader": is_unique_leader,
        "consumed_candidates": n,
        "consumed_token_cost": total_cost,
        "token_cost": total_cost,
    }
    if "is_correct" in winner_cand:
        res["is_correct"] = bool(winner_cand["is_correct"])
    return res


def select_highest_score(
    candidates: List[Dict[str, Any]],
    score_key: str,
) -> Dict[str, Any]:
    """Highest-Score selection: picks candidate with highest quality score.

    Ties broken by earliest candidate in current order.
    """
    scores = [float(c.get(score_key, -float("inf"))) for c in candidates]
    best_idx = 0
    best_score = scores[0]

    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    selected = candidates[best_idx]
    total_cost = sum(get_cand_length(cand) for cand in candidates)
    res = {
        "selected_candidate_idx": best_idx,
        "selected_rollout_id": selected.get("rollout_id", best_idx),
        "selected_answer": get_cand_answer(selected),
        "selected_score": best_score,
        "consumed_candidates": len(candidates),
        "consumed_token_cost": total_cost,
        "token_cost": total_cost,
    }
    if "is_correct" in selected:
        res["is_correct"] = bool(selected["is_correct"])
    return res


def select_rwc(
    candidates: List[Dict[str, Any]],
    score_key: str,
    dataset: str = "math",
) -> Dict[str, Any]:
    """Rank-Weighted Consensus (RWC).

    Protocol:
    1. Descending rank r_i where rank 1 is best, using average ranks for tied scores.
    2. Weights: w_i = (N - r_i) / (N - 1) for N > 1 (w_i = 1.0 for N = 1).
    3. Group vote W(a) = sum_{i: a_i = a} w_i.
    4. Winner is answer group with largest total vote weight.
    5. Tied answer-group votes resolved by earliest supporting rollout in current order.
    6. Margin = (largest_vote - second_largest_vote) / total_vote_weight.
    """
    n = len(candidates)
    total_cost = sum(get_cand_length(cand) for cand in candidates)
    if n == 1:
        c = candidates[0]
        res = {
            "selected_candidate_idx": 0,
            "selected_rollout_id": c.get("rollout_id", 0),
            "selected_answer": get_cand_answer(c),
            "group_votes": [1.0],
            "margin": 1.0,
            "vote_margin": 1.0,
            "consumed_candidates": 1,
            "consumed_token_cost": total_cost,
            "token_cost": total_cost,
        }
        if "is_correct" in c:
            res["is_correct"] = bool(c["is_correct"])
        return res

    scores = np.array([float(c.get(score_key, 0.0)) for c in candidates], dtype=np.float64)
    ranks = rankdata(-scores, method="average")
    weights = (float(n) - ranks) / float(n - 1)

    groups = group_candidates_by_equivalence(candidates, dataset=dataset)
    group_votes = []
    for g in groups:
        vote = float(np.sum(weights[g]))
        group_votes.append((vote, g[0], g))

    group_votes_sorted = sorted(group_votes, key=lambda item: (-item[0], item[1]))

    best_vote, best_rep_idx, best_grp = group_votes_sorted[0]
    second_vote = group_votes_sorted[1][0] if len(group_votes_sorted) > 1 else 0.0
    total_vote = float(np.sum(weights))

    margin = float((best_vote - second_vote) / max(total_vote, 1e-8)) if total_vote > 0 else 0.0
    winner_cand = candidates[best_rep_idx]

    res = {
        "selected_candidate_idx": best_rep_idx,
        "selected_rollout_id": winner_cand.get("rollout_id", best_rep_idx),
        "selected_answer": get_cand_answer(winner_cand),
        "weights": weights.tolist(),
        "group_votes": [item[0] for item in group_votes_sorted],
        "winner_vote": best_vote,
        "runner_up_vote": second_vote,
        "margin": margin,
        "vote_margin": margin,
        "consumed_candidates": n,
        "consumed_token_cost": total_cost,
        "token_cost": total_cost,
    }
    if "is_correct" in winner_cand:
        res["is_correct"] = bool(winner_cand["is_correct"])
    return res


# Alias
select_rank_weighted_consensus = select_rwc
