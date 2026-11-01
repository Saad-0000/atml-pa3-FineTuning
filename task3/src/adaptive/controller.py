"""
Adaptive sampling controllers: Agreement-based and Quality-aware stopping rules.
"""

from typing import Any, Dict, List, Optional, Tuple

from task3.src.scoring.normalization import NO_ANSWER
from task3.src.selection.quality_scores import calibrate_visconf_for_candidates
from task3.src.selection.selectors import select_rwc, select_self_consistency, get_cand_length


class AgreementController:
    """Agreement-based adaptive stopping rule."""

    def __init__(
        self,
        tau: float = 0.75,
        checkpoints: Optional[List[int]] = None,
        max_candidates: int = 32,
        dataset: str = "math",
    ):
        self.tau = tau
        self.checkpoints = checkpoints or [4, 8, 16]
        self.max_candidates = max_candidates
        self.dataset = dataset.lower()
        self.policy_family = "agreement"

    def should_stop(self, revealed_candidates: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        sc_res = select_self_consistency(revealed_candidates, dataset=self.dataset)
        leading_ans = sc_res["selected_answer"]
        support_frac = sc_res["support_fraction"]
        is_unique = sc_res["is_unique_leader"]

        stop = is_unique and (leading_ans != NO_ANSWER) and (support_frac >= self.tau)
        return stop, sc_res

    def evaluate_rollout_sequence(self, ordered_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reveals candidates progressively and checks stop conditions at checkpoints."""
        for n in self.checkpoints:
            if len(ordered_candidates) >= n:
                prefix = ordered_candidates[:n]
                stop, sc_res = self.should_stop(prefix)
                if stop:
                    cost = sum(get_cand_length(c) for c in prefix)
                    res = {
                        "stopped": True,
                        "checkpoint_n": n,
                        "candidates_consumed": n,
                        "selected_answer": sc_res["selected_answer"],
                        "token_cost": cost,
                    }
                    if "is_correct" in sc_res:
                        res["is_correct"] = sc_res["is_correct"]
                    return res

        # Fallback to max_candidates (32)
        n = min(self.max_candidates, len(ordered_candidates))
        prefix = ordered_candidates[:n]
        sc_res = select_self_consistency(prefix, dataset=self.dataset)
        cost = sum(get_cand_length(c) for c in prefix)
        res = {
            "stopped": False,
            "checkpoint_n": n,
            "candidates_consumed": n,
            "selected_answer": sc_res["selected_answer"],
            "token_cost": cost,
        }
        if "is_correct" in sc_res:
            res["is_correct"] = sc_res["is_correct"]
        return res


class QualityAwareController:
    """Quality-aware adaptive stopping rule."""

    def __init__(
        self,
        tau: float = 0.75,
        gamma: float = 0.25,
        quality_score_key: str = "K_i",
        checkpoints: Optional[List[int]] = None,
        max_candidates: int = 32,
        dataset_name: str = "math",
    ):
        self.tau = tau
        self.gamma = gamma
        self.quality_score_key = quality_score_key
        self.checkpoints = checkpoints or [4, 8, 16]
        self.max_candidates = max_candidates
        self.dataset = dataset_name.lower()
        self.policy_family = "quality_aware"

    def should_stop(self, revealed_candidates: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        # 1. Agreement condition
        sc_res = select_self_consistency(revealed_candidates, dataset=self.dataset)
        leading_ans = sc_res["selected_answer"]
        support_frac = sc_res["support_fraction"]
        is_unique = sc_res["is_unique_leader"]

        if not (is_unique and leading_ans != NO_ANSWER and support_frac >= self.tau):
            return False, sc_res

        # 2. RWC consensus condition (calibrated ONLY on revealed candidates)
        cands_to_score = revealed_candidates
        if self.dataset == "mathvista" and self.quality_score_key == "Z_i":
            cands_to_score = calibrate_visconf_for_candidates(revealed_candidates)

        rwc_res = select_rwc(cands_to_score, score_key=self.quality_score_key, dataset=self.dataset)
        rwc_ans = rwc_res["selected_answer"]
        margin = rwc_res.get("margin", 0.0)

        # SC winner == RWC winner and margin >= gamma
        if leading_ans == rwc_ans and margin >= self.gamma:
            return True, sc_res

        return False, sc_res

    def evaluate_rollout_sequence(self, ordered_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reveals candidates progressively and checks stop conditions at checkpoints."""
        for n in self.checkpoints:
            if len(ordered_candidates) >= n:
                prefix = ordered_candidates[:n]
                stop, sc_res = self.should_stop(prefix)
                if stop:
                    cost = sum(get_cand_length(c) for c in prefix)
                    res = {
                        "stopped": True,
                        "checkpoint_n": n,
                        "candidates_consumed": n,
                        "selected_answer": sc_res["selected_answer"],
                        "token_cost": cost,
                    }
                    if "is_correct" in sc_res:
                        res["is_correct"] = sc_res["is_correct"]
                    return res

        # Fallback to max_candidates (32)
        n = min(self.max_candidates, len(ordered_candidates))
        prefix = ordered_candidates[:n]
        sc_res = select_self_consistency(prefix, dataset=self.dataset)
        cost = sum(get_cand_length(c) for c in prefix)
        res = {
            "stopped": False,
            "checkpoint_n": n,
            "candidates_consumed": n,
            "selected_answer": sc_res["selected_answer"],
            "token_cost": cost,
        }
        if "is_correct" in sc_res:
            res["is_correct"] = sc_res["is_correct"]
        return res


# Backward compatibility
AdaptiveController = AgreementController
