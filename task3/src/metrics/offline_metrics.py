"""
Offline evaluation metrics: accuracy, oracle coverage, diversity, and failure decomposition.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from task3.src.scoring.normalization import NO_ANSWER, are_answers_equivalent, grade_answer


class FailureClassification:
    def __init__(self, sampling_failure: bool, selection_failure: bool):
        self.sampling_failure = sampling_failure
        self.selection_failure = selection_failure

    def __iter__(self):
        yield self.sampling_failure
        yield self.selection_failure

    def __getitem__(self, key):
        if key == "sampling_failure":
            return self.sampling_failure
        elif key == "selection_failure":
            return self.selection_failure
        raise KeyError(key)

    def __bool__(self):
        return self.sampling_failure or self.selection_failure


def compute_oracle_coverage(candidates: List[Dict[str, Any]]) -> bool:
    """Returns True if at least one candidate in revealed prefix is correct."""
    return any(bool(c.get("is_correct", False)) for c in candidates)


def count_distinct_answer_groups(candidates: List[Dict[str, Any]], dataset: str = "math") -> int:
    """Counts number of distinct normalized answer groups among candidates."""
    distinct_groups = []
    for cand in candidates:
        ans = cand.get("normalized_answer", cand.get("extracted_answer", NO_ANSWER))
        found = False
        for grp_rep in distinct_groups:
            if are_answers_equivalent(ans, grp_rep, dataset=dataset):
                found = True
                break
        if not found:
            distinct_groups.append(ans)
    return len(distinct_groups)


def compute_prefix_oracle_and_diversity(
    candidates: List[Dict[str, Any]],
    gold_answer: str,
    dataset: str = "math",
) -> Dict[str, Any]:
    """Computes oracle coverage and number of distinct normalized answer groups in prefix."""
    correct_flags = [
        bool(cand.get("is_correct", False)) or grade_answer(cand.get("extracted_answer", ""), gold_answer, dataset=dataset)
        for cand in candidates
    ]
    oracle_covered = any(correct_flags)
    distinct_count = count_distinct_answer_groups(candidates, dataset=dataset)

    return {
        "oracle_covered": oracle_covered,
        "correct_candidates_count": sum(correct_flags),
        "distinct_answer_groups_count": distinct_count,
        "correct_flags": correct_flags,
    }


def classify_failure(
    has_correct_candidate: bool,
    is_correct_selected: bool,
) -> FailureClassification:
    """Decomposes an error into sampling failure vs selection failure.

    Sampling failure: Oracle coverage is False (no correct answer in candidate set).
    Selection failure: Oracle coverage is True, but the method selected an incorrect answer!
    """
    sampling_failure = (not has_correct_candidate) and (not is_correct_selected)
    selection_failure = bool(has_correct_candidate) and (not is_correct_selected)
    return FailureClassification(sampling_failure, selection_failure)
