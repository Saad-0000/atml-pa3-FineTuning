"""Evaluation metrics for Task 2: VQA soft consensus, exact match, and pair metrics.

Assignment Section 7.2:
score(a_hat) = (1/10) * sum_{i=1}^{10} min(1, (1/3) * sum_{j != i} 1[a_hat == a_j])
"""
from typing import List, Sequence, Tuple
from task2.src.data.normalization import normalize_answer


def compute_vqa_soft_score(
    prediction: str,
    human_answers: Sequence[str],
    normalize: bool = True,
) -> float:
    """Compute VQA soft consensus score against 10 human answers.

    Formula:
        score(a_hat) = (1/10) * sum_{i=1}^{10} min(1.0, (1/3) * sum_{j != i} 1[a_hat == a_j])

    Args:
        prediction: Generated model prediction string.
        human_answers: Sequence of human reference answers (typically 10).
        normalize: Whether to apply VQA answer normalization to prediction and references.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if len(human_answers) == 0:
        return 0.0

    pred_norm = normalize_answer(prediction) if normalize else prediction.strip().lower()
    ref_norms = [normalize_answer(a) if normalize else a.strip().lower() for a in human_answers]

    total_matches = sum(1 for a in ref_norms if a == pred_norm)
    total_score = 0.0

    for a_i in ref_norms:
        # Matches among the other 9 annotators
        matches_excluding_i = total_matches - (1 if a_i == pred_norm else 0)
        score_i = min(1.0, matches_excluding_i / 3.0)
        total_score += score_i

    return total_score / float(len(ref_norms))


def compute_exact_match(
    prediction: str,
    reference: str,
    normalize: bool = True,
) -> float:
    """Compute normalized exact match (1.0 if match, else 0.0)."""
    p = normalize_answer(prediction) if normalize else prediction.strip()
    r = normalize_answer(reference) if normalize else reference.strip()
    return 1.0 if p == r else 0.0


def compute_pair_metrics(
    pred1: str,
    pred2: str,
    ref1: str,
    ref2: str,
    human_answers1: Sequence[str] | None = None,
    human_answers2: Sequence[str] | None = None,
    is_vqa: bool = False,
) -> Tuple[bool, bool]:
    """Compute pair accuracy and identical-answer flag for a complementary pair.

    Returns:
        (pair_correct, identical_answers):
            - pair_correct: True if BOTH examples in the pair meet correctness criteria.
            - identical_answers: True if model generated identical normalized strings for both images.
    """
    p1_norm = normalize_answer(pred1)
    p2_norm = normalize_answer(pred2)
    identical_answers = (p1_norm == p2_norm)

    if is_vqa:
        # In VQAv2, score >= 0.9 or full consensus match for binary correctness
        score1 = compute_vqa_soft_score(pred1, human_answers1 or [ref1])
        score2 = compute_vqa_soft_score(pred2, human_answers2 or [ref2])
        # A pair is correct if both have positive consensus accuracy >= 0.5 or match canonical
        pair_correct = (score1 >= 0.5 and score2 >= 0.5)
    else:
        # Exact match on both items
        m1 = compute_exact_match(pred1, ref1)
        m2 = compute_exact_match(pred2, ref2)
        pair_correct = bool(m1 == 1.0 and m2 == 1.0)

    return pair_correct, identical_answers
