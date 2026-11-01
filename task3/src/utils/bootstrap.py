"""Question-level paired bootstrap analysis for statistical significance."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def paired_question_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 6304,
) -> Dict[str, float]:
    """Computes paired bootstrap confidence interval for difference: mean(scores_a) - mean(scores_b).

    Args:
        scores_a: Array of shape [num_questions], where each value is already order-averaged.
        scores_b: Array of shape [num_questions], where each value is already order-averaged.
        num_resamples: Number of bootstrap iterations (default 1,000).
        confidence_level: Desired interval width (default 0.95).
        seed: Deterministic RNG seed.

    Returns:
        Dictionary with:
            diff_point_estimate: float
            ci_lower: float
            ci_upper: float
            mean_a: float
            mean_b: float
    """
    assert len(scores_a) == len(scores_b), "Both score arrays must have the same number of questions!"
    n = len(scores_a)
    if n == 0:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "diff_point_estimate": 0.0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "num_resamples": num_resamples,
            "confidence_level": confidence_level,
        }

    rng = np.random.default_rng(seed)

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    diff_point = mean_a - mean_b

    # Generate bootstrap question indices [num_resamples, n]
    boot_indices = rng.integers(0, n, size=(num_resamples, n))
    boot_diffs = np.zeros(num_resamples, dtype=np.float64)

    for i in range(num_resamples):
        idx = boot_indices[i]
        boot_diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff_point_estimate": diff_point,
        "mean_diff": diff_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "num_resamples": num_resamples,
        "confidence_level": confidence_level,
    }
