"""Paired complete-pair bootstrap confidence intervals.

Assignment Section 7.4:
- 4,000 bootstrap resamples of complete pairs.
- Match sampled pair IDs across conditions within each bootstrap resample.
- 95% CI using the percentile method (2.5th and 97.5th percentiles).
- Never resample individual images/questions independently when they belong to a complementary pair.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def compute_pair_bootstrap_ci(
    pair_ids: Sequence[str | int],
    metric_fn: Callable[[np.ndarray], float],
    n_resamples: int = 4000,
    seed: int = 6304,
    ci_level: float = 0.95,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute point estimate and bootstrap 95% CI over complete pairs.

    Args:
        pair_ids: Array/list of unique pair identifiers.
        metric_fn: Function mapping an array of resampled pair indices (or pair IDs) to a float metric.
        n_resamples: Number of bootstrap iterations (default 4000).
        seed: Random seed for reproducibility (default 6304).
        ci_level: Confidence level (default 0.95 -> [2.5%, 97.5%]).

    Returns:
        point_estimate: Metric computed on the original full sample.
        ci_lower: 2.5th percentile across bootstrap resamples.
        ci_upper: 97.5th percentile across bootstrap resamples.
        boot_estimates: 1D numpy array of all 4,000 bootstrap estimates.
    """
    rng = np.random.RandomState(seed)
    unique_pairs = np.array(list(dict.fromkeys(pair_ids)))  # preserve order of unique pairs
    n_pairs = len(unique_pairs)

    original_indices = np.arange(n_pairs)
    point_estimate = metric_fn(original_indices)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    boot_estimates = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        resampled_indices = rng.choice(n_pairs, size=n_pairs, replace=True)
        boot_estimates[b] = metric_fn(resampled_indices)

    ci_lower = float(np.percentile(boot_estimates, lower_pct))
    ci_upper = float(np.percentile(boot_estimates, upper_pct))

    return float(point_estimate), ci_lower, ci_upper, boot_estimates


def compute_paired_advantage_bootstrap_ci(
    pair_ids: Sequence[str | int],
    eval_fn_cond_a: Callable[[np.ndarray], float],
    eval_fn_cond_b: Callable[[np.ndarray], float],
    n_resamples: int = 4000,
    seed: int = 6304,
    ci_level: float = 0.95,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute paired advantage (Cond A - Cond B) with matched complete-pair bootstrap.

    Both conditions are evaluated on the exact same resampled pair indices for each replicate.

    Returns:
        advantage_point: Point estimate of (Cond A - Cond B).
        ci_lower: 2.5th percentile of difference.
        ci_upper: 97.5th percentile of difference.
        diff_estimates: Array of 4,000 bootstrap differences.
    """
    rng = np.random.RandomState(seed)
    unique_pairs = np.array(list(dict.fromkeys(pair_ids)))
    n_pairs = len(unique_pairs)

    original_indices = np.arange(n_pairs)
    point_a = eval_fn_cond_a(original_indices)
    point_b = eval_fn_cond_b(original_indices)
    advantage_point = point_a - point_b

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    diff_estimates = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        resampled_indices = rng.choice(n_pairs, size=n_pairs, replace=True)
        stat_a = eval_fn_cond_a(resampled_indices)
        stat_b = eval_fn_cond_b(resampled_indices)
        diff_estimates[b] = stat_a - stat_b

    ci_lower = float(np.percentile(diff_estimates, lower_pct))
    ci_upper = float(np.percentile(diff_estimates, upper_pct))

    return float(advantage_point), ci_lower, ci_upper, diff_estimates
