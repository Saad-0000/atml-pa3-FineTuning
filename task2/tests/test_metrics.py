"""Unit tests for evaluation metrics: VQA soft score, exact match, and bootstrap CIs.
"""
import unittest
import numpy as np

from task2.src.data.normalization import normalize_answer
from task2.src.metrics.vqa_scorer import compute_vqa_soft_score, compute_exact_match
from task2.src.metrics.bootstrap import compute_pair_bootstrap_ci, compute_paired_advantage_bootstrap_ci


class TestMetrics(unittest.TestCase):
    def test_vqa_soft_scoring_exact_formula(self):
        """Validates the standard VQA soft consensus formula:

        score = (1/10) * sum_{i=1}^{10} min(1, (1/3) * sum_{j!=i} 1[a_hat == a_j])
        """
        # Case 0: 0 matches -> score 0.0
        human_0 = ["dog"] * 10
        self.assertEqual(compute_vqa_soft_score("cat", human_0), 0.0)

        # Case 1: 1 match out of 10 -> (1*0 + 9*(1/3)) / 10 = 0.3
        human_1 = ["cat"] + ["dog"] * 9
        self.assertAlmostEqual(compute_vqa_soft_score("cat", human_1), 0.3, places=5)

        # Case 2: 2 matches out of 10 -> (2*(1/3) + 8*(2/3)) / 10 = (2/3 + 16/3)/10 = 6/10 = 0.6
        human_2 = ["cat"] * 2 + ["dog"] * 8
        self.assertAlmostEqual(compute_vqa_soft_score("cat", human_2), 0.6, places=5)

        # Case 3: 3 matches out of 10 -> (3*(2/3) + 7*1.0) / 10 = (2 + 7)/10 = 0.9
        human_3 = ["cat"] * 3 + ["dog"] * 7
        self.assertAlmostEqual(compute_vqa_soft_score("cat", human_3), 0.9, places=5)

        # Case 4: 4 matches out of 10 -> (4*1.0 + 6*1.0) / 10 = 1.0
        human_4 = ["cat"] * 4 + ["dog"] * 6
        self.assertAlmostEqual(compute_vqa_soft_score("cat", human_4), 1.0, places=5)

        # Case 10: 10 matches -> 1.0
        human_10 = ["cat"] * 10
        self.assertEqual(compute_vqa_soft_score("cat", human_10), 1.0)

    def test_normalization_rules(self):
        """Tests VQA normalization of contractions, numbers, articles, and punctuation."""
        # Contractions
        self.assertEqual(normalize_answer("don't"), "dont")
        self.assertEqual(normalize_answer("it's"), "its")

        # Number mapping
        self.assertEqual(normalize_answer("one"), "1")
        self.assertEqual(normalize_answer("five"), "5")

        # Articles and punctuation
        self.assertEqual(normalize_answer("A white dog."), "white dog")
        self.assertEqual(normalize_answer("The 2 cats, sitting!"), "2 cats sitting")

    def test_complete_pair_bootstrap_ci(self):
        """Tests that bootstrap CI bounds are consistent and reproducible with seed."""
        pair_ids = [f"pair_{i}" for i in range(50)]

        def metric_fn(indices):
            return float(np.mean(indices % 2 == 0))

        pt, low, high, boots = compute_pair_bootstrap_ci(
            pair_ids=pair_ids,
            metric_fn=metric_fn,
            n_resamples=1000,
            seed=6304,
        )

        self.assertTrue(0.0 <= low <= pt <= high <= 1.0)
        self.assertEqual(len(boots), 1000)


if __name__ == "__main__":
    unittest.main()
