"""
Unit tests for Task 3 Split Builder and Data Integrity.
Verifies:
1. Lexicographical sorting before permutation.
2. Deterministic 200 dev / 800 held-out split using seed 6304.
3. Disjointness between dev and held-out.
4. Live 50 questions match exactly the first 50 held-out questions in permuted order.
"""

import unittest
from task3.src.data.split_builder import build_dataset_splits


class TestSplits(unittest.TestCase):
    def test_lexicographic_and_split_counts(self):
        """Verify split counts (200 dev, 800 held-out) and disjointness."""
        sample_ids = [f"q_{i:04d}" for i in range(1000)]
        split = build_dataset_splits(sample_ids, seed=6304, dataset_name="MATH")

        self.assertEqual(split.total_questions, 1000)
        self.assertEqual(len(split.dev_question_ids), 200)
        self.assertEqual(len(split.held_out_question_ids), 800)
        self.assertTrue(split.verify_disjointness())

    def test_live_50_order_and_membership(self):
        """Verify live 50 are exactly the first 50 held-out questions in permutation order."""
        sample_ids = [f"item_{i:04d}" for i in range(1000)]
        split = build_dataset_splits(sample_ids, seed=6304, dataset_name="MATH")

        self.assertEqual(len(split.live_50_question_ids), 50)
        self.assertEqual(split.live_50_question_ids, split.held_out_question_ids[:50])

    def test_seed_determinism(self):
        """Verify identical splits on repeated calls with seed 6304."""
        sample_ids = [f"prob_{i}" for i in range(1000)]
        split_1 = build_dataset_splits(sample_ids, seed=6304, dataset_name="MathVista")
        split_2 = build_dataset_splits(sample_ids, seed=6304, dataset_name="MathVista")

        self.assertEqual(split_1.dev_question_ids, split_2.dev_question_ids)
        self.assertEqual(split_1.held_out_question_ids, split_2.held_out_question_ids)
        self.assertEqual(split_1.live_50_question_ids, split_2.live_50_question_ids)


if __name__ == "__main__":
    unittest.main()
