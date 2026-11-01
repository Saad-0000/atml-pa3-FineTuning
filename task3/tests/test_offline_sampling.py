"""
Unit tests for offline sampling, nested prefixes, and selection algorithms.
"""

import unittest
from task3.src.sampling.permutation_manager import PermutationManager
from task3.src.selection.selectors import (
    select_first_ans,
    select_self_consistency,
    select_highest_score,
    select_rank_weighted_consensus
)
from task3.src.metrics.offline_metrics import compute_oracle_coverage, classify_failure


class TestOfflineSampling(unittest.TestCase):
    def test_nested_prefixes(self):
        """Verify that smaller budget candidate lists are exact prefixes of larger budgets."""
        mgr = PermutationManager(seed=6304, num_permutations=10)
        qid = "sample_test_001"
        
        p0_b1 = mgr.get_candidate_prefix(qid, 0, 1)
        p0_b2 = mgr.get_candidate_prefix(qid, 0, 2)
        p0_b4 = mgr.get_candidate_prefix(qid, 0, 4)
        p0_b8 = mgr.get_candidate_prefix(qid, 0, 8)

        self.assertEqual(p0_b1, p0_b2[:1])
        self.assertEqual(p0_b2, p0_b4[:2])
        self.assertEqual(p0_b4, p0_b8[:4])

    def test_first_ans_cost_accounting(self):
        """Verify FirstAns charges only the first candidate's tokens."""
        candidates = [
            {"rollout_id": 0, "normalized_answer": "A", "token_length": 100, "is_correct": True},
            {"rollout_id": 1, "normalized_answer": "B", "token_length": 250, "is_correct": False},
            {"rollout_id": 2, "normalized_answer": "A", "token_length": 300, "is_correct": True},
        ]

        res = select_first_ans(candidates)
        self.assertEqual(res["selected_answer"], "A")
        self.assertEqual(res["token_cost"], 100)

    def test_sc_tie_breaking_by_earliest_order(self):
        """Verify SC tie resolves by earliest supporting rollout in current order."""
        candidates = [
            {"rollout_id": 10, "normalized_answer": "X", "token_length": 50},
            {"rollout_id": 11, "normalized_answer": "Y", "token_length": 50},
        ]
        res = select_self_consistency(candidates)
        self.assertEqual(res["selected_answer"], "X")

    def test_rwc_tied_ranks_and_n1(self):
        """Verify RWC with tied scores uses average ranks and handles N=1 safely."""
        # N = 1
        c1 = [{"rollout_id": 0, "normalized_answer": "10", "token_length": 40, "K_i": 2.0}]
        res_1 = select_rank_weighted_consensus(c1, score_key="K_i")
        self.assertEqual(res_1["selected_answer"], "10")
        self.assertEqual(res_1["vote_margin"], 1.0)

        # Tied scores at N=3
        c3 = [
            {"rollout_id": 0, "normalized_answer": "Ans1", "token_length": 40, "K_i": 2.0},
            {"rollout_id": 1, "normalized_answer": "Ans1", "token_length": 40, "K_i": 2.0},
            {"rollout_id": 2, "normalized_answer": "Ans2", "token_length": 40, "K_i": 1.0},
        ]
        res_3 = select_rank_weighted_consensus(c3, score_key="K_i")
        self.assertEqual(res_3["selected_answer"], "Ans1")

    def test_failure_decomposition(self):
        """Verify sampling vs selection failure classification."""
        samp_f, sel_f = classify_failure(has_correct_candidate=True, is_correct_selected=False)
        self.assertFalse(samp_f)
        self.assertTrue(sel_f)

        samp_f, sel_f = classify_failure(has_correct_candidate=False, is_correct_selected=False)
        self.assertTrue(samp_f)
        self.assertFalse(sel_f)

        samp_f, sel_f = classify_failure(has_correct_candidate=True, is_correct_selected=True)
        self.assertFalse(samp_f)
        self.assertFalse(sel_f)


if __name__ == "__main__":
    unittest.main()
