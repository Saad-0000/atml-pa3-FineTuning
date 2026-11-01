"""
Unit tests for adaptive stopping controllers (Agreement & Quality-Aware).
"""

import unittest
from task3.src.adaptive.controller import AgreementController, QualityAwareController


class TestAdaptiveControllers(unittest.TestCase):
    def test_agreement_controller_threshold_and_checkpoints(self):
        """Verify agreement controller stops only at 4, 8, 16 or 32 when tau is met."""
        ctrl = AgreementController(tau=0.75, checkpoints=[4, 8, 16])

        # 4 candidates with 3 matching 'A' (fraction 3/4 = 0.75 >= 0.75)
        candidates_stop_at_4 = [
            {"rollout_id": 0, "normalized_answer": "A", "token_length": 50},
            {"rollout_id": 1, "normalized_answer": "A", "token_length": 50},
            {"rollout_id": 2, "normalized_answer": "A", "token_length": 50},
            {"rollout_id": 3, "normalized_answer": "B", "token_length": 50},
        ] + [{"rollout_id": i, "normalized_answer": "B", "token_length": 50} for i in range(4, 32)]

        dec = ctrl.evaluate_rollout_sequence(candidates_stop_at_4)
        self.assertTrue(dec["stopped"])
        self.assertEqual(dec["checkpoint_n"], 4)
        self.assertEqual(dec["candidates_consumed"], 4)
        self.assertEqual(dec["selected_answer"], "A")

    def test_no_answer_never_triggers_early_stop(self):
        """Verify leading group NO_ANSWER never triggers early stop."""
        ctrl = AgreementController(tau=0.5, checkpoints=[4, 8, 16])

        candidates = [
            {"rollout_id": i, "normalized_answer": "NO_ANSWER", "token_length": 10} for i in range(32)
        ]

        dec = ctrl.evaluate_rollout_sequence(candidates)
        # Falls back to 32
        self.assertEqual(dec["candidates_consumed"], 32)

    def test_quality_aware_controller_agreement_and_margin(self):
        """Verify quality-aware controller requires SC winner == RWC winner and margin >= gamma."""
        ctrl = QualityAwareController(tau=0.75, gamma=0.5, quality_score_key="K_i", checkpoints=[4, 8, 16])

        candidates = [
            {"rollout_id": 0, "normalized_answer": "A", "token_length": 50, "K_i": 3.0},
            {"rollout_id": 1, "normalized_answer": "A", "token_length": 50, "K_i": 2.8},
            {"rollout_id": 2, "normalized_answer": "A", "token_length": 50, "K_i": 2.5},
            {"rollout_id": 3, "normalized_answer": "B", "token_length": 50, "K_i": 0.5},
        ] + [{"rollout_id": i, "normalized_answer": "B", "token_length": 50, "K_i": 0.5} for i in range(4, 32)]

        dec = ctrl.evaluate_rollout_sequence(candidates)
        self.assertTrue(dec["stopped"])
        self.assertEqual(dec["candidates_consumed"], 4)
        self.assertEqual(dec["selected_answer"], "A")


if __name__ == "__main__":
    unittest.main()
