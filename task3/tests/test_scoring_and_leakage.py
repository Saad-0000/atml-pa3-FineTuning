"""
Unit tests for scoring, normalization, and absence of gold-answer leakage.
"""

import unittest
from task3.src.scoring.boxed_extractor import extract_last_boxed_answer
from task3.src.scoring.normalization import normalize_answer, grade_correctness
from task3.src.selection.selectors import select_self_consistency, select_highest_score


class TestScoringAndLeakage(unittest.TestCase):
    def test_boxed_extraction_nested_braces(self):
        """Verify balanced brace extraction on nested formulas."""
        text1 = "Thus, the final answer is \\boxed{\\frac{1}{2}}."
        self.assertEqual(extract_last_boxed_answer(text1), "\\frac{1}{2}")

        text2 = "First \\boxed{old}, but finally \\boxed{\\sqrt{\\frac{a}{b}}}!"
        self.assertEqual(extract_last_boxed_answer(text2), "\\sqrt{\\frac{a}{b}}")

        text_no_box = "The answer is 42."
        self.assertIsNone(extract_last_boxed_answer(text_no_box))

    def test_normalization_and_no_answer(self):
        """Verify NO_ANSWER mapping and incorrectness."""
        self.assertEqual(normalize_answer(None, "MATH"), "NO_ANSWER")
        self.assertEqual(normalize_answer("", "MATH"), "NO_ANSWER")
        self.assertEqual(normalize_answer("   ", "MathVista"), "NO_ANSWER")

        # NO_ANSWER is always incorrect
        self.assertFalse(grade_correctness("NO_ANSWER", "42", "MATH"))
        self.assertFalse(grade_correctness("NO_ANSWER", "NO_ANSWER", "MATH"))

    def test_mathvista_normalization(self):
        """Verify option letter extraction and numeric precision."""
        meta_mc = {"question_type": "multi_choice"}
        self.assertEqual(normalize_answer("(B) 42 units", "MathVista", meta_mc), "B")
        self.assertEqual(normalize_answer("Option C", "MathVista", meta_mc), "C")

        # Numeric float tolerance
        self.assertTrue(grade_correctness("3.14159", "3.14", "MathVista"))

    def test_selector_gold_answer_isolation(self):
        """Verify that selection functions work properly without gold answer fields."""
        candidates_without_gold = [
            {"rollout_id": 0, "normalized_answer": "4", "token_length": 50, "K_i": 2.1},
            {"rollout_id": 1, "normalized_answer": "4", "token_length": 60, "K_i": 2.5},
            {"rollout_id": 2, "normalized_answer": "5", "token_length": 45, "K_i": 3.0},
        ]

        res_sc = select_self_consistency(candidates_without_gold)
        self.assertEqual(res_sc["selected_answer"], "4")
        self.assertNotIn("gold_answer", res_sc)
        self.assertNotIn("is_correct", res_sc)

        res_high = select_highest_score(candidates_without_gold, score_key="K_i")
        self.assertEqual(res_high["selected_answer"], "5")
        self.assertEqual(res_high["selected_rollout_id"], 2)


if __name__ == "__main__":
    unittest.main()
