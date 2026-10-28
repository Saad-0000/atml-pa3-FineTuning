"""Tests for evaluation protocol guardrails and split assertions."""

import unittest
from task1.evaluation import EXPECTED_BENCHMARK_SPECS, assert_evaluation_split_guards


class TestEvaluationProtocol(unittest.TestCase):
    def test_split_guard_enforces_correct_splits(self):
        # Valid passes
        assert_evaluation_split_guards("gsm8k_cot", "test", 1319)
        assert_evaluation_split_guards("hellaswag", "validation", 10042)
        assert_evaluation_split_guards("boolq", "validation", 3270)
        assert_evaluation_split_guards("arc_easy", "test", 2376)

    def test_split_guard_accepts_multi_filter_sample_multiples(self):
        # lm-eval returns 2 * 1319 = 2638 sample records for gsm8k_cot due to multiple filters
        assert_evaluation_split_guards("gsm8k_cot", "test", 2638)

    def test_split_guard_rejects_train_split(self):
        # Must raise ValueError if 'train' is passed
        with self.assertRaises(ValueError):
            assert_evaluation_split_guards("gsm8k_cot", "train", 1319)
        with self.assertRaises(ValueError):
            assert_evaluation_split_guards("hellaswag", "train", 10042)

    def test_split_guard_rejects_truncated_example_count(self):
        # Must raise ValueError if count is partial (e.g. 100 instead of 1319)
        with self.assertRaises(ValueError):
            assert_evaluation_split_guards("gsm8k_cot", "test", 100)

    def test_chat_template_separation_flags(self):
        # gsm8k_cot must have apply_chat_template=True
        self.assertTrue(EXPECTED_BENCHMARK_SPECS["gsm8k"]["apply_chat_template"])

        # Likelihood tasks must have apply_chat_template=False
        self.assertFalse(EXPECTED_BENCHMARK_SPECS["hellaswag"]["apply_chat_template"])
        self.assertFalse(EXPECTED_BENCHMARK_SPECS["boolq"]["apply_chat_template"])
        self.assertFalse(EXPECTED_BENCHMARK_SPECS["arc_easy"]["apply_chat_template"])


if __name__ == "__main__":
    unittest.main()

