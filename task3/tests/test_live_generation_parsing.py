"""
Unit tests for live generation prompts, response reuse, and boxed answer parsing.
"""

import unittest
import tempfile
from pathlib import Path
from task3.src.live_generation.generator import (
    format_initial_messages,
    format_refinement_messages,
    LiveGenerationPipeline
)
from task3.src.scoring.boxed_extractor import extract_last_boxed_answer


class TestLiveGenerationParsing(unittest.TestCase):
    def test_format_initial_messages(self):
        """Verify system message and question prompt format."""
        msgs = format_initial_messages("Find x if 2x = 6.")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are a helpful assistant.")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("Think step by step inside <think> </think> tags", msgs[1]["content"])
        self.assertIn("give your final answer inside \\boxed{}.", msgs[1]["content"])

    def test_format_refinement_messages(self):
        """Verify refinement prompt appends shared response and critique prompt without gold feedback."""
        problem = "Find x if 2x = 6."
        shared_resp = "<think>2x = 6, so x = 3.</think> \\boxed{3}"
        msgs = format_refinement_messages(problem, shared_resp)

        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["content"], shared_resp)
        self.assertEqual(msgs[3]["role"], "user")
        self.assertIn("Review the solution above. Identify and correct any errors", msgs[3]["content"])

    def test_shared_response_reuse_in_pipeline(self):
        """Verify that parallel candidate 0 is reused as the initial response for refinement."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = LiveGenerationPipeline(output_dir=Path(tmp_dir), dry_run=True)
            res = pipeline.run_question({
                "question_id": "test_q01",
                "problem": "Solve 1 + 1.",
                "gold_answer": "2"
            })

            par_c0 = res["strategies"]["parallel"]["candidates"][0]["raw_text"]
            ref_init = res["strategies"]["sequential_refinement"]["initial_response"]
            self.assertEqual(par_c0, ref_init)


if __name__ == "__main__":
    unittest.main()
