"""Tests for chat formatting, loss masking, and collation."""

import unittest
import torch
from task1.formatting import collate_effective_batch, tokenize_training_example


class DummyTokenizer:
    """Mock tokenizer simulating Qwen chat template behavior."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151645  # <|im_end|>
        self.bos_token_id = 151644  # <|im_start|>

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # Emulate tokenizing: prompt has 5 tokens, assistant has 4 tokens + 1 eos
        tokens = [self.bos_token_id, 10, 20]
        for m in messages:
            if m["role"] == "user":
                tokens += [30, 40]
            elif m["role"] == "assistant":
                tokens += [50, 60, 70, self.eos_token_id]
        if add_generation_prompt:
            tokens += [80]  # Generation prompt prefix
        return tokens


class TestFormattingAndMasks(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_prompt_masking_and_eot_supervision(self):
        res = tokenize_training_example(
            tokenizer=self.tokenizer,
            question="What is 2+2?",
            response="4",
            max_seq_length=2048,
        )
        self.assertIsNotNone(res)
        labels = res["labels"]
        prompt_len = res["prompt_len"]

        # Prompt must be masked (-100)
        for i in range(prompt_len):
            self.assertEqual(labels[i], -100)

        # Assistant tokens must not be -100
        supervised = labels[prompt_len:]
        self.assertTrue(all(lbl != -100 for lbl in supervised))
        # End of turn token is supervised
        self.assertEqual(supervised[-1], self.tokenizer.eos_token_id)

    def test_overlength_exclusion_without_truncation(self):
        # Set max length smaller than sequence
        res = tokenize_training_example(
            tokenizer=self.tokenizer,
            question="What is 2+2?",
            response="4",
            max_seq_length=5,  # Too small
        )
        self.assertIsNone(res, "Overlength example must be excluded, not truncated!")

    def test_batch_collation_padding(self):
        ex1 = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
        ex2 = {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]}

        batch = collate_effective_batch([ex1, ex2], pad_token_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["attention_mask"].shape, (2, 3))
        self.assertEqual(batch["labels"].shape, (2, 3))

        # Check padding on ex2
        self.assertEqual(batch["input_ids"][1, 2].item(), 0)
        self.assertEqual(batch["attention_mask"][1, 2].item(), 0)
        self.assertEqual(batch["labels"][1, 2].item(), -100)

    def test_batch_encoding_dict_output_handling(self):
        class DictDummyTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                if add_generation_prompt:
                    return {"input_ids": [151644, 1, 2, 3]}
                return {"input_ids": [151644, 1, 2, 3, 4, 5, 151645]}

        dict_tok = DictDummyTokenizer()
        res, reason, info = tokenize_training_example(
            tokenizer=dict_tok,
            question="Hi",
            response="Hello",
            max_seq_length=2048,
            return_diagnostics=True,
        )
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(res)
        self.assertEqual(res["seq_len"], 7)
        self.assertEqual(res["prompt_len"], 4)
        self.assertEqual(res["num_supervised_tokens"], 3)
        self.assertEqual(res["labels"][:4], [-100, -100, -100, -100])
        self.assertEqual(res["labels"][4:], [4, 5, 151645])

    def test_diagnostic_rejection_reasons(self):
        # Test overlength diagnostic
        res, reason, info = tokenize_training_example(
            tokenizer=self.tokenizer,
            question="What is 2+2?",
            response="4",
            max_seq_length=5,
            return_diagnostics=True,
        )
        self.assertIsNone(res)
        self.assertEqual(reason, "overlength")
        self.assertIn("seq_len", info)
        self.assertGreater(info["seq_len"], 5)

        # Test empty response (prompt_ge_seq)
        class EmptyRespTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                return [1, 2, 3, 4]

        res_empty, reason_empty, _ = tokenize_training_example(
            tokenizer=EmptyRespTokenizer(),
            question="Hi",
            response="",
            return_diagnostics=True,
        )
        self.assertIsNone(res_empty)
        self.assertEqual(reason_empty, "prompt_ge_seq")


if __name__ == "__main__":
    unittest.main()

