"""Unit tests for example-weighted causal cross-entropy loss.

Assignment Section 5.1 & Section 13:
L = (1 / |B|) * sum_{i in B} [(1 / |R_i|) * sum_{t in R_i} -log p(y_{i,t} | I_i, q_i, y_{i,<t})]

Proves that:
1. The loss matches a hand-computed example-weighted value.
2. The loss distinguishes example-weighted objective from token-weighted batch loss
   when response lengths are unequal.
"""
import unittest
import torch
import torch.nn.functional as F

from task2.src.training.loss import compute_example_weighted_loss


class TestExampleWeightedLoss(unittest.TestCase):
    def test_hand_computed_example_weighted_loss(self):
        """Hand-computed test with known logits and labels."""
        batch_size = 2
        seq_len = 4
        vocab_size = 3

        logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        # Example 0: target at index 2 (predicted by logits index 1)
        labels[0, 2] = 0
        logits[0, 1, 0] = 2.0
        ce_0 = -torch.log(torch.exp(torch.tensor(2.0)) / (torch.exp(torch.tensor(2.0)) + 2.0)).item()

        # Example 1: two targets at index 2 and index 3
        labels[1, 2] = 1
        labels[1, 3] = 2
        logits[1, 1, 1] = 1.0
        logits[1, 2, 2] = 3.0
        ce_1_token1 = -torch.log(torch.exp(torch.tensor(1.0)) / (torch.exp(torch.tensor(1.0)) + 2.0)).item()
        ce_1_token2 = -torch.log(torch.exp(torch.tensor(3.0)) / (torch.exp(torch.tensor(3.0)) + 2.0)).item()
        ce_1_mean = (ce_1_token1 + ce_1_token2) / 2.0

        expected_example_weighted_loss = (ce_0 + ce_1_mean) / 2.0

        computed_loss, per_ex = compute_example_weighted_loss(
            logits, labels, return_per_example=True
        )

        self.assertAlmostEqual(computed_loss.item(), expected_example_weighted_loss, places=4)
        self.assertAlmostEqual(per_ex[0].item(), ce_0, places=4)
        self.assertAlmostEqual(per_ex[1].item(), ce_1_mean, places=4)

    def test_example_weighted_vs_token_weighted_divergence(self):
        """Proves that example-weighted loss produces a different value from token-weighted loss

        when response lengths in the batch are unequal.
        """
        batch_size = 2
        seq_len = 12
        vocab_size = 4

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        # Ex 0: 1 target token
        labels[0, 1] = 1

        # Ex 1: 8 target tokens
        for t in range(1, 9):
            labels[1, t] = 2

        computed_example_loss, _ = compute_example_weighted_loss(logits, labels)

        # Global token-weighted cross entropy
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        token_weighted_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # The two losses MUST be distinct
        self.assertGreater(abs(computed_example_loss.item() - token_weighted_loss.item()), 0.01)


if __name__ == "__main__":
    unittest.main()
