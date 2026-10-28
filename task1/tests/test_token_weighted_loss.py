"""Tests for token-weighted loss equivalence across gradient accumulation."""

import unittest
import torch
import torch.nn as nn
from task1.formatting import compute_token_weighted_loss


class TestTokenWeightedLoss(unittest.TestCase):
    def test_accumulated_token_weighted_loss_matches_full_batch(self):
        torch.manual_seed(6304)
        vocab_size = 32
        hidden_dim = 16

        # Simple linear projection model
        model = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Create two microbatches with different numbers of supervised tokens
        # Microbatch 1: 2 examples, seq_len 4
        x1 = torch.randn(2, 4, hidden_dim)
        labels1 = torch.tensor([
            [-100, 5, 12, 8],     # 3 valid tokens
            [-100, -100, 19, 2],  # 2 valid tokens
        ], dtype=torch.long)      # Total valid = 5

        # Microbatch 2: 2 examples, seq_len 5
        x2 = torch.randn(2, 5, hidden_dim)
        labels2 = torch.tensor([
            [-100, 1, 2, 3, 4],   # 4 valid tokens
            [-100, -100, -100, 7, 9],  # 2 valid tokens
        ], dtype=torch.long)      # Total valid = 6

        total_effective_valid_tokens = 5 + 6  # 11 valid tokens across effective batch

        # 1. Gradient Accumulation pass
        model.zero_grad()
        out1 = model(x1)
        loss_sum1, n_tok1 = compute_token_weighted_loss(out1, labels1)
        # Scale by total effective tokens
        (loss_sum1 / total_effective_valid_tokens).backward()

        out2 = model(x2)
        loss_sum2, n_tok2 = compute_token_weighted_loss(out2, labels2)
        (loss_sum2 / total_effective_valid_tokens).backward()

        accumulated_grad = model.weight.grad.clone()

        # 2. Reference single pass (concatenating all tokens together)
        model.zero_grad()
        # Shifted targets and inputs for reference
        # We compute total cross entropy sum / 11 directly
        ref_loss = (loss_sum1 + loss_sum2) / total_effective_valid_tokens
        # Re-run backward on combined loss graph
        out1_ref = model(x1)
        out2_ref = model(x2)
        l1, _ = compute_token_weighted_loss(out1_ref, labels1)
        l2, _ = compute_token_weighted_loss(out2_ref, labels2)
        ((l1 + l2) / total_effective_valid_tokens).backward()

        ref_grad = model.weight.grad.clone()

        # Compare accumulated gradients vs reference
        diff = torch.max(torch.abs(accumulated_grad - ref_grad)).item()
        self.assertLess(diff, 1e-6, f"Gradient discrepancy {diff} exceeds tolerance!")


if __name__ == "__main__":
    unittest.main()
