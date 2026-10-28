"""Tests for LoRA layer construction, initialization, scaling, and weight merging."""

import unittest
import torch
import torch.nn as nn
from task1.lora import LoRALinear, apply_lora_to_model, merge_and_compare_logits


class TestLoRA(unittest.TestCase):
    def test_lora_initialization_and_forward(self):
        torch.manual_seed(6304)
        base = nn.Linear(8, 4, bias=True)
        # Freeze base
        base.weight.requires_grad = False
        base.bias.requires_grad = False

        lora_layer = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)

        # Base must be frozen
        self.assertFalse(lora_layer.base_linear.weight.requires_grad)

        # Adapters must be FP32 and trainable
        self.assertTrue(lora_layer.lora_A.requires_grad)
        self.assertTrue(lora_layer.lora_B.requires_grad)
        self.assertEqual(lora_layer.lora_A.dtype, torch.float32)
        self.assertEqual(lora_layer.lora_B.dtype, torch.float32)

        # B must be initialized to 0
        self.assertAlmostEqual(torch.norm(lora_layer.lora_B).item(), 0.0)

        # Since B is 0, DeltaW must initially be 0
        delta_w = lora_layer.compute_delta_w()
        self.assertAlmostEqual(torch.norm(delta_w).item(), 0.0)

        # Forward pass initially matches base forward exactly
        x = torch.randn(2, 8)
        out_lora = lora_layer(x)
        out_base = base(x)
        diff = torch.max(torch.abs(out_lora - out_base)).item()
        self.assertLess(diff, 1e-6)

    def test_lora_merge_unmerge_equivalence(self):
        torch.manual_seed(6304)
        base = nn.Linear(16, 8, bias=False)
        lora_layer = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)

        # Assign non-zero weights to B to simulate training
        with torch.no_grad():
            lora_layer.lora_B.normal_()

        x = torch.randn(3, 16)
        out_unmerged = lora_layer(x)

        # Merge
        lora_layer.merge_weights()
        self.assertTrue(lora_layer.merged)
        out_merged = lora_layer(x)

        # Difference between merged and unmerged
        diff_merge = torch.max(torch.abs(out_merged - out_unmerged)).item()
        self.assertLess(diff_merge, 1e-5, f"Merged output discrepancy {diff_merge} exceeds tolerance!")

        # Unmerge
        lora_layer.unmerge_weights()
        self.assertFalse(lora_layer.merged)
        out_restored = lora_layer(x)
        diff_restore = torch.max(torch.abs(out_restored - out_unmerged)).item()
        self.assertLess(diff_restore, 1e-5)


if __name__ == "__main__":
    unittest.main()

