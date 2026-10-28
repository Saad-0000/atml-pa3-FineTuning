"""Tests for L2-SP penalty calculation, tied parameter handling, and weight decay grouping."""

import unittest
import torch
import torch.nn as nn
from task1.l2sp import L2SPRegularizer
from task1.model_loader import get_optimizer_param_groups


class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 2, bias=True)
        # Tie fc2.weight to fc1.weight
        self.fc2.weight = self.fc1.weight
        self.norm = nn.LayerNorm(2)


class TestL2SP(unittest.TestCase):
    def test_l2sp_exact_calculation(self):
        model = ToyModelWithTiedWeights()

        # Set known weights
        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            model.fc1.bias.copy_(torch.tensor([0.5, 0.5]))
            model.fc2.bias.copy_(torch.tensor([1.0, 1.0]))
            model.norm.weight.copy_(torch.tensor([1.0, 1.0]))
            model.norm.bias.copy_(torch.tensor([0.0, 0.0]))

        # Reference theta0: all zeros
        theta0 = {k: torch.zeros_like(v) for k, v in model.named_parameters()}

        reg = L2SPRegularizer(model, theta0_dict_or_path=theta0, lambda_val=1e-3)
        omega = reg.compute_omega()

        # Analytical Omega:
        # fc1.weight: 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        # fc1.bias: 0.5^2 + 0.5^2 = 0.5
        # fc2.weight: TIED to fc1.weight -> must be counted ONLY ONCE!
        # fc2.bias: 1^2 + 1^2 = 2
        # norm.weight: 1^2 + 1^2 = 2
        # norm.bias: 0
        # Sum = 30 + 0.5 + 2 + 2 = 34.5
        # 0.5 * Sum = 17.25
        self.assertAlmostEqual(omega.item(), 17.25, places=4)

        # Displacement = sqrt(34.5) = 5.87367
        disp = reg.compute_displacement()
        self.assertAlmostEqual(disp, 34.5**0.5, places=4)

    def test_weight_decay_parameter_grouping(self):
        model = ToyModelWithTiedWeights()
        groups = get_optimizer_param_groups(model, weight_decay=0.01)

        decay_group = groups[0]
        no_decay_group = groups[1]

        self.assertEqual(decay_group["weight_decay"], 0.01)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        # 2D weights without norm/bias must be in decay group
        for p in decay_group["params"]:
            self.assertGreaterEqual(p.ndim, 2)

        # 1D weights, biases, and norms must be in no-decay group
        self.assertGreater(len(no_decay_group["params"]), 0)


if __name__ == "__main__":
    unittest.main()

