"""Tests for OPLoRA SVD basis generation, effective factor projections, and constraints."""

import unittest
import torch
import torch.nn as nn
from task1.oplora import OPLoRALinear, compute_svd_bases


class TestOPLoRA(unittest.TestCase):
    def test_svd_basis_orthonormality(self):
        torch.manual_seed(6304)
        w = torch.randn(32, 24, dtype=torch.float32)
        k = 8
        U_k, V_k, meta = compute_svd_bases(w, k=k)

        self.assertEqual(U_k.shape, (32, k))
        self.assertEqual(V_k.shape, (24, k))

        # Check U_k^T U_k == I_k
        eye_k = torch.eye(k)
        u_diff = torch.max(torch.abs(torch.matmul(U_k.T, U_k) - eye_k)).item()
        self.assertLess(u_diff, 1e-5, f"U_k not orthonormal: diff={u_diff}")

        # Check V_k^T V_k == I_k
        v_diff = torch.max(torch.abs(torch.matmul(V_k.T, V_k) - eye_k)).item()
        self.assertLess(v_diff, 1e-5, f"V_k not orthonormal: diff={v_diff}")

    def test_oplora_projection_constraints(self):
        torch.manual_seed(6304)
        base = nn.Linear(32, 16, bias=False)
        U_k, V_k, _ = compute_svd_bases(base.weight, k=4)

        oplora_layer = OPLoRALinear(base, U_k=U_k, V_k=V_k, r=4, alpha=8.0, k=4, dropout=0.0)

        # Assign arbitrary random values to A and B
        with torch.no_grad():
            oplora_layer.lora_A.normal_()
            oplora_layer.lora_B.normal_()

        # Check effective factors
        A_tilde, B_tilde = oplora_layer.get_effective_factors()

        # Verify U_k^T B_tilde == 0
        uk_b = torch.matmul(U_k.T, B_tilde)
        uk_b_norm = torch.norm(uk_b, p="fro").item()
        self.assertLess(uk_b_norm, 1e-5, f"U_k^T B_tilde != 0: {uk_b_norm}")

        # Verify A_tilde V_k == 0
        a_vk = torch.matmul(A_tilde, V_k)
        a_vk_norm = torch.norm(a_vk, p="fro").item()
        self.assertLess(a_vk_norm, 1e-5, f"A_tilde V_k != 0: {a_vk_norm}")

        # Verify forward matches DeltaW product
        x = torch.randn(2, 32)
        out_forward = oplora_layer(x)

        delta_w = oplora_layer.compute_delta_w()
        out_manual = base(x) + torch.matmul(x, delta_w.T)
        diff = torch.max(torch.abs(out_forward - out_manual)).item()
        self.assertLess(diff, 1e-5, f"OPLoRA forward pass discrepancy: {diff}")


if __name__ == "__main__":
    unittest.main()
