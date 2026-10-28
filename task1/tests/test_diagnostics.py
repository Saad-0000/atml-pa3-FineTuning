"""Tests for weight-space diagnostics (rho_k, eta) and activation diagnostics (R(z))."""

import unittest
import torch
from task1.diagnostics import compute_activation_diagnostic, compute_weight_space_metrics


class TestDiagnostics(unittest.TestCase):
    def test_weight_space_metrics_analytical(self):
        # Orthonormal basis U_k: first 2 columns of 4x4 identity
        eye4 = torch.eye(4, dtype=torch.float32)
        U_k = eye4[:, :2]  # columns 0 and 1

        W0 = 2.0 * eye4  # ||W0||_F = sqrt(4 * 4) = 4.0

        # Case 1: DeltaW completely inside protected subspace (columns 0 and 1)
        DeltaW_in = torch.zeros(4, 4, dtype=torch.float32)
        DeltaW_in[0, 0] = 1.0
        DeltaW_in[1, 1] = 1.0
        # ||DeltaW||_F = sqrt(2)

        res_in = compute_weight_space_metrics(W0=W0, DeltaW=DeltaW_in, U_k=U_k)
        self.assertAlmostEqual(res_in["rho_k"], 1.0, places=5)
        self.assertAlmostEqual(res_in["eta"], (2.0**0.5) / 4.0, places=5)

        # Case 2: DeltaW completely outside protected subspace (columns 2 and 3)
        DeltaW_out = torch.zeros(4, 4, dtype=torch.float32)
        DeltaW_out[2, 2] = 1.0
        DeltaW_out[3, 3] = 1.0

        res_out = compute_weight_space_metrics(W0=W0, DeltaW=DeltaW_out, U_k=U_k)
        self.assertAlmostEqual(res_out["rho_k"], 0.0, places=5)

    def test_zero_update_safeguard(self):
        W0 = torch.eye(4, dtype=torch.float32)
        DeltaW_zero = torch.zeros(4, 4, dtype=torch.float32)
        U_k = torch.eye(4, dtype=torch.float32)[:, :2]

        res = compute_weight_space_metrics(W0=W0, DeltaW=DeltaW_zero, U_k=U_k)
        self.assertEqual(res["rho_k"], 0.0)
        self.assertEqual(res["eta"], 0.0)

    def test_activation_diagnostic_analytical(self):
        W0 = torch.eye(4, dtype=torch.float32)
        DeltaW = 0.1 * torch.eye(4, dtype=torch.float32)
        z = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        r_z = compute_activation_diagnostic(W0=W0, DeltaW=DeltaW, z=z)
        # ||W0 z||_2 = 2.0, ||DeltaW z||_2 = 0.2
        # R(z) = 0.2 / 2.0 = 0.1
        self.assertAlmostEqual(r_z, 0.1, places=5)


if __name__ == "__main__":
    unittest.main()

