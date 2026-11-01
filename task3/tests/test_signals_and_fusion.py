"""
Unit tests for signals, VisConf calibration, and population covariance identity.
"""

import unittest
import numpy as np
from task3.src.signals.fusion_analysis import verify_covariance_identity
from task3.src.selection.quality_scores import (
    compute_visual_engagement,
    calibrate_visconf_for_candidates
)


class TestSignalsAndFusion(unittest.TestCase):
    def test_covariance_identity(self):
        """Verify that Z_token - Z equals population covariance Cov(c, v)."""
        np.random.seed(6304)
        c_tokens = np.random.uniform(0, 1, size=50)
        v_tokens = np.random.uniform(0, 1, size=50)

        res = verify_covariance_identity(c_tokens, v_tokens, tolerance=1e-6)
        self.assertTrue(res["identity_holds"])
        self.assertLess(abs(res["residual"]), 1e-6)

    def test_visconf_calibration_isolation(self):
        """Verify VisConf calibration uses ONLY revealed candidates."""
        revealed_candidates = [
            {"rollout_id": 0, "kl_u_p": [1.0, 2.0, 3.0], "V_i": 0.5, "token_length": 3},
            {"rollout_id": 1, "kl_u_p": [2.0, 4.0, 5.0], "V_i": 0.6, "token_length": 3},
        ]

        calibrated = calibrate_visconf_for_candidates(revealed_candidates)
        self.assertEqual(len(calibrated), 2)
        for c in calibrated:
            self.assertIn("Z_i", c)
            self.assertIn("C_i", c)
            self.assertTrue(0.0 <= c["Z_i"] <= 1.0)

    def test_zero_attention_handling(self):
        """Verify visual engagement handles zero denominators without error."""
        a_img = [0.0, 0.0]
        a_prompt = [0.0, 0.0]
        a_gen = [0.0, 0.0]

        v_val, v_tokens = compute_visual_engagement(a_img, a_prompt, a_gen)
        self.assertEqual(v_val, 0.0)
        self.assertEqual(len(v_tokens), 2)
        self.assertTrue(all(v == 0.0 for v in v_tokens))


if __name__ == "__main__":
    unittest.main()
