"""L2-SP (Starting Point) regularization and displacement metrics."""

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger("task1.l2sp")


class L2SPRegularizer:
    """
    Computes Omega(theta, theta0) = 0.5 * sum_j (theta_j - theta0_j)^2.
    - Sums over all unique parameters including biases and norms.
    - Counts tied parameters only once (via data pointer tracking).
    - Never averages over parameter count.
    - Applied once per optimizer update.
    """

    def __init__(
        self,
        model: nn.Module,
        theta0_dict_or_path: Dict[str, torch.Tensor] | Path | str,
        lambda_val: float = 1.0e-3,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.lambda_val = lambda_val
        self.device = device or next(model.parameters()).device

        if isinstance(theta0_dict_or_path, (str, Path)):
            p = Path(theta0_dict_or_path)
            logger.info(f"Loading theta0 reference weights from {p}...")
            self.theta0: Dict[str, torch.Tensor] = torch.load(str(p), map_location="cpu", weights_only=False)
        else:
            self.theta0 = {k: v.detach().cpu().to(torch.float32) for k, v in theta0_dict_or_path.items()}

        logger.info(f"Initialized L2-SP with lambda={self.lambda_val}, reference params count={len(self.theta0)}")

    def compute_omega(self) -> torch.Tensor:
        """
        Computes 0.5 * sum (theta - theta0)^2 in FP32.
        """
        seen_ptrs: Set[int] = set()
        omega_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for name, param in self.model.named_parameters():
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue  # Skip tied / aliased parameters
            seen_ptrs.add(ptr)

            if name not in self.theta0:
                continue

            ref_param = self.theta0[name].to(device=param.device, dtype=torch.float32)
            diff = param.to(torch.float32) - ref_param
            omega_sum = omega_sum + 0.5 * torch.sum(diff * diff)

        return omega_sum

    def compute_displacement(self) -> float:
        """
        Computes full-model Euclidean displacement ||theta - theta0||_2.
        """
        with torch.no_grad():
            seen_ptrs: Set[int] = set()
            sq_diff_sum = 0.0

            for name, param in self.model.named_parameters():
                ptr = param.data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)

                if name not in self.theta0:
                    continue

                ref_param = self.theta0[name].to(device="cpu", dtype=torch.float32)
                curr_param = param.detach().cpu().to(torch.float32)
                diff = curr_param - ref_param
                sq_diff_sum += float(torch.sum(diff * diff).item())

            return float(sq_diff_sum**0.5)


def compute_full_model_displacement(
    current_model: nn.Module,
    reference_theta0: Dict[str, torch.Tensor],
) -> float:
    """Computes ||theta - theta0||_2 over unique parameters."""
    seen_ptrs: Set[int] = set()
    sq_diff_sum = 0.0

    for name, param in current_model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)

        if name not in reference_theta0:
            continue

        ref_param = reference_theta0[name].to(device="cpu", dtype=torch.float32)
        curr_param = param.detach().cpu().to(torch.float32)
        diff = curr_param - ref_param
        sq_diff_sum += float(torch.sum(diff * diff).item())

    return float(sq_diff_sum**0.5)
