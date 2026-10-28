"""Weight-space diagnostics (rho_k, eta) and activation diagnostics (R(z))."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("task1.diagnostics")


def compute_weight_space_metrics(
    W0: torch.Tensor,
    DeltaW: torch.Tensor,
    U_k: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes weight-space diagnostics in FP32:
    - rho_k = ||U_k U_k^T DeltaW||_F^2 / ||DeltaW||_F^2
    - eta = ||DeltaW||_F / ||W0||_F
    """
    W0_fp32 = W0.detach().to(torch.float32).cpu()
    DeltaW_fp32 = DeltaW.detach().to(torch.float32).cpu()
    U_k_fp32 = U_k.detach().to(torch.float32).cpu()

    norm_w0 = float(torch.norm(W0_fp32, p="fro").item())
    norm_delta = float(torch.norm(DeltaW_fp32, p="fro").item())

    # Safeguard zero update
    if norm_delta < 1e-12:
        return {
            "norm_W0": norm_w0,
            "norm_DeltaW": norm_delta,
            "rho_k": 0.0,
            "eta": 0.0,
        }

    # U_k (d_out, k), U_k^T DeltaW is (k, d_in)
    Uk_T_Delta = torch.matmul(U_k_fp32.T, DeltaW_fp32)
    # ||U_k U_k^T DeltaW||_F^2 == ||U_k^T DeltaW||_F^2 because U_k has orthonormal columns!
    proj_norm_sq = float(torch.norm(Uk_T_Delta, p="fro").item() ** 2)
    total_norm_sq = norm_delta**2

    rho_k = proj_norm_sq / max(total_norm_sq, 1e-12)
    eta = norm_delta / max(norm_w0, 1e-12)

    return {
        "norm_W0": norm_w0,
        "norm_DeltaW": norm_delta,
        "rho_k": rho_k,
        "eta": eta,
    }


def compute_activation_diagnostic(
    W0: torch.Tensor,
    DeltaW: torch.Tensor,
    z: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Computes R(z) = ||DeltaW z||_2 / (||W0 z||_2 + 1e-8) in FP32.
    """
    W0_fp32 = W0.detach().to(torch.float32).cpu()
    DeltaW_fp32 = DeltaW.detach().to(torch.float32).cpu()
    z_fp32 = z.detach().to(torch.float32).cpu()

    # z is shape (d_in,)
    w0_z = torch.matmul(W0_fp32, z_fp32)
    delta_z = torch.matmul(DeltaW_fp32, z_fp32)

    w0_z_norm = float(torch.norm(w0_z, p=2).item())
    delta_z_norm = float(torch.norm(delta_z, p=2).item())

    r_z = delta_z_norm / (w0_z_norm + eps)
    return r_z


def run_full_diagnostics(
    condition_id: str,
    trained_model: nn.Module,
    base_W0_dict: Dict[int, torch.Tensor],
    U_k_dict: Dict[int, torch.Tensor],
    activation_cache: Dict[str, torch.Tensor],
    probes_manifest: List[Dict[str, Any]],
    target_layers: List[int] = [0, 13, 27],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs both weight-space and activation diagnostics for a given model condition.
    Returns:
        weight_records: List of dicts with rho_k, eta per layer
        activation_records: List of dicts with R(z) per probe per layer
    """
    weight_records: List[Dict[str, Any]] = []
    activation_records: List[Dict[str, Any]] = []

    for layer_idx in target_layers:
        down_proj = trained_model.model.layers[layer_idx].mlp.down_proj
        W0 = base_W0_dict[layer_idx]
        U_k = U_k_dict[layer_idx]

        # Extract DeltaW based on layer type
        if hasattr(down_proj, "compute_delta_w"):
            # LoRA or OPLoRA
            DeltaW = down_proj.compute_delta_w()
        else:
            # Full FT: W_trained - W0
            DeltaW = down_proj.weight.detach().to(torch.float32).cpu() - W0.to(torch.float32).cpu()

        # Weight-space
        ws_metrics = compute_weight_space_metrics(W0=W0, DeltaW=DeltaW, U_k=U_k)
        weight_records.append({
            "condition": condition_id,
            "layer": layer_idx,
            "module": "down_proj",
            **ws_metrics,
        })

        # Activation diagnostics on cached z
        for probe in probes_manifest:
            probe_id = probe["probe_id"]
            group = probe["group"]
            cache_key = f"{probe_id}_layer{layer_idx}"
            if cache_key in activation_cache:
                z = activation_cache[cache_key]
                r_z = compute_activation_diagnostic(W0=W0, DeltaW=DeltaW, z=z)
                activation_records.append({
                    "condition": condition_id,
                    "layer": layer_idx,
                    "probe_id": probe_id,
                    "group": group,
                    "R_z": r_z,
                })

    return weight_records, activation_records
