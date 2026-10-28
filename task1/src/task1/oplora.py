"""OPLoRA (Orthogonal Projection LoRA) implementation, SVD basis caching, and constraint verification."""

import math
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear

logger = logging.getLogger("task1.oplora")


def compute_svd_bases(
    weight: torch.Tensor,
    k: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes leading k singular bases U_k and V_k for starting weight W0 in FP32.
    W0 = U Sigma V^T, where U has shape (d_out, d), V has shape (d_in, d).
    Returns:
        U_k: shape (d_out, k), orthonormal columns
        V_k: shape (d_in, k), orthonormal columns
        metadata: SVD timing, singular value stats, orthogonality residuals
    """
    t0 = time.time()
    w_fp32 = weight.detach().to(torch.float32).cpu()
    d_out, d_in = w_fp32.shape

    actual_k = min(k, d_out, d_in)
    if actual_k < k:
        logger.warning(f"Requested k={k} exceeds weight dims ({d_out}, {d_in}). Using k={actual_k}.")

    # Deterministic full or thin SVD in FP32
    U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)
    # Vh is (d, d_in), so V is Vh.T of shape (d_in, d)
    V = Vh.mH

    U_k = U[:, :actual_k].contiguous()  # (d_out, k)
    V_k = V[:, :actual_k].contiguous()  # (d_in, k)

    # Numerical orthogonality checks: ||U_k^T U_k - I||_F, ||V_k^T V_k - I||_F
    eye_k = torch.eye(actual_k, dtype=torch.float32)
    u_res = float(torch.norm(torch.matmul(U_k.T, U_k) - eye_k, p="fro").item())
    v_res = float(torch.norm(torch.matmul(V_k.T, V_k) - eye_k, p="fro").item())

    elapsed = time.time() - t0
    metadata = {
        "k": actual_k,
        "d_out": d_out,
        "d_in": d_in,
        "svd_time_sec": elapsed,
        "u_ortho_residual": u_res,
        "v_ortho_residual": v_res,
        "top1_singular_val": float(S[0].item()),
        "top_k_singular_val": float(S[actual_k - 1].item()),
    }
    return U_k, V_k, metadata


class OPLoRALinear(nn.Module):
    """
    Orthogonal Projection LoRA layer.
    Given frozen starting weight W0 and cached leading bases U_k, V_k:
    B_tilde = B - U_k (U_k^T B)
    A_tilde = A - (A V_k) V_k^T
    DeltaW = (alpha / r) * B_tilde A_tilde

    Implements projection using effective factor operations:
    - Never builds dense d_out x d_out or d_in x d_in projection matrices.
    - Trains FP32 adapter parameters A and B.
    - Keeps U_k, V_k frozen in FP32.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        U_k: torch.Tensor,
        V_k: torch.Tensor,
        r: int = 16,
        alpha: float = 32.0,
        k: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.k = k
        self.scaling = alpha / r

        # Frozen base weight
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Frozen bases U_k (d_out, k), V_k (d_in, k) in FP32
        self.register_buffer("U_k", U_k.to(torch.float32))
        self.register_buffer("V_k", V_k.to(torch.float32))

        # Trainable adapters in FP32
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()
        self.merged = False

    def reset_parameters(self) -> None:
        # Match LoRA initialization: Kaiming uniform A, zero B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_effective_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes projected factors:
        A_tilde = A - (A V_k) V_k^T  (r, d_in)
        B_tilde = B - U_k (U_k^T B)  (d_out, r)
        """
        # A V_k is (r, k) -> (A V_k) V_k^T is (r, d_in)
        A_Vk = torch.matmul(self.lora_A, self.V_k)
        A_tilde = self.lora_A - torch.matmul(A_Vk, self.V_k.T)

        # U_k^T B is (k, r) -> U_k (U_k^T B) is (d_out, r)
        Uk_B = torch.matmul(self.U_k.T, self.lora_B)
        B_tilde = self.lora_B - torch.matmul(self.U_k, Uk_B)

        return A_tilde, B_tilde

    def compute_delta_w(self) -> torch.Tensor:
        """Computes (alpha / r) * B_tilde A_tilde in FP32."""
        A_tilde, B_tilde = self.get_effective_factors()
        return self.scaling * torch.matmul(B_tilde, A_tilde)

    def check_projection_constraints(self) -> Dict[str, float]:
        """
        Verifies numerical constraints:
        ||U_k^T B_tilde||_F, ||A_tilde V_k||_F, and protected interference rho_k.
        """
        with torch.no_grad():
            A_tilde, B_tilde = self.get_effective_factors()
            uk_b_res = float(torch.norm(torch.matmul(self.U_k.T, B_tilde), p="fro").item())
            a_vk_res = float(torch.norm(torch.matmul(A_tilde, self.V_k), p="fro").item())
            b_norm = float(torch.norm(B_tilde, p="fro").item()) + 1e-12
            a_norm = float(torch.norm(A_tilde, p="fro").item()) + 1e-12

            return {
                "b_proj_residual": uk_b_res,
                "a_proj_residual": a_vk_res,
                "b_proj_relative": uk_b_res / b_norm,
                "a_proj_relative": a_vk_res / a_norm,
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_linear(x)

        result = self.base_linear(x)

        x_fp32 = x.to(torch.float32)
        dropped_x = self.lora_dropout(x_fp32)

        # Effective factor forward:
        # A_tilde x = A x - (A V_k) (V_k^T x)
        A_x = F.linear(dropped_x, self.lora_A)  # (..., r)
        # V_k is (d_in, k), so V_k^T x is (..., k)
        Vk_x = F.linear(dropped_x, self.V_k.T)  # (..., k)
        # A V_k is (r, k)
        A_Vk = torch.matmul(self.lora_A, self.V_k)  # (r, k)
        # (A V_k) (V_k^T x) is (..., r)
        proj_x = F.linear(Vk_x, A_Vk)  # (..., r)
        intermediate = A_x - proj_x

        # B_tilde intermediate = B intermediate - U_k (U_k^T (B intermediate))
        B_inter = F.linear(intermediate, self.lora_B)  # (..., d_out)
        # U_k is (d_out, k), so U_k^T B_inter is (..., k)
        Uk_inter = F.linear(B_inter, self.U_k.T)  # (..., k)
        proj_inter = F.linear(Uk_inter, self.U_k)  # (..., d_out)
        adapter_out = B_inter - proj_inter

        result = result + (self.scaling * adapter_out).to(result.dtype)
        return result

    def merge_weights(self) -> None:
        """Merges projected DeltaW into base_linear.weight."""
        if self.merged:
            return
        delta_w = self.compute_delta_w().to(self.base_linear.weight.dtype)
        self.base_linear.weight.data.add_(delta_w)
        self.merged = True

    def unmerge_weights(self) -> None:
        """Subtracts projected DeltaW from base_linear.weight."""
        if not self.merged:
            return
        delta_w = self.compute_delta_w().to(self.base_linear.weight.dtype)
        self.base_linear.weight.data.sub_(delta_w)
        self.merged = False


def apply_oplora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    k: int = 128,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    basis_cache_dir: Optional[Path | str] = None,
) -> Tuple[nn.Module, Dict[str, OPLoRALinear], Dict[str, Any]]:
    """
    Freezes backbone and replaces target nn.Linear with OPLoRALinear.
    Computes or loads cached FP32 SVD bases for each target module.
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    if basis_cache_dir is not None:
        cache_p = Path(basis_cache_dir)
        cache_p.mkdir(parents=True, exist_ok=True)
    else:
        cache_p = None

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    oplora_layers: Dict[str, OPLoRALinear] = {}
    basis_metadata: Dict[str, Any] = {
        "protection_rank_k": k,
        "adapter_rank_r": r,
        "alpha": alpha,
        "modules": {},
    }

    t0_total = time.time()
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, (LoRALinear, OPLoRALinear)):
                clean_name = name.replace(".", "_")
                u_cache = cache_p / f"{clean_name}_U_k{k}.pt" if cache_p else None
                v_cache = cache_p / f"{clean_name}_V_k{k}.pt" if cache_p else None

                if u_cache and v_cache and u_cache.exists() and v_cache.exists():
                    logger.info(f"Loading cached bases for {name}...")
                    U_k = torch.load(str(u_cache), map_location="cpu", weights_only=False)
                    V_k = torch.load(str(v_cache), map_location="cpu", weights_only=False)
                    meta = {"cached": True}
                else:
                    logger.info(f"Computing SVD bases for {name} (shape {module.weight.shape}, k={k})...")
                    U_k, V_k, meta = compute_svd_bases(module.weight, k=k)
                    if u_cache and v_cache:
                        torch.save(U_k, str(u_cache))
                        torch.save(V_k, str(v_cache))

                basis_metadata["modules"][name] = meta

                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[1] if "." in name else name

                parent = model.get_submodule(parent_name) if parent_name else model
                oplora_layer = OPLoRALinear(
                    base_linear=module,
                    U_k=U_k,
                    V_k=V_k,
                    r=r,
                    alpha=alpha,
                    k=k,
                    dropout=dropout,
                )
                setattr(parent, child_name, oplora_layer)
                oplora_layers[name] = oplora_layer
                break

    basis_metadata["total_preprocessing_time_sec"] = time.time() - t0_total
    logger.info(
        f"Applied OPLoRA (r={r}, alpha={alpha}, k={k}) to {len(oplora_layers)} modules in "
        f"{basis_metadata['total_preprocessing_time_sec']:.2f}s."
    )
    return model, oplora_layers, basis_metadata
