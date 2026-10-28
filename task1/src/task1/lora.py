"""LoRA (Low-Rank Adaptation) custom module and model injection."""

import math
import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("task1.lora")


class LoRALinear(nn.Module):
    """
    Linear layer augmented with LoRA adapter:
    output = W0 x + (alpha / r) * B(A(dropout(x)))
    - W0 is frozen (typically FP16)
    - A, B are trainable parameters in FP32
    - A initialized with Gaussian / Kaiming uniform
    - B initialized to zeros
    - No trainable biases
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        if r > 0:
            # Adapters in FP32
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.merged = False

    def reset_parameters(self) -> None:
        if self.r > 0:
            # Kaiming uniform initialization for A, matching standard LoRA
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def compute_delta_w(self) -> torch.Tensor:
        """Computes (alpha / r) * B A in FP32."""
        if self.r <= 0:
            return torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
        return self.scaling * torch.matmul(self.lora_B, self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_linear(x)

        # Base forward
        result = self.base_linear(x)

        if self.r > 0:
            # Cast input to float32 for adapter pass to maintain FP32 stability
            x_fp32 = x.to(torch.float32)
            adapter_out = self.lora_dropout(x_fp32)
            adapter_out = F.linear(adapter_out, self.lora_A)
            adapter_out = F.linear(adapter_out, self.lora_B)
            # Scale and cast back to result's dtype
            result = result + (self.scaling * adapter_out).to(result.dtype)

        return result

    def merge_weights(self) -> None:
        """Merges DeltaW into base_linear.weight."""
        if self.merged:
            return
        if self.r > 0:
            delta_w = self.compute_delta_w().to(self.base_linear.weight.dtype)
            self.base_linear.weight.data.add_(delta_w)
        self.merged = True

    def unmerge_weights(self) -> None:
        """Subtracts DeltaW from base_linear.weight."""
        if not self.merged:
            return
        if self.r > 0:
            delta_w = self.compute_delta_w().to(self.base_linear.weight.dtype)
            self.base_linear.weight.data.sub_(delta_w)
        self.merged = False


def apply_lora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, LoRALinear]]:
    """
    Freezes backbone and replaces matching nn.Linear with LoRALinear.
    Default targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
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

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    lora_layers: Dict[str, LoRALinear] = {}

    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                # Locate parent
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[1] if "." in name else name

                parent = model.get_submodule(parent_name) if parent_name else model
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                lora_layers[name] = lora_layer
                break

    logger.info(f"Applied LoRA (r={r}, alpha={alpha}) to {len(lora_layers)} modules.")
    return model, lora_layers


def merge_and_compare_logits(
    model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    tolerance: float = 1.0e-3,
) -> Tuple[float, float, bool]:
    """
    Verifies that merged model logits match unmerged model logits on a sample batch.
    Returns: (max_abs_diff, max_rel_diff, passed_check)
    """
    model.eval()
    with torch.no_grad():
        # 1. Unmerged logits
        out_unmerged = model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch.get("attention_mask"),
        ).logits

        # 2. Merge all LoRALinear layers
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.merge_weights()

        # 3. Merged logits
        out_merged = model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch.get("attention_mask"),
        ).logits

        # 4. Unmerge back
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.unmerge_weights()

        diff = torch.abs(out_merged.to(torch.float32) - out_unmerged.to(torch.float32))
        max_abs = float(diff.max().item())
        denom = torch.abs(out_unmerged.to(torch.float32)).clamp(min=1e-6)
        max_rel = float((diff / denom).max().item())

        passed = max_abs <= tolerance
        logger.info(f"LoRA merge check: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, passed={passed}")
        return max_abs, max_rel, passed
