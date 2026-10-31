"""LoRA adapter implementation for Qwen3-0.6B language model in Stages 2 and 3."""

import math
import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("task2.models.lora")


class LoRALinear(nn.Module):
    """
    LoRA linear replacement for transformer attention projection weights.
    W = W0 + (alpha / r) * B * A
    Backbone W0 is frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        lora_alpha: Optional[float] = None,
        lora_dropout: Optional[float] = None,
    ):
        super().__init__()
        effective_alpha = float(lora_alpha if lora_alpha is not None else alpha)
        effective_dropout = float(lora_dropout if lora_dropout is not None else dropout)
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = effective_alpha
        self.scaling = effective_alpha / float(r)

        # Freeze base layer parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        # Trainable adapter weights in FP32
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))

        self.merged = False
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform init for A, zero init for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        # Compute adapter delta: (x @ A^T) @ B^T * scaling
        x_fp32 = x.to(torch.float32)
        dropped = self.dropout(x_fp32)
        adapter_out = F.linear(dropped, self.lora_A)  # [..., r]
        adapter_out = F.linear(adapter_out, self.lora_B) * self.scaling  # [..., out_features]

        return base_out + adapter_out.to(orig_dtype)

    def merge_weights(self):
        """Merges LoRA weight delta into base layer weight for inference."""
        if not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta.to(self.base_layer.weight.dtype)
            self.merged = True

    def unmerge_weights(self):
        """Restores unmerged base layer weight."""
        if self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta.to(self.base_layer.weight.dtype)
            self.merged = False


def apply_lora_to_model(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Applies LoRA to specified attention projections across all transformer layers.
    Targets: q_proj, k_proj, v_proj, o_proj.
    """
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    replaced_names: List[str] = []

    for name, module in model.named_modules():
        for t in target_modules:
            if hasattr(module, t):
                child = getattr(module, t)
                if isinstance(child, nn.Linear):
                    lora_mod = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, t, lora_mod)
                    replaced_names.append(f"{name}.{t}")

    logger.info(f"Applied LoRA (r={r}, alpha={alpha}) to {len(replaced_names)} modules.")
    return model, replaced_names


def mark_only_lora_and_connector_as_trainable(
    model: nn.Module,
    connector: nn.Module,
    enable_lora: bool = True,
) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in connector.parameters():
        p.requires_grad = True
    if enable_lora:
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
