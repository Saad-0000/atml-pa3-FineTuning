"""Two-layer MLP connector mapping SigLIP patch features into Qwen text embedding space."""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger("task2.models.connector")


class MultimodalConnector(nn.Module):
    """
    Two-layer MLP connector:
    Linear(768, 1024) -> GELU -> Linear(1024, 1024)
    Maps SigLIP patch features [B, 196, 768] into Qwen embedding space [B, 196, 1024].
    Maintains a calibration scale factor s initialized to 1.0.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        activation: str = "gelu",
        init_scale: float = 1.0,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        in_d = input_dim if input_dim is not None else in_dim
        out_d = output_dim if output_dim is not None else out_dim
        self.in_dim = in_d
        self.hidden_dim = hidden_dim
        self.out_dim = out_d

        self.linear1 = nn.Linear(in_d, hidden_dim)
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

        self.linear2 = nn.Linear(hidden_dim, out_dim)

        # Scale factor calibrated against text embedding magnitude
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform initialization for stable initial representations
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, 196, 768] float16 or float32 tensor
        Returns:
            visual_tokens: [B, 196, 1024] in same dtype as visual_features
        """
        orig_dtype = visual_features.dtype
        # Compute in float32 for maximum precision then project to target dtype
        x = visual_features.to(torch.float32)
        h = self.act(self.linear1(x))
        out = self.linear2(h)
        out = out * self.scale
        return out.to(orig_dtype)

    def calibrate_scale(self, target_norm: float, sample_features: torch.Tensor) -> float:
        """
        Calibrates scale s such that E[||Z||_2] matches target text embedding norm.
        s = target_norm / mean_unscaled_norm
        """
        self.eval()
        with torch.no_grad():
            x = sample_features.to(self.linear1.weight.device, dtype=torch.float32)
            # Temporarily set scale to 1.0 to measure unscaled norm
            h = self.act(self.linear1(x))
            unscaled = self.linear2(h)
            token_norms = torch.norm(unscaled, p=2, dim=-1)  # [B, 196]
            mean_norm = float(token_norms.mean().item())

            if mean_norm > 1e-6:
                new_scale = float(target_norm / mean_norm)
                self.scale.copy_(torch.tensor(new_scale, dtype=torch.float32))
                logger.info(
                    f"Connector scale calibrated: unscaled norm={mean_norm:.4f}, "
                    f"target norm={target_norm:.4f}, scale s={new_scale:.4f}"
                )
                return new_scale
        return float(self.scale.item())

    def set_scale(self, scale_value: float) -> None:
        self.scale.fill_(float(scale_value))


MLPConnector = MultimodalConnector


def calibrate_connector_scale(connector: MultimodalConnector, target_norm: float, sample_features: torch.Tensor) -> float:
    return connector.calibrate_scale(target_norm, sample_features)
