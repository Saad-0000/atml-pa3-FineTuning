"""Per-stage learning rate scheduler with warmup and cosine decay to 10% peak LR.
"""
import math
from typing import List
import torch


class StageCosineLRScheduler:
    """Stage-local scheduler: 5% linear warmup, then cosine decay to 10% peak LR.

    Assignment Section 5:
    'Per-stage 5% warmup, then cosine decay to 10% of peak LR'
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_stage_updates: int,
        peak_lrs: List[float],
        min_lr_ratio: float = 0.1,
        warmup_ratio: float = 0.05,
        last_step: int = -1,
    ):
        self.optimizer = optimizer
        self.total_stage_updates = total_stage_updates
        self.peak_lrs = peak_lrs
        self.min_lr_ratio = min_lr_ratio
        self.warmup_steps = max(1, round(warmup_ratio * total_stage_updates))
        self.current_step = last_step

    def get_lr_for_step(self, step: int, peak_lr: float) -> float:
        """Compute learning rate for a specific stage-local update step (0-indexed)."""
        if step < self.warmup_steps:
            # Linear warmup from 0 to peak_lr
            return peak_lr * float(step + 1) / float(self.warmup_steps)
        else:
            # Cosine decay from peak_lr down to min_lr (0.1 * peak_lr)
            decay_steps = self.total_stage_updates - self.warmup_steps
            if decay_steps <= 0:
                return peak_lr * self.min_lr_ratio
            progress = float(step - self.warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return peak_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_mult)

    def step(self, step: int | None = None) -> List[float]:
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            peak_lr = self.peak_lrs[i] if i < len(self.peak_lrs) else self.peak_lrs[-1]
            lr = self.get_lr_for_step(self.current_step, peak_lr)
            param_group["lr"] = lr
            lrs.append(lr)
        return lrs

    def state_dict(self) -> dict:
        return {
            "total_stage_updates": self.total_stage_updates,
            "peak_lrs": self.peak_lrs,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_stage_updates = state_dict["total_stage_updates"]
        self.peak_lrs = state_dict["peak_lrs"]
        self.min_lr_ratio = state_dict["min_lr_ratio"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.current_step = state_dict["current_step"]
