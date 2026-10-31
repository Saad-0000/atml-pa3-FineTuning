"""Example-weighted causal cross-entropy loss function.

Assignment Section 5.1:
L = (1 / |B|) * sum_{i in B} [(1 / |R_i|) * sum_{t in R_i} -log p(y_{i,t} | I_i, q_i, y_{i,<t})]

Computes loss averaged within each example over its own target tokens,
then averaged across examples in the batch (not a global token average).
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_example_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    return_per_example: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute example-weighted causal cross-entropy loss.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size].
        labels: Tensor of shape [batch_size, seq_len] with target token IDs and -100 for ignored positions.
        ignore_index: Index representing positions to ignore (-100).
        return_per_example: If True, also returns a 1D tensor of shape [batch_size]
            containing the per-example mean NLL for each item.

    Returns:
        loss: Scalar tensor representing example-weighted batch loss.
        per_example_nll: Optional tensor of shape [batch_size] containing per-example NLLs.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size = shift_labels.size(0)
    per_example_losses = []

    for i in range(batch_size):
        item_logits = shift_logits[i]  # [seq_len - 1, vocab_size]
        item_labels = shift_labels[i]  # [seq_len - 1]

        mask = item_labels != ignore_index
        n_targets = mask.sum()

        if n_targets > 0:
            target_logits = item_logits[mask]
            target_labels = item_labels[mask]
            item_loss = F.cross_entropy(
                target_logits,
                target_labels,
                reduction="mean",  # Divides by n_targets (|R_i|)
            )
            per_example_losses.append(item_loss)
        else:
            per_example_losses.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))

    per_example_tensor = torch.stack(per_example_losses)
    batch_loss = per_example_tensor.mean()

    if return_per_example:
        return batch_loss, per_example_tensor
    return batch_loss, None


class ExampleWeightedLoss(nn.Module):
    """Module wrapper for example-weighted causal cross-entropy loss."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_per_example: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return compute_example_weighted_loss(
            logits,
            labels,
            ignore_index=self.ignore_index,
            return_per_example=return_per_example,
        )
