"""Model loading, tokenizer initialization, and parameter grouping utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("task1.model_loader")


def load_tokenizer(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    revision: str = "main",
    enable_thinking: bool = False,
) -> PreTrainedTokenizerBase:
    """Loads tokenizer with pad_token configured and non-thinking chat template."""
    logger.info(f"Loading tokenizer from {model_name_or_path} (revision={revision})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|extra_0|>"
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    revision: str = "main",
    torch_dtype: torch.dtype = torch.float16,
    gradient_checkpointing: bool = True,
    device: Optional[torch.device] = None,
) -> PreTrainedModel:
    """Loads the base causal language model."""
    logger.info(f"Loading base model from {model_name_or_path} (dtype={torch_dtype}, rev={revision})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if device is not None:
        model.to(device)

    return model


def get_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Separates trainable parameters into decay and no-decay groups.
    Rule: Exclude biases and parameters with fewer than two dimensions from weight decay.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name.lower() or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return groups


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    pct_trainable = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "percentage_trainable": pct_trainable,
    }


def save_theta0_reference(
    model: nn.Module,
    save_path: Path | str,
) -> None:
    """Saves immutable FP32 reference state dict for L2-SP and displacement calculations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving immutable theta0 reference to {save_path}...")
    theta0 = {name: param.detach().cpu().to(torch.float32) for name, param in model.named_parameters()}
    torch.save(theta0, str(save_path))
    logger.info(f"Saved {len(theta0)} parameters to {save_path}.")

