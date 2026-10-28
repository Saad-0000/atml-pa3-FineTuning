"""Shared training engine supporting all five fine-tuning methods with exact token-weighted loss."""

import math
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def _create_grad_scaler(device_type: str = "cuda", enabled: bool = True):
    """Instantiates modern torch.amp.GradScaler with fallback to legacy torch.cuda.amp."""
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (TypeError, AttributeError):
        from torch.cuda.amp import GradScaler as LegacyGradScaler
        return LegacyGradScaler(enabled=enabled)


def _get_autocast_context(device_type: str = "cuda", dtype: torch.dtype = torch.float16, enabled: bool = True):
    """Instantiates modern torch.amp.autocast with fallback to legacy torch.cuda.amp."""
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    except (TypeError, AttributeError):
        from torch.cuda.amp import autocast as legacy_autocast
        return legacy_autocast(enabled=enabled, dtype=dtype)

from .checkpointing import CheckpointManager, set_rng_states
from .formatting import collate_effective_batch, compute_token_weighted_loss
from .l2sp import L2SPRegularizer
from .logging_utils import MetricLogger, get_gpu_memory_stats, reset_gpu_memory_stats
from .model_loader import count_parameters, get_optimizer_param_groups

logger = logging.getLogger("task1.training")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine learning rate decay with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train_single_method(
    method_name: str,
    model: nn.Module,
    train_examples: List[Dict[str, Any]],
    pad_token_id: int,
    peak_lr: float = 2.0e-5,
    lambda_l2sp: float = 0.0,
    l2sp_regularizer: Optional[L2SPRegularizer] = None,
    total_updates: int = 1250,
    warmup_updates: int = 38,
    microbatch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_manager: Optional[CheckpointManager] = None,
    metric_logger: Optional[MetricLogger] = None,
    resume_mode: str = "auto",
    resume_path: Optional[Path | str] = None,
    emergency_save_interval: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes one epoch of training (40,000 examples, 1,250 optimizer updates).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    # 1. Optimizer & LR Scheduler
    param_groups = get_optimizer_param_groups(model, weight_decay=weight_decay)
    optimizer = AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.999), eps=1.0e-8)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    scaler = _create_grad_scaler(device_type=device.type, enabled=(device.type == "cuda"))

    start_update = 0
    examples_seen = 0
    data_position = 0
    skipped_updates_count = 0

    # 2. Resumption Logic
    ckpt_to_load = None
    if resume_mode == "auto" and checkpoint_manager is not None:
        ckpt_to_load = checkpoint_manager.find_latest_checkpoint()
    elif resume_mode == "path" and resume_path is not None:
        ckpt_to_load = Path(resume_path)

    if ckpt_to_load is not None and checkpoint_manager is not None:
        logger.info(f"Resuming {method_name} from checkpoint: {ckpt_to_load}")
        state = checkpoint_manager.load_checkpoint(ckpt_to_load)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("lr_scheduler_state_dict") and lr_scheduler is not None:
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        if state.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler_state_dict"])
        if state.get("rng_states"):
            set_rng_states(state["rng_states"])

        start_update = state["update_step"]
        examples_seen = state["examples_seen"]
        data_position = state["data_position"]
        logger.info(
            f"Resumed at update {start_update}/{total_updates}, examples seen: {examples_seen}, data pos: {data_position}"
        )

    effective_batch_size = microbatch_size * gradient_accumulation_steps
    num_effective_batches = len(train_examples) // effective_batch_size
    effective_updates_target = min(total_updates, num_effective_batches)

    reset_gpu_memory_stats()
    train_start_time = time.time()
    last_ckpt_time = train_start_time

    pbar = tqdm(
        total=effective_updates_target,
        initial=start_update,
        desc=f"Training {method_name}",
        dynamic_ncols=True,
    )

    model.train()

    for update_idx in range(start_update, effective_updates_target):
        current_update = update_idx + 1
        eff_start_idx = update_idx * effective_batch_size
        eff_end_idx = eff_start_idx + effective_batch_size
        eff_batch_examples = train_examples[eff_start_idx:eff_end_idx]

        # Count total valid response tokens across the entire effective batch of 32 examples
        total_valid_response_tokens = sum(ex["num_supervised_tokens"] for ex in eff_batch_examples)
        denom_tokens = max(1, total_valid_response_tokens)

        optimizer.zero_grad(set_to_none=True)
        eff_sft_loss_sum = 0.0
        eff_tokens_processed = 0

        # Microbatch accumulation
        for mb_idx in range(gradient_accumulation_steps):
            mb_start = mb_idx * microbatch_size
            mb_end = mb_start + microbatch_size
            mb_examples = eff_batch_examples[mb_start:mb_end]

            collated = collate_effective_batch(mb_examples, pad_token_id=pad_token_id)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)

            with _get_autocast_context(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_sum, num_tokens = compute_token_weighted_loss(outputs.logits, labels)
                # Normalize by effective batch total tokens
                loss_microbatch = loss_sum / denom_tokens

            scaler.scale(loss_microbatch).backward()

            eff_sft_loss_sum += float(loss_sum.detach().item())
            eff_tokens_processed += num_tokens

            # Explicitly deallocate intermediate microbatch tensors to keep VRAM footprint low
            del outputs, loss_sum, loss_microbatch, input_ids, attention_mask, labels, collated

        # SFT loss normalized across entire effective batch
        sft_loss_val = eff_sft_loss_sum / denom_tokens

        # L2-SP Regularization (added once per optimizer update)
        omega_val = 0.0
        lambda_omega_val = 0.0
        total_loss_val = sft_loss_val

        if lambda_l2sp > 0.0 and l2sp_regularizer is not None:
            omega_tensor = l2sp_regularizer.compute_omega()
            omega_val = float(omega_tensor.detach().item())
            lambda_omega_tensor = lambda_l2sp * omega_tensor
            lambda_omega_val = float(lambda_omega_tensor.detach().item())
            total_loss_val += lambda_omega_val
            # Backward for L2-SP penalty
            scaler.scale(lambda_omega_tensor).backward()
            del omega_tensor, lambda_omega_tensor

        # Unscale gradients before norm clipping
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())

        # Optimizer step
        old_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale() if scaler.is_enabled() else 1.0

        # Deallocate parameter gradients immediately after optimizer step to reclaim VRAM
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and (current_update % 20 == 0):
            torch.cuda.empty_cache()

        step_skipped = new_scale < old_scale
        if step_skipped:
            skipped_updates_count += 1

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        examples_seen += len(eff_batch_examples)
        data_position = eff_end_idx

        # GPU Memory
        mem_stats = get_gpu_memory_stats()
        elapsed_sec = time.time() - train_start_time
        updates_done = current_update - start_update
        sec_per_update = elapsed_sec / max(1, updates_done)
        remaining_sec_est = sec_per_update * (effective_updates_target - current_update)

        step_metrics = {
            "update": current_update,
            "examples_seen": examples_seen,
            "data_position": data_position,
            "sft_loss": sft_loss_val,
            "omega": omega_val,
            "lambda_omega": lambda_omega_val,
            "total_loss": total_loss_val,
            "learning_rate": current_lr,
            "grad_norm": grad_norm,
            "scale": new_scale,
            "skipped_updates_total": skipped_updates_count,
            "tokens_in_effective_batch": eff_tokens_processed,
            "elapsed_seconds": elapsed_sec,
            "remaining_seconds_est": remaining_sec_est,
            "tokens_per_sec": eff_tokens_processed / max(1e-4, sec_per_update),
            **mem_stats,
        }

        if metric_logger is not None:
            metric_logger.log(step_metrics)

        # Progress bar update
        pbar.set_postfix({
            "sft_loss": f"{sft_loss_val:.4f}",
            "lr": f"{current_lr:.2e}",
            "gnorm": f"{grad_norm:.2f}",
            "skip": skipped_updates_count,
            "gpu_alloc_mb": f"{mem_stats['gpu_allocated_mib']:.0f}",
            "rem_min_est": f"{remaining_sec_est / 60.0:.1f}",
        })
        pbar.update(1)

        # Checkpoint Saving: Official Milestones + Periodic Emergency
        is_official = current_update in (checkpoint_manager.official_milestones if checkpoint_manager else [])
        is_emergency = (current_update % emergency_save_interval == 0)

        if (is_official or is_emergency) and checkpoint_manager is not None:
            method_cfg = {
                "method_name": method_name,
                "peak_lr": peak_lr,
                "lambda_l2sp": lambda_l2sp,
                "weight_decay": weight_decay,
            }
            checkpoint_manager.save_checkpoint(
                update_step=current_update,
                examples_seen=examples_seen,
                data_position=data_position,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                method_config=method_cfg,
            )
            last_ckpt_time = time.time()

    pbar.close()

    total_training_time = time.time() - train_start_time
    logger.info(f"Training for {method_name} completed in {total_training_time:.2f} seconds.")

    return {
        "method_name": method_name,
        "completed_updates": effective_updates_target,
        "total_examples_seen": examples_seen,
        "total_training_time_sec": total_training_time,
        "skipped_updates": skipped_updates_count,
        "final_gpu_memory": get_gpu_memory_stats(),
    }

