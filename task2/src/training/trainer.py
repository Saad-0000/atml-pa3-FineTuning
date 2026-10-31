"""Training loop orchestrator for Task 2 three-stage training.

Executes:
- Stage 1: Caption alignment (2,250 updates, connector only, peak LR 1e-3)
- Stage 2: Visual instruction tuning (1,500 updates, connector + LoRA, peak LRs 1e-4 / 2e-5)
- Stage 3: Natural VQA specialization (750 updates, connector + LoRA, peak LRs 5e-5 / 1e-5)
Total: 4,500 updates, 72,000 example presentations.
Microbatch: 8, Gradient accumulation: 2, Effective batch: 16.
AdamW, weight decay 0.01, clip global norm 1.0 (unscale before clip), FP16 autocast.
"""
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from task2.src.config import Task2Config
from task2.src.models.vlm import VLMModel
from task2.src.training.checkpointing import CheckpointManager
from task2.src.training.loss import compute_example_weighted_loss
from task2.src.training.scheduler import StageCosineLRScheduler

logger = logging.getLogger(__name__)


class VLMTrainer:
    """Orchestrates 3-stage training, gradient accumulation, logging, and checkpointing."""

    def __init__(
        self,
        vlm_model: VLMModel,
        config: Task2Config,
        checkpoint_manager: CheckpointManager,
        output_dir: str | Path,
        dev_eval_callback: Optional[Callable[[int, int], Dict[str, Any]]] = None,
    ):
        self.vlm = vlm_model
        self.config = config
        self.ckpt_manager = checkpoint_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_jsonl = self.logs_dir / "training_log.jsonl"
        self.train_log_csv = self.logs_dir / "training_log.csv"

        self.dev_eval_callback = dev_eval_callback

        # State tracking
        self.global_update = 0
        self.examples_presented = 0
        self.smoothed_loss = None
        self.smoothing_factor = 0.95

        # Initialize CSV header if needed
        self._init_csv_log()

    def _init_csv_log(self) -> None:
        if not self.train_log_csv.exists():
            with open(self.train_log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_update",
                    "stage",
                    "stage_update",
                    "examples_presented",
                    "train_loss",
                    "smoothed_loss",
                    "connector_lr",
                    "lora_lr",
                    "grad_norm",
                    "scaler_scale",
                    "step_time_sec",
                    "throughput_ex_per_sec",
                    "gpu_mem_allocated_mb",
                    "gpu_mem_peak_mb",
                ])

    def _get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.vlm.parameters() if p.requires_grad]

    def _setup_optimizer_and_scheduler(
        self,
        stage: int,
        total_stage_updates: int,
        connector_peak_lr: float,
        lora_peak_lr: Optional[float] = None,
    ) -> Tuple[torch.optim.Optimizer, StageCosineLRScheduler]:
        """Configure AdamW with per-component parameter groups and stage-local scheduler."""
        trainable_params = self._get_trainable_parameters()
        if not trainable_params:
            raise ValueError(f"No trainable parameters found for stage {stage}!")

        param_groups = []
        peak_lrs = []

        # Connector group
        connector_params = [p for p in self.vlm.connector.parameters() if p.requires_grad]
        if connector_params:
            param_groups.append({
                "params": connector_params,
                "lr": 0.0,
                "weight_decay": self.config.training.weight_decay,
            })
            peak_lrs.append(connector_peak_lr)

        # LoRA group (if Stage 2 or 3)
        if stage in (2, 3) and lora_peak_lr is not None:
            lora_params = [
                p for name, p in self.vlm.qwen.named_parameters()
                if p.requires_grad and ("lora_A" in name or "lora_B" in name)
            ]
            if lora_params:
                param_groups.append({
                    "params": lora_params,
                    "lr": 0.0,
                    "weight_decay": self.config.training.weight_decay,
                })
                peak_lrs.append(lora_peak_lr)

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.training.weight_decay)
        scheduler = StageCosineLRScheduler(
            optimizer=optimizer,
            total_stage_updates=total_stage_updates,
            peak_lrs=peak_lrs,
            min_lr_ratio=0.10,
            warmup_ratio=0.05,
        )
        return optimizer, scheduler

    def train_stage(
        self,
        stage: int,
        batch_generator: Any,  # yields List of dicts (microbatch of size 8)
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run training for a specific stage (1, 2, or 3)."""
        stage_cfg = self.config.stages.get(stage)
        total_stage_updates = stage_cfg.updates
        connector_peak_lr = stage_cfg.connector_lr
        lora_peak_lr = stage_cfg.lora_lr

        logger.info(
            f"=== Starting Stage {stage}: {stage_cfg.name} ==="
            f" ({total_stage_updates} updates, connector LR={connector_peak_lr}, LoRA LR={lora_peak_lr})"
        )

        # Configure model trainability for this stage
        if stage == 1:
            self.vlm.set_stage_trainability(enable_lora=False)
        else:
            if not self.vlm.lora_enabled:
                self.vlm.attach_lora()
            self.vlm.set_stage_trainability(enable_lora=True)

        optimizer, scheduler = self._setup_optimizer_and_scheduler(
            stage=stage,
            total_stage_updates=total_stage_updates,
            connector_peak_lr=connector_peak_lr,
            lora_peak_lr=lora_peak_lr,
        )

        scaler = GradScaler("cuda", enabled=self.vlm.device.type == "cuda")

        stage_update = 0

        # Resumption handling
        if resume_from is not None:
            logger.info("Restoring trainer state from checkpoint...")
            self.global_update = resume_from["global_update"]
            stage_update = resume_from["stage_update"]
            self.examples_presented = resume_from["examples_presented"]
            optimizer.load_state_dict(resume_from["optimizer_state"])
            if resume_from.get("scheduler_state"):
                scheduler.load_state_dict(resume_from["scheduler_state"])
            if resume_from.get("scaler_state") and scaler.is_enabled():
                scaler.load_state_dict(resume_from["scaler_state"])
            logger.info(f"Resumed at global update {self.global_update}, stage {stage} update {stage_update}")

        # Trigger update 0 dev evaluation if at very beginning
        if self.global_update == 0 and self.dev_eval_callback is not None:
            logger.info("Executing scheduled dev evaluation at update 0 (baseline)...")
            self.dev_eval_callback(self.global_update, stage)

        grad_accum_steps = self.config.training.gradient_accumulation_steps
        microbatch_size = self.config.training.microbatch_size
        effective_batch_size = self.config.training.effective_batch_size

        self.vlm.train()
        optimizer.zero_grad()

        accum_step = 0
        running_loss = 0.0
        start_step_time = time.time()

        for raw_microbatch in batch_generator:
            if stage_update >= total_stage_updates:
                break

            # Prepare multimodal inputs
            batch_inputs = self.vlm.prepare_multimodal_inputs(raw_microbatch)
            inputs_embeds = batch_inputs["inputs_embeds"].to(self.vlm.device)
            attention_mask = batch_inputs["attention_mask"].to(self.vlm.device)
            labels = batch_inputs["labels"].to(self.vlm.device)
            position_ids = batch_inputs.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(self.vlm.device)

            # Mixed precision forward pass
            use_amp = self.vlm.device.type == "cuda"
            with autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = self.vlm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                microbatch_loss, _ = compute_example_weighted_loss(
                    logits=logits,
                    labels=labels,
                    ignore_index=-100,
                )
                scaled_loss = microbatch_loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            running_loss += microbatch_loss.item()
            accum_step += 1

            if accum_step % grad_accum_steps == 0:
                # Complete effective batch (16 examples)
                self.examples_presented += effective_batch_size
                trainable_params = self._get_trainable_parameters()

                # Unscale before clipping norm to 1.0
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=self.config.training.max_grad_norm,
                ).item()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Step scheduler
                lrs = scheduler.step()
                connector_lr = lrs[0]
                lora_lr = lrs[1] if len(lrs) > 1 else 0.0

                stage_update += 1
                self.global_update += 1

                step_duration = time.time() - start_step_time
                throughput = effective_batch_size / max(step_duration, 1e-4)

                current_loss = running_loss / grad_accum_steps
                running_loss = 0.0
                start_step_time = time.time()

                if self.smoothed_loss is None:
                    self.smoothed_loss = current_loss
                else:
                    self.smoothed_loss = (
                        self.smoothing_factor * self.smoothed_loss
                        + (1 - self.smoothing_factor) * current_loss
                    )

                # GPU memory statistics
                allocated_mb = 0.0
                peak_mb = 0.0
                if torch.cuda.is_available():
                    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                scaler_scale = scaler.get_scale() if scaler.is_enabled() else 1.0

                # Log to JSONL and CSV
                log_entry = {
                    "global_update": self.global_update,
                    "stage": stage,
                    "stage_update": stage_update,
                    "examples_presented": self.examples_presented,
                    "train_loss": round(current_loss, 5),
                    "smoothed_loss": round(self.smoothed_loss, 5),
                    "connector_lr": connector_lr,
                    "lora_lr": lora_lr,
                    "grad_norm": round(grad_norm, 4),
                    "scaler_scale": scaler_scale,
                    "step_time_sec": round(step_duration, 3),
                    "throughput_ex_per_sec": round(throughput, 1),
                    "gpu_mem_allocated_mb": round(allocated_mb, 1),
                    "gpu_mem_peak_mb": round(peak_mb, 1),
                }

                with open(self.train_log_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                with open(self.train_log_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.global_update,
                        stage,
                        stage_update,
                        self.examples_presented,
                        round(current_loss, 5),
                        round(self.smoothed_loss, 5),
                        f"{connector_lr:.2e}",
                        f"{lora_lr:.2e}",
                        round(grad_norm, 4),
                        scaler_scale,
                        round(step_duration, 3),
                        round(throughput, 1),
                        round(allocated_mb, 1),
                        round(peak_mb, 1),
                    ])

                if self.global_update % 50 == 0 or stage_update == total_stage_updates:
                    logger.info(
                        f"Stage {stage} [{stage_update}/{total_stage_updates}] "
                        f"Global: {self.global_update}/4500 | "
                        f"Loss: {current_loss:.4f} (smooth: {self.smoothed_loss:.4f}) | "
                        f"Norm: {grad_norm:.2f} | Conn LR: {connector_lr:.2e} | LoRA LR: {lora_lr:.2e} | "
                        f"{throughput:.1f} ex/s"
                    )

                # Checkpoint saving every 250 updates or stage boundary
                if self.global_update % 250 == 0 or stage_update == total_stage_updates:
                    sampler_state = getattr(batch_generator, "get_state", lambda: {})()
                    lora_state = self.vlm.get_lora_state_dict() if self.vlm.lora_enabled else None
                    self.ckpt_manager.save_checkpoint(
                        global_update=self.global_update,
                        stage=stage,
                        stage_update=stage_update,
                        examples_presented=self.examples_presented,
                        connector=self.vlm.connector,
                        lora_state=lora_state,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if scaler.is_enabled() else None,
                        sampler_state=sampler_state,
                        config=self.config.raw_config,
                        extra_meta={"loss": current_loss, "smoothed_loss": self.smoothed_loss},
                    )

                # Scheduled dev evaluation
                dev_schedule = self.config.evaluation.dev_eval_schedule
                if self.global_update in dev_schedule and self.dev_eval_callback is not None:
                    logger.info(f"Triggering scheduled dev evaluation at global update {self.global_update}...")
                    self.vlm.eval()
                    self.dev_eval_callback(self.global_update, stage)
                    self.vlm.train()

        # End of stage: export standalone connector and adapter weights
        lora_state = self.vlm.get_lora_state_dict() if self.vlm.lora_enabled else None
        self.ckpt_manager.export_stage_weights(
            stage=stage,
            global_update=self.global_update,
            connector=self.vlm.connector,
            lora_state=lora_state,
            output_dir=self.output_dir / "artifacts" / "model_exports",
        )
        logger.info(f"=== Stage {stage} completed at global update {self.global_update} ===")
