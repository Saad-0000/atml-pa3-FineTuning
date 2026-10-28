#!/usr/bin/env python3
"""
Method training runner for Task 1:
Executes one of the five independent training conditions from the untouched starting checkpoint.
Options: full_ft, l2sp, lora4, lora16, oplora16
"""

import os
# Configure PyTorch memory allocator to avoid fragmentation on Tesla T4 GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK1_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK1_ROOT / "src"))

import torch
from task1.checkpointing import CheckpointManager
from task1.config import get_default_configs
from task1.data import prepare_metamathqa_subset
from task1.l2sp import L2SPRegularizer
from task1.logging_utils import MetricLogger
from task1.lora import apply_lora_to_model
from task1.model_loader import load_base_model, load_tokenizer
from task1.oplora import apply_oplora_to_model
from task1.training import train_single_method

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("train_method")


def main():
    parser = argparse.ArgumentParser(description="Train one of five fine-tuning methods on MetaMathQA")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["full_ft", "l2sp", "lora4", "lora16", "oplora16"],
        help="Training method to execute",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        choices=["auto", "path", "none"],
        help="Checkpoint resumption mode",
    )
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint folder to resume from")
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    m_cfg = methods_cfg[args.method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"=== Starting Training for Method: {args.method} ({m_cfg['name']}) ===")

    # 1. Load Tokenizer
    tokenizer = load_tokenizer(
        model_name_or_path=t1_cfg.model_name,
        revision=t1_cfg.model_revision,
        enable_thinking=t1_cfg.enable_thinking,
    )

    # 2. Load 40,000 Filtered Examples
    data_cache_dir = t1_cfg.artifacts_dir / "data_cache"
    examples, _ = prepare_metamathqa_subset(
        tokenizer=tokenizer,
        source_dataset_name=t1_cfg.source_dataset,
        target_count=t1_cfg.budget_examples,
        max_seq_length=t1_cfg.max_seq_length,
        enable_thinking=t1_cfg.enable_thinking,
        cache_dir=data_cache_dir,
    )

    # 3. Load Fresh Starting Model (Untouched Qwen3-0.6B)
    is_adapter = args.method in ["lora4", "lora16", "oplora16"]
    backbone_dtype = torch.float16 if is_adapter else torch.float32

    logger.info(f"Loading fresh starting model in {backbone_dtype}...")
    model = load_base_model(
        model_name_or_path=t1_cfg.model_name,
        revision=t1_cfg.model_revision,
        torch_dtype=backbone_dtype,
        gradient_checkpointing=t1_cfg.gradient_checkpointing,
        device=device,
    )

    l2sp_reg = None
    if args.method == "l2sp":
        ref_path = t1_cfg.artifacts_dir / "theta0_reference" / "theta0.pt"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference theta0 weights missing at {ref_path}! Run evaluate_baseline first.")
        l2sp_reg = L2SPRegularizer(model, theta0_dict_or_path=ref_path, lambda_val=m_cfg["lambda_l2sp"], device=device)

    elif args.method in ["lora4", "lora16"]:
        model, _ = apply_lora_to_model(
            model=model,
            r=m_cfg["r"],
            alpha=m_cfg["alpha"],
            dropout=m_cfg["dropout"],
            target_modules=m_cfg["target_modules"],
        )

    elif args.method == "oplora16":
        basis_cache_dir = t1_cfg.artifacts_dir / "oplora_bases"
        model, _, _ = apply_oplora_to_model(
            model=model,
            r=m_cfg["r"],
            alpha=m_cfg["alpha"],
            k=m_cfg["protection_rank_k"],
            dropout=m_cfg["dropout"],
            target_modules=m_cfg["target_modules"],
            basis_cache_dir=basis_cache_dir,
        )

    # 4. Initialize Checkpoint and Metric Managers
    ckpt_mgr = CheckpointManager(
        save_dir=t1_cfg.checkpoints_dir,
        method_name=args.method,
        official_milestones=t1_cfg.milestones_updates,
        keep_emergency_copies=t1_cfg.keep_latest,
    )
    metric_logger = MetricLogger(
        output_dir=t1_cfg.results_dir / "raw",
        method_name=args.method,
    )

    # 5. Train
    train_results = train_single_method(
        method_name=args.method,
        model=model,
        train_examples=examples,
        pad_token_id=tokenizer.pad_token_id,
        peak_lr=m_cfg["peak_lr"],
        lambda_l2sp=m_cfg.get("lambda_l2sp", 0.0),
        l2sp_regularizer=l2sp_reg,
        total_updates=t1_cfg.total_optimizer_updates,
        warmup_updates=t1_cfg.warmup_updates,
        microbatch_size=t1_cfg.microbatch_size,
        gradient_accumulation_steps=t1_cfg.gradient_accumulation_steps,
        weight_decay=t1_cfg.weight_decay,
        max_grad_norm=t1_cfg.max_grad_norm,
        checkpoint_manager=ckpt_mgr,
        metric_logger=metric_logger,
        resume_mode=args.resume,
        resume_path=args.resume_path,
        emergency_save_interval=t1_cfg.emergency_save_interval_updates,
        device=device,
    )

    metric_logger.close()
    logger.info(f"Training for {args.method} finished successfully: {train_results}")


if __name__ == "__main__":
    main()

