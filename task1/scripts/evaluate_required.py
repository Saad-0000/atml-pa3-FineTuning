#!/usr/bin/env python3
"""
Required evaluation runner for Task 1:
Evaluates the 7 required model conditions on all 4 benchmarks:
1. Baseline (unchanged starting model)
2. Full FT Midpoint (step 625)
3. Full FT Final (step 1250)
4. Full FT + L2-SP Final (step 1250)
5. LoRA-4 Final (step 1250)
6. LoRA-16 Final (step 1250)
7. OPLoRA-16 Final (step 1250)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK1_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK1_ROOT / "src"))

import torch
from task1.config import get_default_configs
from task1.evaluation import evaluate_with_lm_eval
from task1.lora import apply_lora_to_model, merge_and_compare_logits
from task1.model_loader import load_base_model, load_tokenizer
from task1.oplora import apply_oplora_to_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_required")


def load_model_for_condition(
    condition_id: str,
    t1_cfg: Any,
    methods_cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    """Loads and reconstructs model state for evaluation."""
    tokenizer = load_tokenizer(t1_cfg.model_name, revision=t1_cfg.model_revision, enable_thinking=t1_cfg.enable_thinking)

    if condition_id == "baseline":
        model = load_base_model(t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=torch.float16, device=device)
        return model, tokenizer

    # Map condition to method and step
    if condition_id == "full_ft_midpoint":
        method, step = "full_ft", 625
    elif condition_id == "full_ft_final":
        method, step = "full_ft", 1250
    elif condition_id == "l2sp_final":
        method, step = "l2sp", 1250
    elif condition_id == "lora4_final":
        method, step = "lora4", 1250
    elif condition_id == "lora16_final":
        method, step = "lora16", 1250
    elif condition_id == "oplora16_final":
        method, step = "oplora16", 1250
    else:
        raise ValueError(f"Unknown condition: {condition_id}")

    ckpt_path = t1_cfg.checkpoints_dir / method / f"checkpoint_step_{step:04d}" / "training_state.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint for {condition_id} missing at {ckpt_path}!")

    logger.info(f"Loading checkpoint for {condition_id} from {ckpt_path}...")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    is_adapter = method in ["lora4", "lora16", "oplora16"]
    backbone_dtype = torch.float16

    model = load_base_model(t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=backbone_dtype, device=device)

    if method in ["lora4", "lora16"]:
        m_cfg = methods_cfg[method]
        model, _ = apply_lora_to_model(
            model=model,
            r=m_cfg["r"],
            alpha=m_cfg["alpha"],
            dropout=0.0,
            target_modules=m_cfg["target_modules"],
        )
    elif method == "oplora16":
        m_cfg = methods_cfg[method]
        basis_cache_dir = t1_cfg.artifacts_dir / "oplora_bases"
        model, _, _ = apply_oplora_to_model(
            model=model,
            r=m_cfg["r"],
            alpha=m_cfg["alpha"],
            k=m_cfg["protection_rank_k"],
            dropout=0.0,
            target_modules=m_cfg["target_modules"],
            basis_cache_dir=basis_cache_dir,
        )

    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # For adapter methods: merge weights for evaluation
    if is_adapter:
        logger.info(f"Merging adapter weights into backbone for {condition_id}...")
        for m in model.modules():
            if hasattr(m, "merge_weights"):
                m.merge_weights()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate required checkpoints across 4 benchmarks")
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=[
            "all",
            "baseline",
            "full_ft_midpoint",
            "full_ft_final",
            "l2sp_final",
            "lora4_final",
            "lora16_final",
            "oplora16_final",
        ],
        help="Evaluation condition to run",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditions_to_run = (
        ["baseline", "full_ft_midpoint", "full_ft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
        if args.condition == "all"
        else [args.condition]
    )

    benchmarks = ["gsm8k", "hellaswag", "boolq", "arc_easy"]

    for cond_id in conditions_to_run:
        logger.info(f"=== Evaluating Condition: {cond_id} ===")
        out_dir = t1_cfg.results_dir / "raw" / cond_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            model, tokenizer = load_model_for_condition(cond_id, t1_cfg, methods_cfg, device)
            for task_key in benchmarks:
                logger.info(f"Running {task_key} for {cond_id}...")
                evaluate_with_lm_eval(
                    model=model,
                    tokenizer=tokenizer,
                    task_key=task_key,
                    device=str(device),
                    batch_size=args.batch_size,
                    output_dir=out_dir,
                )
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Evaluation failed or blocked for {cond_id}: {e}")


if __name__ == "__main__":
    main()
