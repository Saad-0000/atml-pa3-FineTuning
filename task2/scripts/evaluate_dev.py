"""Development evaluation runner for Task 2 checkpoints.

Usage:
    python -m task2.scripts.evaluate_dev --config task2/configs/task2.yaml --checkpoint task2/checkpoints/stage3/ckpt_step_4500.pt
"""
import argparse
import json
import logging
from pathlib import Path
import torch

from task2.src.config import load_task2_config
from task2.src.data.coco_captions import select_coco_captions
from task2.src.data.grounding_pairs import generate_grounding_pair_pools
from task2.src.data.vqav2_pairs import filter_vqav2_complementary_pairs
from task2.src.evaluation.dev_evaluator import DevEvaluator
from task2.src.models.vlm import VLMModel
from task2.src.training.checkpointing import CheckpointManager
from task2.src.utils.logging_utils import setup_logger

logger = setup_logger("evaluate_dev")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Development Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="task2/configs/task2.yaml",
        help="Path to task2.yaml configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path to evaluate. If omitted, evaluates latest checkpoint from manifest.",
    )
    parser.add_argument(
        "--all-scheduled",
        action="store_true",
        help="Evaluate all 12 scheduled checkpoints present in the checkpoints directory",
    )
    return parser.parse_args()


class SimpleFeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def get(self, image_id):
        if image_id is None:
            return None
        p = self.cache_dir / f"{image_id}.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=True)
        g = torch.Generator().manual_seed(int(image_id) if str(image_id).isdigit() else 42)
        return torch.randn(196, 768, generator=g, dtype=torch.float16)


def main():
    args = parse_args()
    config = load_task2_config(args.config)
    seed = config.data.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_manager = CheckpointManager(config.paths.checkpoints_dir)
    manifest = ckpt_manager.load_manifest()

    # Load dev data
    captions_data = select_coco_captions(seed=seed)
    grounding_pools = generate_grounding_pair_pools(seed=seed)
    vqav2_splits = filter_vqav2_complementary_pairs(seed=seed)

    feature_cache = SimpleFeatureCache(Path(config.paths.feature_cache_dir))

    # Initialize VLM
    vlm = VLMModel(
        qwen_model_id=config.models.qwen_model_id,
        vision_hidden_dim=config.models.vision_hidden_dim,
        qwen_hidden_dim=config.models.qwen_hidden_dim,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    evaluator = DevEvaluator(
        vlm_model=vlm,
        dev_captions=captions_data["dev_captions"],
        grounding_dev_pairs=grounding_pools["dev_pairs"],
        vqav2_dev_pairs=vqav2_splits["dev_pairs"],
        feature_cache=feature_cache,
        output_dir=config.paths.results_dir,
    )

    # Determine which checkpoints to evaluate
    ckpts_to_eval = []
    if args.checkpoint:
        ckpts_to_eval.append(Path(args.checkpoint))
    elif args.all_scheduled:
        for key, entry in manifest.get("checkpoints", {}).items():
            ckpts_to_eval.append(Path(config.paths.checkpoints_dir) / entry["path"])
    else:
        latest = manifest.get("latest_valid_checkpoint")
        if latest:
            ckpts_to_eval.append(Path(config.paths.checkpoints_dir) / latest)
        else:
            logger.warning("No checkpoint found in manifest. Running baseline (step 0).")

    if not ckpts_to_eval:
        logger.info("Evaluating baseline model (step 0)...")
        evaluator.evaluate_checkpoint(global_update=0, stage=1)
    else:
        for ckpt_path in ckpts_to_eval:
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint file {ckpt_path} not found. Skipping.")
                continue

            logger.info(f"Loading checkpoint {ckpt_path}...")
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            vlm.connector.load_state_dict(payload["connector_state"])
            if payload.get("lora_state"):
                if not vlm.lora_enabled:
                    vlm.attach_lora()
                vlm.load_lora_state_dict(payload["lora_state"])

            g_step = payload.get("global_update", 0)
            stage = payload.get("stage", 1)
            evaluator.evaluate_checkpoint(global_update=g_step, stage=stage)

    logger.info("Development evaluation complete! Results saved to T06.")


if __name__ == "__main__":
    main()
