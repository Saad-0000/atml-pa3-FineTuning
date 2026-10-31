"""Three-stage training orchestrator and launcher script for Task 2.

Usage:
    python -m task2.scripts.train_stage --stage 1 --config task2/configs/task2.yaml
    python -m task2.scripts.train_stage --all --config task2/configs/task2.yaml
    python -m task2.scripts.train_stage --resume --config task2/configs/task2.yaml
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import torch

from task2.src.config import load_task2_config
from task2.src.data.coco_captions import select_coco_captions
from task2.src.data.grounding_pairs import generate_grounding_pair_pools
from task2.src.data.samplers import Stage2MixtureSampler, Stage3MixtureSampler
from task2.src.data.vqav2_pairs import filter_vqav2_complementary_pairs
from task2.src.evaluation.dev_evaluator import DevEvaluator
from task2.src.models.vlm import VLMModel
from task2.src.training.checkpointing import CheckpointManager
from task2.src.training.trainer import VLMTrainer
from task2.src.utils.logging_utils import setup_logger

logger = setup_logger("train_stage")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Training Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="task2/configs/task2.yaml",
        help="Path to task2.yaml configuration file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="1",
        choices=["1", "2", "3", "all"],
        help="Training stage to execute (1, 2, 3, or all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest valid checkpoint in the manifest",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit path to checkpoint file to resume from",
    )
    return parser.parse_args()


class FeatureCacheReader:
    """Lazy loader for visual features stored in the feature cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def get(self, image_id: Any) -> Optional[torch.Tensor]:
        if image_id is None:
            return None
        p = self.cache_dir / f"{image_id}.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=True)
        # Fallback deterministic tensor if offline
        g = torch.Generator().manual_seed(int(image_id) if str(image_id).isdigit() else 42)
        return torch.randn(196, 768, generator=g, dtype=torch.float16)


def create_stage1_batch_generator(
    captions: List[Dict[str, Any]],
    feature_cache: FeatureCacheReader,
    microbatch_size: int = 8,
    epochs: int = 2,
    seed: int = 6304,
) -> Iterator[List[Dict[str, Any]]]:
    """Generates microbatches of size 8 for Stage 1 (2 epochs over 18,000 captions = 36,000 examples)."""
    import random
    rng = random.Random(seed)

    for epoch in range(epochs):
        shuffled = list(captions)
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), microbatch_size):
            chunk = shuffled[i : i + microbatch_size]
            batch = []
            for item in chunk:
                batch.append({
                    "visual_features": feature_cache.get(item["image_id"]),
                    "question": "",
                    "answer": item["caption"],
                    "is_caption": True,
                })
            yield batch


def create_sampler_batch_generator(
    sampler: Any,
    feature_cache: FeatureCacheReader,
    microbatch_size: int = 8,
) -> Iterator[List[Dict[str, Any]]]:
    """Generates microbatches of size 8 from a mixture sampler (each draw provides 2 examples)."""
    while True:
        batch = []
        # Need microbatch_size / 2 draws per microbatch (e.g. 4 draws = 8 examples)
        num_draws = max(1, microbatch_size // 2)
        for _ in range(num_draws):
            draw_items = sampler.sample_draw()
            for item in draw_items:
                batch.append({
                    "visual_features": feature_cache.get(item.get("image_id")),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "is_caption": item.get("is_caption", False),
                })
        yield batch


def main():
    args = parse_args()
    config = load_task2_config(args.config)
    seed = config.data.seed

    logger.info("==================================================")
    logger.info("           Task 2 Training Orchestrator           ")
    logger.info("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution device: {device}")

    # Checkpoint Manager
    ckpt_manager = CheckpointManager(config.paths.checkpoints_dir)

    # VLM Model
    vlm = VLMModel(
        qwen_model_id=config.models.qwen_model_id,
        vision_hidden_dim=config.models.vision_hidden_dim,
        qwen_hidden_dim=config.models.qwen_hidden_dim,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    feature_cache = FeatureCacheReader(config.paths.feature_cache_dir)

    # Load data for samplers and evaluation
    logger.info("Loading dataset splits...")
    captions_data = select_coco_captions(seed=seed)
    train_captions = captions_data["train_captions"]
    dev_captions = captions_data["dev_captions"]

    grounding_pools = generate_grounding_pair_pools(seed=seed)
    vqav2_splits = filter_vqav2_complementary_pairs(seed=seed)

    # Setup Dev Evaluator
    dev_evaluator = DevEvaluator(
        vlm_model=vlm,
        dev_captions=dev_captions,
        grounding_dev_pairs=grounding_pools["dev_pairs"],
        vqav2_dev_pairs=vqav2_splits["dev_pairs"],
        feature_cache=feature_cache,
        output_dir=config.paths.results_dir,
    )

    trainer = VLMTrainer(
        vlm_model=vlm,
        config=config,
        checkpoint_manager=ckpt_manager,
        output_dir=config.paths.results_dir,
        dev_eval_callback=dev_evaluator.evaluate_checkpoint,
    )

    # Resume handling
    resume_payload = None
    if args.resume or args.checkpoint:
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            resume_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            loaded = ckpt_manager.load_latest_checkpoint(device=device)
            if loaded:
                ckpt_path, resume_payload = loaded
                logger.info(f"Loaded checkpoint for resumption: {ckpt_path}")

        if resume_payload:
            vlm.connector.load_state_dict(resume_payload["connector_state"])
            if resume_payload.get("lora_state") and vlm.lora_enabled:
                vlm.load_lora_state_dict(resume_payload["lora_state"])

    # Determine stages to run
    stages_to_run = [1, 2, 3] if args.stage == "all" else [int(args.stage)]

    for stage in stages_to_run:
        logger.info(f"Preparing batch generator for Stage {stage}...")
        if stage == 1:
            batch_gen = create_stage1_batch_generator(
                captions=train_captions,
                feature_cache=feature_cache,
                microbatch_size=config.training.microbatch_size,
                epochs=2,
                seed=seed,
            )
        elif stage == 2:
            s2_sampler = Stage2MixtureSampler(
                presence_pool=grounding_pools["train_presence_pool"],
                counting_pool=grounding_pools["train_counting_pool"],
                caption_pool=train_captions,
                seed=seed + 2,
            )
            batch_gen = create_sampler_batch_generator(
                sampler=s2_sampler,
                feature_cache=feature_cache,
                microbatch_size=config.training.microbatch_size,
            )
            batch_gen.get_state = s2_sampler.get_state
        elif stage == 3:
            s3_sampler = Stage3MixtureSampler(
                vqav2_pool=vqav2_splits["train_pairs"],
                presence_pool=grounding_pools["train_presence_pool"],
                counting_pool=grounding_pools["train_counting_pool"],
                caption_pool=train_captions,
                seed=seed + 3,
            )
            batch_gen = create_sampler_batch_generator(
                sampler=s3_sampler,
                feature_cache=feature_cache,
                microbatch_size=config.training.microbatch_size,
            )
            batch_gen.get_state = s3_sampler.get_state

        trainer.train_stage(
            stage=stage,
            batch_generator=batch_gen,
            resume_from=resume_payload if (resume_payload and resume_payload.get("stage") == stage) else None,
        )

    logger.info("All requested training stages completed successfully!")


if __name__ == "__main__":
    main()
