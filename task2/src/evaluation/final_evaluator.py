"""Final evaluation protocol for Task 2.

Evaluates the completed Stage 3 model (at 4,500 updates) on both test suites:
- Annotation-derived Grounding: 64 pairs / 128 questions.
- Natural VQAv2: 96 pairs / 192 questions.

Under 4 image conditions:
1. Correct: Original image.
2. Complementary swap: Partner image with different reference answer.
3. Mismatched: Image from another pair in the same suite (excluding both original pair images).
4. No image: Visual tokens omitted completely.

Greedy decoding, enable_thinking=False, max 16 new tokens, batch size 8.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from task2.src.data.normalization import normalize_answer
from task2.src.evaluation.dev_evaluator import compute_example_answer_nll
from task2.src.models.vlm import VLMModel

logger = logging.getLogger(__name__)


def generate_fixed_mismatched_mapping(
    pairs: List[Dict[str, Any]],
    seed: int = 6304,
) -> Dict[str, int]:
    """Generates a deterministic mismatched image mapping.

    For each pair, selects an image from another pair in the same suite,
    strictly excluding BOTH images of the original pair.
    """
    rng = np.random.RandomState(seed)
    mapping = {}
    pair_ids = [p["pair_id"] for p in pairs]

    # Collect all image IDs available
    all_images = []
    for p in pairs:
        all_images.append(p["image_id_1"])
        all_images.append(p["image_id_2"])
    all_images = list(dict.fromkeys(all_images))

    for p in pairs:
        pid = p["pair_id"]
        img1 = p["image_id_1"]
        img2 = p["image_id_2"]
        # Candidate images: all images not belonging to this pair
        candidates = [img for img in all_images if img != img1 and img != img2]
        if not candidates:
            raise ValueError(f"No candidate mismatched images available for pair {pid}")
        chosen_img = int(rng.choice(candidates))
        mapping[pid] = chosen_img

    return mapping


class FinalEvaluator:
    """Runs final generation and scoring across all four image conditions."""

    def __init__(
        self,
        vlm_model: VLMModel,
        grounding_test_pairs: List[Dict[str, Any]],
        vqav2_test_pairs: List[Dict[str, Any]],
        feature_cache: Dict[str | int, torch.Tensor],
        output_dir: str | Path,
        seed: int = 6304,
    ):
        self.vlm = vlm_model
        self.grounding_pairs = grounding_test_pairs
        self.vqav2_pairs = vqav2_test_pairs
        self.feature_cache = feature_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Generate and save fixed mismatched mappings
        self.grounding_mismatched_map = generate_fixed_mismatched_mapping(
            self.grounding_pairs, seed=self.seed
        )
        self.vqav2_mismatched_map = generate_fixed_mismatched_mapping(
            self.vqav2_pairs, seed=self.seed + 1
        )
        self._save_mismatched_mappings()

    def _save_mismatched_mappings(self) -> None:
        data_dir = self.output_dir / "results" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_file = data_dir / "mismatched_image_mapping.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "grounding": self.grounding_mismatched_map,
                "vqav2": self.vqav2_mismatched_map,
                "seed": self.seed,
            }, f, indent=2)

    @torch.no_grad()
    def evaluate_suite(
        self,
        pairs: List[Dict[str, Any]],
        suite_name: str,
        mismatched_map: Dict[str, int],
        batch_size: int = 8,
        max_new_tokens: int = 16,
    ) -> pd.DataFrame:
        """Evaluates a test suite under all 4 conditions with greedy generation and reference NLL."""
        self.vlm.eval()
        records: List[Dict[str, Any]] = []

        conditions = ["correct", "complementary_swap", "mismatched", "no_image"]

        logger.info(f"Evaluating {suite_name} ({len(pairs)} pairs / {len(pairs)*2} questions) across 4 conditions...")

        for cond in conditions:
            logger.info(f"  Condition: {cond}")
            for pair in pairs:
                pid = pair["pair_id"]
                q = pair["question"]
                img1_id = pair["image_id_1"]
                img2_id = pair["image_id_2"]
                ans1 = pair["answer_1"]
                ans2 = pair["answer_2"]
                q_type = pair.get("answer_type", pair.get("pair_type", "unknown"))

                # For each question in the pair
                for item_idx, (orig_img_id, partner_img_id, target_ans, human_ans) in enumerate([
                    (img1_id, img2_id, ans1, pair.get("human_answers_1", [ans1])),
                    (img2_id, img1_id, ans2, pair.get("human_answers_2", [ans2])),
                ]):
                    ex_id = f"{pid}_ex{item_idx+1}"

                    # Determine input image based on condition
                    if cond == "correct":
                        supplied_img_id = orig_img_id
                    elif cond == "complementary_swap":
                        supplied_img_id = partner_img_id
                    elif cond == "mismatched":
                        supplied_img_id = mismatched_map[pid]
                    elif cond == "no_image":
                        supplied_img_id = None
                    else:
                        raise ValueError(f"Unknown condition: {cond}")

                    feat = self.feature_cache.get(supplied_img_id) if supplied_img_id is not None else None

                    # 1. Greedy text generation
                    t0 = time.time()
                    gen_out = self.vlm.generate(
                        visual_features=feat,
                        question=q,
                        max_new_tokens=max_new_tokens,
                    )
                    gen_duration = time.time() - t0

                    pred_text = gen_out["text"]
                    pred_norm = normalize_answer(pred_text)
                    is_empty = len(pred_text.strip()) == 0
                    is_trunc = gen_out["is_truncated"]

                    # 2. Reference-answer NLL computation
                    ref_inputs = self.vlm.prepare_multimodal_inputs([{
                        "visual_features": feat,
                        "question": q,
                        "answer": target_ans,
                        "is_caption": False,
                    }])
                    logits = self.vlm(
                        inputs_embeds=ref_inputs["inputs_embeds"].to(self.vlm.device),
                        attention_mask=ref_inputs["attention_mask"].to(self.vlm.device),
                    )
                    ref_nll = compute_example_answer_nll(logits, ref_inputs["labels"].to(self.vlm.device))[0]

                    records.append({
                        "suite": suite_name,
                        "condition": cond,
                        "pair_id": pid,
                        "example_id": ex_id,
                        "question": q,
                        "original_image_id": orig_img_id,
                        "supplied_image_id": supplied_img_id,
                        "pair_type": pair.get("pair_type", "unknown"),
                        "answer_type": pair.get("answer_type", pair.get("pair_type", "unknown")),
                        "reference": target_ans,
                        "human_answers": human_ans,
                        "prediction": pred_text,
                        "normalized_prediction": pred_norm,
                        "is_empty": is_empty,
                        "is_truncated": is_trunc,
                        "generation_length": gen_out["generated_tokens_count"],
                        "generation_time_sec": round(gen_duration, 4),
                        "reference_nll": round(ref_nll, 4),
                    })

        df = pd.DataFrame(records)
        return df

    def run_all(self) -> Dict[str, pd.DataFrame]:
        """Runs final evaluation for both suites and saves parquet files."""
        preds_dir = self.output_dir / "results" / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)

        df_grounding = self.evaluate_suite(
            self.grounding_pairs,
            suite_name="grounding",
            mismatched_map=self.grounding_mismatched_map,
        )
        df_grounding.to_parquet(preds_dir / "final_grounding_predictions.parquet", index=False)

        df_vqav2 = self.evaluate_suite(
            self.vqav2_pairs,
            suite_name="vqav2",
            mismatched_map=self.vqav2_mismatched_map,
        )
        df_vqav2.to_parquet(preds_dir / "final_vqav2_predictions.parquet", index=False)

        logger.info(f"Final predictions saved to {preds_dir}")
        return {
            "grounding": df_grounding,
            "vqav2": df_vqav2,
        }
