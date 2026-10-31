"""Development evaluation protocol for Task 2.

Evaluates at scheduled checkpoints:
0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500.

Computes:
1. Caption reference loss on the 96-image dev caption set.
2. Reference-answer NLL on Grounding dev (24 pairs / 48 questions) and
   Natural VQAv2 dev (32 pairs / 64 questions) under 3 conditions:
   - Correct image
   - Complementary partner image
   - No-image input (visual tokens omitted)
3. Image-swap penalty: DeltaNLL = NLL(partner) - NLL(correct).
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from task2.src.models.vlm import VLMModel
from task2.src.training.loss import compute_example_weighted_loss

logger = logging.getLogger(__name__)


def compute_example_answer_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> List[float]:
    """Compute per-example mean NLL over target tokens (excluding termination tokens if flagged)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size = shift_labels.size(0)
    per_example_nlls = []

    for i in range(batch_size):
        item_logits = shift_logits[i]
        item_labels = shift_labels[i]

        mask = item_labels != ignore_index
        if mask.sum() > 0:
            target_logits = item_logits[mask]
            target_labels = item_labels[mask]
            # Average within the example over target tokens
            nll = F.cross_entropy(target_logits, target_labels, reduction="mean").item()
            per_example_nlls.append(nll)
        else:
            per_example_nlls.append(0.0)

    return per_example_nlls


class DevEvaluator:
    """Evaluates development metrics across the 12 scheduled training checkpoints."""

    def __init__(
        self,
        vlm_model: VLMModel,
        dev_captions: List[Dict[str, Any]],
        grounding_dev_pairs: List[Dict[str, Any]],
        vqav2_dev_pairs: List[Dict[str, Any]],
        feature_cache: Dict[str | int, torch.Tensor],
        output_dir: str | Path,
    ):
        self.vlm = vlm_model
        self.dev_captions = dev_captions
        self.grounding_dev_pairs = grounding_dev_pairs
        self.vqav2_dev_pairs = vqav2_dev_pairs
        self.feature_cache = feature_cache
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, Any]] = []

    @torch.no_grad()
    def evaluate_checkpoint(
        self,
        global_update: int,
        stage: int,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Runs the full dev evaluation suite for a specific checkpoint."""
        start_time = time.time()
        self.vlm.eval()

        logger.info(f"Running development evaluation at global update {global_update} (Stage {stage})...")

        # 1. Caption reference loss on 96 dev captions
        caption_losses = []
        for i in range(0, len(self.dev_captions), batch_size):
            batch_data = self.dev_captions[i : i + batch_size]
            prepared = []
            for item in batch_data:
                feat = self.feature_cache.get(item["image_id"])
                prepared.append({
                    "visual_features": feat,
                    "question": "",
                    "answer": item["caption"],
                    "is_caption": True,
                })
            inputs = self.vlm.prepare_multimodal_inputs(prepared)
            inputs_embeds = inputs["inputs_embeds"].to(self.vlm.device)
            attention_mask = inputs["attention_mask"].to(self.vlm.device)
            labels = inputs["labels"].to(self.vlm.device)

            logits = self.vlm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            _, per_ex_loss = compute_example_weighted_loss(logits, labels, return_per_example=True)
            caption_losses.extend(per_ex_loss.cpu().tolist())

        mean_caption_loss = float(np.mean(caption_losses)) if caption_losses else 0.0

        # Helper to evaluate QA suite under 3 conditions
        def eval_qa_suite(
            pairs: List[Dict[str, Any]],
            suite_name: str,
        ) -> Dict[str, Any]:
            suite_records = []
            for pair in pairs:
                q = pair["question"]
                img1_id = pair["image_id_1"]
                img2_id = pair["image_id_2"]
                ans1 = pair["answer_1"]
                ans2 = pair["answer_2"]
                pair_id = pair.get("pair_id", f"{suite_name}_{img1_id}_{img2_id}")
                q_type = pair.get("answer_type", pair.get("pair_type", "unknown"))

                # Evaluate both items of the pair under correct, partner, no_image
                for item_idx, (img_corr_id, img_part_id, target_ans) in enumerate([
                    (img1_id, img2_id, ans1),
                    (img2_id, img1_id, ans2),
                ]):
                    ex_id = f"{pair_id}_ex{item_idx+1}"

                    # Condition: correct
                    feat_corr = self.feature_cache.get(img_corr_id)
                    inputs_corr = self.vlm.prepare_multimodal_inputs([{
                        "visual_features": feat_corr,
                        "question": q,
                        "answer": target_ans,
                        "is_caption": False,
                    }])
                    logits_corr = self.vlm(
                        inputs_embeds=inputs_corr["inputs_embeds"].to(self.vlm.device),
                        attention_mask=inputs_corr["attention_mask"].to(self.vlm.device),
                    )
                    nll_corr = compute_example_answer_nll(logits_corr, inputs_corr["labels"].to(self.vlm.device))[0]

                    # Condition: partner (complementary swap)
                    feat_part = self.feature_cache.get(img_part_id)
                    inputs_part = self.vlm.prepare_multimodal_inputs([{
                        "visual_features": feat_part,
                        "question": q,
                        "answer": target_ans,
                        "is_caption": False,
                    }])
                    logits_part = self.vlm(
                        inputs_embeds=inputs_part["inputs_embeds"].to(self.vlm.device),
                        attention_mask=inputs_part["attention_mask"].to(self.vlm.device),
                    )
                    nll_part = compute_example_answer_nll(logits_part, inputs_part["labels"].to(self.vlm.device))[0]

                    # Condition: no_image
                    inputs_noimg = self.vlm.prepare_multimodal_inputs([{
                        "visual_features": None,
                        "question": q,
                        "answer": target_ans,
                        "is_caption": False,
                    }])
                    logits_noimg = self.vlm(
                        inputs_embeds=inputs_noimg["inputs_embeds"].to(self.vlm.device),
                        attention_mask=inputs_noimg["attention_mask"].to(self.vlm.device),
                    )
                    nll_noimg = compute_example_answer_nll(logits_noimg, inputs_noimg["labels"].to(self.vlm.device))[0]

                    # Image-swap penalty DeltaNLL = NLL_partner - NLL_correct
                    delta_nll = nll_part - nll_corr

                    suite_records.append({
                        "global_update": global_update,
                        "stage": stage,
                        "suite": suite_name,
                        "pair_id": pair_id,
                        "example_id": ex_id,
                        "question_type": q_type,
                        "nll_correct": nll_corr,
                        "nll_partner": nll_part,
                        "nll_no_image": nll_noimg,
                        "delta_nll": delta_nll,
                    })

            df_suite = pd.DataFrame(suite_records)
            return {
                "records": suite_records,
                "mean_nll_correct": float(df_suite["nll_correct"].mean()) if len(df_suite) > 0 else 0.0,
                "mean_nll_partner": float(df_suite["nll_partner"].mean()) if len(df_suite) > 0 else 0.0,
                "mean_nll_no_image": float(df_suite["nll_no_image"].mean()) if len(df_suite) > 0 else 0.0,
                "mean_delta_nll": float(df_suite["delta_nll"].mean()) if len(df_suite) > 0 else 0.0,
            }

        grounding_res = eval_qa_suite(self.grounding_dev_pairs, "grounding")
        vqav2_res = eval_qa_suite(self.vqav2_dev_pairs, "vqav2")

        eval_duration = time.time() - start_time

        summary_row = {
            "global_update": global_update,
            "stage": stage,
            "caption_loss": round(mean_caption_loss, 4),
            "grounding_nll_correct": round(grounding_res["mean_nll_correct"], 4),
            "grounding_nll_partner": round(grounding_res["mean_nll_partner"], 4),
            "grounding_nll_no_image": round(grounding_res["mean_nll_no_image"], 4),
            "grounding_delta_nll": round(grounding_res["mean_delta_nll"], 4),
            "vqav2_nll_correct": round(vqav2_res["mean_nll_correct"], 4),
            "vqav2_nll_partner": round(vqav2_res["mean_nll_partner"], 4),
            "vqav2_nll_no_image": round(vqav2_res["mean_nll_no_image"], 4),
            "vqav2_delta_nll": round(vqav2_res["mean_delta_nll"], 4),
            "eval_duration_sec": round(eval_duration, 2),
        }

        self.records.append(summary_row)
        self._save_results(grounding_res["records"] + vqav2_res["records"])

        logger.info(
            f"Dev Eval @ Update {global_update}: Cap Loss={mean_caption_loss:.4f} | "
            f"Grounding DeltaNLL={grounding_res['mean_delta_nll']:.4f} | "
            f"VQAv2 DeltaNLL={vqav2_res['mean_delta_nll']:.4f} ({eval_duration:.1f}s)"
        )
        return summary_row

    def _save_results(self, per_example_records: List[Dict[str, Any]]) -> None:
        """Persist development evaluation tables and per-example parquet."""
        tables_dir = self.output_dir / "results" / "tables"
        preds_dir = self.output_dir / "results" / "predictions"
        tables_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

        # Save T06 table
        df_summary = pd.DataFrame(self.records)
        df_summary.to_csv(tables_dir / "T06_development_evaluation_by_checkpoint.csv", index=False)

        # Append / save per-example parquet
        per_ex_file = preds_dir / "development_per_example.parquet"
        new_df = pd.DataFrame(per_example_records)
        if per_ex_file.exists():
            try:
                old_df = pd.read_parquet(per_ex_file)
                combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
                    subset=["global_update", "example_id"]
                )
                combined.to_parquet(per_ex_file, index=False)
            except Exception:
                new_df.to_parquet(per_ex_file, index=False)
        else:
            new_df.to_parquet(per_ex_file, index=False)
