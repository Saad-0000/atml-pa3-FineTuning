"""
Deterministic dataset split management for Task 3.

Protocol:
1. Collect 1,000 question IDs per dataset.
2. Sort lexicographically as strings (preserving string representations).
3. Permute with np.random.default_rng(6304).
4. Assign first 200 -> development split.
5. Assign remaining 800 -> held-out evaluation split.
6. Select first 50 MATH held-out questions in that exact order for live generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Optional
import numpy as np

from task3.src.utils.common import atomic_save_json, load_json, setup_logger

logger = setup_logger("task3.data.splits")


class DatasetSplit:
    def __init__(
        self,
        dataset_name: str,
        total_questions: int,
        dev_question_ids: List[str],
        held_out_question_ids: List[str],
        live_50_question_ids: List[str],
        sorted_original_ids: List[str],
        seed: int = 6304,
    ):
        self.dataset_name = dataset_name
        self.total_questions = total_questions
        self.dev_question_ids = dev_question_ids
        self.held_out_question_ids = held_out_question_ids
        self.live_50_question_ids = live_50_question_ids
        self.sorted_original_ids = sorted_original_ids
        self.seed = seed

    def verify_disjointness(self) -> bool:
        dev_set = set(self.dev_question_ids)
        held_out_set = set(self.held_out_question_ids)
        if len(dev_set.intersection(held_out_set)) != 0:
            return False
        if len(dev_set) != len(self.dev_question_ids) or len(held_out_set) != len(self.held_out_question_ids):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "total_questions": self.total_questions,
            "dev_question_ids": self.dev_question_ids,
            "held_out_question_ids": self.held_out_question_ids,
            "live_50_question_ids": self.live_50_question_ids,
            "sorted_original_ids": self.sorted_original_ids,
        }


def build_dataset_splits(
    question_ids: List[Any],
    seed: int = 6304,
    dataset_name: str = "MATH",
) -> DatasetSplit:
    """Creates deterministic 200 dev / 800 held-out split from question IDs."""
    str_ids = [str(qid) for qid in question_ids]
    sorted_ids = sorted(list(dict.fromkeys(str_ids)))

    rng = np.random.default_rng(seed)
    permuted_ids = rng.permutation(sorted_ids).tolist()

    if len(sorted_ids) >= 1000:
        dev_ids = permuted_ids[:200]
        held_out_ids = permuted_ids[200:]
        live_50_ids = held_out_ids[:50] if dataset_name.upper() == "MATH" else []
    else:
        n_dev = max(1, int(len(sorted_ids) * 0.2))
        dev_ids = permuted_ids[:n_dev]
        held_out_ids = permuted_ids[n_dev:]
        live_50_ids = held_out_ids[: min(50, len(held_out_ids))] if dataset_name.upper() == "MATH" else []

    split = DatasetSplit(
        dataset_name=dataset_name,
        total_questions=len(sorted_ids),
        dev_question_ids=dev_ids,
        held_out_question_ids=held_out_ids,
        live_50_question_ids=live_50_ids,
        sorted_original_ids=sorted_ids,
        seed=seed,
    )
    return split


def save_splits_manifest(split: DatasetSplit, output_path: str | Path):
    out_p = Path(output_path)
    atomic_save_json(split.to_dict(), out_p)
    logger.info(f"Saved {split.dataset_name} split manifest to {out_p}")


def load_splits_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    return load_json(manifest_path)


class SplitManager:
    """Backward compatibility wrapper."""
    def __init__(self, seed: int = 6304):
        self.seed = seed

    def build_splits(self, question_ids: List[Any]) -> Dict[str, Any]:
        sp = build_dataset_splits(question_ids, seed=self.seed)
        return sp.to_dict()
