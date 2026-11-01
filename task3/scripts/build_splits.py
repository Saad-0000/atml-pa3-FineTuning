"""
Task 3 Split Generation and Validation Script
Generates deterministic 200 development / 800 held-out splits using seed 6304,
and selects the first 50 MATH held-out questions for live generation.
Generates Table T01 (Split manifest summary).
"""

import sys
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from task3.src.utils.common import get_logger, atomic_save_json
from task3.src.data.split_builder import build_dataset_splits, save_splits_manifest
from task3.src.data.parquet_loader import load_dataset_rollouts
from task3.src.visualization.tables import build_table_t01

logger = get_logger("build_splits")


def get_question_ids(data_dir: Path, dataset_name: str) -> list:
    """Extract distinct question IDs from dataset rollouts or generate standard 1000 IDs."""
    parquet_path = resolve_dataset_dir(data_dir, dataset_name) / "generations.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=["sample_id"])
        ids = df["sample_id"].astype(str).unique().tolist()
        return ids
    else:
        logger.warning(f"{parquet_path} not found. Generating default standard 1000 question IDs for {dataset_name}.")
        return [f"{dataset_name.lower()}_{i:04d}" for i in range(1000)]


def main():
    parser = argparse.ArgumentParser(description="Build deterministic splits for Task 3")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to ATML-PA3-Data")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests output directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output directory")
    parser.add_argument("--seed", type=int, default=6304, help="Random seed for default_rng")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifests_dir = Path(args.manifests_dir)
    tables_dir = Path(args.tables_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    math_ids = get_question_ids(data_dir, "MATH")
    mathvista_ids = get_question_ids(data_dir, "MathVista")

    math_split = build_dataset_splits(math_ids, seed=args.seed, dataset_name="MATH")
    mathvista_split = build_dataset_splits(mathvista_ids, seed=args.seed, dataset_name="MathVista")

    save_splits_manifest(math_split, manifests_dir / "math_splits.json")
    save_splits_manifest(mathvista_split, manifests_dir / "mathvista_splits.json")

    # Build T01 table
    split_summaries = [
        {
            "dataset": "MATH",
            "total_questions": math_split.total_questions,
            "dev_questions": len(math_split.dev_question_ids),
            "held_out_questions": len(math_split.held_out_question_ids),
            "live_math_questions": len(math_split.live_50_question_ids),
            "seed": math_split.seed,
            "disjoint": math_split.verify_disjointness(),
        },
        {
            "dataset": "MathVista",
            "total_questions": mathvista_split.total_questions,
            "dev_questions": len(mathvista_split.dev_question_ids),
            "held_out_questions": len(mathvista_split.held_out_question_ids),
            "live_math_questions": 0,
            "seed": mathvista_split.seed,
            "disjoint": mathvista_split.verify_disjointness(),
        }
    ]

    df_t01 = build_table_t01(split_summaries)
    csv_path = tables_dir / "T01_split_manifest_summary.csv"
    df_t01.to_csv(csv_path, index=False)
    logger.info(f"Saved Table T01 to {csv_path}")


if __name__ == "__main__":
    main()
