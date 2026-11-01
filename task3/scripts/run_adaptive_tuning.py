"""
Task 3 Adaptive Threshold Tuning Script
Tunes agreement (tau) and quality-aware (tau, gamma) stopping policies on the Development split (200 questions).
Selects the policy with lowest token cost within 1 pp of fixed-32 SC.
Freezes selected settings to manifests/adaptive_selected_policies.json and outputs Table T12.
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
from task3.src.data.split_builder import load_splits_manifest
from task3.src.adaptive.grid_search import run_development_grid_search
from task3.src.visualization.tables import build_table_t12

logger = get_logger("run_adaptive_tuning")


def main():
    parser = argparse.ArgumentParser(description="Tune adaptive stopping policies on development split")
    parser.add_argument("--rollouts_dir", type=str, default="task3/results/per_rollout", help="Rollouts directory")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output directory")
    parser.add_argument("--seed", type=int, default=6304, help="Random seed")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    manifests_dir = Path(args.manifests_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    math_splits = load_splits_manifest(manifests_dir / "math_splits.json")
    mathvista_splits = load_splits_manifest(manifests_dir / "mathvista_splits.json")

    math_dev_ids = set(math_splits.get("dev_question_ids", []))
    mathvista_dev_ids = set(mathvista_splits.get("dev_question_ids", []))

    df_math = pd.read_parquet(rollouts_dir / "math_rollouts.parquet")
    df_math_dev = df_math[df_math["sample_id"].isin(math_dev_ids)].copy()

    df_mathvista = pd.read_parquet(rollouts_dir / "mathvista_rollouts.parquet")
    df_mathvista_dev = df_mathvista[df_mathvista["sample_id"].isin(mathvista_dev_ids)].copy()

    logger.info(f"MATH dev questions: {df_math_dev['sample_id'].nunique()}, MathVista dev questions: {df_mathvista_dev['sample_id'].nunique()}")

    # Run grid search
    math_grid = run_development_grid_search(df_math_dev, "MATH", seed=args.seed)
    mathvista_grid = run_development_grid_search(df_mathvista_dev, "MathVista", seed=args.seed)

    # Save frozen policy manifest
    selected_manifest = {
        "MATH": math_grid["selected_policies"],
        "MathVista": mathvista_grid["selected_policies"],
        "selection_metadata": {
            "dev_questions_math": len(math_dev_ids),
            "dev_questions_mathvista": len(mathvista_dev_ids),
            "feasibility_margin": -0.01,
            "seed": args.seed,
        }
    }
    manifest_path = manifests_dir / "adaptive_selected_policies.json"
    atomic_save_json(selected_manifest, manifest_path)
    logger.info(f"Saved frozen adaptive policy selections to {manifest_path}")

    # Build Table T12
    t12_records = []
    for g in [math_grid, mathvista_grid]:
        t12_records.extend(g["grid_results"])

    df_t12 = build_table_t12(t12_records)
    df_t12.to_csv(tables_dir / "T12_adaptive_dev_grid.csv", index=False)
    logger.info(f"Saved Table T12 to {tables_dir / 'T12_adaptive_dev_grid.csv'}")


if __name__ == "__main__":
    main()
