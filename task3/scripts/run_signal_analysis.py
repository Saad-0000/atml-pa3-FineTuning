"""
Task 3 Signal Analysis and Tokenwise Fusion Script (Experiment C)
Analyzes length vs score vs correctness associations (T06),
confidence-vision correlations (T07), and tokenwise vs trajectory fusion (T08).
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from task3.src.utils.common import get_logger, atomic_save_json
from task3.src.data.split_builder import load_splits_manifest
from task3.src.signals.quartile_analysis import compute_quartile_analysis_for_dataset
from task3.src.signals.correlation_analysis import compute_mathvista_correlations
from task3.src.signals.fusion_analysis import compute_fusion_comparison_for_dataset
from task3.src.visualization.tables import build_table_t06, build_table_t07, build_table_t08

logger = get_logger("run_signal_analysis")


def main():
    parser = argparse.ArgumentParser(description="Run Task 3 Signal and Fusion Analysis (Exp C)")
    parser.add_argument("--rollouts_dir", type=str, default="task3/results/per_rollout", help="Canonical rollouts directory")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output directory")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    manifests_dir = Path(args.manifests_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    math_splits = load_splits_manifest(manifests_dir / "math_splits.json")
    mathvista_splits = load_splits_manifest(manifests_dir / "mathvista_splits.json")

    math_heldout_ids = set(math_splits.get("held_out_question_ids", []))
    mathvista_heldout_ids = set(mathvista_splits.get("held_out_question_ids", []))

    df_math = pd.read_parquet(rollouts_dir / "math_rollouts.parquet")
    df_math_heldout = df_math[df_math["sample_id"].isin(math_heldout_ids)].copy()

    df_mathvista = pd.read_parquet(rollouts_dir / "mathvista_rollouts.parquet")
    df_mathvista_heldout = df_mathvista[df_mathvista["sample_id"].isin(mathvista_heldout_ids)].copy()

    # 1. Quartile Analysis (T06)
    logger.info("Computing within-question quartile analysis for MATH and MathVista...")
    math_signals = [("length", "token_length"), ("L_i", "L_i"), ("K_i", "K_i")]
    mathvista_signals = [("length", "token_length"), ("L_i", "L_i"), ("K_i", "K_i"), ("V_i", "V_i")]

    t06_rows = []
    math_q_res = compute_quartile_analysis_for_dataset(df_math_heldout, "MATH", math_signals)
    t06_rows.extend(math_q_res)

    mathvista_q_res = compute_quartile_analysis_for_dataset(df_mathvista_heldout, "MathVista", mathvista_signals)
    t06_rows.extend(mathvista_q_res)

    df_t06 = build_table_t06(t06_rows)
    df_t06.to_csv(tables_dir / "T06_signal_length_correctness.csv", index=False)
    logger.info(f"Saved Table T06 to {tables_dir / 'T06_signal_length_correctness.csv'}")

    # 2. MathVista Confidence-Vision Relationship (T07)
    logger.info("Computing tokenwise confidence-vision correlation for MathVista...")
    corr_summary = compute_mathvista_correlations(df_mathvista_heldout)
    t07_rows = [corr_summary]
    df_t07 = build_table_t07(t07_rows)
    df_t07.to_csv(tables_dir / "T07_confidence_vision_relationship.csv", index=False)
    logger.info(f"Saved Table T07 to {tables_dir / 'T07_confidence_vision_relationship.csv'}")

    # 3. Tokenwise vs Trajectory Fusion (T08)
    logger.info("Computing tokenwise vs trajectory-level fusion comparison...")
    fusion_summary = compute_fusion_comparison_for_dataset(df_mathvista_heldout)
    df_t08 = build_table_t08(fusion_summary)
    df_t08.to_csv(tables_dir / "T08_fusion_comparison.csv", index=False)
    logger.info(f"Saved Table T08 to {tables_dir / 'T08_fusion_comparison.csv'}")


if __name__ == "__main__":
    main()
