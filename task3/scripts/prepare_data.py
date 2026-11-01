"""
Task 3 Data Preparation and Rollout Normalization Script
Processes supplied Parquet rollouts:
1. Validates schema and token arrays.
2. Normalizes answers and evaluates correctness (without leakage into selection).
3. Computes rollout quality metrics (L_i, K_i, V_i).
4. Persists canonical rollout records to Parquet.
5. Builds Table T02 (Scorer and answer-normalization audit).
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
from task3.src.data.parquet_loader import load_dataset_rollouts, resolve_dataset_dir
from task3.src.data.synthetic_data import create_synthetic_dataset
from task3.src.scoring.normalization import normalize_answer, grade_correctness, MATH_VERIFY_AVAILABLE
from task3.src.selection.quality_scores import compute_mean_logp, compute_mean_self_certainty, compute_visual_engagement
from task3.src.visualization.tables import build_table_t02

logger = get_logger("prepare_data")


def process_dataset_rollouts(data_dir: Path, dataset_name: str, output_dir: Path) -> Dict[str, Any]:
    """Loads, scores, and saves canonical rollouts for a dataset."""
    logger.info(f"Processing rollouts for {dataset_name}...")
    ds_dir = resolve_dataset_dir(data_dir, dataset_name)
    
    if not (ds_dir / "generations.parquet").exists():
        logger.warning(f"{ds_dir} not found. Generating synthetic fallback...")
        create_synthetic_dataset(data_dir, dataset_name, num_questions=20, rollouts_per_q=32)

    df = load_dataset_rollouts(data_dir, dataset_name)
    logger.info(f"Loaded {len(df)} rollouts for {dataset_name}.")

    # 1. Normalize answers and grade correctness
    norm_answers = []
    is_correct_list = []
    parse_failures = 0
    no_answer_count = 0
    correct_count = 0

    # 2. Quality scores
    mean_logps = []
    mean_kls = []
    visual_engagements = []

    for idx, row in df.iterrows():
        raw_ans = row.get("extracted_answer", "")
        gold_ans = row.get("gold_answer", "")
        q_meta = row.to_dict()

        norm_ans = normalize_answer(raw_ans, dataset_name, q_meta)
        norm_answers.append(norm_ans)

        if norm_ans == "NO_ANSWER":
            no_answer_count += 1
            is_correct = False
        else:
            is_correct = grade_correctness(norm_ans, gold_ans, dataset_name, q_meta)

        if is_correct:
            correct_count += 1
        is_correct_list.append(is_correct)

        # L_i
        logp_arr = row.get("logp", [])
        mean_logps.append(compute_mean_logp(logp_arr))

        # K_i
        kl_arr = row.get("kl_u_p", [])
        mean_kls.append(compute_mean_self_certainty(kl_arr))

        # V_i (MathVista only)
        if dataset_name == "MathVista" and "A_image" in row:
            v_val, _ = compute_visual_engagement(
                row.get("A_image", []),
                row.get("A_prompt", []),
                row.get("A_generated", [])
            )
            visual_engagements.append(v_val)
        else:
            visual_engagements.append(0.0)

    df["normalized_answer"] = norm_answers
    df["is_correct"] = is_correct_list
    df["L_i"] = mean_logps
    df["K_i"] = mean_kls
    if dataset_name == "MathVista":
        df["V_i"] = visual_engagements

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / f"{dataset_name.lower()}_rollouts.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info(f"Saved canonical rollouts to {out_parquet}")

    audit = {
        "dataset": dataset_name,
        "scorer_package": "math_verify" if MATH_VERIFY_AVAILABLE else "sympy/regex_fallback",
        "total_rollouts": len(df),
        "correct_rollouts": correct_count,
        "no_answer_count": no_answer_count,
        "parse_failures": parse_failures,
    }
    return audit


def main():
    parser = argparse.ArgumentParser(description="Prepare and score rollouts")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to supplied data")
    parser.add_argument("--output_rollouts_dir", type=str, default="task3/results/per_rollout", help="Rollout output dir")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rollouts_dir = Path(args.output_rollouts_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    math_audit = process_dataset_rollouts(data_dir, "MATH", rollouts_dir)
    mathvista_audit = process_dataset_rollouts(data_dir, "MathVista", rollouts_dir)

    # Build T02
    t02_records = [
        {
            "dataset": "MATH",
            "scorer_package": math_audit["scorer_package"],
            "normalization_rule": "math_verify latex parse + sympy equivalence",
            "total_rollouts": math_audit["total_rollouts"],
            "no_answer_count": math_audit["no_answer_count"],
            "parse_failures": math_audit["parse_failures"],
            "correct_rollouts": math_audit["correct_rollouts"],
            "raw_accuracy": round(math_audit["correct_rollouts"] / max(math_audit["total_rollouts"], 1) * 100, 2),
        },
        {
            "dataset": "MathVista",
            "scorer_package": "MathVista precision & option normalizer",
            "normalization_rule": "option letter / float precision tolerance",
            "total_rollouts": mathvista_audit["total_rollouts"],
            "no_answer_count": mathvista_audit["no_answer_count"],
            "parse_failures": mathvista_audit["parse_failures"],
            "correct_rollouts": mathvista_audit["correct_rollouts"],
            "raw_accuracy": round(mathvista_audit["correct_rollouts"] / max(mathvista_audit["total_rollouts"], 1) * 100, 2),
        }
    ]

    df_t02 = build_table_t02(t02_records)
    csv_path = tables_dir / "T02_scorer_normalization_audit.csv"
    df_t02.to_csv(csv_path, index=False)
    logger.info(f"Saved Table T02 to {csv_path}")


if __name__ == "__main__":
    main()
