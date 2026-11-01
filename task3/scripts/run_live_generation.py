"""
Task 3 Live Generation Script
Executes the live-generation experiment on the first 50 MATH held-out questions.
Compares:
1. Single longer response (cap 3072)
2. Parallel sampling (3 x 1024)
3. Sequential refinement (1024 + 2048) with shared initial response reuse.
Generates Tables T09, T10, and T11.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from task3.src.utils.common import get_logger, atomic_save_json
from task3.src.data.split_builder import load_splits_manifest
from task3.src.live_generation.generator import LiveGenerationPipeline
from task3.src.visualization.tables import build_table_t09, build_table_t10, build_table_t11

logger = get_logger("run_live_generation")


def main():
    parser = argparse.ArgumentParser(description="Task 3 Live Generation Experiment")
    parser.add_argument("--config", type=str, default="task3/configs/live_generation.yaml", help="Config path")
    parser.add_argument("--manifests_dir", type=str, default="task3/manifests", help="Manifests directory")
    parser.add_argument("--output_dir", type=str, default="task3/results/live_generation", help="Live generation output dir")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output dir")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to ATML-PA3-Data")
    parser.add_argument("--dry_run", action="store_true", help="Run with mock generator for testing without GPU")
    parser.add_argument("--max_questions", type=int, default=50, help="Max questions to run (default 50)")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    math_splits = load_splits_manifest(manifests_dir / "math_splits.json")
    live_ids = math_splits.get("live_50_question_ids", [])[:args.max_questions]
    logger.info(f"Targeting {len(live_ids)} live generation questions from MATH held-out.")

    # Load question text and gold answers from parquet or create fallbacks
    data_dir = Path(args.data_dir)
    math_parquet = data_dir / "MATH" / "generations.parquet"
    
    questions_data = {}
    if math_parquet.exists():
        df_math = pd.read_parquet(math_parquet).drop_duplicates("sample_id")
        for _, row in df_math.iterrows():
            sid = str(row["sample_id"])
            if sid in live_ids:
                prob_text = row.get("prompt", row.get("problem", f"Math problem for {sid}"))
                questions_data[sid] = {
                    "question_id": sid,
                    "problem": prob_text,
                    "gold_answer": str(row.get("gold_answer", "0")),
                }
    else:
        logger.warning(f"{math_parquet} not found. Using question stubs.")
        for sid in live_ids:
            questions_data[sid] = {
                "question_id": sid,
                "problem": f"Solve this math problem for {sid}: What is 2 + 2?",
                "gold_answer": "4",
            }

    ordered_questions = [questions_data[qid] for qid in live_ids if qid in questions_data]
    logger.info(f"Prepared {len(ordered_questions)} questions for live generation.")

    # Initialize live pipeline
    pipeline = LiveGenerationPipeline(
        config_path=args.config,
        output_dir=output_dir,
        dry_run=args.dry_run
    )

    all_results = pipeline.run_all_questions(ordered_questions)
    summary = pipeline.summarize_results(all_results)

    # Build T09 (Per-question results)
    t09_records = []
    for qid, res in all_results.items():
        for strat in ["single_long", "parallel", "sequential_refinement"]:
            s_data = res.get("strategies", {}).get(strat, {})
            t09_records.append({
                "question_id": qid,
                "strategy": strat,
                "selected_answer": s_data.get("selected_answer", ""),
                "is_correct": s_data.get("is_correct", False),
                "input_tokens": s_data.get("input_tokens", 0),
                "output_tokens": s_data.get("output_tokens", 0),
                "runtime_seconds": round(s_data.get("runtime_seconds", 0.0), 3),
                "peak_memory_mb": round(s_data.get("peak_memory_mb", 0.0), 1),
                "extraction_success": s_data.get("extraction_success", True),
                "truncated": s_data.get("truncated", False),
            })

    df_t09 = build_table_t09(t09_records)
    df_t09.to_csv(tables_dir / "T09_live_question_results.csv", index=False)
    logger.info(f"Saved Table T09 to {tables_dir / 'T09_live_question_results.csv'}")

    # Build T10 (Strategy summary)
    t10_records = []
    for strat, s_sum in summary.get("strategies", {}).items():
        t10_records.append({
            "strategy": strat,
            "accuracy": s_sum.get("accuracy", 0.0),
            "mean_input_tokens": s_sum.get("mean_input_tokens", 0.0),
            "mean_output_tokens": s_sum.get("mean_output_tokens", 0.0),
            "mean_runtime_sec": s_sum.get("mean_runtime_sec", 0.0),
            "total_runtime_sec": s_sum.get("total_runtime_sec", 0.0),
            "extraction_rate": s_sum.get("extraction_rate", 1.0),
            "truncation_rate": s_sum.get("truncation_rate", 0.0),
            "peak_memory_mb": s_sum.get("peak_memory_mb", 0.0),
            "num_questions": s_sum.get("num_questions", len(ordered_questions)),
        })

    df_t10 = build_table_t10(t10_records)
    df_t10.to_csv(tables_dir / "T10_live_strategy_summary.csv", index=False)
    logger.info(f"Saved Table T10 to {tables_dir / 'T10_live_strategy_summary.csv'}")

    # Build T11 (Refinement transition matrix)
    ref_matrix = summary.get("refinement_transitions", {})
    t11_records = [
        {"transition": cat, "count": ref_matrix.get(cat, 0), "rate": round(ref_matrix.get(cat, 0) / max(len(ordered_questions), 1), 3)}
        for cat in ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]
    ]
    df_t11 = build_table_t11(t11_records)
    df_t11.to_csv(tables_dir / "T11_refinement_transition_matrix.csv", index=False)
    logger.info(f"Saved Table T11 to {tables_dir / 'T11_refinement_transition_matrix.csv'}")


if __name__ == "__main__":
    main()
