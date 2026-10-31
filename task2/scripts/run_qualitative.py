"""Qualitative example selection and human inspection sheet generator for Task 2.

Usage:
    python -m task2.scripts.run_qualitative --config task2/configs/task2.yaml
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

from task2.src.config import load_task2_config
from task2.src.utils.logging_utils import setup_logger
from task2.src.visualization.qualitative import (
    select_qualitative_examples,
    export_human_review_sheet,
    export_qualitative_html_viewer,
)

logger = setup_logger("run_qualitative")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Qualitative Diagnostics")
    parser.add_argument(
        "--config",
        type=str,
        default="task2/configs/task2.yaml",
        help="Path to task2.yaml configuration file",
    )
    parser.add_argument(
        "--num-per-cat",
        type=int,
        default=2,
        help="Number of pairs to sample per qualitative behavior category",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_task2_config(args.config)
    seed = config.data.seed

    preds_dir = Path(config.paths.results_dir) / "predictions"
    qual_dir = Path(config.paths.results_dir) / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    g_file = preds_dir / "final_grounding_predictions.parquet"
    v_file = preds_dir / "final_vqav2_predictions.parquet"

    if not g_file.exists() or not v_file.exists():
        logger.warning(
            f"Predictions not found at {g_file} and {v_file}. "
            "Please run evaluate_final first or verify result paths."
        )
        return

    df_grounding = pd.read_parquet(g_file)
    df_vqav2 = pd.read_parquet(v_file)

    logger.info("Selecting qualitative examples across behavior categories...")
    df_selected = select_qualitative_examples(
        df_grounding=df_grounding,
        df_vqav2=df_vqav2,
        num_per_category=args.num_per_cat,
        seed=seed,
    )

    manifest_csv = qual_dir / "qualitative_example_manifest.csv"
    df_selected.to_csv(manifest_csv, index=False)
    logger.info(f"Saved qualitative manifest: {manifest_csv} ({len(df_selected)} rows)")

    review_sheet_md = qual_dir / "human_review_sheet.md"
    export_human_review_sheet(df_selected, review_sheet_md)

    viewer_html = qual_dir / "aligned_complementary_examples.html"
    export_qualitative_html_viewer(df_selected, viewer_html)

    logger.info("Qualitative diagnostics generated successfully!")


if __name__ == "__main__":
    main()
