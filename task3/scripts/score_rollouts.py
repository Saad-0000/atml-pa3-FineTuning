"""
Task 3 Rollout Scoring Script
Runs standalone extraction, normalization, and scoring on rollout files.
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task3.scripts.prepare_data import process_dataset_rollouts
from task3.src.utils.common import get_logger

logger = get_logger("score_rollouts")


def main():
    parser = argparse.ArgumentParser(description="Score rollout files standalone")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to supplied data")
    parser.add_argument("--dataset", type=str, default="MATH", choices=["MATH", "MathVista", "all"], help="Dataset to score")
    parser.add_argument("--output_dir", type=str, default="task3/results/per_rollout", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    datasets = ["MATH", "MathVista"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        audit = process_dataset_rollouts(data_dir, ds, out_dir)
        logger.info(f"Scoring summary for {ds}: {audit}")


if __name__ == "__main__":
    main()
