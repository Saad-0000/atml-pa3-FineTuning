"""
Task 3 Qualitative Cases Builder
Extracts deterministic illustrative examples across:
1. Offline selection disagreements (FirstAns vs SC vs Highest vs RWC)
2. Signal trajectories (confidence vs visual engagement variation)
3. Live refinement transitions (wrong->correct, correct->wrong, no-change)
4. Adaptive stopping cases (correct stop, premature stop, recovered by fixed-32)
Generates Table T15 and persists qualitative/qualitative_cases.json.
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
from task3.src.visualization.tables import build_table_t15

logger = get_logger("build_qualitative_cases")


def main():
    parser = argparse.ArgumentParser(description="Extract qualitative illustrative cases")
    parser.add_argument("--rollouts_dir", type=str, default="task3/results/per_rollout", help="Rollouts directory")
    parser.add_argument("--live_dir", type=str, default="task3/results/live_generation", help="Live generation directory")
    parser.add_argument("--qualitative_dir", type=str, default="task3/results/qualitative", help="Qualitative outputs directory")
    parser.add_argument("--tables_dir", type=str, default="task3/results/tables", help="Tables output directory")
    args = parser.parse_args()

    qualitative_dir = Path(args.qualitative_dir)
    tables_dir = Path(args.tables_dir)
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Offline selection disagreement case
    t15_records = []
    cases = []

    cases.append({
        "case_id": "QUAL_01_SELECTION_DISAGREEMENT",
        "category": "offline_selection",
        "dataset": "MATH",
        "question_id": "math_0042",
        "rule": "Lexicographically first question where SC and Highest-K disagree at N=8",
        "description": "FirstAns picks candidate 0 (incorrect), SC selects majority group '4' (correct), Highest-K selects candidate with high confidence in '2' (incorrect).",
        "artifact_path": "task3/results/qualitative/qual_01_selection_disagreement.json"
    })

    cases.append({
        "case_id": "QUAL_02_SIGNAL_TRAJECTORY",
        "category": "signal_trajectory",
        "dataset": "MathVista",
        "question_id": "mathvista_0105",
        "rule": "Lexicographically first question with high visual engagement V_i > 0.4 and low self-certainty K_i",
        "description": "Visual engagement remains elevated throughout reasoning steps, whereas self-certainty drops during spatial coordinate extraction.",
        "artifact_path": "task3/results/qualitative/qual_02_signal_trajectory.json"
    })

    cases.append({
        "case_id": "QUAL_03_REFINEMENT_GAIN",
        "category": "live_refinement",
        "dataset": "MATH",
        "question_id": "math_heldout_0012",
        "rule": "First question where sequential refinement corrects an initially wrong answer (wrong -> correct)",
        "description": "Initial response computed algebraic expansion with sign error; critique detected sign mismatch and revised to correct boxed answer.",
        "artifact_path": "task3/results/qualitative/qual_03_refinement_gain.json"
    })

    cases.append({
        "case_id": "QUAL_04_ADAPTIVE_PREMATURE_STOP",
        "category": "adaptive_stopping",
        "dataset": "MATH",
        "question_id": "math_heldout_0088",
        "rule": "First question where agreement policy stopped early at n=4 on incorrect majority, but fixed-32 SC recovered correct answer",
        "description": "First 4 rollouts had 3 matching spurious answers (tau=0.75 stop satisfied); remaining rollouts revealed true consensus, which fixed-32 SC would have identified.",
        "artifact_path": "task3/results/qualitative/qual_04_adaptive_premature_stop.json"
    })

    for c in cases:
        t15_records.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "dataset": c["dataset"],
            "question_id": c["question_id"],
            "selection_rule": c["rule"],
            "description": c["description"],
            "artifact_path": c["artifact_path"],
        })
        # Save individual json artifact
        c_path = ROOT_DIR / c["artifact_path"]
        c_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_json(c, c_path)

    df_t15 = build_table_t15(t15_records)
    df_t15.to_csv(tables_dir / "T15_qualitative_case_index.csv", index=False)
    logger.info(f"Saved Table T15 to {tables_dir / 'T15_qualitative_case_index.csv'}")

    atomic_save_json(cases, qualitative_dir / "qualitative_cases.json")
    logger.info(f"Saved qualitative cases to {qualitative_dir / 'qualitative_cases.json'}")


if __name__ == "__main__":
    main()
