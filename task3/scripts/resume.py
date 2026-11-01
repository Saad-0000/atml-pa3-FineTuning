"""
Task 3 Pipeline Orchestrator and Crash Resume Script
Inspects completed steps and artifacts, resuming execution from the earliest uncompleted stage.
Enables crash-safe execution on Google Colab or local workstations.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task3.src.utils.common import get_logger

logger = get_logger("resume")


def run_stage(cmd: List[str], stage_name: str) -> bool:
    """Run a sub-script and log output."""
    logger.info(f">>> Running stage: {stage_name} ({' '.join(cmd)})")
    try:
        ret = subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
        logger.info(f"Stage '{stage_name}' completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage '{stage_name}' failed with return code {e.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Task 3 Crash Recovery and Pipeline Resume")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to supplied data")
    parser.add_argument("--dry_run", action="store_true", help="Run live generation in dry-run/mock mode")
    parser.add_argument("--force_all", action="store_true", help="Force rerun of all stages regardless of completed artifacts")
    args = parser.parse_args()

    task3_dir = Path("task3")
    manifests_dir = task3_dir / "manifests"
    results_dir = task3_dir / "results"
    tables_dir = results_dir / "tables"
    plots_dir = results_dir / "plots"
    rollouts_dir = results_dir / "per_rollout"

    python_exe = sys.executable

    stages = [
        # Stage 1: Preflight
        {
            "name": "Preflight Check",
            "artifact": task3_dir / "logs" / "preflight_summary.json",
            "cmd": [python_exe, "task3/scripts/preflight.py", "--data_dir", args.data_dir]
        },
        # Stage 2: Data Inspection
        {
            "name": "Data Inspection (T00)",
            "artifact": tables_dir / "T00_source_data_preflight.csv",
            "cmd": [python_exe, "task3/scripts/inspect_data.py", "--data_dir", args.data_dir]
        },
        # Stage 3: Split Manifests
        {
            "name": "Split Generation (T01)",
            "artifact": manifests_dir / "math_splits.json",
            "cmd": [python_exe, "task3/scripts/build_splits.py", "--data_dir", args.data_dir]
        },
        # Stage 4: Rollout Scoring & Normalization
        {
            "name": "Rollout Scoring (T02)",
            "artifact": rollouts_dir / "math_rollouts.parquet",
            "cmd": [python_exe, "task3/scripts/prepare_data.py", "--data_dir", args.data_dir]
        },
        # Stage 5: Offline Sampling (T03, T04, T05)
        {
            "name": "Offline Sampling (Exp A & B)",
            "artifact": tables_dir / "T04_offline_method_comparison.csv",
            "cmd": [python_exe, "task3/scripts/run_offline_sampling.py"]
        },
        # Stage 6: Signal Analysis (T06, T07, T08)
        {
            "name": "Signal Analysis (Exp C)",
            "artifact": tables_dir / "T08_fusion_comparison.csv",
            "cmd": [python_exe, "task3/scripts/run_signal_analysis.py"]
        },
        # Stage 7: Live Generation (T09, T10, T11)
        {
            "name": "Live Generation (50 MATH)",
            "artifact": tables_dir / "T10_live_strategy_summary.csv",
            "cmd": [python_exe, "task3/scripts/run_live_generation.py", "--data_dir", args.data_dir] + (["--dry_run"] if args.dry_run else [])
        },
        # Stage 8: Adaptive Dev Tuning (T12)
        {
            "name": "Adaptive Policy Tuning (Dev Split)",
            "artifact": manifests_dir / "adaptive_selected_policies.json",
            "cmd": [python_exe, "task3/scripts/run_adaptive_tuning.py"]
        },
        # Stage 9: Adaptive Held-out Test (T13, T14)
        {
            "name": "Adaptive Held-out Evaluation",
            "artifact": tables_dir / "T13_adaptive_heldout_results.csv",
            "cmd": [python_exe, "task3/scripts/run_adaptive_test.py"]
        },
        # Stage 10: Qualitative Cases (T15)
        {
            "name": "Qualitative Case Index",
            "artifact": tables_dir / "T15_qualitative_case_index.csv",
            "cmd": [python_exe, "task3/scripts/build_qualitative_cases.py"]
        },
        # Stage 11: Tables & Plots (T16, T17, P01-P16)
        {
            "name": "Tables and Figures Generation",
            "artifact": tables_dir / "T17_evidence_completeness_audit.csv",
            "cmd": [python_exe, "task3/scripts/make_tables.py"]
        },
        {
            "name": "Plot Rendering",
            "artifact": plots_dir / "P01_math_accuracy_oracle.png",
            "cmd": [python_exe, "task3/scripts/make_plots.py"]
        },
        # Stage 12: Final Evidence Audit
        {
            "name": "Evidence Manifest Audit",
            "artifact": results_dir / "EVIDENCE_AUDIT.md",
            "cmd": [python_exe, "task3/scripts/audit_evidence.py"]
        },
    ]

    logger.info("=" * 60)
    logger.info("Task 3 Execution & Resume Pipeline")
    logger.info("=" * 60)

    for st in stages:
        art = Path(st["artifact"])
        if art.exists() and not args.force_all:
            logger.info(f"Skipping completed stage: {st['name']} (Artifact found: {art})")
            continue

        success = run_stage(st["cmd"], st["name"])
        if not success:
            logger.error(f"Pipeline stopped at stage: {st['name']}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Task 3 Pipeline Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
