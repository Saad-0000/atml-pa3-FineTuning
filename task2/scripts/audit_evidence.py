"""Evidence completeness audit runner and dictionary manifest builder for Task 2.

Usage:
    python -m task2.scripts.audit_evidence --config task2/configs/task2.yaml
"""
import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from task2.src.config import load_task2_config
from task2.src.utils.logging_utils import setup_logger
from task2.src.utils.manifests import audit_evidence_completeness, build_dictionary_manifest
from task2.src.visualization.tables import TableBuilder

logger = setup_logger("audit_evidence")


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Evidence Audit")
    parser.add_argument(
        "--config",
        type=str,
        default="task2/configs/task2.yaml",
        help="Path to task2.yaml configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_task2_config(args.config)
    task2_root = Path(__file__).resolve().parent.parent

    logger.info("==================================================")
    logger.info("      Task 2 Evidence Completeness Audit          ")
    logger.info("==================================================")

    # 1. Build Dictionary Manifest
    logger.info("Building dictionary manifest...")
    dict_manifest = build_dictionary_manifest(task2_root)

    # 2. Run Evidence Audit
    logger.info("Auditing Required Evidence items...")
    evidence_report = audit_evidence_completeness(task2_root)

    # 3. Build Table T12
    table_builder = TableBuilder(config.paths.results_dir)
    t12_rows = []
    for item in evidence_report["items"]:
        t12_rows.append({
            "requirement_id": item["requirement_id"],
            "title": item["title"],
            "mandatory": item["mandatory"],
            "script": item["script"],
            "status": item["status"],
        })

    table_builder.build_t12_evidence_manifest(t12_rows)

    logger.info("==================================================")
    logger.info(f"Audit Complete! Overall Status: {evidence_report['overall_status']}")
    for item in evidence_report["items"]:
        logger.info(f"  [{item['status']}] {item['requirement_id']}: {item['title']}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
