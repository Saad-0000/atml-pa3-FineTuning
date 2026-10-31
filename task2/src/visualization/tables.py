"""Renderers for all 12 required tables (T01 - T12) in CSV and Markdown formats.

Assignment Section 8:
T01 - Dataset and split inventory
T02 - Data filtering and pair-selection audit
T03 - Model/interface verification
T04 - Training configuration and stage schedule
T05 - Stage sampling and mixture counts
T06 - Development evaluation by checkpoint
T07 - Final annotation-derived grounding results
T08 - Final natural VQAv2 results
T09 - Correct-image advantage and paired bootstrap comparisons
T10 - Resource and computational-cost comparison
T11 - Checkpoint and recovery audit
T12 - Evidence completeness and reproducibility manifest
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def df_to_markdown_string(df: pd.DataFrame) -> str:
    """Formats DataFrame as a GitHub-flavored Markdown table without external dependencies."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val).replace("\n", " ") for val in row.values) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows) + "\n"


def save_table_csv_and_markdown(
    df: pd.DataFrame,
    table_id: str,
    title: str,
    output_dir: str | Path,
) -> Tuple[Path, Path]:
    """Saves DataFrame as both CSV and Markdown table."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{table_id}_{title}.csv"
    md_path = out_dir / f"{table_id}_{title}.md"

    df.to_csv(csv_path, index=False)

    md_content = f"# {table_id}: {title.replace('_', ' ').title()}\n\n"
    md_content += df_to_markdown_string(df)
    md_content += "\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, md_path


class TableBuilder:
    """Builds and exports tables T01 - T12."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.tables_dir = self.results_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def build_t01_inventory(self, inventory_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """T01 - Dataset and split inventory."""
        df = pd.DataFrame(inventory_data)
        save_table_csv_and_markdown(df, "T01", "dataset_split_inventory", self.tables_dir)
        return df

    def build_t02_filtering_audit(self, audit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """T02 - Data filtering and pair-selection audit."""
        df = pd.DataFrame(audit_data)
        save_table_csv_and_markdown(df, "T02", "filtering_pair_selection_audit", self.tables_dir)
        return df

    def build_t03_model_verification(self, verification_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T03 - Model/interface verification."""
        df = pd.DataFrame(verification_rows)
        save_table_csv_and_markdown(df, "T03", "model_interface_verification", self.tables_dir)
        return df

    def build_t04_training_config(self, config_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T04 - Training configuration and stage schedule."""
        df = pd.DataFrame(config_rows)
        save_table_csv_and_markdown(df, "T04", "training_configuration", self.tables_dir)
        return df

    def build_t05_sampling_mixture(self, mixture_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T05 - Stage sampling and mixture counts."""
        df = pd.DataFrame(mixture_rows)
        save_table_csv_and_markdown(df, "T05", "stage_sampling_mixture", self.tables_dir)
        return df

    def build_t06_dev_evaluation(self, dev_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """T06 - Development evaluation by checkpoint."""
        df = pd.DataFrame(dev_records)
        save_table_csv_and_markdown(df, "T06", "development_evaluation_by_checkpoint", self.tables_dir)
        return df

    def build_t07_final_grounding(self, condition_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T07 - Final annotation-derived grounding results."""
        df = pd.DataFrame(condition_rows)
        save_table_csv_and_markdown(df, "T07", "final_grounding_results", self.tables_dir)
        return df

    def build_t08_final_vqav2(self, condition_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T08 - Final natural VQAv2 results."""
        df = pd.DataFrame(condition_rows)
        save_table_csv_and_markdown(df, "T08", "final_vqav2_results", self.tables_dir)
        return df

    def build_t09_correct_image_advantage(self, advantage_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T09 - Correct-image advantage and paired bootstrap comparisons."""
        df = pd.DataFrame(advantage_rows)
        save_table_csv_and_markdown(df, "T09", "correct_image_advantage_bootstrap", self.tables_dir)
        return df

    def build_t10_resources(self, resource_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T10 - Resource and computational-cost comparison."""
        df = pd.DataFrame(resource_rows)
        save_table_csv_and_markdown(df, "T10", "resource_comparison", self.tables_dir)
        return df

    def build_t11_checkpoint_audit(self, checkpoint_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T11 - Checkpoint and recovery audit."""
        df = pd.DataFrame(checkpoint_rows)
        save_table_csv_and_markdown(df, "T11", "checkpoint_recovery_audit", self.tables_dir)
        return df

    def build_t12_evidence_manifest(self, evidence_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """T12 - Evidence completeness and reproducibility manifest."""
        df = pd.DataFrame(evidence_rows)
        save_table_csv_and_markdown(df, "T12", "evidence_completeness", self.tables_dir)
        return df
