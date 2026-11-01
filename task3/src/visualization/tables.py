"""Table builder for Task 3 required evidence tables (T00 through T17)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from task3.src.utils.common import setup_logger

logger = setup_logger("task3.visualization.tables")


def save_table(df: pd.DataFrame, table_id: str, title: str, output_dir: Path) -> Path:
    """Saves DataFrame as CSV and Markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{table_id}_{title}.csv"
    md_path = output_dir / f"{table_id}_{title}.md"

    df.to_csv(csv_path, index=False)
    try:
        df.to_markdown(md_path, index=False)
    except Exception:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

    logger.info(f"Saved {table_id} ({title}) to {csv_path}")
    return csv_path


class TableBuilder:
    """Generates all mandatory tables T00 through T17."""

    def __init__(self, tables_dir: str | Path = "task3/results/tables"):
        self.tables_dir = Path(tables_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def build_t00_preflight(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T00", "source_data_preflight_and_validation", self.tables_dir)
        return df

    def build_t01_splits(self, split_records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(split_records)
        save_table(df, "T01", "split_manifest_summary", self.tables_dir)
        return df

    def build_t02_normalization_audit(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T02", "scorer_and_answer_normalization_audit", self.tables_dir)
        return df

    def build_t03_first_ans_oracle_diversity(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T03", "first_ans_oracle_coverage_and_diversity", self.tables_dir)
        return df

    def build_t04_offline_methods(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T04", "offline_method_comparison", self.tables_dir)
        return df

    def build_t05_bootstrap_diffs(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T05", "main_paired_accuracy_differences_and_bootstrap_intervals", self.tables_dir)
        return df

    def build_t06_signal_length(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T06", "signal_length_correctness_analysis", self.tables_dir)
        return df

    def build_t07_confidence_vision(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T07", "mathvista_confidence_vision_relationship", self.tables_dir)
        return df

    def build_t08_fusion(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T08", "tokenwise_versus_trajectory_fusion", self.tables_dir)
        return df

    def build_t09_live_questions(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T09", "live_generation_question_level_results", self.tables_dir)
        return df

    def build_t10_live_summary(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T10", "live_strategy_summary", self.tables_dir)
        return df

    def build_t11_transitions(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T11", "refinement_transition_matrix", self.tables_dir)
        return df

    def build_t12_adaptive_grid(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T12", "adaptive_development_grid_and_selected_thresholds", self.tables_dir)
        return df

    def build_t13_adaptive_held_out(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T13", "adaptive_held_out_results", self.tables_dir)
        return df

    def build_t14_premature_stops(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T14", "premature_stop_and_recovery_analysis", self.tables_dir)
        return df

    def build_t15_qualitative_index(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T15", "qualitative_case_index", self.tables_dir)
        return df

    def build_t16_resources(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T16", "resource_and_run_inventory", self.tables_dir)
        return df

    def build_t17_audit(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        save_table(df, "T17", "evidence_completeness_audit", self.tables_dir)
        return df


# Convenient module-level functional builders
def build_table_t00(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t01(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t02(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t03(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t04(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t05(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t06(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t07(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t08(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t09(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t10(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t11(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t12(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t13(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t14(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t15(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t16(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)
def build_table_t17(records: List[Dict[str, Any]]) -> pd.DataFrame: return pd.DataFrame(records)

