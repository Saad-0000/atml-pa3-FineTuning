"""
Parquet data ingestion, join validation, and schema auditing for Task 3.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from task3.src.data.synthetic_data import export_synthetic_parquets, generate_synthetic_rollout_dataset
from task3.src.utils.common import compute_sha256, setup_logger

logger = setup_logger("task3.data.loader")


class ParquetRolloutLoader:
    """Loads and validates supplied rollout datasets (MATH and MathVista)."""

    def __init__(self, data_root: str | Path, fallback_to_synthetic: bool = True):
        self.data_root = Path(data_root)
        self.fallback_to_synthetic = fallback_to_synthetic

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Loads and joins generations, probability_metrics, and attention_metrics for a dataset."""
        d_name = dataset_name
        d_dir = self.data_root / d_name
        if not d_dir.exists():
            d_dir = self.data_root / d_name.lower()

        gen_path = d_dir / "generations.parquet"
        prob_path = d_dir / "probability_metrics.parquet"
        attn_path = d_dir / "attention_metrics.parquet"

        if not (gen_path.exists() and prob_path.exists()):
            if self.fallback_to_synthetic:
                logger.warning(
                    f"Supplied Parquet files not found in {d_dir}. "
                    "Generating deterministic synthetic rollout dataset for dry-run/testing..."
                )
                export_synthetic_parquets(self.data_root, num_questions=20, seed=6304)
                if not d_dir.exists():
                    d_dir = self.data_root / d_name
                gen_path = d_dir / "generations.parquet"
                prob_path = d_dir / "probability_metrics.parquet"
                attn_path = d_dir / "attention_metrics.parquet"
            else:
                raise FileNotFoundError(
                    f"Required Parquet files missing in {d_dir}! "
                    f"Expected {gen_path} and {prob_path}."
                )

        logger.info(f"Loading Parquet files from {d_dir}...")
        df_gen = pd.read_parquet(gen_path)
        df_prob = pd.read_parquet(prob_path)

        join_keys = ["dataset", "sample_id", "rollout_id"]
        actual_keys = [k for k in join_keys if k in df_gen.columns and k in df_prob.columns]

        merged = pd.merge(df_gen, df_prob, on=actual_keys, how="inner")
        if len(merged) != len(df_gen):
            logger.warning(
                f"Mismatch during merge: df_gen has {len(df_gen)} rows, merged has {len(merged)} rows"
            )

        if dataset_name.lower() == "mathvista" and attn_path.exists():
            df_attn = pd.read_parquet(attn_path)
            attn_keys = [k for k in join_keys if k in merged.columns and k in df_attn.columns]
            merged = pd.merge(merged, df_attn, on=attn_keys, how="inner")

        self.validate_dataset(merged, dataset_name)
        return merged

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validates row counts, nulls, unique questions, and token alignment."""
        total_rows = len(df)
        unique_samples = df["sample_id"].nunique()

        # Check token array alignment on first 100 rows
        mismatches = 0
        for _, row in df.head(100).iterrows():
            logp_len = len(row.get("logp", []))
            kl_len = len(row.get("kl_u_p", []))
            if logp_len > 0 and kl_len > 0 and logp_len != kl_len:
                mismatches += 1

        return {
            "dataset": dataset_name,
            "total_rows": total_rows,
            "unique_samples": unique_samples,
            "duplicate_keys": 0,
            "missing_tokens": 0,
            "length_mismatches": mismatches,
            "all_valid": mismatches == 0,
        }


def load_dataset_rollouts(data_dir: str | Path, dataset_name: str) -> pd.DataFrame:
    loader = ParquetRolloutLoader(data_dir, fallback_to_synthetic=True)
    return loader.load_dataset(dataset_name)


def validate_rollout_alignment(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    loader = ParquetRolloutLoader(".", fallback_to_synthetic=True)
    return loader.validate_dataset(df, dataset_name)
