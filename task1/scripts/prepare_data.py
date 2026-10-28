#!/usr/bin/env python3
"""
Data preparation script for Task 1:
- Filters MetaMathQA in preserved source order (non-empty, <=2048 formatted tokens, no GSM8K test leakage).
- Selects first 40,000 eligible examples.
- Samples fixed 64 activation probes (32 GSM8K + 32 HellaSwag) using seed 6304.
- Persists data and probe manifests to persistent storage.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK1_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK1_ROOT / "src"))

from task1.config import get_default_configs
from task1.data import prepare_metamathqa_subset
from task1.model_loader import load_tokenizer
from task1.probes import extract_probe_inputs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare filtered training dataset and probe sets")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of filtered cache")
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)

    logger.info("Initializing tokenizer for chat-template length checks...")
    tokenizer = load_tokenizer(
        model_name_or_path=t1_cfg.model_name,
        revision=t1_cfg.model_revision,
        enable_thinking=t1_cfg.enable_thinking,
    )

    data_cache_dir = t1_cfg.artifacts_dir / "data_cache"
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter MetaMathQA (first 40,000 eligible)
    logger.info("Building 40,000 eligible MetaMathQA dataset...")
    examples, data_manifest = prepare_metamathqa_subset(
        tokenizer=tokenizer,
        source_dataset_name=t1_cfg.source_dataset,
        target_count=t1_cfg.budget_examples,
        max_seq_length=t1_cfg.max_seq_length,
        enable_thinking=t1_cfg.enable_thinking,
        cache_dir=data_cache_dir,
        force_rebuild=args.force_rebuild,
    )
    logger.info(f"Dataset prepared successfully: {len(examples)} examples.")

    # 2. Extract activation probes (32 GSM8K + 32 HellaSwag)
    logger.info("Sampling 64 fixed activation probes with seed 6304...")
    probe_cache_file = t1_cfg.artifacts_dir / "probe_manifest.json"
    probes, probe_manifest = extract_probe_inputs(
        tokenizer=tokenizer,
        seed=t1_cfg.seed,
        cache_path=probe_cache_file,
    )
    logger.info(f"Probe manifest prepared: {len(probes)} probes.")

    # 3. Update dictionary manifest
    dict_manifest_file = t1_cfg.results_dir / "dictionary_manifest.json"
    dict_entries = []
    if dict_manifest_file.exists():
        try:
            with open(dict_manifest_file, "r", encoding="utf-8") as f:
                dict_entries = json.load(f).get("entries", [])
        except Exception:
            dict_entries = []

    # Add data manifest & probe manifest
    dict_entries = [e for e in dict_entries if e.get("name") not in ["metamathqa_manifest", "probe_manifest"]]
    dict_entries.append({
        "name": "metamathqa_manifest",
        "relative_path": "artifacts/data_cache/metamathqa_filtered_40000_manifest.json",
        "schema_version": "1.0",
        "format": "json",
        "purpose": "Filter statistics, SHA-256 hash, and eligibility counts for 40k MetaMathQA dataset",
    })
    dict_entries.append({
        "name": "probe_manifest",
        "relative_path": "artifacts/probe_manifest.json",
        "schema_version": "1.0",
        "format": "json",
        "purpose": "Exact prompt tokens, text, and hashes for 64 activation probes (seed 6304)",
    })

    with open(dict_manifest_file, "w", encoding="utf-8") as f:
        json.dump({"entries": dict_entries}, f, indent=2)

    logger.info(f"Dictionary manifest updated at {dict_manifest_file}.")
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
