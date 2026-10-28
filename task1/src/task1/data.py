"""Data loading, deterministic text normalization, GSM8K test leakage filtering, and cache builder."""

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datasets import load_dataset
from tqdm import tqdm

from .formatting import tokenize_training_example

logger = logging.getLogger("task1.data")


def normalize_text(text: str) -> str:
    """
    Deterministic text normalization for leakage detection and comparison:
    1. Unicode NFKC normalization
    2. Lowercase conversion
    3. Whitespace normalization (collapsing all consecutive whitespace to a single space, stripping)
    4. Standardizing curly quotes and dashes
    """
    if not text:
        return ""
    # Unicode NFKC
    norm = unicodedata.normalize("NFKC", text)
    # Lowercase
    norm = norm.lower()
    # Normalize common quotation marks and dashes
    norm = norm.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    norm = norm.replace("—", "-").replace("–", "-")
    # Collapse whitespace
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def load_gsm8k_test_questions(
    dataset_name: str = "gsm8k",
    config_name: str = "main",
    split: str = "test",
) -> Set[str]:
    """
    Loads GSM8K test questions and returns a set of normalized question strings.
    """
    logger.info(f"Loading GSM8K {split} split from {dataset_name}:{config_name}...")
    ds = load_dataset(dataset_name, config_name, split=split)
    normalized_questions = set()
    for row in ds:
        q = row.get("question", "")
        norm_q = normalize_text(q)
        if norm_q:
            normalized_questions.add(norm_q)
    logger.info(f"Loaded {len(normalized_questions)} normalized GSM8K test questions for leakage filtering.")
    return normalized_questions


def prepare_metamathqa_subset(
    tokenizer: Any,
    source_dataset_name: str = "meta-math/MetaMathQA",
    target_count: int = 40000,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
    cache_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scans MetaMathQA in preserved source order, applying eligibility filters:
    1. Empty query or response exclusion.
    2. GSM8K test contamination exclusion (checking both query and original_question).
    3. Sequence length <= 2048 formatted tokens (no truncation).
    4. Selects first target_count (40,000) eligible examples.

    Returns:
        examples: List of 40,000 filtered, tokenized examples.
        manifest: Dictionary with filter statistics, hashes, and audit counts.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / f"metamathqa_filtered_{target_count}.jsonl"
        manifest_file = cache_dir / f"metamathqa_filtered_{target_count}_manifest.json"

        if not force_rebuild and cached_file.exists() and manifest_file.exists():
            logger.info(f"Loading cached filtered dataset from {cached_file}...")
            examples = []
            with cached_file.open("r", encoding="utf-8") as f:
                for line in f:
                    examples.append(json.loads(line))
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if len(examples) == target_count:
                logger.info(f"Successfully loaded {len(examples)} cached examples.")
                return examples, manifest
            else:
                logger.warning(f"Cached example count {len(examples)} != target {target_count}. Rebuilding...")

    # Load GSM8K test leakage set
    gsm8k_test_set = load_gsm8k_test_questions()

    # Load MetaMathQA
    logger.info(f"Streaming/loading {source_dataset_name} in source order...")
    ds = load_dataset(source_dataset_name, split="train")

    selected_examples: List[Dict[str, Any]] = []
    stats = {
        "scanned_total": 0,
        "excluded_empty": 0,
        "excluded_gsm8k_leakage": 0,
        "excluded_overlength": 0,
        "selected_total": 0,
        "target_count": target_count,
    }

    pbar = tqdm(total=target_count, desc="Filtering MetaMathQA")

    for idx, row in enumerate(ds):
        stats["scanned_total"] += 1

        query = row.get("query", "")
        response = row.get("response", "")
        orig_q = row.get("original_question", "")
        q_type = row.get("type", "")

        # 1. Non-empty check
        if not query or not response or not query.strip() or not response.strip():
            stats["excluded_empty"] += 1
            continue

        # 2. Leakage check
        norm_query = normalize_text(query)
        norm_orig = normalize_text(orig_q) if orig_q else ""
        if norm_query in gsm8k_test_set or (norm_orig and norm_orig in gsm8k_test_set):
            stats["excluded_gsm8k_leakage"] += 1
            continue

        # 3. Formatted tokenization & length cap check (no truncation)
        tokenized = tokenize_training_example(
            tokenizer=tokenizer,
            question=query,
            response=response,
            max_seq_length=max_seq_length,
            enable_thinking=enable_thinking,
        )
        if tokenized is None:
            stats["excluded_overlength"] += 1
            continue

        # Retained example record
        record = {
            "source_idx": idx,
            "query": query,
            "response": response,
            "original_question": orig_q,
            "type": q_type,
            "seq_len": tokenized["seq_len"],
            "prompt_len": tokenized["prompt_len"],
            "num_supervised_tokens": tokenized["num_supervised_tokens"],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
        }
        selected_examples.append(record)
        stats["selected_total"] += 1
        pbar.update(1)

        if len(selected_examples) >= target_count:
            break

    pbar.close()

    if len(selected_examples) < target_count:
        raise RuntimeError(
            f"Insufficient eligible training examples: collected {len(selected_examples)}, required {target_count}!"
        )

    # Compute manifest hash
    hasher = hashlib.sha256()
    for ex in selected_examples:
        hasher.update(str(ex["source_idx"]).encode("utf-8"))
        hasher.update(ex["query"].encode("utf-8"))
    data_hash = hasher.hexdigest()

    manifest = {
        "source_dataset": source_dataset_name,
        "target_count": target_count,
        "max_seq_length": max_seq_length,
        "seed": 6304,
        "data_sha256": data_hash,
        "filter_statistics": stats,
    }

    # Save cache if requested
    if cache_dir is not None:
        logger.info(f"Saving filtered dataset to {cached_file}...")
        with cached_file.open("w", encoding="utf-8") as f:
            for ex in selected_examples:
                f.write(json.dumps(ex) + "\n")
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Dataset cache successfully saved.")

    return selected_examples, manifest

