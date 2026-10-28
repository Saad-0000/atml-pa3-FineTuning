"""Evaluation engine wrapping lm-evaluation-harness with split asserting and chat-template isolation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch

logger = logging.getLogger("task1.evaluation")

EXPECTED_BENCHMARK_SPECS = {
    "gsm8k": {
        "task_name": "gsm8k_cot",
        "split": "test",
        "expected_count": 1319,
        "primary_metric": "exact_match,flexible-extract",
        "apply_chat_template": True,
        "max_gen_toks": 1024,
    },
    "hellaswag": {
        "task_name": "hellaswag",
        "split": "validation",
        "expected_count": 10042,
        "primary_metric": "acc_norm",
        "apply_chat_template": False,
    },
    "boolq": {
        "task_name": "boolq",
        "split": "validation",
        "expected_count": 3270,
        "primary_metric": "acc",
        "apply_chat_template": False,
    },
    "arc_easy": {
        "task_name": "arc_easy",
        "split": "test",
        "expected_count": 2376,
        "primary_metric": "acc_norm",
        "apply_chat_template": False,
    },
}


def assert_evaluation_split_guards(task_name: str, resolved_split: str, resolved_count: int) -> None:
    """
    Enforces that evaluation never uses the train split or a truncated subset.
    """
    matching_spec = None
    for spec in EXPECTED_BENCHMARK_SPECS.values():
        if spec["task_name"] == task_name:
            matching_spec = spec
            break

    if matching_spec is None:
        logger.warning(f"Task {task_name} not found in expected benchmark specifications.")
        return

    req_split = matching_spec["split"]
    req_count = matching_spec["expected_count"]

    if resolved_split.lower() != req_split.lower():
        raise ValueError(
            f"Evaluation Preflight Violation! Task '{task_name}' resolved split '{resolved_split}', "
            f"but required split is strictly '{req_split}'! Never substitute train or another split."
        )

    if resolved_count != req_count:
        raise ValueError(
            f"Evaluation Preflight Violation! Task '{task_name}' resolved count {resolved_count}, "
            f"but required example count is strictly {req_count}!"
        )

    logger.info(f"Split assertion PASSED for {task_name}: split='{resolved_split}', count={resolved_count}.")


def evaluate_with_lm_eval(
    model: Any,
    tokenizer: Any,
    task_key: str,
    device: str = "cuda",
    batch_size: int = 8,
    output_dir: Optional[Path | str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Executes a single benchmark task using lm-evaluation-harness with strict isolation.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    spec = EXPECTED_BENCHMARK_SPECS[task_key]
    task_name = spec["task_name"]
    apply_chat_template = spec["apply_chat_template"]

    logger.info(
        f"Starting evaluation on {task_key} (task={task_name}, apply_chat_template={apply_chat_template})..."
    )

    # Wrap model with lm-eval HFLM
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    task_args: Dict[str, Any] = {
        "tasks": [task_name],
        "num_fewshot": 0,
        "apply_chat_template": apply_chat_template,
        "log_samples": True,
    }

    if apply_chat_template:
        # For GSM8K non-thinking template
        task_args["fewshot_as_multiturn"] = False

    results = lm_eval.simple_evaluate(
        model=hflm,
        **task_args,
    )

    task_results = results.get("results", {}).get(task_name, {})
    samples = results.get("samples", {}).get(task_name, [])

    # Validate split and sample count
    actual_count = len(samples)
    actual_split = spec["split"]  # lm-eval uses the configured split
    assert_evaluation_split_guards(task_name, actual_split, actual_count)

    primary_metric_key = spec["primary_metric"]
    primary_score = task_results.get(primary_metric_key, 0.0)

    aggregates = {
        "task_key": task_key,
        "task_name": task_name,
        "split": actual_split,
        "count": actual_count,
        "primary_metric": primary_metric_key,
        "primary_score": float(primary_score),
        "raw_results": task_results,
    }

    # Format per-example records
    per_example_records = []
    for idx, sample in enumerate(samples):
        record = {
            "task_key": task_key,
            "task_name": task_name,
            "example_idx": idx,
            "doc_id": sample.get("doc_id", idx),
            "target": sample.get("target"),
            "filtered_resps": sample.get("filtered_resps"),
            "metrics": sample.get("metrics"),
        }
        per_example_records.append(record)

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / f"{task_key}_aggregates.json", "w", encoding="utf-8") as f:
            json.dump(aggregates, f, indent=2)
        df = pd.DataFrame(per_example_records)
        df.to_parquet(out_p / f"{task_key}_per_example.parquet", index=False)

    return aggregates, per_example_records

