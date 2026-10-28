"""Result table builders, plot generators, and evidence manifest generator."""

import csv
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("task1.results")


def export_environment_metadata(output_path: Path | str) -> Dict[str, Any]:
    """Captures software versions, hardware specs, and runtime details."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except ImportError:
        meta["transformers_version"] = "not_installed"

    try:
        import peft
        meta["peft_version"] = peft.__version__
    except ImportError:
        meta["peft_version"] = "not_installed"

    try:
        import datasets
        meta["datasets_version"] = datasets.__version__
    except ImportError:
        meta["datasets_version"] = "not_installed"

    try:
        import lm_eval
        meta["lm_eval_version"] = getattr(lm_eval, "__version__", "installed")
    except ImportError:
        meta["lm_eval_version"] = "not_installed"

    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Exported environment metadata to {p}.")
    return meta


def build_evidence_manifest(
    results_dir: Path | str,
    status_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Constructs the 5 top-level Required Evidence items according to the assignment:
    1. Benchmark results
    2. Learning and regularization
    3. Resources
    4. Diagnostics
    5. Qualitative findings
    """
    res_p = Path(results_dir)
    status_overrides = status_overrides or {}

    def get_status(target_file: Path, fallback_key: str) -> str:
        if fallback_key in status_overrides:
            return status_overrides[fallback_key]
        return "complete" if target_file.exists() and target_file.stat().st_size > 0 else "blocked"

    items = [
        {
            "category": "Benchmark results",
            "description": "One table covering all seven evaluation conditions and all four benchmarks, with primary scores and changes from baseline in percentage points.",
            "target_files": [
                str(res_p / "tables/benchmark_results_all_conditions.csv"),
                str(res_p / "tables/benchmark_results_all_conditions.md"),
                str(res_p / "plots/benchmark_comparison_all_conditions.png"),
            ],
            "generation_command": "python scripts/build_results.py",
            "status": get_status(res_p / "tables/benchmark_results_all_conditions.csv", "benchmarks"),
        },
        {
            "category": "Learning and regularization",
            "description": "Combined SFT-loss plot for the five runs, with a concise comparison of Full FT and L2-SP penalty strength and final parameter displacement.",
            "target_files": [
                str(res_p / "plots/training_sft_loss_all_methods.png"),
                str(res_p / "plots/l2sp_objective_components.png"),
                str(res_p / "tables/l2sp_and_displacement.csv"),
            ],
            "generation_command": "python scripts/build_results.py",
            "status": get_status(res_p / "tables/l2sp_and_displacement.csv", "learning"),
        },
        {
            "category": "Resources",
            "description": "Compact comparison of trainable parameters, training time, peak GPU memory, and checkpoint sizes for the five runs.",
            "target_files": [
                str(res_p / "tables/resource_comparison.csv"),
                str(res_p / "plots/resource_comparison.png"),
            ],
            "generation_command": "python scripts/build_results.py",
            "status": get_status(res_p / "tables/resource_comparison.csv", "resources"),
        },
        {
            "category": "Diagnostics",
            "description": "Compact figure or table summarizing subspace interference, relative update magnitude, and math/non-math activation corrections.",
            "target_files": [
                str(res_p / "tables/weight_space_diagnostics.csv"),
                str(res_p / "tables/activation_diagnostics_summary.csv"),
                str(res_p / "plots/weight_space_diagnostics.png"),
                str(res_p / "plots/activation_corrections.png"),
            ],
            "generation_command": "python scripts/run_diagnostics.py",
            "status": get_status(res_p / "tables/weight_space_diagnostics.csv", "diagnostics"),
        },
        {
            "category": "Qualitative findings",
            "description": "Summary of the fixed 42-response annotation subset and aligned excerpts comparing the same question and format across models.",
            "target_files": [
                str(res_p / "tables/qualitative_42_annotations.csv"),
                str(res_p / "qualitative/all_168_responses.jsonl"),
            ],
            "generation_command": "python scripts/run_qualitative.py",
            "status": get_status(res_p / "tables/qualitative_42_annotations.csv", "qualitative"),
        },
    ]

    manifest = {
        "assignment": "ATML Fall 2026 PA3 Task 1",
        "evidence_categories": items,
    }

    with open(res_p / "evidence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Write checklist CSV
    with open(res_p / "tables/evidence_checklist.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "status", "generation_command", "target_files"])
        writer.writeheader()
        for it in items:
            writer.writerow({
                "category": it["category"],
                "status": it["status"],
                "generation_command": it["generation_command"],
                "target_files": "; ".join(it["target_files"]),
            })

    return manifest

