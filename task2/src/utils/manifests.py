"""Evidence completeness audit and dictionary manifest builders for Task 2.

Assignment Section 14 & 15:
- dictionary_manifest.json: describes all saved ID maps, pair maps, canonical targets, etc.
- evidence_manifest.json: validates every Required Evidence bullet with exact status:
  PASS, FAIL, INCOMPLETE, BLOCKED, or NOT_RUN_OPTIONAL.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def build_dictionary_manifest(task2_dir: str | Path) -> Dict[str, Any]:
    """Catalogs all persistent metadata, split, and mapping dictionaries."""
    base_dir = Path(task2_dir)
    data_dir = base_dir / "results" / "data"

    dictionaries = {
        "split_image_ids": {
            "path": "results/data/split_image_ids.json",
            "description": "Mutually disjoint image ID sets for train2014, dev, and test.",
            "format": "json",
            "schema_version": "1.0",
        },
        "selected_example_ids": {
            "path": "results/data/selected_example_ids.json",
            "description": "IDs for 18,000 train captions, 96 dev captions, and QA examples.",
            "format": "json",
            "schema_version": "1.0",
        },
        "pair_inventory": {
            "path": "results/data/pair_inventory.csv",
            "description": "Complete inventory of all grounding and VQAv2 complementary pairs.",
            "format": "csv",
            "schema_version": "1.0",
        },
        "mismatched_image_mapping": {
            "path": "results/data/mismatched_image_mapping.json",
            "description": "Fixed deterministic mapping for the Mismatched image evaluation condition.",
            "format": "json",
            "schema_version": "1.0",
        },
        "preprocessing_and_normalization": {
            "path": "results/data/preprocessing_and_normalization.json",
            "description": "SigLIP resolution, normalization constants, and VQA text normalizer rules.",
            "format": "json",
            "schema_version": "1.0",
        },
    }

    manifest = {
        "schema_version": "1.0",
        "dictionaries": {},
    }

    for name, info in dictionaries.items():
        rel_path = info["path"]
        abs_path = base_dir / rel_path
        exists = abs_path.exists()
        manifest["dictionaries"][name] = {
            **info,
            "exists": exists,
            "sha256": compute_file_sha256(abs_path) if exists else None,
            "size_bytes": abs_path.stat().st_size if exists else 0,
        }

    out_file = base_dir / "manifests" / "dictionary_manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dictionary manifest saved to {out_file}")
    return manifest


def audit_evidence_completeness(
    task2_dir: str | Path,
    repo_commit: str = "HEAD",
    config_hash: str = "",
) -> Dict[str, Any]:
    """Audits every Required Evidence bullet from the assignment."""
    base_dir = Path(task2_dir)
    results_dir = base_dir / "results"

    evidence_items = [
        {
            "requirement_id": "RE-01",
            "title": "Main Training Loss Curve (Stage Boundaries at 2250, 3750)",
            "output_files": ["results/plots/P01_training_loss.png", "results/plots/P01_training_loss_data.csv"],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-02",
            "title": "Development Caption Reference Loss Curve",
            "output_files": ["results/plots/P02_development_caption_loss.png", "results/plots/P02_development_caption_loss_data.csv"],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-03",
            "title": "Development Answer NLL Curves Across 3 Conditions",
            "output_files": ["results/plots/P03_development_answer_nll.png", "results/plots/P03_development_answer_nll_data.csv"],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-04",
            "title": "Development Image-Swap Penalty Curves (DeltaNLL)",
            "output_files": ["results/plots/P04_development_image_swap_penalty.png", "results/plots/P04_development_image_swap_penalty_data.csv"],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-05",
            "title": "Development Checkpoint Evaluation Table (T06)",
            "output_files": ["results/tables/T06_development_evaluation_by_checkpoint.csv"],
            "script": "task2/scripts/evaluate_dev.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-06",
            "title": "Final Grounding Results Table and Plots (4 Conditions, 95% CIs)",
            "output_files": [
                "results/tables/T07_final_grounding_results.csv",
                "results/plots/P05_final_grounding_accuracy_by_condition.png",
                "results/plots/P07_grounding_pair_diagnostics.png",
            ],
            "script": "task2/scripts/evaluate_final.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-07",
            "title": "Final Natural VQAv2 Results Table and Plots (4 Conditions, 95% CIs)",
            "output_files": [
                "results/tables/T08_final_vqav2_results.csv",
                "results/plots/P06_final_vqav2_accuracy_by_condition.png",
                "results/plots/P08_vqav2_pair_diagnostics.png",
            ],
            "script": "task2/scripts/evaluate_final.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-08",
            "title": "Correct-Image Advantage Over Swaps (Table T09, Plot P10)",
            "output_files": [
                "results/tables/T09_correct_image_advantage_bootstrap.csv",
                "results/plots/P10_correct_image_advantages.png",
            ],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-09",
            "title": "Qualitative Aligned Complementary Inspection Sheet & Manifest",
            "output_files": [
                "results/qualitative/qualitative_example_manifest.csv",
                "results/qualitative/human_review_sheet.md",
                "results/qualitative/aligned_complementary_examples.html",
            ],
            "script": "task2/scripts/run_qualitative.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-10",
            "title": "Resource and Computational Cost Comparison (T10, P12)",
            "output_files": [
                "results/tables/T10_resource_comparison.csv",
                "results/plots/P12_resource_usage.png",
            ],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-11",
            "title": "Dataset Inventory & Filtering Audit (T01, T02)",
            "output_files": [
                "results/tables/T01_dataset_split_inventory.csv",
                "results/tables/T02_filtering_pair_selection_audit.csv",
            ],
            "script": "task2/scripts/prepare_data.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-12",
            "title": "Model/Interface Verification & Scale Calibration (T03)",
            "output_files": ["results/tables/T03_model_interface_verification.csv"],
            "script": "task2/scripts/preflight.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-13",
            "title": "Training Configuration and Realized Mixture Counts (T04, T05, P11)",
            "output_files": [
                "results/tables/T04_training_configuration.csv",
                "results/tables/T05_stage_sampling_mixture.csv",
                "results/plots/P11_training_mixture_composition.png",
            ],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-14",
            "title": "Checkpoint and Recovery Audit (T11)",
            "output_files": ["results/tables/T11_checkpoint_recovery_audit.csv"],
            "script": "task2/scripts/build_results.py",
            "mandatory": True,
        },
        {
            "requirement_id": "RE-15",
            "title": "Optional 49-Token Visual Compression Comparison",
            "output_files": ["results/tables/T13_optional_compression_comparison.csv"],
            "script": "task2/scripts/optional_compression.py",
            "mandatory": False,
        },
    ]

    audit_records = []
    overall_status = "PASS"

    for item in evidence_items:
        paths = [base_dir / p for p in item["output_files"]]
        all_exist = all(p.exists() and p.stat().st_size > 0 for p in paths)

        if all_exist:
            status = "PASS"
        elif not item["mandatory"]:
            status = "NOT_RUN_OPTIONAL"
        else:
            status = "INCOMPLETE"
            overall_status = "INCOMPLETE"

        audit_records.append({
            "requirement_id": item["requirement_id"],
            "title": item["title"],
            "status": status,
            "mandatory": item["mandatory"],
            "output_files": item["output_files"],
            "script": item["script"],
        })

    audit_result = {
        "repo_commit": repo_commit,
        "config_hash": config_hash,
        "overall_status": overall_status,
        "items": audit_records,
    }

    out_file = base_dir / "results" / "evidence_manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    logger.info(f"Evidence completeness audit completed with overall status: {overall_status}")
    return audit_result
