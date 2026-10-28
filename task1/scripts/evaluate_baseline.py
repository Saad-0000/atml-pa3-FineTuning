#!/usr/bin/env python3
"""
Baseline evaluation and asset caching script:
- Evaluates the unchanged starting model (Qwen/Qwen3-0.6B) once across all four benchmarks.
- Saves per-example results and aggregates under results/raw/baseline/.
- Caches base model down_proj activations z for the 64 probes (layers 0, 13, 27).
- Caches immutable theta0 reference weights for L2-SP and displacement.
- Precomputes and caches FP32 leading SVD bases (k=128) for OPLoRA.
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

import torch
from task1.config import get_default_configs
from task1.evaluation import evaluate_with_lm_eval
from task1.model_loader import load_base_model, load_tokenizer, save_theta0_reference
from task1.oplora import compute_svd_bases
from task1.probes import cache_base_model_activations, extract_probe_inputs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_baseline")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unchanged baseline model and cache reference assets")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-eval", action="store_true", help="Skip lm-eval benchmarks, cache assets only")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-evaluation of previously cached tasks")
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading base model {t1_cfg.model_name} on {device}...")
    tokenizer = load_tokenizer(
        model_name_or_path=t1_cfg.model_name,
        revision=t1_cfg.model_revision,
        enable_thinking=t1_cfg.enable_thinking,
    )
    model = load_base_model(
        model_name_or_path=t1_cfg.model_name,
        revision=t1_cfg.model_revision,
        torch_dtype=torch.float16,
        gradient_checkpointing=False,
        device=device,
    )

    # 1. Save immutable theta0 reference weights
    ref_path = t1_cfg.artifacts_dir / "theta0_reference" / "theta0.pt"
    if not ref_path.exists():
        logger.info(f"Saving immutable theta0 reference to {ref_path}...")
        save_theta0_reference(model, ref_path)
    else:
        logger.info(f"theta0 reference already exists at {ref_path}.")

    # 2. Precompute and cache OPLoRA SVD bases (k=128) in FP32
    oplora_bases_dir = t1_cfg.artifacts_dir / "oplora_bases"
    oplora_bases_dir.mkdir(parents=True, exist_ok=True)
    target_modules = methods_cfg.get("oplora16", {}).get("target_modules", ["down_proj"])

    logger.info("Checking/precomputing OPLoRA SVD bases (k=128)...")
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, torch.nn.Linear):
                clean_name = name.replace(".", "_")
                u_cache = oplora_bases_dir / f"{clean_name}_U_k128.pt"
                v_cache = oplora_bases_dir / f"{clean_name}_V_k128.pt"
                if not (u_cache.exists() and v_cache.exists()):
                    logger.info(f"Computing SVD for {name} ({module.weight.shape})...")
                    U_k, V_k, _ = compute_svd_bases(module.weight, k=128)
                    torch.save(U_k, str(u_cache))
                    torch.save(V_k, str(v_cache))

    # 3. Cache base model activation probes (layers 0, 13, 27)
    logger.info("Caching base model activation vectors z on 64 probes...")
    probe_cache_file = t1_cfg.artifacts_dir / "probe_manifest.json"
    probes, _ = extract_probe_inputs(tokenizer, seed=t1_cfg.seed, cache_path=probe_cache_file)
    cache_base_model_activations(
        base_model=model,
        probes=probes,
        target_layers=[0, 13, 27],
        output_dir=t1_cfg.artifacts_dir / "activation_cache",
        device=device,
    )

    # 4. Evaluate 4 benchmarks
    if not args.skip_eval:
        baseline_out_dir = t1_cfg.results_dir / "raw" / "baseline"
        baseline_out_dir.mkdir(parents=True, exist_ok=True)

        for task_key in ["gsm8k", "hellaswag", "boolq", "arc_easy"]:
            agg_file = baseline_out_dir / f"{task_key}_aggregates.json"
            parquet_file = baseline_out_dir / f"{task_key}_per_example.parquet"
            if not args.force_rerun and agg_file.exists() and parquet_file.exists():
                logger.info(f"Task {task_key} already evaluated at {agg_file}. Skipping re-evaluation.")
                continue

            logger.info(f"Evaluating baseline on {task_key}...")
            evaluate_with_lm_eval(
                model=model,
                tokenizer=tokenizer,
                task_key=task_key,
                device=str(device),
                batch_size=args.batch_size,
                output_dir=baseline_out_dir,
            )

    logger.info("Baseline evaluation and asset caching completed successfully.")


if __name__ == "__main__":
    main()

