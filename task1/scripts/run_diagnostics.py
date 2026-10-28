#!/usr/bin/env python3
"""
Diagnostics calculation script:
- Weight-space diagnostics (rho_k, eta) on down_proj at layers 0, 13, 27.
- Activation diagnostics (R(z)) on down_proj at layers 0, 13, 27 across 64 fixed probe vectors.
- Covers the 6 required model states:
  Full FT midpoint, Full FT final, L2-SP final, LoRA-4 final, LoRA-16 final, OPLoRA-16 final.
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

import pandas as pd
import torch
from task1.config import get_default_configs
from task1.diagnostics import run_full_diagnostics
from task1.lora import apply_lora_to_model
from task1.model_loader import load_base_model
from task1.oplora import apply_oplora_to_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("run_diagnostics")


def main():
    parser = argparse.ArgumentParser(description="Compute weight-space and activation diagnostics")
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    device = torch.device("cpu")  # Diagnostics run stably in FP32 on CPU

    target_layers = [0, 13, 27]

    # 1. Load starting base model W0 weights and U_k bases
    logger.info("Loading starting model reference W0 and U_k bases...")
    base_model = load_base_model(
        t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=torch.float32, gradient_checkpointing=False, device=device
    )

    base_W0_dict = {}
    U_k_dict = {}
    oplora_bases_dir = t1_cfg.artifacts_dir / "oplora_bases"

    for layer_idx in target_layers:
        down_proj = base_model.model.layers[layer_idx].mlp.down_proj
        base_W0_dict[layer_idx] = down_proj.weight.detach().clone().to(torch.float32)

        # Load U_k basis
        clean_name = f"model_layers_{layer_idx}_mlp_down_proj"
        u_cache = oplora_bases_dir / f"{clean_name}_U_k128.pt"
        if not u_cache.exists():
            raise FileNotFoundError(f"Missing cached basis {u_cache}! Run evaluate_baseline first.")
        U_k_dict[layer_idx] = torch.load(str(u_cache), map_location="cpu", weights_only=False).to(torch.float32)

    # 2. Load cached activation vectors z
    act_cache_file = t1_cfg.artifacts_dir / "activation_cache" / "base_activations_z.pt"
    probe_manifest_file = t1_cfg.artifacts_dir / "probe_manifest.json"

    if not act_cache_file.exists() or not probe_manifest_file.exists():
        raise FileNotFoundError("Missing activation probe cache! Run evaluate_baseline first.")

    activation_cache = torch.load(str(act_cache_file), map_location="cpu", weights_only=False)
    with open(probe_manifest_file, "r", encoding="utf-8") as f:
        probe_manifest = json.load(f)["probes"]

    conditions = [
        ("full_ft_midpoint", "full_ft", 625),
        ("full_ft_final", "full_ft", 1250),
        ("l2sp_final", "l2sp", 1250),
        ("lora4_final", "lora4", 1250),
        ("lora16_final", "lora16", 1250),
        ("oplora16_final", "oplora16", 1250),
    ]

    all_weight_records = []
    all_activation_records = []

    for cond_id, method, step in conditions:
        ckpt_path = t1_cfg.checkpoints_dir / method / f"checkpoint_step_{step:04d}" / "training_state.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint for {cond_id} missing at {ckpt_path}. Skipping.")
            continue

        logger.info(f"Computing diagnostics for {cond_id} from {ckpt_path}...")
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Load fresh base architecture
        model = load_base_model(
            t1_cfg.model_name, revision=t1_cfg.model_revision, torch_dtype=torch.float32, gradient_checkpointing=False, device=device
        )

        if method in ["lora4", "lora16"]:
            m_cfg = methods_cfg[method]
            model, _ = apply_lora_to_model(
                model=model, r=m_cfg["r"], alpha=m_cfg["alpha"], dropout=0.0, target_modules=m_cfg["target_modules"]
            )
        elif method == "oplora16":
            m_cfg = methods_cfg[method]
            model, _, _ = apply_oplora_to_model(
                model=model,
                r=m_cfg["r"],
                alpha=m_cfg["alpha"],
                k=m_cfg["protection_rank_k"],
                dropout=0.0,
                target_modules=m_cfg["target_modules"],
                basis_cache_dir=oplora_bases_dir,
            )

        model.load_state_dict(state["model_state_dict"])

        w_recs, a_recs = run_full_diagnostics(
            condition_id=cond_id,
            trained_model=model,
            base_W0_dict=base_W0_dict,
            U_k_dict=U_k_dict,
            activation_cache=activation_cache,
            probes_manifest=probe_manifest,
            target_layers=target_layers,
        )
        all_weight_records.extend(w_recs)
        all_activation_records.extend(a_recs)

    # 3. Save raw outputs and summary tables
    raw_dir = t1_cfg.results_dir / "raw"
    tables_dir = t1_cfg.results_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if all_weight_records:
        df_w = pd.DataFrame(all_weight_records)
        df_w.to_csv(raw_dir / "weight_space_diagnostics.csv", index=False)
        df_w.to_csv(tables_dir / "weight_space_diagnostics.csv", index=False)
        logger.info(f"Saved weight-space diagnostics ({len(df_w)} rows).")

    if all_activation_records:
        df_a = pd.DataFrame(all_activation_records)
        df_a.to_csv(raw_dir / "activation_corrections.csv", index=False)

        # Aggregate summary table: condition x layer x group
        summary = (
            df_a.groupby(["condition", "layer", "group"])["R_z"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .reset_index()
        )
        summary.to_csv(tables_dir / "activation_diagnostics_summary.csv", index=False)
        logger.info(f"Saved activation diagnostics summary ({len(summary)} rows).")

    logger.info("Diagnostics computation finished.")


if __name__ == "__main__":
    main()
