#!/usr/bin/env python3
"""
Preflight verification script for Task 1.
Checks GPU architecture, VRAM feasibility, persistent storage, dependencies, and data accessibility.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TASK1_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TASK1_ROOT / "src"))

import torch
from task1.config import get_default_configs
from task1.results import export_environment_metadata

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("preflight")


def check_gpu_and_vram() -> dict:
    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        return {
            "status": "blocked",
            "reason": "No CUDA GPU detected! Assignment Task 1 requires an NVIDIA GPU (target: Tesla T4 16GB).",
            "gpu_name": "None",
            "vram_gib": 0.0,
        }

    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gib = vram_bytes / (1024**3)

    is_t4 = "T4" in device_name.upper()
    sufficient_vram = vram_gib >= 14.0

    return {
        "status": "pass" if (cuda_avail and sufficient_vram) else "warning",
        "gpu_name": device_name,
        "vram_gib": vram_gib,
        "is_target_t4": is_t4,
        "sufficient_vram": sufficient_vram,
    }


def check_persistent_storage(persistent_root: Path) -> dict:
    exists = persistent_root.exists()
    free_gib = 0.0
    if exists:
        try:
            total, used, free = shutil.disk_usage(persistent_root)
            free_gib = free / (1024**3)
        except Exception:
            free_gib = 0.0

    # Task 1 requires ~15 GB persistent storage for all 5 methods' checkpoints, reference, and cache
    sufficient_space = free_gib >= 15.0
    return {
        "path": str(persistent_root),
        "exists": exists,
        "free_gib": free_gib,
        "sufficient_space": sufficient_space,
        "status": "pass" if (exists and sufficient_space) else "warning",
    }


def run_gpu_smoke_test(seq_len: int = 2048, microbatch: int = 2) -> dict:
    """Simulates a microbatch forward/backward pass with synthetic tensors at full 2048 seq cap."""
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "No CUDA available"}

    logger.info(f"Running T4 memory smoke test (seq_len={seq_len}, microbatch={microbatch})...")
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    try:
        # Approximate Qwen3-0.6B transformer layer dimensions (hidden=1024, ffn=3072)
        lin1 = torch.nn.Linear(1024, 3072).to(device)
        lin2 = torch.nn.Linear(3072, 1024).to(device)
        x = torch.randn(microbatch, seq_len, 1024, device=device, dtype=torch.float16)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            h = lin1(x)
            out = lin2(h)
            loss = out.sum()

        loss.backward()

        peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

        del x, h, out, loss, lin1, lin2
        torch.cuda.empty_cache()

        return {
            "status": "pass",
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"status": "blocked", "reason": f"OOM during smoke test: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Preflight check for ATML PA3 Task 1")
    parser.add_argument("--persistent-root", type=str, default=None)
    args = parser.parse_args()

    t1_cfg, methods_cfg, eval_cfg = get_default_configs(TASK1_ROOT)
    persist_root = Path(args.persistent_root) if args.persistent_root else t1_cfg.persistent_root

    logger.info("=== Starting Task 1 Preflight Verification ===")

    # 1. GPU Check
    gpu_res = check_gpu_and_vram()
    logger.info(f"GPU check: {gpu_res}")

    # 2. Storage Check
    storage_res = check_persistent_storage(persist_root)
    logger.info(f"Storage check: {storage_res}")

    # 3. Smoke Test
    smoke_res = run_gpu_smoke_test(seq_len=t1_cfg.max_seq_length, microbatch=t1_cfg.microbatch_size)
    logger.info(f"Smoke test: {smoke_res}")

    # 4. Export environment
    env_meta = export_environment_metadata(t1_cfg.results_dir / "environment.json")

    # 5. Check qualitative prompt source
    prompt_file = Path(eval_cfg.get("qualitative", {}).get("prompts_source", "shared_assets/inputs/first_plot_questions.yaml"))
    if not prompt_file.is_absolute():
        prompt_file = REPO_ROOT / prompt_file
    prompts_available = prompt_file.exists()

    overall_status = "PASS" if (gpu_res["status"] == "pass" and smoke_res.get("status") == "pass") else "BLOCKED_OR_WARNING"

    status_lines = [
        "# Task 1 Execution Status & Preflight Audit",
        f"- Timestamp: {env_meta.get('platform')}",
        f"- Overall Feasibility Gate: **{overall_status}**",
        "",
        "## Preflight Results",
        f"- **GPU**: {gpu_res['gpu_name']} ({gpu_res['vram_gib']:.2f} GiB VRAM)",
        f"- **Tesla T4 Target**: {'Yes' if gpu_res.get('is_target_t4') else 'No (Detected: ' + gpu_res['gpu_name'] + ')'}",
        f"- **VRAM Feasibility**: {'Sufficient' if gpu_res.get('sufficient_vram') else 'Insufficient for 16GB T4 requirement'}",
        f"- **Persistent Storage Path**: `{storage_res['path']}` (Free: {storage_res['free_gib']:.2f} GiB)",
        f"- **GPU Smoke Test**: {smoke_res.get('status')}",
        f"- **Qualitative Prompts Available**: {prompts_available} (`{prompt_file}`)",
        "",
        "## Experiment Status",
        "- **Unchanged Baseline**: Pending execution in Colab",
        "- **Full Fine-Tuning**: Pending execution in Colab",
        "- **Full FT + L2-SP**: Pending execution in Colab",
        "- **LoRA Rank 4**: Pending execution in Colab",
        "- **LoRA Rank 16**: Pending execution in Colab",
        "- **OPLoRA Rank 16**: Pending execution in Colab",
    ]

    status_file = t1_cfg.results_dir / "STATUS.md"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines) + "\n")

    logger.info(f"Preflight finished. Status written to {status_file}.")
    print("\n" + "\n".join(status_lines))


if __name__ == "__main__":
    main()
