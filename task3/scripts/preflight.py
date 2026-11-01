"""
Task 3 Preflight Verification Script
Inspects environment, dependencies, GPU (T4 checks), and data availability.
Outputs structured diagnostics and verifies environment readiness.
"""

import os
import sys
import argparse
import platform
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task3.src.utils.common import get_logger, get_file_sha256

logger = get_logger("preflight")


def check_gpu():
    """Verify GPU device and memory."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_info = {
            "cuda_available": cuda_avail,
            "device_count": torch.cuda.device_count() if cuda_avail else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else "None",
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0,
            "torch_version": torch.__version__,
        }
    except Exception as e:
        gpu_info = {"cuda_available": False, "error": str(e)}
    return gpu_info


def check_packages():
    """Check critical pinned packages quickly without full module initialization."""
    pkg_status = {}
    for pkg in ["torch", "transformers", "accelerate", "pandas", "pyarrow", "scipy", "math_verify", "sympy"]:
        spec = importlib.util.find_spec(pkg)
        if spec is not None:
            pkg_status[pkg] = "installed"
        else:
            pkg_status[pkg] = "MISSING"
    return pkg_status


def check_data_directory(data_dir: Path):
    """Check presence of supplied ATML-PA3-Data files."""
    if not data_dir.exists():
        return {"exists": False, "path": str(data_dir), "files": {}}
    
    expected_files = [
        "README.MD",
        "MATH/generations.parquet",
        "MATH/probability_metrics.parquet",
        "MathVista/generations.parquet",
        "MathVista/probability_metrics.parquet",
        "MathVista/attention_metrics.parquet",
    ]
    
    file_info = {}
    for rel_path in expected_files:
        p = data_dir / rel_path
        if p.exists():
            file_info[rel_path] = {
                "present": True,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            }
        else:
            file_info[rel_path] = {"present": False}
            
    return {"exists": True, "path": str(data_dir), "files": file_info}


def run_preflight(data_dir_path: str, output_path: Optional[str] = None):
    logger.info("=" * 60)
    logger.info("Running Task 3 Preflight Verification")
    logger.info("=" * 60)

    data_dir = Path(data_dir_path)
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
    }
    logger.info(f"Python: {env_info['python_version']} on {env_info['platform']}")

    gpu_info = check_gpu()
    logger.info(f"GPU: {gpu_info.get('device_name', 'None')} (CUDA: {gpu_info.get('cuda_available', False)}, Mem: {gpu_info.get('total_memory_gb', 0)} GB)")

    pkg_info = check_packages()
    missing_pkgs = [p for p, v in pkg_info.items() if v == "MISSING"]
    if missing_pkgs:
        logger.warning(f"Missing packages: {missing_pkgs}")
    else:
        logger.info("All essential packages installed.")

    data_info = check_data_directory(data_dir)
    logger.info(f"Data directory: {data_dir} (Exists: {data_info['exists']})")
    for f, finfo in data_info.get("files", {}).items():
        logger.info(f"  - {f}: {'FOUND (' + str(finfo.get('size_mb')) + ' MB)' if finfo.get('present') else 'MISSING'}")

    summary = {
        "status": "PASS" if not missing_pkgs else "WARN",
        "environment": env_info,
        "gpu": gpu_info,
        "packages": pkg_info,
        "data": data_info,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved preflight summary to {out_p}")

    logger.info("=" * 60)
    logger.info(f"Preflight completed with status: {summary['status']}")
    logger.info("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Task 3 Preflight Verification")
    parser.add_argument("--data_dir", type=str, default="data/ATML-PA3-Data", help="Path to supplied ATML-PA3-Data")
    parser.add_argument("--output", type=str, default="task3/logs/preflight_summary.json", help="Output summary JSON path")
    args = parser.parse_args()

    run_preflight(args.data_dir, args.output)


if __name__ == "__main__":
    main()
