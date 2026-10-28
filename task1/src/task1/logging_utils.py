"""Logging, metrics tracking, and GPU memory monitoring utilities."""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


def get_logger(name: str = "task1", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


class MetricLogger:
    """Logs training metrics to CSV and JSONL with explicit flushing."""

    def __init__(self, output_dir: Path, method_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name

        self.csv_path = self.output_dir / f"training_{method_name}.csv"
        self.jsonl_path = self.output_dir / f"training_{method_name}.jsonl"

        self._csv_file = None
        self._csv_writer = None
        self._csv_headers: Optional[List[str]] = None

    def log(self, metrics: Dict[str, Any]) -> None:
        metrics = dict(metrics)
        metrics["timestamp"] = time.time()
        metrics["method"] = self.method_name

        # JSONL append
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
            f.flush()

        # CSV append
        keys = list(metrics.keys())
        if self._csv_file is None:
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._csv_headers = keys
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_headers, extrasaction="ignore")
            if write_header:
                self._csv_writer.writeheader()
        elif set(keys) != set(self._csv_headers or []):
            # Extend headers if new keys appeared
            new_keys = [k for k in keys if k not in (self._csv_headers or [])]
            if new_keys:
                self._csv_headers = (self._csv_headers or []) + new_keys
                self._csv_file.close()
                self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_headers, extrasaction="ignore")

        if self._csv_writer is not None:
            self._csv_writer.writerow(metrics)
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()


def get_gpu_memory_stats() -> Dict[str, float]:
    """Returns allocated and reserved GPU memory in MiB."""
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_mib": 0.0,
            "gpu_reserved_mib": 0.0,
            "gpu_max_allocated_mib": 0.0,
            "gpu_max_reserved_mib": 0.0,
        }
    return {
        "gpu_allocated_mib": torch.cuda.memory_allocated() / (1024**2),
        "gpu_reserved_mib": torch.cuda.memory_reserved() / (1024**2),
        "gpu_max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "gpu_max_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def reset_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

