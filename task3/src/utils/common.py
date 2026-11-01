"""Utility functions for Task 3: logging, file hashing, atomic writes, and seed management."""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np


def setup_logger(
    name: str = "task3",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures structured console and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def compute_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def atomic_save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Saves JSON data atomically using a temporary file and rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_file = p.with_suffix(f".tmp_{os.getpid()}")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    temp_file.replace(p)


def load_json(path: str | Path) -> Any:
    """Loads JSON data from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_deterministic_rng(seed: int, sub_index: int = 0) -> np.random.Generator:
    """Derives a deterministic NumPy Generator from a primary seed and sub-index."""
    combined_seed = (seed * 10007 + sub_index) % (2**32 - 1)
    return np.random.default_rng(combined_seed)


# Aliases
get_logger = setup_logger
get_file_sha256 = compute_sha256
