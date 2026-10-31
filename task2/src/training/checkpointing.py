"""Atomic, verified checkpoint saving and resumption for Task 2.

Supports saving:
- Connector state
- LoRA state
- Optimizer state
- Scheduler state and update counters
- GradScaler state
- Sampler mixture counters and RNG states (Python, NumPy, PyTorch CPU/CUDA)
- Resolved config and commit hash
- SHA-256 checksums and atomic commit
"""
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "2.0.0"


def compute_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_rng_states() -> Dict[str, Any]:
    """Capture current RNG states for Python, NumPy, and PyTorch (CPU & CUDA)."""
    states = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states for Python, NumPy, and PyTorch (CPU & CUDA)."""
    if "python_rng_state" in states:
        random.setstate(states["python_rng_state"])
    if "numpy_rng_state" in states:
        np.random.set_state(states["numpy_rng_state"])
    if "torch_cpu_rng_state" in states:
        torch.set_rng_state(states["torch_cpu_rng_state"])
    if "torch_cuda_rng_state" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda_rng_state"])


class CheckpointManager:
    """Manages atomic saving, verification, and loading of training checkpoints."""

    def __init__(self, checkpoints_dir: str | Path):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.checkpoints_dir / "checkpoint_manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if not self.manifest_file.exists():
            initial_manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoints": {},
                "latest_valid_checkpoint": None,
                "stage_exports": {},
            }
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(initial_manifest, f, indent=2)

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_file.exists():
            self._ensure_manifest()
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        tmp_path = self.manifest_file.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        tmp_path.replace(self.manifest_file)

    def save_checkpoint(
        self,
        global_update: int,
        stage: int,
        stage_update: int,
        examples_presented: int,
        connector: torch.nn.Module,
        lora_state: Optional[Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.amp.GradScaler],
        sampler_state: Dict[str, Any],
        config: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically saves a complete checkpoint and updates the manifest.

        Args:
            global_update: Global optimizer update counter (1..4500).
            stage: Stage number (1, 2, 3).
            stage_update: Stage-local update counter.
            examples_presented: Total examples presented so far.
            connector: MLPConnector module.
            lora_state: State dict of LoRA adapter weights (None in Stage 1).
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler instance.
            scaler: GradScaler instance (if using AMP).
            sampler_state: Realized mixture counters and positions.
            config: Resolved configuration dictionary.
            extra_meta: Additional metadata (e.g. loss, learning rates).

        Returns:
            Path to the saved checkpoint file.
        """
        stage_dir = self.checkpoints_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        ckpt_filename = f"ckpt_step_{global_update:04d}.pt"
        final_path = stage_dir / ckpt_filename
        tmp_path = stage_dir / f"{ckpt_filename}.tmp"

        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "global_update": global_update,
            "stage": stage,
            "stage_update": stage_update,
            "examples_presented": examples_presented,
            "connector_state": connector.state_dict(),
            "lora_state": lora_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "sampler_state": sampler_state,
            "rng_states": get_rng_states(),
            "config": config,
            "extra_meta": extra_meta or {},
        }

        # Step 1: Write to temporary file
        torch.save(payload, tmp_path)

        # Step 2: Compute checksum and verify file readability
        sha256_hash = compute_sha256(tmp_path)
        file_size_bytes = tmp_path.stat().st_size

        try:
            # Smoke test load
            _ = torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Checkpoint verification failed during save: {e}")

        # Step 3: Atomic commit
        tmp_path.replace(final_path)

        # Step 4: Write companion metadata JSON
        meta_info = {
            "checkpoint_file": str(final_path.name),
            "global_update": global_update,
            "stage": stage,
            "stage_update": stage_update,
            "examples_presented": examples_presented,
            "file_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "verified": True,
            "extra_meta": extra_meta or {},
        }
        meta_path = final_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)

        # Step 5: Update master manifest
        manifest = self.load_manifest()
        key = f"step_{global_update}"
        manifest["checkpoints"][key] = {
            "path": str(final_path.relative_to(self.checkpoints_dir)),
            "global_update": global_update,
            "stage": stage,
            "stage_update": stage_update,
            "file_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "verified": True,
        }
        manifest["latest_valid_checkpoint"] = str(final_path.relative_to(self.checkpoints_dir))
        self._save_manifest(manifest)

        logger.info(
            f"Successfully saved and verified checkpoint at update {global_update}: "
            f"{final_path} ({file_size_bytes / 1e6:.1f} MB, sha256: {sha256_hash[:8]}...)"
        )
        return final_path

    def export_stage_weights(
        self,
        stage: int,
        global_update: int,
        connector: torch.nn.Module,
        lora_state: Optional[Dict[str, torch.Tensor]],
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Path]:
        """Export standalone connector and adapter weights at stage completion."""
        target_dir = Path(output_dir) if output_dir else self.checkpoints_dir / "exports"
        target_dir.mkdir(parents=True, exist_ok=True)

        connector_path = target_dir / f"stage{stage}_connector_step_{global_update}.pt"
        torch.save(connector.state_dict(), connector_path)

        exported = {"connector": connector_path}

        if lora_state is not None:
            lora_path = target_dir / f"stage{stage}_lora_adapters_step_{global_update}.pt"
            torch.save(lora_state, lora_path)
            exported["lora"] = lora_path

        manifest = self.load_manifest()
        manifest["stage_exports"][f"stage_{stage}"] = {
            "global_update": global_update,
            "connector_path": str(connector_path),
            "lora_path": str(exported.get("lora", "")),
        }
        self._save_manifest(manifest)
        return exported

    def load_latest_checkpoint(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """Loads and verifies the latest valid checkpoint from manifest."""
        manifest = self.load_manifest()
        latest_rel_path = manifest.get("latest_valid_checkpoint")
        if not latest_rel_path:
            return None

        ckpt_path = self.checkpoints_dir / latest_rel_path
        if not ckpt_path.exists():
            logger.warning(f"Manifest pointed to {ckpt_path}, but file does not exist.")
            return None

        # Verify checksum
        recorded_sha = manifest["checkpoints"].get(
            f"step_{ckpt_path.stem.split('_')[-1]}", {}
        ).get("sha256")
        if recorded_sha:
            actual_sha = compute_sha256(ckpt_path)
            if actual_sha != recorded_sha:
                raise ValueError(
                    f"Checksum mismatch for checkpoint {ckpt_path}! "
                    f"Recorded: {recorded_sha}, Actual: {actual_sha}"
                )

        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        return ckpt_path, payload
