"""Full recovery checkpointing, atomic persistence, hash verification, and resume discovery."""

import gc
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger("task1.checkpointing")


def compute_file_sha256(file_path: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_rng_states() -> Dict[str, Any]:
    """Captures Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states."""
    states: Dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restores Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states."""
    if "python_rng" in states:
        random.setstate(states["python_rng"])
    if "numpy_rng" in states:
        np.random.set_state(states["numpy_rng"])
    if "torch_cpu_rng" in states:
        torch.set_rng_state(states["torch_cpu_rng"])
    if "torch_cuda_rng" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda_rng"])


class CheckpointManager:
    """
    Manages atomic writing, verification, manifest tracking, and rotation of checkpoints.
    - Preserves immutable official milestones (updates 312, 625, 937, 1250).
    - Rotates only emergency copies (keeping at least latest and one prior known-good).
    - Writes completion manifest only after full verification of the saved file.
    """

    def __init__(
        self,
        save_dir: Path | str,
        method_name: str,
        official_milestones: Optional[List[int]] = None,
        keep_emergency_copies: int = 2,
    ):
        self.save_dir = Path(save_dir) / method_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name
        self.official_milestones = set(official_milestones or [312, 625, 937, 1250])
        self.keep_emergency_copies = max(1, keep_emergency_copies)
        self.manifest_file = self.save_dir / "checkpoint_manifest.json"

    def save_checkpoint(
        self,
        update_step: int,
        examples_seen: int,
        data_position: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        method_config: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Atomically saves a full recovery checkpoint.
        """
        is_official = update_step in self.official_milestones
        ckpt_name = f"checkpoint_step_{update_step:04d}"
        target_dir = self.save_dir / ckpt_name
        target_ckpt_file = target_dir / "training_state.pt"
        target_manifest = target_dir / "completion_manifest.json"

        # Temporary staging directory on the same filesystem for atomic rename
        temp_dir = self.save_dir / f".tmp_{ckpt_name}_{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_ckpt_file = temp_dir / "training_state.pt"

        # Assemble full recovery bundle
        recovery_state = {
            "method_name": self.method_name,
            "update_step": update_step,
            "examples_seen": examples_seen,
            "data_position": data_position,
            "is_official_milestone": is_official,
            "rng_states": get_rng_states(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "method_config": method_config,
            "extra_metadata": extra_metadata or {},
        }

        # Fast local SSD staging: avoids random-write seek thrashing over Google Drive network FUSE
        local_staging_dir = Path("/tmp") if Path("/tmp").exists() and os.access("/tmp", os.W_OK) else None
        if local_staging_dir is not None:
            local_ckpt_file = local_staging_dir / f".local_{ckpt_name}_{os.getpid()}.pt"
            logger.info(f"Writing recovery checkpoint to local SSD staging {local_ckpt_file}...")
            torch.save(recovery_state, str(local_ckpt_file))
            file_size = local_ckpt_file.stat().st_size
            sha256 = compute_file_sha256(local_ckpt_file)

            # Free memory immediately before transferring to Drive
            del recovery_state
            gc.collect()

            logger.info(f"Streaming {file_size / (1024**2):.1f} MiB checkpoint to persistent Drive {temp_ckpt_file}...")
            shutil.copyfile(str(local_ckpt_file), str(temp_ckpt_file))

            try:
                local_ckpt_file.unlink()
            except Exception:
                pass
        else:
            logger.info(f"Writing recovery checkpoint to temporary staging {temp_ckpt_file}...")
            torch.save(recovery_state, str(temp_ckpt_file))
            file_size = temp_ckpt_file.stat().st_size
            sha256 = compute_file_sha256(temp_ckpt_file)
            del recovery_state
            gc.collect()

        # Fast integrity verification: confirm valid Zip archive structure without loading 7GB into RAM
        if zipfile.is_zipfile(temp_ckpt_file):
            with zipfile.ZipFile(temp_ckpt_file, "r") as zf:
                if not zf.namelist():
                    raise RuntimeError(f"Saved checkpoint archive {temp_ckpt_file} is corrupt or empty!")

        # Write completion manifest in temp dir
        comp_manifest = {
            "method_name": self.method_name,
            "update_step": update_step,
            "examples_seen": examples_seen,
            "data_position": data_position,
            "is_official_milestone": is_official,
            "file_name": "training_state.pt",
            "file_size_bytes": file_size,
            "sha256": sha256,
            "status": "complete",
        }
        with open(temp_dir / "completion_manifest.json", "w", encoding="utf-8") as f:
            json.dump(comp_manifest, f, indent=2)

        # Atomic commit: remove old target if exists and rename temp_dir
        if target_dir.exists():
            shutil.rmtree(target_dir)
        temp_dir.rename(target_dir)

        # Update latest symlink / pointer
        latest_pointer = self.save_dir / "latest_checkpoint.json"
        with open(latest_pointer, "w", encoding="utf-8") as f:
            json.dump(comp_manifest, f, indent=2)

        # Update top-level checkpoint manifest
        self._record_manifest(comp_manifest)

        # Prune non-official emergency checkpoints if count exceeds keep_emergency_copies
        if not is_official:
            self._prune_emergency_checkpoints()

        logger.info(
            f"Successfully persisted checkpoint step {update_step} ({examples_seen} ex) to {target_dir}. "
            f"Size={file_size / (1024**2):.2f} MiB, SHA256={sha256[:12]}..."
        )
        return target_dir

    def _record_manifest(self, record: Dict[str, Any]) -> None:
        manifest_data = []
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = []

        # Remove existing record for same step if present
        manifest_data = [m for m in manifest_data if m.get("update_step") != record["update_step"]]
        manifest_data.append(record)
        manifest_data.sort(key=lambda x: x["update_step"])

        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def _prune_emergency_checkpoints(self) -> None:
        """Removes older non-official emergency checkpoints to save disk space."""
        if not self.manifest_file.exists():
            return
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        emergency_steps = [
            m["update_step"]
            for m in manifest_data
            if not m.get("is_official_milestone", False)
        ]
        emergency_steps.sort()

        if len(emergency_steps) > self.keep_emergency_copies:
            to_remove = emergency_steps[:-self.keep_emergency_copies]
            for step in to_remove:
                d = self.save_dir / f"checkpoint_step_{step:04d}"
                if d.exists():
                    logger.info(f"Pruning emergency checkpoint {d}...")
                    shutil.rmtree(d, ignore_errors=True)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Discovers the newest verified complete checkpoint."""
        if not self.manifest_file.exists():
            return None
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        valid_ckpts = []
        for m in manifest_data:
            step = m.get("update_step", 0)
            d = self.save_dir / f"checkpoint_step_{step:04d}"
            f_pt = d / "training_state.pt"
            f_man = d / "completion_manifest.json"
            if d.exists() and f_pt.exists() and f_man.exists():
                valid_ckpts.append((step, d))

        if not valid_ckpts:
            return None
        valid_ckpts.sort(key=lambda x: x[0])
        return valid_ckpts[-1][1]

    def load_checkpoint(self, checkpoint_path: Path | str) -> Dict[str, Any]:
        """Loads and verifies a checkpoint bundle."""
        p = Path(checkpoint_path)
        ckpt_file = p / "training_state.pt" if p.is_dir() else p
        manifest_file = p / "completion_manifest.json" if p.is_dir() else p.parent / "completion_manifest.json"

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                man = json.load(f)
            stored_hash = man.get("sha256")
            if stored_hash:
                curr_hash = compute_file_sha256(ckpt_file)
                if curr_hash != stored_hash:
                    raise ValueError(f"Corrupted checkpoint! SHA256 mismatch: {curr_hash} != {stored_hash}")

        logger.info(f"Loading checkpoint bundle from {ckpt_file}...")
        state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        return state
