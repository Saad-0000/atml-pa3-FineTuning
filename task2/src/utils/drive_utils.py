"""Google Drive persistence and atomic file utilities for Task 2.

Handles:
- Drive path resolution and mounting checks
- Atomic file writes and copy operations
- File integrity checks via SHA-256
"""
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_drive_mounted(drive_root: str | Path) -> bool:
    """Check if Google Drive mount exists and is writable."""
    path = Path(drive_root)
    if not path.exists():
        logger.warning(f"Drive root does not exist: {path}")
        return False
    try:
        test_file = path / ".write_test.tmp"
        with open(test_file, "w") as f:
            f.write("drive_mount_ok")
        test_file.unlink()
        return True
    except Exception as e:
        logger.error(f"Drive root is not writable: {e}")
        return False


def atomic_copy_file(src_path: str | Path, dst_path: str | Path) -> Path:
    """Copies a file atomically using a temporary destination and checksum verification."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_dst = dst.with_suffix(f"{dst.suffix}.tmp")
    shutil.copy2(src, tmp_dst)

    src_hash = compute_sha256(src)
    dst_hash = compute_sha256(tmp_dst)

    if src_hash != dst_hash:
        tmp_dst.unlink()
        raise IOError(f"Checksum mismatch copying {src} to {dst}!")

    tmp_dst.replace(dst)
    return dst
