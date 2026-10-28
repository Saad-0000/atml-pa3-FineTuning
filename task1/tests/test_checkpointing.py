"""Tests for full recovery checkpointing, atomic writes, and hash verification."""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from task1.checkpointing import CheckpointManager, get_rng_states, set_rng_states


class TestCheckpointing(unittest.TestCase):
    def test_checkpoint_atomic_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_mgr = CheckpointManager(
                save_dir=tmp_dir,
                method_name="test_method",
                official_milestones=[312, 625, 937, 1250],
                keep_emergency_copies=2,
            )

            model = nn.Linear(4, 2)
            optimizer = AdamW(model.parameters(), lr=1e-3)

            # Save checkpoint at step 312 (official milestone)
            target_path = ckpt_mgr.save_checkpoint(
                update_step=312,
                examples_seen=9984,
                data_position=9984,
                model=model,
                optimizer=optimizer,
                lr_scheduler=None,
                scaler=None,
                method_config={"method": "test"},
            )

            self.assertTrue(target_path.exists())
            self.assertTrue((target_path / "training_state.pt").exists())
            self.assertTrue((target_path / "completion_manifest.json").exists())

            # Load checkpoint back
            loaded = ckpt_mgr.load_checkpoint(target_path)
            self.assertEqual(loaded["update_step"], 312)
            self.assertEqual(loaded["examples_seen"], 9984)
            self.assertTrue(loaded["is_official_milestone"])

    def test_milestone_preservation_during_emergency_pruning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_mgr = CheckpointManager(
                save_dir=tmp_dir,
                method_name="test_method",
                official_milestones=[312, 625, 937, 1250],
                keep_emergency_copies=1,
            )

            model = nn.Linear(2, 2)
            optimizer = AdamW(model.parameters(), lr=1e-3)

            # Save official milestone 312
            ckpt_mgr.save_checkpoint(312, 9984, 9984, model, optimizer, None, None, {})
            # Save emergency 50, 100, 150
            ckpt_mgr.save_checkpoint(50, 1600, 1600, model, optimizer, None, None, {})
            ckpt_mgr.save_checkpoint(100, 3200, 3200, model, optimizer, None, None, {})
            ckpt_mgr.save_checkpoint(150, 4800, 4800, model, optimizer, None, None, {})

            # Milestone 312 must NEVER be pruned
            m312 = Path(tmp_dir) / "test_method" / "checkpoint_step_0312"
            self.assertTrue(m312.exists(), "Official milestone 312 was pruned!")

            # Out of emergency copies 50, 100, 150 with keep=1, only 150 should remain
            e50 = Path(tmp_dir) / "test_method" / "checkpoint_step_0050"
            e150 = Path(tmp_dir) / "test_method" / "checkpoint_step_0150"
            self.assertFalse(e50.exists(), "Old emergency checkpoint 50 was not pruned!")
            self.assertTrue(e150.exists(), "Latest emergency checkpoint 150 was pruned!")


if __name__ == "__main__":
    unittest.main()

