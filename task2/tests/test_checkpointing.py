"""Unit tests for atomic checkpoint saving and resumption.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
import torch

from task2.src.models.connector import MLPConnector
from task2.src.training.checkpointing import CheckpointManager


class TestCheckpointing(unittest.TestCase):
    def test_atomic_checkpoint_save_and_resume(self):
        """Validates atomic save, manifest updates, and state recovery."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            manager = CheckpointManager(temp_dir)
            connector = MLPConnector(768, 1024, 1024)
            optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-3)

            # Save checkpoint at update 250
            saved_path = manager.save_checkpoint(
                global_update=250,
                stage=1,
                stage_update=250,
                examples_presented=4000,
                connector=connector,
                lora_state=None,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                sampler_state={"draws": 2000},
                config={"seed": 6304},
            )

            self.assertTrue(saved_path.exists())
            self.assertFalse(saved_path.with_suffix(".pt.tmp").exists())

            # Check manifest
            manifest = manager.load_manifest()
            self.assertIn("step_250", manifest["checkpoints"])
            self.assertIsNotNone(manifest["latest_valid_checkpoint"])

            # Resume
            resumed_path, payload = manager.load_latest_checkpoint()
            self.assertEqual(resumed_path, saved_path)
            self.assertEqual(payload["global_update"], 250)
            self.assertEqual(payload["stage"], 1)
            self.assertEqual(payload["examples_presented"], 4000)
            self.assertEqual(payload["sampler_state"]["draws"], 2000)

            # Load weights into new connector
            new_connector = MLPConnector(768, 1024, 1024)
            new_connector.load_state_dict(payload["connector_state"])
            for p1, p2 in zip(connector.parameters(), new_connector.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
