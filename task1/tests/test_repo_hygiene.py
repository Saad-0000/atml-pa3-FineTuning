"""Test repository structure hygiene and placeholder boundaries."""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestRepositoryHygiene(unittest.TestCase):
    def test_task2_placeholder_strictly_clean(self):
        t2 = REPO_ROOT / "task2"
        self.assertTrue(t2.exists(), "task2 directory must exist")
        files = [f.name for f in t2.iterdir()]
        self.assertEqual(
            files,
            [".gitkeep"],
            f"task2/ must contain ONLY .gitkeep! Found: {files}",
        )

    def test_task3_placeholder_strictly_clean(self):
        t3 = REPO_ROOT / "task3"
        self.assertTrue(t3.exists(), "task3 directory must exist")
        files = [f.name for f in t3.iterdir()]
        self.assertEqual(
            files,
            [".gitkeep"],
            f"task3/ must contain ONLY .gitkeep! Found: {files}",
        )

    def test_manifests_exist(self):
        m_json = REPO_ROOT / "manifests" / "future_task_readiness.json"
        m_md = REPO_ROOT / "FUTURE_TASK_READINESS.md"
        self.assertTrue(m_json.exists(), "future_task_readiness.json must exist")
        self.assertTrue(m_md.exists(), "FUTURE_TASK_READINESS.md must exist")


if __name__ == "__main__":
    unittest.main()
