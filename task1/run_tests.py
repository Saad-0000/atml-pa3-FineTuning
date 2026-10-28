import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(repo_root / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())

