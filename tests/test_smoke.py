import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from pipeline import run_srs_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_example_file(self) -> None:
        config = load_config(
            "contest",
            config_file=PROJECT_ROOT / "examples" / "contest.json",
        )
        result = run_srs_pipeline(
            PROJECT_ROOT / "input" / "test_accelerometer_xyz.csv",
            config,
        )
        self.assertEqual(result.status, "event_processed")
        self.assertIsNotNone(result.event_frame)
        self.assertEqual(result.srs_table().shape, (94, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
