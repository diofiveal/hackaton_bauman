"""Frozen float64 results captured from the uploaded code before refactoring."""

import unittest

import numpy as np

from config_io import build_config
from paths import PROJECT_DIR
from pipeline import run_srs_pipeline


class RegressionTest(unittest.TestCase):
    def test_five_original_records_and_both_tail_modes(self):
        with np.load(PROJECT_DIR / "tests/fixtures/baseline_srs.npz") as expected:
            for path in sorted((PROJECT_DIR / "input").glob("raw_signal*.csv")):
                for tail in ("frame_only", "zero_tail"):
                    with self.subTest(record=path.name, tail=tail):
                        config = build_config(
                            cli_overrides={
                                "processing_mode": "validation",
                                "tail_mode": tail,
                                "tail_time_constants": 1.0,
                            }
                        )
                        result = run_srs_pipeline(path, config)
                        self.assertEqual(result.status, "validation_completed")
                        self.assertEqual(result.srs_table().shape, (94, 5))
                        np.testing.assert_array_equal(
                            result.srs_table(), expected[f"{path.stem}_{tail}"]
                        )

    def test_confirmed_rest_event_and_both_tail_modes(self):
        with np.load(PROJECT_DIR / "tests/fixtures/baseline_srs.npz") as expected:
            for tail in ("frame_only", "zero_tail"):
                with self.subTest(tail=tail):
                    config = build_config(
                        cli_overrides={
                            "compensation.method": "confirmed_rest",
                            "compensation.magnitude_mode": "dynamic_magnitude",
                            "compensation.rest_start_s": 0.0,
                            "compensation.rest_end_s": 0.01,
                            "tail_mode": tail,
                            "tail_time_constants": 1.0,
                        }
                    )
                    result = run_srs_pipeline(
                        PROJECT_DIR / "tests/fixtures/confirmed_rest.csv", config
                    )
                    self.assertEqual(result.status, "event_processed")
                    self.assertEqual(
                        (result.event_frame.start_index, result.event_frame.end_index),
                        (123, 147),
                    )
                    self.assertEqual(result.event_frame.amax_g, 5.0)
                    np.testing.assert_array_equal(
                        result.srs_table(), expected[f"confirmed_rest_{tail}"]
                    )

    def test_raw_contest_requires_known_gravity_handling(self):
        with self.assertRaisesRegex(ValueError, "excluding gravity"):
            run_srs_pipeline(PROJECT_DIR / "input/raw_signal_0051.csv", build_config())


if __name__ == "__main__":
    unittest.main()
