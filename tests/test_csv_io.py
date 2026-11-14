"""Input contracts: units, delimiters, timestamps and invalid samples."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from csv_io import load_acceleration_csv


class CsvTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "signal.csv"

    def load(self, text, **kwargs):
        self.path.write_text(text, encoding="utf-8")
        return load_acceleration_csv(self.path, **kwargs)

    def test_seconds_keep_the_original_time_origin(self):
        result = self.load("time_s,ax,ay,az\n12,1,2,3\n12.001,1,2,3\n12.002,1,2,3\n")
        np.testing.assert_array_equal(result.time_s, [12.0, 12.001, 12.002])
        self.assertAlmostEqual(result.sample_rate_hz, 1000.0)

    def test_microseconds_are_converted_without_rebasing(self):
        result = self.load(
            "t,ax,ay,az\n84362138,0,0,1\n84362288,0,0,1\n84362438,0,0,1\n",
            expected_sample_rate_hz=6660,
        )
        self.assertEqual(result.detected_time_unit, "microseconds")
        self.assertAlmostEqual(result.time_s[0], 84.362138)

    def test_three_delimiters_and_aliases(self):
        for delimiter in (",", ";", "\t"):
            with self.subTest(delimiter=delimiter):
                text = "\n".join(
                    delimiter.join(row)
                    for row in (
                        ("time_s", "accel_x", "accel_y", "accel_z"),
                        ("0", "1", "2", "3"),
                        ("0.001", "1", "2", "3"),
                        ("0.002", "1", "2", "3"),
                    )
                )
                self.assertEqual(self.load(text).sample_count, 3)

    def test_legacy_two_column_input(self):
        result = self.load("Time_sec;Acceleration_g\n0;1\n0.001;2\n0.002;3\n")
        self.assertEqual(result.source_format, "legacy_2_column")
        np.testing.assert_array_equal(result.ax_g, [1, 2, 3])
        np.testing.assert_array_equal(result.ay_g, [0, 0, 0])

    def test_nonfinite_values(self):
        for value in ("nan", "inf", "-inf"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "NaN or infinity"),
            ):
                self.load(f"t,ax,ay,az\n0,0,0,0\n0.001,{value},0,0\n0.002,0,0,0\n")

    def test_nonmonotonic_time(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.load("t,ax,ay,az\n0,0,0,0\n0.001,0,0,0\n0.001,0,0,0\n")

    def test_timing_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Timing validation"):
            self.load(
                "time_s,ax,ay,az\n0,0,0,0\n0.001,0,0,0\n0.002,0,0,0\n0.005,0,0,0\n"
            )

    def test_timing_warnings_can_be_inspected_explicitly(self):
        result = self.load(
            "time_s,ax,ay,az\n0,0,0,0\n0.001,0,0,0\n0.002,0,0,0\n0.005,0,0,0\n",
            strict_timing=False,
        )
        self.assertEqual(result.estimated_missing_samples, 2)
        self.assertTrue(result.warnings)

    def test_conflicting_time_units(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.load(
                "time_s,ax,ay,az\n0,0,0,0\n1,0,0,0\n2,0,0,0\n", time_unit="microseconds"
            )

    def test_srs_csv_is_not_a_time_record(self):
        with self.assertRaisesRegex(ValueError, "time column"):
            self.load("f,gx,gy,gz\n10,1,2,3\n20,1,2,3\n30,1,2,3\n")

    def test_duplicate_headers(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load("t,ax,ax,az\n0,0,0,0\n1,0,0,0\n2,0,0,0\n")
