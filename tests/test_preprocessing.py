"""Compensation, explicit rest and per-axis event detector contracts."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from config_io import build_config
from csv_io import load_acceleration_csv
from event_detection import detect_event_frame
from models import AccelerationData
from paths import PROJECT_DIR
from pipeline import run_srs_pipeline
from preprocessing import apply_static_compensation, estimate_static_compensation
from smallwood import calculate_srs_1d, calculate_total_acceleration


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.data = load_acceleration_csv(
            PROJECT_DIR / "tests/fixtures/confirmed_rest.csv",
            expected_sample_rate_hz=6660,
        )

    def test_rest_vector_is_applied_once(self):
        compensation = estimate_static_compensation(self.data, 0.0, 0.01)
        np.testing.assert_array_equal(compensation.vector_g, [0, 0, 1])
        dynamic = apply_static_compensation(self.data, compensation)
        np.testing.assert_array_equal(dynamic.az_g, self.data.az_g - 1)
        np.testing.assert_array_equal(self.data.az_g[:60], np.ones(60))

    def test_raw_and_dynamic_magnitudes(self):
        compensation = estimate_static_compensation(self.data, 0.0, 0.01)
        raw = calculate_total_acceleration(self.data)
        dynamic = calculate_total_acceleration(
            apply_static_compensation(self.data, compensation)
        )
        np.testing.assert_array_equal(raw[:60], np.ones(60))
        np.testing.assert_array_equal(dynamic[:60], np.zeros(60))

    def test_rest_interval_outside_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the recording"):
            estimate_static_compensation(self.data, 50.0, 51.0)

    def test_too_short_rest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            estimate_static_compensation(self.data, 0.0, 0.0001)

    def test_noisy_rest_is_rejected(self):
        noisy = replace(self.data, ax_g=np.tile([0.0, 0.5], 150))
        with self.assertRaisesRegex(ValueError, "too noisy"):
            estimate_static_compensation(noisy, 0.0, 0.01, maximum_rest_std_g=0.1)

    def test_bad_gravity_vector_is_rejected(self):
        bad = replace(self.data, az_g=self.data.az_g * 3)
        with self.assertRaisesRegex(ValueError, "implausible"):
            estimate_static_compensation(
                bad, 0.0, 0.01, maximum_gravity_norm_deviation_g=0.35
            )

    def test_rest_peak_to_peak_limit(self):
        noisy = replace(self.data, ax_g=np.tile([0.0, 0.6], 150))
        with self.assertRaisesRegex(ValueError, "peak-to-peak"):
            estimate_static_compensation(
                noisy, 0.0, 0.01, maximum_rest_peak_to_peak_g=0.5
            )

    def test_detector_uses_axes_instead_of_total_magnitude(self):
        data = replace(
            self.data,
            ax_g=np.full(300, 2.0),
            ay_g=np.full(300, 2.0),
            az_g=np.full(300, 2.0),
        )
        self.assertGreater(calculate_total_acceleration(data)[0], 3.0)
        self.assertIsNone(detect_event_frame(data))

    def test_exact_threshold_does_not_trigger(self):
        self.assertIsNone(
            detect_event_frame(
                replace(
                    self.data,
                    ax_g=np.full(300, 3.0),
                    ay_g=np.zeros(300),
                    az_g=np.zeros(300),
                )
            )
        )

    def test_inclusive_frame_and_negative_axis_trigger(self):
        zeros = np.zeros(7)
        values = np.array([0, -1, -2, -4, -2, -1, 0], dtype=float)
        data = AccelerationData(np.arange(7) / 6660, zeros, values, zeros, 6660.0)
        frame = detect_event_frame(data)
        self.assertEqual(
            (frame.start_index, frame.end_index, frame.sample_count), (1, 5, 5)
        )
        self.assertEqual(frame.amax_g, 4.0)

    def test_validation_ignores_contest_compensation_settings(self):
        config = build_config(
            cli_overrides={
                "compensation.method": "confirmed_rest",
                "compensation.rest_start_s": 0.0,
                "compensation.rest_end_s": 0.01,
                "compensation.magnitude_mode": "dynamic_magnitude",
            }
        )
        result = run_srs_pipeline(
            PROJECT_DIR / "input/raw_signal_0051.csv", config, mode="validation"
        )
        self.assertIsNone(result.compensation)
        self.assertEqual(result.status, "validation_completed")

    def test_contest_can_report_no_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quiet.csv"
            values = np.zeros(40)
            np.savetxt(
                path,
                np.column_stack((np.arange(40) / 6660, values, values, values)),
                delimiter=",",
                header="t,ax,ay,az",
                comments="",
                fmt="%.17g",
            )
            config = build_config(cli_overrides={"input.gravity_compensated": True})
            result = run_srs_pipeline(path, config)
            self.assertEqual(result.status, "no_event")
            self.assertIsNone(result.srs_result)

    def test_double_compensation_is_rejected(self):
        config = build_config(
            cli_overrides={
                "input.gravity_compensated": True,
                "compensation.method": "confirmed_rest",
                "compensation.rest_start_s": 0.0,
                "compensation.rest_end_s": 0.01,
            }
        )
        with self.assertRaisesRegex(ValueError, "second compensation"):
            run_srs_pipeline(PROJECT_DIR / "tests/fixtures/confirmed_rest.csv", config)

    def test_explicit_raw_srs_policy_keeps_raw_axes(self):
        config = build_config(
            cli_overrides={
                "compensation.method": "confirmed_rest",
                "compensation.magnitude_mode": "raw_magnitude",
                "compensation.rest_start_s": 0.0,
                "compensation.rest_end_s": 0.01,
            }
        )
        result = run_srs_pipeline(
            PROJECT_DIR / "tests/fixtures/confirmed_rest.csv", config
        )
        expected = calculate_srs_1d(
            result.frame_data.az_g,
            result.srs_result.frequencies_hz,
            result.source_data.sample_rate_hz,
            10.0,
        )
        np.testing.assert_array_equal(result.srs_result.gz_g, expected)
