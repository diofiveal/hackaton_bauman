"""Configuration precedence, fixed contest settings and research grid modes."""

import unittest

import numpy as np

from config import FrequencyGridConfig
from config_io import build_config
from frequency_grid import frequencies_from_config, load_frequency_grid_csv
from paths import PROJECT_DIR


class ConfigGridTest(unittest.TestCase):
    def test_configuration_precedence(self):
        result = build_config(
            "custom",
            file_overrides={"sample_rate_hz": 5000},
            cli_overrides={"sample_rate_hz": 8000},
        )
        self.assertEqual(result.sample_rate_hz, 8000)

    def test_q_and_damping_overrides(self):
        by_q = build_config("custom", cli_overrides={"quality_factor_q": 5})
        by_zeta = build_config("custom", cli_overrides={"damping_ratio_zeta": 0.1})
        self.assertEqual(by_q.damping_ratio_zeta, 0.1)
        self.assertEqual(by_zeta.quality_factor_q, 5)

    def test_contest_settings_are_locked(self):
        with self.assertRaisesRegex(ValueError, "fixed"):
            build_config(cli_overrides={"quality_factor_q": 5})

    def test_unknown_configuration_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration parameter"):
            build_config(cli_overrides={"typo": 1})

    def test_fixed_csv_is_not_regenerated(self):
        expected = np.loadtxt(PROJECT_DIR / "data/frequency_grid.csv")
        result, warnings = frequencies_from_config(FrequencyGridConfig())
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.size, 94)
        self.assertTrue(warnings)

    def test_three_research_grid_sources(self):
        for source in ("explicit", "fractional_octave", "logarithmic"):
            with self.subTest(source=source):
                config = FrequencyGridConfig(
                    source=source, frequencies_hz=(10, 20, 40), point_count=15
                )
                grid, _ = frequencies_from_config(config)
                self.assertTrue(np.all(np.diff(grid) > 0))
                if source == "explicit":
                    np.testing.assert_array_equal(grid, [10, 20, 40])
                if source == "logarithmic":
                    self.assertEqual(grid.size, 15)
                    self.assertEqual(grid[-1], 2500)

    def test_expected_frequency_count(self):
        with self.assertRaisesRegex(ValueError, "expected 90"):
            load_frequency_grid_csv(
                PROJECT_DIR / "data/frequency_grid.csv", expected_count=90
            )

    def test_invalid_modes_are_rejected(self):
        for overrides in (
            {"tail_mode": "bad"},
            {"input.acceleration_unit": "m/s2"},
            {"compensation.magnitude_mode": "bad"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                build_config("custom", cli_overrides=overrides)
