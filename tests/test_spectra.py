"""Round-trip CSV, envelope compatibility, curves and exported graphics."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from envelope import calculate_envelope
from models import SRSResult
from plots.comparison import plot_srs_comparison
from plots.envelope import plot_envelope
from plots.limits import plot_srs_limit, srs_limit_g
from srs_io import read_srs_csv, save_envelope_csv, save_srs_csv


class SpectraTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spectrum = SRSResult(
            np.array([10.0, 100.0, 1000.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
        )

    def test_srs_csv_round_trip(self):
        path = self.root / "srs.csv"
        save_srs_csv(self.spectrum, path)
        np.testing.assert_array_equal(
            read_srs_csv(path).as_table(), self.spectrum.as_table()
        )

    def test_additional_total_is_ignored_by_xyz_loader(self):
        spectrum = replace(self.spectrum, g_total_g=np.array([5.0, 6.0, 7.0]))
        path = self.root / "total.csv"
        save_srs_csv(spectrum, path)
        self.assertEqual(spectrum.as_table().shape, (3, 5))
        self.assertEqual(read_srs_csv(path).as_table().shape, (3, 4))

    def test_pointwise_envelope_and_export(self):
        second = replace(self.spectrum, gx_g=np.array([4.0, 1.0, 5.0]))
        envelope = calculate_envelope([self.spectrum, second])
        np.testing.assert_array_equal(envelope.gx_g, [4, 2, 5])
        path = self.root / "envelope.csv"
        save_envelope_csv(envelope, path)
        self.assertEqual(path.read_text().splitlines()[0], "f,ax,ay,az")
        np.testing.assert_array_equal(
            read_srs_csv(path).as_table(), envelope.as_table()
        )

    def test_envelope_rejects_rounded_frequency_grid(self):
        second = replace(
            self.spectrum, frequencies_hz=self.spectrum.frequencies_hz + 0.001
        )
        with self.assertRaisesRegex(ValueError, "different frequency grid"):
            calculate_envelope([self.spectrum, second])

    def test_invalid_spectrum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            replace(self.spectrum, gx_g=np.array([-1.0, 1.0, 1.0]))

    def test_known_limit_values_and_undefined_range(self):
        actual = srs_limit_g(np.array([5, 10, 100, 1000, 1500, 2000, 2500]))
        np.testing.assert_array_equal(actual[1:-1], [10, 20, 20, 35, 50])
        self.assertTrue(np.isnan(actual[0]) and np.isnan(actual[-1]))

    def test_comparison_envelope_and_limit_plots(self):
        source = self.root / "srs.csv"
        save_srs_csv(self.spectrum, source)
        for layout in ("combined", "stacked"):
            path = self.root / f"comparison_{layout}.png"
            plot_srs_comparison(source, source, path, layout=layout)
            self.assertGreater(path.stat().st_size, 1000)
        limit = self.root / "limit.png"
        plot_srs_limit(limit)
        envelope = self.root / "envelope.png"
        plot_envelope([self.spectrum], self.spectrum, ["test"], envelope)
        self.assertGreater(limit.stat().st_size, 1000)
        self.assertGreater(envelope.stat().st_size, 1000)
