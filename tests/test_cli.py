"""Real command-line launches verify wiring and serialized outputs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paths import PROJECT_DIR


class CliTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(PROJECT_DIR / "main.py"), *map(str, arguments)],
            cwd=self.output,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def output_args(self):
        return [
            "--output",
            self.output / "srs.csv",
            "--metadata-output",
            self.output / "metadata.json",
            "--plot-output",
            self.output / "acceleration.png",
            "--envelope-output",
            self.output / "envelope.csv",
            "--envelope-plot",
            self.output / "envelope.png",
        ]

    def test_validation_cli_from_another_working_directory(self):
        result = self.run_cli(
            PROJECT_DIR / "input/raw_signal_0051.csv",
            "--mode",
            "validation",
            *self.output_args(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.output / "metadata.json").read_text())
        self.assertEqual(metadata["processing_mode"], "validation")
        self.assertEqual(metadata["sample_count"], 450)
        self.assertFalse(metadata["compensation_applied"])
        self.assertTrue((self.output / "acceleration.png").is_file())
        self.assertFalse((self.output / "envelope.csv").exists())

    def test_contest_cli_with_confirmed_rest(self):
        result = self.run_cli(
            PROJECT_DIR / "tests/fixtures/confirmed_rest.csv",
            "--mode",
            "contest",
            "--rest-start",
            "0",
            "--rest-end",
            "0.01",
            *self.output_args(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.output / "metadata.json").read_text())
        self.assertEqual(metadata["event"]["start_index"], 123)
        self.assertEqual(metadata["event"]["end_index"], 147)
        self.assertTrue(metadata["compensation_applied"])
        self.assertTrue((self.output / "envelope.csv").is_file())
        self.assertTrue((self.output / "envelope.png").is_file())

    def test_invalid_flag_combination_returns_error(self):
        result = self.run_cli(
            "--mode", "validation", "--rest-start", "0", "--rest-end", "0.01"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no compensation flags", result.stderr)

    def test_raw_contest_returns_actionable_error(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("verified rest interval", result.stderr)
