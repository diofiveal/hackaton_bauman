"""Command-line arguments grouped by processing responsibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from paths import (
    DEFAULT_ACCELERATION_PLOT_PATH,
    DEFAULT_ENVELOPE_PATH,
    DEFAULT_ENVELOPE_PLOT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_SRS_PATH,
)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a shock event and calculate Maximax SRS.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=("Input [t, ax, ay, az] CSV (default: input/raw_signal_0051.csv)"),
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "contest"),
        default=None,
        help="Processing mode: validation=whole raw record; contest=detected 3g shock frame",
    )
    parser.add_argument(
        "--profile",
        choices=("contest", "custom"),
        default="contest",
        help="Base configuration profile (default: contest)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional JSON configuration file",
    )
    _add_compensation_arguments(parser)
    _add_tail_arguments(parser)
    _add_output_arguments(parser)
    _add_envelope_arguments(parser)
    return parser


def _add_compensation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rest-start",
        type=float,
        help="Confirmed rest interval start in seconds",
    )
    parser.add_argument(
        "--rest-end",
        type=float,
        help="Confirmed rest interval end in seconds",
    )
    parser.add_argument(
        "--no-compensation",
        action="store_true",
        help="Disable rest compensation (contest then needs --input-gravity-compensated)",
    )
    parser.add_argument(
        "--input-gravity-compensated",
        action="store_true",
        help="Assert CSV already excludes gravity and bias (do not use on raw data)",
    )


def _add_tail_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tail-mode",
        choices=("frame_only", "zero_tail"),
        help="Override oscillator tail mode",
    )
    parser.add_argument(
        "--tail-time-constants",
        type=float,
        help="Decay duration in time constants for zero_tail",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SRS_PATH,
        help="Output SRS CSV path",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_ACCELERATION_PLOT_PATH,
        help="Output PNG path for acceleration visualization",
    )


def _add_envelope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--envelope-inputs",
        nargs="*",
        default=[],
        metavar="SRS_CSV",
        help=(
            "Previously saved SRS CSVs to combine with the new SRS. "
            "Accepts files and quoted glob patterns."
        ),
    )
    parser.add_argument(
        "--envelope-output",
        type=Path,
        default=DEFAULT_ENVELOPE_PATH,
        help="Output CSV for the SRS envelope",
    )
    parser.add_argument(
        "--envelope-plot",
        type=Path,
        default=DEFAULT_ENVELOPE_PLOT_PATH,
        help="Output PNG for the SRS envelope",
    )
    parser.add_argument(
        "--no-envelope",
        action="store_true",
        help="Skip automatic envelope generation",
    )
