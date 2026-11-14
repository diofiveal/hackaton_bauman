"""Pointwise maxima across independently calculated XYZ spectra."""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from models import SRSResult
from paths import DEFAULT_ENVELOPE_PATH, DEFAULT_ENVELOPE_PLOT_PATH
from srs_io import read_srs_csv, save_envelope_csv


def _expand_csv_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Expand CSV paths, directories, and quoted wildcard patterns."""
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        entry = str(item)
        candidate = Path(entry)
        if candidate.is_dir():
            paths = sorted(candidate.glob("*.csv"))
        elif any(char in entry for char in "*?["):
            paths = [Path(value) for value in sorted(glob.glob(entry))]
        else:
            paths = [candidate]

        if not paths:
            raise FileNotFoundError(f"No SRS CSV files match: {entry}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"SRS CSV file not found: {path}")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Expected a .csv file: {path}")
            resolved = path.resolve()
            if resolved not in seen:
                found.append(path)
                seen.add(resolved)
    if not found:
        raise ValueError("Provide at least one precomputed SRS CSV file")
    return found


def calculate_envelope(spectra: Sequence[SRSResult]) -> SRSResult:
    """Compute separate pointwise maxima without rounding/interpolation."""
    if not spectra:
        raise ValueError("Cannot build an SRS envelope without spectra")
    grid = np.asarray(spectra[0].frequencies_hz, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("Invalid frequency grid")
    for i, spectrum in enumerate(spectra[1:], start=2):
        if not np.array_equal(np.asarray(spectrum.frequencies_hz), grid):
            raise ValueError(
                f"Spectrum #{i} has a different frequency grid. "
                "All frequencies must match exactly, in the same order."
            )
    for spectrum in spectra:
        if any(
            np.asarray(channel).shape != grid.shape
            for channel in (spectrum.gx_g, spectrum.gy_g, spectrum.gz_g)
        ):
            raise ValueError("A spectrum channel does not match the frequency grid")
    return SRSResult(
        frequencies_hz=grid.copy(),
        gx_g=np.maximum.reduce([s.gx_g for s in spectra]),
        gy_g=np.maximum.reduce([s.gy_g for s in spectra]),
        gz_g=np.maximum.reduce([s.gz_g for s in spectra]),
    )


def build_envelope_from_files(
    inputs: Sequence[str | Path],
    output_path: str | Path = DEFAULT_ENVELOPE_PATH,
    plot_path: str | Path = DEFAULT_ENVELOPE_PLOT_PATH,
    *,
    show: bool = False,
) -> SRSResult:
    """Read N ready-made SRS files, calculate envelope, export CSV and PNG."""
    from plots.envelope import plot_envelope

    paths = _expand_csv_paths(inputs)
    output_resolved = Path(output_path).resolve()
    if any(path.resolve() == output_resolved for path in paths):
        raise ValueError("Envelope output must not overwrite an input SRS CSV")
    spectra = [read_srs_csv(path) for path in paths]
    envelope = calculate_envelope(spectra)
    save_envelope_csv(envelope, output_path)
    plot_envelope(
        spectra, envelope, [path.name for path in paths], plot_path, show=show
    )
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make an SRS envelope from one or more ALREADY CALCULATED SRS CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="SRS CSV paths, directory, or quoted glob such as 'output/shock*_srs.csv'",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ENVELOPE_PATH, help="Envelope CSV output"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_ENVELOPE_PLOT_PATH,
        help="Envelope PNG output",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show Matplotlib plot after saving"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        paths = _expand_csv_paths(args.inputs)
        envelope = build_envelope_from_files(
            paths, output_path=args.output, plot_path=args.plot, show=args.show
        )
        print(f"SRS spectra: {len(paths)}")
        print(f"Frequency points: {envelope.frequencies_hz.size}")
        print(f"Envelope CSV: {args.output}")
        print(f"Envelope plot: {args.plot}")
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
