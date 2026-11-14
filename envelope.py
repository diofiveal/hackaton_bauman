"""Build a pointwise Maximax SRS envelope from ALREADY CALCULATED SRS CSV files.

No acceleration CSV and no Smallwood recalculation are needed.

Supported input examples:
    f_hz,gx_g,gy_g,gz_g,g_total_g    (main.py output)
    f,gx,gy,gz                     (STM32 SRS converted to CSV)
    f,ax,ay,az                     (previously exported envelope)

Each input should represent the SRS spectrum of ONE independent shock.
All input spectra must use exactly the same frequency grid and SRS settings.
CSV files do not contain Q or calculation mode; ensure they match yourself.

Usage (from the project root):
    python envelope.py output/srs_result.csv
    python envelope.py output/shock1_srs.csv output/shock2_srs.csv
    python envelope.py "output/shock*_srs.csv" --show

Outputs:
    output/srs_envelope.csv         (f,ax,ay,az)
    output/srs_envelope.png         (individual SRS spectra and envelope)
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "output" / "srs_envelope.csv"
DEFAULT_PLOT = PROJECT_DIR / "output" / "srs_envelope.png"


@dataclass(frozen=True)
class SpectrumCSV:
    """One precomputed three-axis SRS spectrum (all accelerations in g)."""

    frequencies_hz: np.ndarray
    gx_g: np.ndarray
    gy_g: np.ndarray
    gz_g: np.ndarray


@dataclass(frozen=True)
class EnvelopeResult:
    """Pointwise maximum across all supplied three-axis SRS spectra."""

    frequencies_hz: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray

    def as_table(self) -> np.ndarray:
        return np.column_stack((self.frequencies_hz, self.ax_g, self.ay_g, self.az_g))


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


def read_srs_csv(path: str | Path) -> SpectrumCSV:
    """Read a precomputed SRS CSV, validating headers, grid, and values.

    The fifth ``g_total_g`` column written by main.py is ignored: the
    contest envelope is calculated separately for X, Y, and Z.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        first_lines = stream.read(4096)
        if not first_lines.strip():
            raise ValueError(f"Empty SRS CSV: {path}")
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(first_lines, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")
        original_headers = list(reader.fieldnames)
        headers = [header.strip().lower() for header in original_headers]
        if len(set(headers)) != len(headers):
            raise ValueError(f"Duplicate CSV column in {path}")
        normalized = dict(zip(headers, original_headers))

        accepted_headers = (
            ("f_hz", "gx_g", "gy_g", "gz_g"),
            ("f", "gx", "gy", "gz"),
            ("f", "ax", "ay", "az"),
            ("f_hz", "gx", "gy", "gz"),
        )
        selected = next(
            (option for option in accepted_headers if all(key in normalized for key in option)),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Unsupported SRS CSV header in {path}: {original_headers}. "
                "Expected f_hz,gx_g,gy_g,gz_g[,g_total_g] or f,gx,gy,gz. "
                "An acceleration CSV (t,ax,ay,az) is not an SRS spectrum."
            )

        rows: list[list[float]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row or all(v is None or not str(v).strip() for v in row.values()):
                continue
            try:
                numeric = [float(row[normalized[key]]) for key in selected]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid SRS data in {path} at line {line_no}") from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"Non-finite SRS values in {path} at line {line_no}")
            rows.append(numeric)

    if not rows:
        raise ValueError(f"No SRS data rows in {path}")
    values = np.asarray(rows, dtype=np.float64)
    frequencies = values[:, 0]
    if np.any(frequencies <= 0) or not np.all(np.diff(frequencies) > 0):
        raise ValueError(f"Frequencies must be positive and strictly increasing: {path}")
    if np.any(values[:, 1:] < 0):
        raise ValueError(f"Maximax SRS must be nonnegative: {path}")

    return SpectrumCSV(
        frequencies_hz=frequencies.copy(),
        gx_g=values[:, 1].copy(),
        gy_g=values[:, 2].copy(),
        gz_g=values[:, 3].copy(),
    )


def calculate_envelope(spectra: Sequence[SpectrumCSV]) -> EnvelopeResult:
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
        if any(np.asarray(channel).shape != grid.shape for channel in (
            spectrum.gx_g, spectrum.gy_g, spectrum.gz_g
        )):
            raise ValueError("A spectrum channel does not match the frequency grid")
    return EnvelopeResult(
        frequencies_hz=grid.copy(),
        ax_g=np.maximum.reduce([s.gx_g for s in spectra]),
        ay_g=np.maximum.reduce([s.gy_g for s in spectra]),
        az_g=np.maximum.reduce([s.gz_g for s in spectra]),
    )


def save_envelope_csv(envelope: EnvelopeResult, output_path: str | Path) -> None:
    """Export the requested contest format f,ax,ay,az."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        envelope.as_table(),
        delimiter=",",
        header="f,ax,ay,az",
        comments="",
        fmt="%.17g",
    )


def plot_envelope(
    spectra: Sequence[SpectrumCSV],
    envelope: EnvelopeResult,
    source_names: Sequence[str],
    output_path: str | Path,
    *,
    dpi: int = 160,
    show: bool = False,
) -> None:
    """Display every input SRS (thin gray) and the envelope (bold color)."""
    import matplotlib.pyplot as plt

    if len(spectra) != len(source_names):
        raise ValueError("Each SRS spectrum needs a source name")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    series = (
        ("X", "gx_g", envelope.ax_g, "tab:blue"),
        ("Y", "gy_g", envelope.ay_g, "tab:green"),
        ("Z", "gz_g", envelope.az_g, "tab:red"),
    )
    for ax, (axis_name, value_attr, maximum, color) in zip(axes, series):
        for i, (spectrum, name) in enumerate(zip(spectra, source_names)):
            label = name if len(spectra) <= 6 else ("Individual shocks" if i == 0 else None)
            ax.plot(
                envelope.frequencies_hz,
                getattr(spectrum, value_attr),
                color="0.55",
                linewidth=1.1,
                alpha=0.6,
                label=label,
            )
        ax.plot(
            envelope.frequencies_hz,
            maximum,
            color=color,
            linewidth=2.5,
            label=f"Envelope {axis_name}",
            zorder=10,
        )
        ax.set_xscale("log")
        ax.set_ylabel(f"SRS {axis_name}, g")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Frequency, Hz")
    axes[-1].set_xlim(float(envelope.frequencies_hz[0]), float(envelope.frequencies_hz[-1]))
    fig.suptitle(f"Maximax SRS envelope: {len(spectra)} shock(s)")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def build_envelope_from_files(
    inputs: Sequence[str | Path],
    output_path: str | Path = DEFAULT_CSV,
    plot_path: str | Path = DEFAULT_PLOT,
    *,
    show: bool = False,
) -> EnvelopeResult:
    """Read N ready-made SRS files, calculate envelope, export CSV and PNG."""
    paths = _expand_csv_paths(inputs)
    output_resolved = Path(output_path).resolve()
    if any(path.resolve() == output_resolved for path in paths):
        raise ValueError("Envelope output must not overwrite an input SRS CSV")
    spectra = [read_srs_csv(path) for path in paths]
    envelope = calculate_envelope(spectra)
    save_envelope_csv(envelope, output_path)
    plot_envelope(spectra, envelope, [path.name for path in paths], plot_path, show=show)
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make an SRS envelope from one or more ALREADY CALCULATED SRS CSV files."
    )
    parser.add_argument(
        "inputs", nargs="+", help="SRS CSV paths, directory, or quoted glob such as 'output/shock*_srs.csv'"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV, help="Envelope CSV output")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="Envelope PNG output")
    parser.add_argument("--show", action="store_true", help="Show Matplotlib plot after saving")
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
