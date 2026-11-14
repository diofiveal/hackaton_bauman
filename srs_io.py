"""Read/write calculated spectra, with distinct acceleration and SRS headers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from models import SRSResult


def read_srs_csv(path: str | Path) -> SRSResult:
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
        normalized = dict(zip(headers, original_headers, strict=True))

        accepted_headers = (
            ("f_hz", "gx_g", "gy_g", "gz_g"),
            ("f", "gx", "gy", "gz"),
            ("f", "ax", "ay", "az"),
            ("f_hz", "gx", "gy", "gz"),
        )
        selected = next(
            (
                option
                for option in accepted_headers
                if all(key in normalized for key in option)
            ),
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
                raise ValueError(
                    f"Invalid SRS data in {path} at line {line_no}"
                ) from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"Non-finite SRS values in {path} at line {line_no}")
            rows.append(numeric)

    if not rows:
        raise ValueError(f"No SRS data rows in {path}")
    values = np.asarray(rows, dtype=np.float64)
    frequencies = values[:, 0]
    if np.any(frequencies <= 0) or not np.all(np.diff(frequencies) > 0):
        raise ValueError(
            f"Frequencies must be positive and strictly increasing: {path}"
        )
    if np.any(values[:, 1:] < 0):
        raise ValueError(f"Maximax SRS must be nonnegative: {path}")

    return SRSResult(
        frequencies_hz=frequencies.copy(),
        gx_g=values[:, 1].copy(),
        gy_g=values[:, 2].copy(),
        gz_g=values[:, 3].copy(),
    )


def save_srs_csv(spectrum: SRSResult, output_path: str | Path) -> None:
    """Export float64 SRS values using the original 12-significant-digit format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "f_hz,gx_g,gy_g,gz_g"
    if spectrum.g_total_g is not None:
        header += ",g_total_g"
    np.savetxt(
        path,
        spectrum.as_table(),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12g",
    )


def save_envelope_csv(envelope: SRSResult, output_path: str | Path) -> None:
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
