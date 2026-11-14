"""Read accelerometer CSV columns; validate samples without filtering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from models import AccelerationData
from timing import TimeUnit, choose_time_unit, validate_timing

_TIME_ALIASES = {
    "t",
    "time",
    "time_s",
    "time_sec",
    "timestamp",
    "timestamp_s",
    "t_us",
    "time_us",
    "timestamp_us",
    "time_usec",
}
_AX_ALIASES = {"ax", "ax_g", "accel_x", "acceleration_x"}
_AY_ALIASES = {"ay", "ay_g", "accel_y", "acceleration_y"}
_AZ_ALIASES = {"az", "az_g", "accel_z", "acceleration_z"}
_LEGACY_ACCEL_ALIASES = {
    "acceleration_g",
    "acceleration",
    "accel_g",
    "accel",
}


def _normalise_header(name: str) -> str:
    return name.lstrip("\ufeff").strip().lower().replace(" ", "_")


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
    if not sample.strip():
        raise ValueError(f"CSV file is empty: {path}")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        counts = {delimiter: first_line.count(delimiter) for delimiter in ",;\t"}
        delimiter = max(counts, key=counts.get)
        if counts[delimiter] == 0:
            raise ValueError(
                "Cannot detect CSV delimiter; expected comma, semicolon, or tab"
            ) from None
        return delimiter


def _find_column(headers: list[str], aliases: set[str], label: str) -> int:
    matches = [index for index, header in enumerate(headers) if header in aliases]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label} column, found {len(matches)}")
    return matches[0]


def _read_numeric_rows(path: Path, delimiter: str) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            raw_headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc

        headers = [_normalise_header(header) for header in raw_headers]
        if not headers or any(not header for header in headers):
            raise ValueError("CSV header contains an empty column name")
        if len(set(headers)) != len(headers):
            raise ValueError("CSV header contains duplicate column names")

        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"Line {line_number}: expected {len(headers)} values, got {len(row)}"
                )
            try:
                rows.append([float(value.strip()) for value in row])
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: non-numeric value") from exc

    if len(rows) < 3:
        raise ValueError("CSV must contain at least three data rows")

    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad_row, bad_column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(
            "CSV contains NaN or infinity at data row "
            f"{bad_row + 2}, column '{headers[bad_column]}'"
        )
    return headers, values


def _read_axes(
    headers: list[str], values: np.ndarray, allow_legacy_two_column: bool
) -> tuple:
    """Select all three axes, or map the explicit legacy channel to X."""
    has_ax = any(header in _AX_ALIASES for header in headers)
    has_ay = any(header in _AY_ALIASES for header in headers)
    has_az = any(header in _AZ_ALIASES for header in headers)
    has_legacy = any(header in _LEGACY_ACCEL_ALIASES for header in headers)

    if has_ax or has_ay or has_az:
        ax_index = _find_column(headers, _AX_ALIASES, "ax")
        ay_index = _find_column(headers, _AY_ALIASES, "ay")
        az_index = _find_column(headers, _AZ_ALIASES, "az")
        ax_g = values[:, ax_index].copy()
        ay_g = values[:, ay_index].copy()
        az_g = values[:, az_index].copy()
        source_format: Literal["triaxial", "legacy_2_column"] = "triaxial"
    elif has_legacy and allow_legacy_two_column:
        acceleration_index = _find_column(
            headers, _LEGACY_ACCEL_ALIASES, "legacy acceleration"
        )
        ax_g = values[:, acceleration_index].copy()
        ay_g = np.zeros_like(ax_g)
        az_g = np.zeros_like(ax_g)
        source_format = "legacy_2_column"
    else:
        raise ValueError(
            "Expected columns [t, ax, ay, az] or legacy [Time_sec, Acceleration_g]"
        )
    return ax_g, ay_g, az_g, source_format


def load_acceleration_csv(
    path: str | Path,
    *,
    time_unit: TimeUnit = "auto",
    expected_sample_rate_hz: float | None = None,
    sample_rate_tolerance_fraction: float = 0.02,
    interval_tolerance_fraction: float = 0.02,
    strict_timing: bool = True,
    allow_legacy_two_column: bool = True,
) -> AccelerationData:
    """Read XYZ in g and time in seconds, preserving the timestamp origin.

    Header suffixes or time_unit select seconds/microseconds. For a generic
    header the expected sample rate is used. Timing warnings raise in strict
    mode. Legacy two-column input maps its acceleration to X and zeros Y/Z.
    No resampling, filtering or gravity removal is performed here.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    if time_unit not in {"auto", "seconds", "microseconds"}:
        raise ValueError("time_unit must be 'auto', 'seconds', or 'microseconds'")

    delimiter = _detect_delimiter(input_path)
    headers, values = _read_numeric_rows(input_path, delimiter)
    time_index = _find_column(headers, _TIME_ALIASES, "time")
    time_header = headers[time_index]

    ax_g, ay_g, az_g, source_format = _read_axes(
        headers, values, allow_legacy_two_column
    )

    raw_time = values[:, time_index].copy()
    if not np.all(np.diff(raw_time) > 0):
        index = int(np.flatnonzero(np.diff(raw_time) <= 0)[0])
        raise ValueError(
            f"Time must be strictly increasing; violation between samples {index} and {index + 1}"
        )

    detected_unit = choose_time_unit(
        raw_time, time_header, time_unit, expected_sample_rate_hz
    )
    time_s = raw_time * (1.0e-6 if detected_unit == "microseconds" else 1.0)

    sample_rate_hz, max_deviation_pct, missing, warnings = validate_timing(
        time_s,
        expected_sample_rate_hz=expected_sample_rate_hz,
        sample_rate_tolerance_fraction=sample_rate_tolerance_fraction,
        interval_tolerance_fraction=interval_tolerance_fraction,
        strict_timing=strict_timing,
    )

    return AccelerationData(
        time_s=time_s,
        ax_g=ax_g,
        ay_g=ay_g,
        az_g=az_g,
        sample_rate_hz=sample_rate_hz,
        source_format=source_format,
        detected_time_unit=detected_unit,
        max_interval_deviation_percent=max_deviation_pct,
        estimated_missing_samples=missing,
        warnings=warnings,
    )
