"""Reading and validating accelerometer CSV files.

Supported input formats
-----------------------
New format::

    t, ax, ay, az

Legacy regression format::

    Time_sec;Acceleration_g

All returned time values are seconds and all acceleration values are g.
The loader never resamples, interpolates, filters, or removes gravity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


TimeUnit = Literal["auto", "seconds", "microseconds"]


@dataclass(frozen=True)
class AccelerationData:
    """Validated accelerometer samples in common units."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    sample_rate_hz: float
    source_format: Literal["triaxial", "legacy_2_column"]
    detected_time_unit: Literal["seconds", "microseconds"]
    max_interval_deviation_percent: float
    estimated_missing_samples: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)


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
_SECONDS_HEADERS = {"time_s", "time_sec", "timestamp_s"}
_MICROSECONDS_HEADERS = {"t_us", "time_us", "timestamp_us", "time_usec"}


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
            raise ValueError("Cannot detect CSV delimiter; expected comma, semicolon, or tab")
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


def _header_time_unit(time_header: str) -> Literal["seconds", "microseconds"] | None:
    if time_header in _SECONDS_HEADERS:
        return "seconds"
    if time_header in _MICROSECONDS_HEADERS:
        return "microseconds"
    return None


def _choose_time_unit(
    raw_time: np.ndarray,
    time_header: str,
    requested_unit: TimeUnit,
    expected_sample_rate_hz: float | None,
) -> Literal["seconds", "microseconds"]:
    header_unit = _header_time_unit(time_header)
    if requested_unit != "auto":
        if header_unit is not None and requested_unit != header_unit:
            raise ValueError(
                f"Requested time unit '{requested_unit}' conflicts with header '{time_header}'"
            )
        return requested_unit

    if header_unit is not None:
        return header_unit

    raw_dt = float(np.median(np.diff(raw_time)))
    if expected_sample_rate_hz is not None:
        if not np.isfinite(expected_sample_rate_hz) or expected_sample_rate_hz <= 0:
            raise ValueError("expected_sample_rate_hz must be positive and finite")
        candidate_rates = {
            "seconds": 1.0 / raw_dt,
            "microseconds": 1.0e6 / raw_dt,
        }
        return min(
            candidate_rates,
            key=lambda unit: abs(candidate_rates[unit] / expected_sample_rate_hz - 1.0),
        )

    # A generic column named 't' is common. This conservative heuristic covers
    # normal high-rate measurements while rejecting genuinely ambiguous scales.
    if raw_dt < 0.1:
        return "seconds"
    if raw_dt >= 1.0:
        return "microseconds"
    raise ValueError(
        "Ambiguous time unit in generic time column; pass time_unit='seconds' "
        "or time_unit='microseconds'"
    )


def _validate_timing(
    time_s: np.ndarray,
    *,
    expected_sample_rate_hz: float | None,
    sample_rate_tolerance_fraction: float,
    interval_tolerance_fraction: float,
    strict_timing: bool,
) -> tuple[float, float, int, tuple[str, ...]]:
    if not np.all(np.diff(time_s) > 0):
        index = int(np.flatnonzero(np.diff(time_s) <= 0)[0])
        raise ValueError(
            f"Time must be strictly increasing; violation between samples {index} and {index + 1}"
        )
    if not 0 <= sample_rate_tolerance_fraction < 1:
        raise ValueError("sample_rate_tolerance_fraction must be in [0, 1)")
    if not 0 <= interval_tolerance_fraction < 1:
        raise ValueError("interval_tolerance_fraction must be in [0, 1)")

    intervals = np.diff(time_s)
    nominal_dt = float(np.median(intervals))
    # Use the whole record for the measured rate so that integer-microsecond
    # timestamp rounding (for example alternating 150/151 us) does not bias it.
    sample_rate_hz = float((time_s.size - 1) / (time_s[-1] - time_s[0]))
    relative_deviation = np.abs(intervals - nominal_dt) / nominal_dt
    maximum_deviation = float(np.max(relative_deviation))

    large_gaps = intervals > 1.5 * nominal_dt
    estimated_missing = int(
        np.sum(np.maximum(np.rint(intervals[large_gaps] / nominal_dt).astype(int) - 1, 0))
    )

    warnings: list[str] = []
    if estimated_missing:
        warnings.append(f"Estimated missing samples: {estimated_missing}")
    if maximum_deviation > interval_tolerance_fraction:
        warnings.append(
            "Maximum sampling-interval deviation is "
            f"{100.0 * maximum_deviation:.4f}%"
        )

    if expected_sample_rate_hz is not None:
        if not np.isfinite(expected_sample_rate_hz) or expected_sample_rate_hz <= 0:
            raise ValueError("expected_sample_rate_hz must be positive and finite")
        rate_error = abs(sample_rate_hz / expected_sample_rate_hz - 1.0)
        if rate_error > sample_rate_tolerance_fraction:
            warnings.append(
                f"Measured sample rate {sample_rate_hz:.6f} Hz differs from expected "
                f"{expected_sample_rate_hz:.6f} Hz by {100.0 * rate_error:.4f}%"
            )

    if strict_timing and warnings:
        raise ValueError("Timing validation failed: " + "; ".join(warnings))

    return (
        sample_rate_hz,
        100.0 * maximum_deviation,
        estimated_missing,
        tuple(warnings),
    )


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
    """Load and validate a three-axis or legacy accelerometer CSV.

    Parameters
    ----------
    path:
        Input CSV path.
    time_unit:
        ``auto``, ``seconds``, or ``microseconds``. Explicit header suffixes
        have priority in auto mode. For a generic ``t`` header, the expected
        sample rate is used when supplied.
    expected_sample_rate_hz:
        Optional nominal rate, for example ``DEFAULT_CONFIG.sample_rate_hz``.
        The measured rate is always returned; the nominal value is only used
        for unit selection and validation.
    strict_timing:
        If true, missing samples, excessive interval jitter, and excessive
        sample-rate error raise ``ValueError``. If false, they are reported in
        ``AccelerationData.warnings`` without modifying samples.
    allow_legacy_two_column:
        Accept ``Time_sec;Acceleration_g``. The legacy acceleration is mapped
        to ``ax_g`` and zero arrays are returned for ``ay_g`` and ``az_g``.
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

    raw_time = values[:, time_index].copy()
    if not np.all(np.diff(raw_time) > 0):
        index = int(np.flatnonzero(np.diff(raw_time) <= 0)[0])
        raise ValueError(
            f"Time must be strictly increasing; violation between samples {index} and {index + 1}"
        )

    detected_unit = _choose_time_unit(
        raw_time, time_header, time_unit, expected_sample_rate_hz
    )
    time_s = raw_time * (1.0e-6 if detected_unit == "microseconds" else 1.0)

    sample_rate_hz, max_deviation_pct, missing, warnings = _validate_timing(
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
