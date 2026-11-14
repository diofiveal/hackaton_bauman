"""Time-unit selection and sampling-grid checks; timestamps keep their origin."""

from __future__ import annotations

from typing import Literal

import numpy as np

TimeUnit = Literal["auto", "seconds", "microseconds"]
_SECONDS_HEADERS = {"time_s", "time_sec", "timestamp_s"}
_MICROSECONDS_HEADERS = {"t_us", "time_us", "timestamp_us", "time_usec"}


def _header_time_unit(time_header: str) -> Literal["seconds", "microseconds"] | None:
    if time_header in _SECONDS_HEADERS:
        return "seconds"
    if time_header in _MICROSECONDS_HEADERS:
        return "microseconds"
    return None


def choose_time_unit(
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


def validate_timing(
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
        np.sum(
            np.maximum(np.rint(intervals[large_gaps] / nominal_dt).astype(int) - 1, 0)
        )
    )

    warnings: list[str] = []
    if estimated_missing:
        warnings.append(f"Estimated missing samples: {estimated_missing}")
    if maximum_deviation > interval_tolerance_fraction:
        warnings.append(
            f"Maximum sampling-interval deviation is {100.0 * maximum_deviation:.4f}%"
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
