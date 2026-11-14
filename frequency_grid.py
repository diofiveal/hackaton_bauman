"""Loading and generating validated SRS oscillator-frequency grids."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .data_models import FrequencyGrid
except ImportError:  # Direct execution with files in one directory.
    from data_models import FrequencyGrid


def _validated_frequencies(values: Iterable[float] | np.ndarray) -> np.ndarray:
    frequencies = np.asarray(tuple(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("Frequency grid must be a one-dimensional array of at least two values")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency grid contains NaN or infinity")
    if np.any(frequencies <= 0):
        raise ValueError("Frequency grid values must be positive")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequency grid must be strictly increasing")
    return frequencies


def fractional_octave_step_error(
    frequencies_hz: Iterable[float] | np.ndarray,
    octave_fraction: int,
) -> float:
    """Return the maximum relative error from an ideal 1/N-octave ratio."""
    frequencies = _validated_frequencies(frequencies_hz)
    if octave_fraction < 1:
        raise ValueError("octave_fraction must be a positive integer")
    ideal_ratio = 2.0 ** (1.0 / octave_fraction)
    actual_ratios = frequencies[1:] / frequencies[:-1]
    return float(np.max(np.abs(actual_ratios / ideal_ratio - 1.0)))


def _check_expected_value(
    actual: float,
    expected: float | None,
    name: str,
    tolerance_fraction: float,
) -> None:
    if expected is None:
        return
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError(f"{name} expectation must be positive and finite")
    relative_error = abs(actual / expected - 1.0)
    if relative_error > tolerance_fraction:
        raise ValueError(
            f"Frequency grid {name} is {actual:.12g} Hz, expected "
            f"{expected:.12g} Hz (relative error {relative_error:.6g})"
        )


def load_frequency_grid_csv(
    path: str | Path,
    *,
    octave_fraction: int = 12,
    expected_count: int | None = None,
    expected_start_hz: float | None = None,
    expected_stop_hz: float | None = None,
    bound_tolerance_fraction: float = 1.0e-5,
    step_tolerance_fraction: float = 5.0e-4,
) -> FrequencyGrid:
    """Load the exact one-column competition grid without modifying values.

    Blank lines are ignored.  An optional first-line text header is accepted.
    Every other non-numeric line is rejected.  The fixed source values are
    returned unchanged after conversion to float64.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Frequency-grid CSV not found: {input_path}")
    if octave_fraction < 1:
        raise ValueError("octave_fraction must be a positive integer")
    if bound_tolerance_fraction < 0 or step_tolerance_fraction < 0:
        raise ValueError("Validation tolerances must be nonnegative")

    values: list[float] = []
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # The supplied file is one-column.  Reject extra columns instead
            # of accidentally selecting one value from a malformed table.
            if "," in line or ";" in line or "\t" in line:
                raise ValueError(
                    f"Line {line_number}: expected exactly one frequency value"
                )
            try:
                value = float(line)
            except ValueError as exc:
                if not values and line_number == 1:
                    continue  # Optional header such as Frequency_Hz.
                raise ValueError(
                    f"Line {line_number}: invalid frequency value {line!r}"
                ) from exc
            values.append(value)

    frequencies = _validated_frequencies(values)
    if expected_count is not None and frequencies.size != expected_count:
        raise ValueError(
            f"Frequency grid contains {frequencies.size} values, expected {expected_count}"
        )

    _check_expected_value(
        float(frequencies[0]), expected_start_hz, "start", bound_tolerance_fraction
    )
    _check_expected_value(
        float(frequencies[-1]), expected_stop_hz, "stop", bound_tolerance_fraction
    )

    step_error = fractional_octave_step_error(frequencies, octave_fraction)
    if step_error > step_tolerance_fraction:
        raise ValueError(
            f"Frequency-grid step differs from ideal 1/{octave_fraction} octave: "
            f"maximum relative ratio error {step_error:.6g}, limit "
            f"{step_tolerance_fraction:.6g}"
        )

    return FrequencyGrid(
        frequencies_hz=frequencies,
        source="csv",
        octave_fraction=octave_fraction,
        source_name=input_path.name,
        maximum_step_error_fraction=step_error,
    )


def generate_fractional_octave_grid(
    start_hz: float,
    stop_hz: float,
    octave_fraction: int = 12,
) -> FrequencyGrid:
    """Generate a research 1/N-octave grid, never the competition grid."""
    if not np.isfinite(start_hz) or not np.isfinite(stop_hz) or not 0 < start_hz < stop_hz:
        raise ValueError("Require finite 0 < start_hz < stop_hz")
    if octave_fraction < 1:
        raise ValueError("octave_fraction must be a positive integer")
    ratio = 2.0 ** (1.0 / octave_fraction)
    point_count = int(np.floor(np.log(stop_hz / start_hz) / np.log(ratio))) + 1
    frequencies = start_hz * ratio ** np.arange(point_count, dtype=np.float64)
    return FrequencyGrid(
        frequencies_hz=frequencies,
        source="fractional_octave",
        octave_fraction=octave_fraction,
        maximum_step_error_fraction=fractional_octave_step_error(
            frequencies, octave_fraction
        ),
    )


def frequency_grid_from_values(
    values_hz: Iterable[float] | np.ndarray,
) -> FrequencyGrid:
    """Create a validated research grid from explicit frequencies."""
    return FrequencyGrid(
        frequencies_hz=_validated_frequencies(values_hz),
        source="explicit",
    )


def generate_logarithmic_grid(
    start_hz: float,
    stop_hz: float,
    point_count: int,
) -> FrequencyGrid:
    """Generate a logarithmic research grid with an exact endpoint."""
    if not np.isfinite(start_hz) or not np.isfinite(stop_hz) or not 0 < start_hz < stop_hz:
        raise ValueError("Require finite 0 < start_hz < stop_hz")
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    return FrequencyGrid(
        frequencies_hz=np.geomspace(start_hz, stop_hz, point_count),
        source="logarithmic",
    )
