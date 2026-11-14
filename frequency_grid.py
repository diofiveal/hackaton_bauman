"""Load the exact competition grid or generate a research grid."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from config import FrequencyGridConfig
from paths import PROJECT_DIR
from validation import frequency_array


def fractional_octave_step_error(
    frequencies_hz: Iterable[float] | np.ndarray,
    octave_fraction: int,
) -> float:
    """Return the maximum relative error from an ideal 1/N-octave ratio."""
    frequencies = frequency_array(frequencies_hz, minimum_size=2)
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
) -> np.ndarray:
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

    frequencies = frequency_array(values, minimum_size=2)
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

    return frequencies


def generate_fractional_octave_grid(
    start_hz: float,
    stop_hz: float,
    octave_fraction: int = 12,
) -> np.ndarray:
    """Generate a research 1/N-octave grid, never the competition grid."""
    if (
        not np.isfinite(start_hz)
        or not np.isfinite(stop_hz)
        or not 0 < start_hz < stop_hz
    ):
        raise ValueError("Require finite 0 < start_hz < stop_hz")
    if octave_fraction < 1:
        raise ValueError("octave_fraction must be a positive integer")
    ratio = 2.0 ** (1.0 / octave_fraction)
    point_count = int(np.floor(np.log(stop_hz / start_hz) / np.log(ratio))) + 1
    frequencies = start_hz * ratio ** np.arange(point_count, dtype=np.float64)
    return frequencies


def frequency_grid_from_values(
    values_hz: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Create a validated research grid from explicit frequencies."""
    return frequency_array(values_hz, minimum_size=2)


def generate_logarithmic_grid(
    start_hz: float,
    stop_hz: float,
    point_count: int,
) -> np.ndarray:
    """Generate a logarithmic research grid with an exact endpoint."""
    if (
        not np.isfinite(start_hz)
        or not np.isfinite(stop_hz)
        or not 0 < start_hz < stop_hz
    ):
        raise ValueError("Require finite 0 < start_hz < stop_hz")
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    return np.geomspace(start_hz, stop_hz, point_count)


def frequencies_from_config(
    config: FrequencyGridConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the configured grid, preserving exact CSV frequency values."""
    if config.source == "csv":
        path = config.source_file
        if not path.is_absolute() and not path.is_file():
            path = PROJECT_DIR / path
        frequencies = load_frequency_grid_csv(
            path,
            octave_fraction=config.octave_fraction,
            expected_start_hz=config.start_hz,
        )
    elif config.source == "fractional_octave":
        frequencies = generate_fractional_octave_grid(
            config.start_hz, config.stop_hz, config.octave_fraction
        )
    elif config.source == "explicit":
        frequencies = frequency_grid_from_values(config.frequencies_hz)
    elif config.source == "logarithmic":
        frequencies = generate_logarithmic_grid(
            config.start_hz, config.stop_hz, config.point_count
        )
    else:
        raise ValueError(f"Unsupported frequency-grid source: {config.source!r}")
    warnings = ()
    if config.source == "csv" and frequencies[-1] < config.stop_hz:
        warnings = (
            f"Fixed grid ends at {frequencies[-1]:.9g} Hz, below configured fmax={config.stop_hz:.9g} Hz",
        )
    return frequencies, warnings
