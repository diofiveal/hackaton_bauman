"""Smallwood recursive calculation of the Maximax shock response spectrum.

The same single-oscillator recurrence is used for one sample, one complete
signal, the full frequency grid, and the ax/ay/az/resultant channels.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

try:
    from .accelerometer import (
        StaticCompensation,
        prepare_acceleration_channels,
    )
    from .data_models import (
        SRSCalculationInfo,
        SRSResult,
        SmallwoodCoefficients,
        SmallwoodCoefficientTable,
        SmallwoodState,
    )
    from .frequency_grid import (
        frequency_grid_from_values,
        generate_fractional_octave_grid,
        generate_logarithmic_grid,
        load_frequency_grid_csv,
    )
except ImportError:  # Direct execution with files in one directory.
    from accelerometer import StaticCompensation, prepare_acceleration_channels
    from data_models import (
        SRSCalculationInfo,
        SRSResult,
        SmallwoodCoefficients,
        SmallwoodCoefficientTable,
        SmallwoodState,
    )
    from frequency_grid import (
        frequency_grid_from_values,
        generate_fractional_octave_grid,
        generate_logarithmic_grid,
        load_frequency_grid_csv,
    )


class AccelerationRecord(Protocol):
    """Minimum input interface required by :func:`calculate_srs_xyz`."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class SRS1DResult:
    """Maximax SRS of one acceleration channel."""

    frequencies_hz: np.ndarray
    srs_g: np.ndarray

    def __post_init__(self) -> None:
        frequencies = _validated_frequencies(self.frequencies_hz)
        spectrum = _validated_signal(self.srs_g, "srs_g")
        if spectrum.shape != frequencies.shape:
            raise ValueError("srs_g must have the same shape as frequencies_hz")
        if np.any(spectrum < 0.0):
            raise ValueError("Maximax SRS values must be nonnegative")
        object.__setattr__(self, "frequencies_hz", frequencies.copy())
        object.__setattr__(self, "srs_g", spectrum.copy())


def _sin_over_argument(value: float) -> float:
    """Evaluate sin(value)/value stably, including value close to zero."""
    return float(np.sinc(value / np.pi))


def calculate_smallwood_coefficients(
    frequency_hz: float,
    sample_rate_hz: float,
    quality_factor_q: float,
) -> SmallwoodCoefficients:
    """Calculate the five coefficients from the case-document formulas.

    Returned ``a1`` and ``a2`` follow the document recurrence:

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]

    They are not converted to the CMSIS plus-feedback convention here.
    """
    parameters = np.array(
        [frequency_hz, sample_rate_hz, quality_factor_q], dtype=np.float64
    )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("frequency_hz, sample_rate_hz, and Q must be finite")
    if frequency_hz <= 0 or sample_rate_hz <= 0:
        raise ValueError("frequency_hz and sample_rate_hz must be positive")
    if quality_factor_q <= 0.5:
        raise ValueError("quality_factor_q must be greater than 0.5")

    sample_period_s = 1.0 / sample_rate_hz
    omega_n = 2.0 * np.pi * frequency_hz
    a_value = omega_n * sample_period_s / (2.0 * quality_factor_q)
    b_value = (
        omega_n
        * sample_period_s
        * np.sqrt(1.0 - 1.0 / (4.0 * quality_factor_q**2))
    )
    exponential = np.exp(-a_value)
    sinc_value = _sin_over_argument(float(b_value))

    b0 = 1.0 - exponential * sinc_value
    b1 = 2.0 * exponential * (sinc_value - np.cos(b_value))
    b2 = np.exp(-2.0 * a_value) - exponential * sinc_value
    a1 = -2.0 * exponential * np.cos(b_value)
    a2 = np.exp(-2.0 * a_value)

    coefficients = SmallwoodCoefficients(
        frequency_hz=float(frequency_hz),
        sample_rate_hz=float(sample_rate_hz),
        quality_factor_q=float(quality_factor_q),
        b0=float(b0),
        b1=float(b1),
        b2=float(b2),
        a1=float(a1),
        a2=float(a2),
    )
    if not np.all(np.isfinite(coefficients.numerator())) or not np.all(
        np.isfinite(coefficients.denominator())
    ):
        raise FloatingPointError(
            "Smallwood coefficient calculation produced nonfinite values"
        )
    if not coefficients.is_stable:
        raise ValueError(
            f"Unstable oscillator coefficients for {frequency_hz:g} Hz, "
            f"fs={sample_rate_hz:g} Hz, Q={quality_factor_q:g}"
        )
    return coefficients


def precompute_smallwood_coefficients(
    frequencies_hz: Iterable[float] | np.ndarray,
    sample_rate_hz: float,
    quality_factor_q: float,
) -> SmallwoodCoefficientTable:
    """Calculate coefficients for every frequency without changing the grid."""
    frequencies = np.asarray(
        tuple(frequencies_hz)
        if not isinstance(frequencies_hz, np.ndarray)
        else frequencies_hz,
        dtype=np.float64,
    )
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain positive finite values")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies_hz must be strictly increasing")

    rows = [
        calculate_smallwood_coefficients(
            float(frequency), sample_rate_hz, quality_factor_q
        )
        for frequency in frequencies
    ]
    return SmallwoodCoefficientTable(
        frequencies_hz=frequencies,
        sample_rate_hz=float(sample_rate_hz),
        quality_factor_q=float(quality_factor_q),
        b0=np.array([row.b0 for row in rows], dtype=np.float64),
        b1=np.array([row.b1 for row in rows], dtype=np.float64),
        b2=np.array([row.b2 for row in rows], dtype=np.float64),
        a1=np.array([row.a1 for row in rows], dtype=np.float64),
        a2=np.array([row.a2 for row in rows], dtype=np.float64),
    )


def _validated_signal(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    """Return a finite, nonempty, one-dimensional float64 signal."""
    signal = np.asarray(
        tuple(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.float64,
    )
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} contains NaN or infinity")
    return signal


def _validated_frequencies(
    frequencies_hz: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Validate an increasing oscillator-frequency grid."""
    frequencies = _validated_signal(frequencies_hz, "frequencies_hz")
    if np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain positive values")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_hz must be strictly increasing")
    return frequencies


def update_oscillator_state(
    sample_g: float,
    coefficients: SmallwoodCoefficients,
    state: SmallwoodState,
) -> float:
    """Process one input sample and update one oscillator state.

    The function implements exactly one recurrence step::

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]

    The returned value is ``y[k]``.  ``state.peak_abs_g`` stores Maximax.
    """
    sample = float(sample_g)
    if not np.isfinite(sample):
        raise ValueError("sample_g must be finite")

    response_g = (
        coefficients.b0 * sample
        + coefficients.b1 * state.x1_g
        + coefficients.b2 * state.x2_g
        - coefficients.a1 * state.y1_g
        - coefficients.a2 * state.y2_g
    )
    if not np.isfinite(response_g):
        raise FloatingPointError("Oscillator response became NaN or infinity")

    state.x2_g = state.x1_g
    state.x1_g = sample
    state.y2_g = state.y1_g
    state.y1_g = float(response_g)
    state.peak_abs_g = max(state.peak_abs_g, abs(float(response_g)))
    return float(response_g)


def _tail_sample_count(
    coefficients: SmallwoodCoefficients,
    tail_mode: str,
    tail_time_constants: float,
) -> int:
    """Return zero-tail length from the damped-oscillator time constant."""
    if tail_mode == "frame_only":
        return 0
    if tail_mode != "zero_tail":
        raise ValueError("tail_mode must be 'frame_only' or 'zero_tail'")
    if not np.isfinite(tail_time_constants) or tail_time_constants <= 0.0:
        raise ValueError("zero_tail requires positive finite tail_time_constants")

    # Envelope: exp(-zeta*omega_n*t), therefore tau = Q/(pi*f_n).
    time_constant_s = coefficients.quality_factor_q / (
        np.pi * coefficients.frequency_hz
    )
    tail_duration_s = tail_time_constants * time_constant_s
    return int(np.ceil(tail_duration_s * coefficients.sample_rate_hz))


def calculate_one_oscillator(
    acceleration_g: Iterable[float] | np.ndarray,
    coefficients: SmallwoodCoefficients,
    *,
    state: SmallwoodState | None = None,
    reset_state: bool = True,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> np.ndarray:
    """Calculate the complete response of one virtual oscillator.

    A caller may provide ``state`` for streaming processing.  By default that
    state is reset before the signal is processed, so repeated calculations
    are independent.  Set ``reset_state=False`` only to continue a previous
    block of the same signal.
    """
    signal = _validated_signal(acceleration_g, "acceleration_g")
    oscillator_state = state if state is not None else SmallwoodState()
    if reset_state:
        oscillator_state.reset()

    tail_count = _tail_sample_count(
        coefficients,
        tail_mode,
        tail_time_constants,
    )
    response_g = np.empty(signal.size + tail_count, dtype=np.float64)
    for index, sample_g in enumerate(signal):
        response_g[index] = update_oscillator_state(
            float(sample_g), coefficients, oscillator_state
        )
    for tail_index in range(tail_count):
        response_g[signal.size + tail_index] = update_oscillator_state(
            0.0,
            coefficients,
            oscillator_state,
        )
    return response_g


def calculate_one_oscillator_maximax(
    acceleration_g: Iterable[float] | np.ndarray,
    coefficients: SmallwoodCoefficients,
    *,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> float:
    """Return ``max(abs(y))`` for one signal and one oscillator."""
    state = SmallwoodState()
    calculate_one_oscillator(
        acceleration_g,
        coefficients,
        state=state,
        reset_state=True,
        tail_mode=tail_mode,
        tail_time_constants=tail_time_constants,
    )
    return float(state.peak_abs_g)


def _calculate_srs_from_table(
    acceleration_g: np.ndarray,
    coefficient_table: SmallwoodCoefficientTable,
    *,
    tail_mode: str,
    tail_time_constants: float,
) -> np.ndarray:
    """Apply the same single-oscillator kernel to every table row."""
    return np.array(
        [
            calculate_one_oscillator_maximax(
                acceleration_g,
                coefficient_table.row(index),
                tail_mode=tail_mode,
                tail_time_constants=tail_time_constants,
            )
            for index in range(coefficient_table.point_count)
        ],
        dtype=np.float64,
    )


def calculate_srs_1d(
    acceleration_g: Iterable[float] | np.ndarray,
    frequencies_hz: Iterable[float] | np.ndarray,
    sample_rate_hz: float,
    q: float,
    *,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> SRS1DResult:
    """Calculate the Maximax SRS of one acceleration channel."""
    signal = _validated_signal(acceleration_g, "acceleration_g")
    frequencies = _validated_frequencies(frequencies_hz)
    coefficient_table = precompute_smallwood_coefficients(
        frequencies,
        sample_rate_hz,
        q,
    )
    return SRS1DResult(
        frequencies_hz=frequencies,
        srs_g=_calculate_srs_from_table(
            signal,
            coefficient_table,
            tail_mode=tail_mode,
            tail_time_constants=tail_time_constants,
        ),
    )


def _resolve_grid_path(source_file: str | Path) -> Path:
    """Find a configured grid relative to the CWD or this module."""
    path = Path(source_file)
    candidates = (
        [path]
        if path.is_absolute()
        else [Path.cwd() / path, Path(__file__).parent / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Frequency-grid CSV not found: {source_file}")


def _frequencies_from_config(config: object) -> tuple[np.ndarray, tuple[str, ...]]:
    """Obtain a grid selected by the current or future configuration model."""
    grid_config = getattr(config, "grid")
    source = getattr(
        grid_config,
        "source",
        "csv" if getattr(grid_config, "fixed", True) else "fractional_octave",
    )
    start_hz = float(getattr(grid_config, "start_hz"))
    stop_hz = float(getattr(grid_config, "stop_hz"))
    octave_fraction = int(getattr(grid_config, "octave_fraction"))

    if source == "csv":
        grid = load_frequency_grid_csv(
            _resolve_grid_path(getattr(grid_config, "source_file")),
            octave_fraction=octave_fraction,
            expected_start_hz=start_hz,
        )
    elif source == "fractional_octave":
        grid = generate_fractional_octave_grid(
            start_hz,
            stop_hz,
            octave_fraction,
        )
    elif source == "explicit":
        grid = frequency_grid_from_values(getattr(grid_config, "frequencies_hz"))
    elif source == "logarithmic":
        grid = generate_logarithmic_grid(
            start_hz,
            stop_hz,
            int(getattr(grid_config, "point_count")),
        )
    else:
        raise ValueError(f"Unsupported frequency-grid source: {source!r}")

    warnings: list[str] = []
    if source == "csv" and grid.maximum_hz < stop_hz:
        warnings.append(
            f"Fixed grid ends at {grid.maximum_hz:.9g} Hz, "
            f"below configured fmax={stop_hz:.9g} Hz"
        )
    return grid.frequencies_hz, tuple(warnings)


def _validated_xyz_data(
    data: AccelerationRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the three required acceleration channels."""
    ax_g = _validated_signal(data.ax_g, "ax_g")
    ay_g = _validated_signal(data.ay_g, "ay_g")
    az_g = _validated_signal(data.az_g, "az_g")
    if not (ax_g.shape == ay_g.shape == az_g.shape):
        raise ValueError("ax_g, ay_g, and az_g must have equal lengths")
    return ax_g, ay_g, az_g


def calculate_srs_xyz(
    data: AccelerationRecord,
    config: object,
    *,
    compensation: StaticCompensation | None = None,
) -> SRSResult:
    """Calculate SRS columns ``f, gx, gy, gz, g_total``.

    The required axis spectra remain independent.  The resultant is an
    additional raw-magnitude channel and does not replace them or affect the
    contest trigger.
    """
    prepared = prepare_acceleration_channels(
        data,
        getattr(config, "compensation"),
        compensation=compensation,
    )
    ax_g, ay_g, az_g = _validated_xyz_data(prepared)
    measured_sample_rate_hz = float(data.sample_rate_hz)
    configured_sample_rate_hz = float(getattr(config, "sample_rate_hz"))
    if not np.isfinite(measured_sample_rate_hz) or measured_sample_rate_hz <= 0.0:
        raise ValueError("data.sample_rate_hz must be positive and finite")
    rate_error = abs(measured_sample_rate_hz / configured_sample_rate_hz - 1.0)
    rate_tolerance = float(
        getattr(getattr(config, "limits"), "sample_rate_tolerance_fraction")
    )
    if rate_error > rate_tolerance:
        raise ValueError(
            f"Measured sample rate {measured_sample_rate_hz:.9g} Hz differs "
            f"from configured {configured_sample_rate_hz:.9g} Hz by more than "
            f"{100.0 * rate_tolerance:.6g}%"
        )
    if getattr(config, "spectrum_type", "maximax") != "maximax":
        raise ValueError("Only Maximax SRS is implemented")

    frequencies_hz, warnings = _frequencies_from_config(config)
    max_samples = getattr(getattr(config, "limits"), "max_samples")
    if max_samples is not None and ax_g.size > int(max_samples):
        raise ValueError(
            f"Input contains {ax_g.size} samples, limit is {int(max_samples)}"
        )
    max_frequency_count = getattr(
        getattr(config, "limits"),
        "max_frequency_count",
    )
    if max_frequency_count is not None and frequencies_hz.size > int(
        max_frequency_count
    ):
        raise ValueError(
            f"Grid contains {frequencies_hz.size} frequencies, limit is "
            f"{int(max_frequency_count)}"
        )
    quality_factor_q = float(getattr(config, "quality_factor_q"))
    coefficient_table = precompute_smallwood_coefficients(
        frequencies_hz,
        measured_sample_rate_hz,
        quality_factor_q,
    )

    tail_mode = getattr(config, "tail_mode", "frame_only")
    tail_time_constants = float(getattr(config, "tail_time_constants", 0.0))
    spectra = [
        _calculate_srs_from_table(
            channel,
            coefficient_table,
            tail_mode=tail_mode,
            tail_time_constants=tail_time_constants,
        )
        for channel in (ax_g, ay_g, az_g, prepared.a_total_g)
    ]
    return SRSResult(
        frequencies_hz=frequencies_hz,
        gx_g=spectra[0],
        gy_g=spectra[1],
        gz_g=spectra[2],
        g_total_g=spectra[3],
        info=SRSCalculationInfo(
            sample_rate_hz=measured_sample_rate_hz,
            quality_factor_q=quality_factor_q,
            spectrum_type="maximax",
            tail_mode=tail_mode,
            tail_time_constants=tail_time_constants,
        ),
        warnings=warnings,
    )
