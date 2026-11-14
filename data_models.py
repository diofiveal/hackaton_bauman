"""Shared data structures for the SRS calculation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


SpectrumType = Literal["maximax"]
TailMode = Literal["frame_only", "zero_tail"]
GridSource = Literal["csv", "fractional_octave", "explicit", "logarithmic"]


def _finite_1d(values: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class FrequencyGrid:
    """Validated natural-frequency grid used by the virtual oscillators."""

    frequencies_hz: np.ndarray
    source: GridSource
    octave_fraction: int | None = None
    source_name: str | None = None
    maximum_step_error_fraction: float | None = None

    def __post_init__(self) -> None:
        frequencies = _finite_1d(self.frequencies_hz, "frequencies_hz")
        if frequencies.size < 2:
            raise ValueError("Frequency grid must contain at least two points")
        if np.any(frequencies <= 0):
            raise ValueError("Frequency grid values must be positive")
        if np.any(np.diff(frequencies) <= 0):
            raise ValueError("Frequency grid must be strictly increasing")
        if self.octave_fraction is not None and self.octave_fraction < 1:
            raise ValueError("octave_fraction must be a positive integer")
        object.__setattr__(self, "frequencies_hz", frequencies.copy())

    @property
    def point_count(self) -> int:
        return int(self.frequencies_hz.size)

    @property
    def minimum_hz(self) -> float:
        return float(self.frequencies_hz[0])

    @property
    def maximum_hz(self) -> float:
        return float(self.frequencies_hz[-1])


@dataclass(frozen=True)
class SmallwoodCoefficients:
    """Ramp-invariant coefficients for one virtual oscillator.

    The recurrence uses the case-document convention:

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]
    """

    frequency_hz: float
    sample_rate_hz: float
    quality_factor_q: float
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def numerator(self) -> np.ndarray:
        return np.array([self.b0, self.b1, self.b2], dtype=np.float64)

    def denominator(self) -> np.ndarray:
        return np.array([1.0, self.a1, self.a2], dtype=np.float64)

    def cmsis_feedback(self) -> tuple[float, float]:
        """Return feedback values for CMSIS' plus-feedback convention."""
        return -self.a1, -self.a2

    def pole_magnitudes(self) -> np.ndarray:
        return np.abs(np.roots(self.denominator()))

    @property
    def is_stable(self) -> bool:
        return bool(np.all(self.pole_magnitudes() < 1.0))


@dataclass(frozen=True)
class SmallwoodCoefficientTable:
    """Precomputed coefficients for all frequencies in one grid."""

    frequencies_hz: np.ndarray
    sample_rate_hz: float
    quality_factor_q: float
    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    a1: np.ndarray
    a2: np.ndarray

    def __post_init__(self) -> None:
        frequencies = _finite_1d(self.frequencies_hz, "frequencies_hz")
        arrays = {
            "b0": _finite_1d(self.b0, "b0"),
            "b1": _finite_1d(self.b1, "b1"),
            "b2": _finite_1d(self.b2, "b2"),
            "a1": _finite_1d(self.a1, "a1"),
            "a2": _finite_1d(self.a2, "a2"),
        }
        if any(array.shape != frequencies.shape for array in arrays.values()):
            raise ValueError("All coefficient arrays must match frequencies_hz")
        object.__setattr__(self, "frequencies_hz", frequencies.copy())
        for name, array in arrays.items():
            object.__setattr__(self, name, array.copy())

    @property
    def point_count(self) -> int:
        return int(self.frequencies_hz.size)

    def row(self, index: int) -> SmallwoodCoefficients:
        return SmallwoodCoefficients(
            frequency_hz=float(self.frequencies_hz[index]),
            sample_rate_hz=float(self.sample_rate_hz),
            quality_factor_q=float(self.quality_factor_q),
            b0=float(self.b0[index]),
            b1=float(self.b1[index]),
            b2=float(self.b2[index]),
            a1=float(self.a1[index]),
            a2=float(self.a2[index]),
        )

    def as_matrix(self) -> np.ndarray:
        """Return columns [f, b0, b1, b2, a1, a2]."""
        return np.column_stack(
            (self.frequencies_hz, self.b0, self.b1, self.b2, self.a1, self.a2)
        )


@dataclass
class SmallwoodState:
    """Mutable streaming state for one oscillator; recurrence comes later."""

    x1_g: float = 0.0
    x2_g: float = 0.0
    y1_g: float = 0.0
    y2_g: float = 0.0
    peak_abs_g: float = 0.0

    def reset(self) -> None:
        self.x1_g = 0.0
        self.x2_g = 0.0
        self.y1_g = 0.0
        self.y2_g = 0.0
        self.peak_abs_g = 0.0


@dataclass(frozen=True)
class EventFrame:
    """Indices and metadata for one selected shock frame."""

    start_index: int
    end_index: int
    amax_g: float
    valid: bool = True
    status: str = "valid"

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("Invalid event-frame indices")
        if not np.isfinite(self.amax_g) or self.amax_g < 0:
            raise ValueError("amax_g must be finite and nonnegative")

    @property
    def sample_count(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class SRSCalculationInfo:
    """Settings that must accompany a calculated SRS result."""

    sample_rate_hz: float
    quality_factor_q: float
    spectrum_type: SpectrumType = "maximax"
    tail_mode: TailMode = "frame_only"
    tail_time_constants: float = 0.0


@dataclass(frozen=True)
class SRSResult:
    """SRS spectra for the required axes and optional resultant channel."""

    frequencies_hz: np.ndarray
    gx_g: np.ndarray
    gy_g: np.ndarray
    gz_g: np.ndarray
    info: SRSCalculationInfo
    g_total_g: np.ndarray | None = None
    valid: bool = True
    status: str = "valid"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        frequencies = _finite_1d(self.frequencies_hz, "frequencies_hz")
        spectra = {
            "gx_g": _finite_1d(self.gx_g, "gx_g"),
            "gy_g": _finite_1d(self.gy_g, "gy_g"),
            "gz_g": _finite_1d(self.gz_g, "gz_g"),
        }
        if self.g_total_g is not None:
            spectra["g_total_g"] = _finite_1d(self.g_total_g, "g_total_g")
        if any(values.shape != frequencies.shape for values in spectra.values()):
            raise ValueError("Every SRS spectrum must match frequencies_hz")
        if any(np.any(values < 0) for values in spectra.values()):
            raise ValueError("Maximax SRS values must be nonnegative")
        object.__setattr__(self, "frequencies_hz", frequencies.copy())
        for name, values in spectra.items():
            object.__setattr__(self, name, values.copy())

    def as_table(self) -> np.ndarray:
        """Return [f, gx, gy, gz] and optional g_total as the final column."""
        columns = [self.frequencies_hz, self.gx_g, self.gy_g, self.gz_g]
        if self.g_total_g is not None:
            columns.append(self.g_total_g)
        return np.column_stack(columns)
