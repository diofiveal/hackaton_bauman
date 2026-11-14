"""Signal, compensation, frame and SRS results shared by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from config import SpectrumType, TailMode
from validation import finite_signal


@dataclass(frozen=True)
class AccelerationData:
    """Validated accelerometer samples in common units."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    sample_rate_hz: float
    source_format: Literal["triaxial", "legacy_2_column"] = "triaxial"
    detected_time_unit: Literal["seconds", "microseconds"] = "seconds"
    max_interval_deviation_percent: float = 0.0
    estimated_missing_samples: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)


@dataclass(frozen=True)
class StaticCompensation:
    """Static gravity-plus-offset vector measured during confirmed rest.

    The three mean values contain the combined effect of gravity projection
    and sensor zero offset.  A rest measurement alone cannot separate them.
    """

    ax_g: float
    ay_g: float
    az_g: float
    rest_start_s: float
    rest_end_s: float
    sample_count: int
    std_ax_g: float
    std_ay_g: float
    std_az_g: float
    max_rest_peak_to_peak_g: float = 0.0
    gravity_vector_norm_g: float = 0.0

    @property
    def vector_g(self) -> np.ndarray:
        return np.array([self.ax_g, self.ay_g, self.az_g], dtype=np.float64)

    @property
    def maximum_rest_std_g(self) -> float:
        return max(self.std_ax_g, self.std_ay_g, self.std_az_g)


@dataclass(frozen=True)
class EventFrame:
    """Indices and metadata for one selected shock frame."""

    start_index: int
    end_index: int
    amax_g: float

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
    info: SRSCalculationInfo | None = None
    g_total_g: np.ndarray | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        frequencies = finite_signal(self.frequencies_hz, "frequencies_hz")
        spectra = {
            "gx_g": finite_signal(self.gx_g, "gx_g"),
            "gy_g": finite_signal(self.gy_g, "gy_g"),
            "gz_g": finite_signal(self.gz_g, "gz_g"),
        }
        if self.g_total_g is not None:
            spectra["g_total_g"] = finite_signal(self.g_total_g, "g_total_g")
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
