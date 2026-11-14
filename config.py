"""Central configuration for the SRS processing pipeline.

Configuration priority is implemented as:

    profile -> configuration file -> command-line overrides

The contest profile keeps the jury parameters fixed. The custom profile is a
starting point for research calculations and may be overridden explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ProfileName = Literal["contest", "custom"]
ProcessingMode = Literal["validation", "contest"]
SpectrumType = Literal["maximax"]
ResponseType = Literal["absolute_acceleration"]
GridSource = Literal["csv", "fractional_octave", "explicit", "logarithmic"]
TailMode = Literal["frame_only", "zero_tail"]
MagnitudeMode = Literal["raw_magnitude", "dynamic_magnitude"]
CompensationMethod = Literal["none", "confirmed_rest"]


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class FrequencyGridConfig:
    """Frequency-grid selection and source-specific parameters."""

    source: GridSource = "csv"
    start_hz: float = 10.0
    stop_hz: float = 2500.0
    octave_fraction: int = 12
    source_file: Path = Path("data/frequency_grid.csv")
    frequencies_hz: tuple[float, ...] = ()
    point_count: int = 94

    def __post_init__(self) -> None:
        if self.source not in {"csv", "fractional_octave", "explicit", "logarithmic"}:
            raise ValueError(f"Unsupported frequency-grid source: {self.source!r}")
        _require_positive_finite(self.start_hz, "grid.start_hz")
        _require_positive_finite(self.stop_hz, "grid.stop_hz")
        if self.start_hz >= self.stop_hz:
            raise ValueError("grid.start_hz must be less than grid.stop_hz")
        if self.octave_fraction < 1:
            raise ValueError("grid.octave_fraction must be at least 1")
        if self.point_count < 2:
            raise ValueError("grid.point_count must be at least 2")

        object.__setattr__(self, "source_file", Path(self.source_file))
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        object.__setattr__(self, "frequencies_hz", frequencies)
        if self.source == "explicit":
            if len(frequencies) < 2:
                raise ValueError(
                    "grid.frequencies_hz needs at least two values for explicit mode"
                )
            if any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
                raise ValueError("Explicit frequencies must be positive and finite")
            if any(
                right <= left
                for left, right in zip(frequencies, frequencies[1:], strict=False)
            ):
                raise ValueError("Explicit frequencies must be strictly increasing")


@dataclass(frozen=True)
class EventDetectionConfig:
    """Rules for detecting and extracting a shock event."""

    trigger_g: float = 3.0
    trigger_mode: Literal["any_axis"] = "any_axis"
    gravity_excluded_from_trigger: bool = True
    frame_level_fraction_of_amax: float = 0.25

    def __post_init__(self) -> None:
        if self.trigger_mode != "any_axis":
            raise ValueError("Only the any_axis trigger is implemented")
        _require_positive_finite(self.trigger_g, "event.trigger_g")
        if not 0.0 < self.frame_level_fraction_of_amax <= 1.0:
            raise ValueError(
                "event.frame_level_fraction_of_amax must be in the interval (0, 1]"
            )


@dataclass(frozen=True)
class StaticCompensationConfig:
    """Rules for offset and gravity compensation.

    Compensation is never estimated automatically. Dynamic magnitude may be
    enabled only when the caller supplies a confirmed rest interval.
    """

    magnitude_mode: MagnitudeMode = "raw_magnitude"
    method: CompensationMethod = "none"
    rest_start_s: float | None = None
    rest_end_s: float | None = None
    # Conservative engineering limits (not specified by the hackathon jury).
    minimum_rest_samples: int = 32
    maximum_rest_std_g: float | None = 0.10
    maximum_rest_peak_to_peak_g: float | None = 0.50
    maximum_gravity_norm_deviation_g: float | None = 0.35

    def __post_init__(self) -> None:
        if self.method not in {"none", "confirmed_rest"}:
            raise ValueError(f"Unsupported compensation method: {self.method!r}")
        if self.magnitude_mode not in {"raw_magnitude", "dynamic_magnitude"}:
            raise ValueError(f"Unsupported magnitude mode: {self.magnitude_mode!r}")
        if self.minimum_rest_samples < 3:
            raise ValueError("compensation.minimum_rest_samples must be at least 3")
        for field_name in (
            "maximum_rest_std_g",
            "maximum_rest_peak_to_peak_g",
            "maximum_gravity_norm_deviation_g",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_finite(value, f"compensation.{field_name}")

        interval_is_set = self.rest_start_s is not None or self.rest_end_s is not None
        if interval_is_set:
            if self.rest_start_s is None or self.rest_end_s is None:
                raise ValueError("Both rest interval boundaries must be supplied")
            if not (
                math.isfinite(self.rest_start_s)
                and math.isfinite(self.rest_end_s)
                and 0.0 <= self.rest_start_s < self.rest_end_s
            ):
                raise ValueError("Require 0 <= rest_start_s < rest_end_s")

        if self.method == "none" and interval_is_set:
            raise ValueError(
                "Do not specify rest interval when compensation.method='none'"
            )
        if self.method == "confirmed_rest" and not interval_is_set:
            raise ValueError("confirmed_rest requires rest_start_s and rest_end_s")

        if self.magnitude_mode == "dynamic_magnitude":
            if self.method != "confirmed_rest" or not interval_is_set:
                raise ValueError(
                    "dynamic_magnitude requires method='confirmed_rest' and "
                    "an explicit rest interval"
                )


@dataclass(frozen=True)
class ComputationLimitsConfig:
    """Safety limits for memory use and input validation."""

    max_samples: int | None = None
    max_frequency_count: int | None = 94
    sample_rate_tolerance_fraction: float = 0.02
    interval_tolerance_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("limits.max_samples must be positive or None")
        if self.max_frequency_count is not None and self.max_frequency_count < 1:
            raise ValueError("limits.max_frequency_count must be positive or None")
        for name, value in (
            ("sample_rate_tolerance_fraction", self.sample_rate_tolerance_fraction),
            ("interval_tolerance_fraction", self.interval_tolerance_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"limits.{name} must be in the interval [0, 1)")


@dataclass(frozen=True)
class InputConfig:
    """Expected input table and physical units."""

    time_unit: Literal["auto", "seconds", "microseconds"] = "auto"
    acceleration_unit: Literal["g"] = "g"
    # True ONLY when external preprocessing has already removed gravity and bias.
    # This flag does not perform any numerical correction.
    gravity_compensated: bool = False

    def __post_init__(self) -> None:
        if self.acceleration_unit != "g":
            raise ValueError("Acceleration input must be in g")
        if self.time_unit not in {"auto", "seconds", "microseconds"}:
            raise ValueError("Unsupported input time unit")
        if type(self.gravity_compensated) is not bool:
            raise ValueError("input.gravity_compensated must be a boolean")


@dataclass(frozen=True)
class SRSConfig:
    """Single immutable configuration structure for the SRS pipeline."""

    profile: ProfileName = "contest"
    processing_mode: ProcessingMode = "contest"
    sample_rate_hz: float = 6660.0
    quality_factor_q: float | None = 10.0
    damping_ratio_zeta: float | None = 0.05
    spectrum_type: SpectrumType = "maximax"
    response_type: ResponseType = "absolute_acceleration"
    tail_mode: TailMode = "frame_only"
    tail_time_constants: float = 5.0
    grid: FrequencyGridConfig = field(default_factory=FrequencyGridConfig)
    event: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    compensation: StaticCompensationConfig = field(
        default_factory=StaticCompensationConfig
    )
    limits: ComputationLimitsConfig = field(default_factory=ComputationLimitsConfig)
    input: InputConfig = field(default_factory=InputConfig)

    def __post_init__(self) -> None:
        if self.spectrum_type != "maximax":
            raise ValueError("Only Maximax SRS is implemented")
        if self.response_type != "absolute_acceleration":
            raise ValueError("Only absolute acceleration response is implemented")
        if self.tail_mode not in {"frame_only", "zero_tail"}:
            raise ValueError("tail_mode must be 'frame_only' or 'zero_tail'")
        if self.processing_mode not in {"contest", "validation"}:
            raise ValueError("processing_mode must be 'contest' or 'validation'")
        _require_positive_finite(self.sample_rate_hz, "sample_rate_hz")

        q = self.quality_factor_q
        zeta = self.damping_ratio_zeta
        if q is None and zeta is None:
            raise ValueError("Specify quality_factor_q or damping_ratio_zeta")
        if q is None:
            _require_positive_finite(float(zeta), "damping_ratio_zeta")
            q = 1.0 / (2.0 * float(zeta))
            object.__setattr__(self, "quality_factor_q", q)
        if zeta is None:
            _require_positive_finite(float(q), "quality_factor_q")
            zeta = 1.0 / (2.0 * float(q))
            object.__setattr__(self, "damping_ratio_zeta", zeta)

        _require_positive_finite(float(q), "quality_factor_q")
        _require_positive_finite(float(zeta), "damping_ratio_zeta")
        if float(q) <= 0.5:
            raise ValueError("quality_factor_q must be greater than 0.5")
        expected_zeta = 1.0 / (2.0 * float(q))
        if not math.isclose(float(zeta), expected_zeta, rel_tol=1.0e-9):
            raise ValueError("quality_factor_q and damping_ratio_zeta are inconsistent")

        if (
            not math.isfinite(self.tail_time_constants)
            or self.tail_time_constants < 0.0
        ):
            raise ValueError("tail_time_constants must be finite and nonnegative")
        if self.tail_mode == "zero_tail" and self.tail_time_constants <= 0.0:
            raise ValueError("zero_tail requires tail_time_constants > 0")
