"""Central configuration for the SRS processing pipeline.

Configuration priority is implemented as:

    profile -> configuration file -> command-line overrides

The contest profile keeps the jury parameters fixed. The custom profile is a
starting point for research calculations and may be overridden explicitly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal


ProfileName = Literal["contest", "custom"]
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
    fixed: bool = True
    strict_file_validation: bool = True

    def __post_init__(self) -> None:
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
            if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
                raise ValueError("Explicit frequencies must be strictly increasing")

    @property
    def fmin_hz(self) -> float:
        """Alias used in the requirements for the lower grid bound."""
        return self.start_hz

    @property
    def fmax_hz(self) -> float:
        """Alias used in the requirements for the upper grid bound."""
        return self.stop_hz


@dataclass(frozen=True)
class EventDetectionConfig:
    """Rules for detecting and extracting a shock event."""

    trigger_g: float = 3.0
    trigger_mode: Literal["any_axis"] = "any_axis"
    gravity_excluded_from_trigger: bool = True
    frame_level_fraction_of_amax: float = 0.25

    def __post_init__(self) -> None:
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
    minimum_rest_samples: int = 3
    maximum_rest_std_g: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_rest_samples < 3:
            raise ValueError("compensation.minimum_rest_samples must be at least 3")
        if self.maximum_rest_std_g is not None:
            _require_positive_finite(
                self.maximum_rest_std_g,
                "compensation.maximum_rest_std_g",
            )

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

    columns: tuple[str, ...] = ("t", "ax", "ay", "az")
    time_unit: Literal["auto", "seconds", "microseconds"] = "auto"
    acceleration_unit: Literal["g"] = "g"
    required_axis_channels: tuple[str, ...] = ("ax", "ay", "az")
    calculate_resultant_acceleration: bool = True


@dataclass(frozen=True)
class SRSConfig:
    """Single immutable configuration structure for the SRS pipeline."""

    profile: ProfileName = "contest"
    sample_rate_hz: float = 6660.0
    quality_factor_q: float | None = 10.0
    damping_ratio_zeta: float | None = 0.05
    spectrum_type: SpectrumType = "maximax"
    response_type: ResponseType = "absolute_acceleration"
    tail_mode: TailMode = "frame_only"
    tail_time_constants: float = 5.0
    calculate_each_axis_separately: bool = True
    smallwood_formulas_unchanged: bool = True
    coefficients_may_be_precomputed: bool = True
    grid: FrequencyGridConfig = field(default_factory=FrequencyGridConfig)
    event: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    compensation: StaticCompensationConfig = field(
        default_factory=StaticCompensationConfig
    )
    limits: ComputationLimitsConfig = field(default_factory=ComputationLimitsConfig)
    input: InputConfig = field(default_factory=InputConfig)

    def __post_init__(self) -> None:
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


CONTEST_CONFIG = SRSConfig()

CUSTOM_CONFIG = SRSConfig(
    profile="custom",
    grid=FrequencyGridConfig(
        source="fractional_octave",
        fixed=False,
        strict_file_validation=False,
    ),
    limits=ComputationLimitsConfig(max_frequency_count=None),
)

DEFAULT_CONFIG = CONTEST_CONFIG

PROFILE_CONFIGS: Mapping[str, SRSConfig] = {
    "contest": CONTEST_CONFIG,
    "custom": CUSTOM_CONFIG,
}


def get_profile_config(profile: ProfileName) -> SRSConfig:
    """Return one immutable base profile."""
    try:
        return PROFILE_CONFIGS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown configuration profile: {profile!r}") from exc


def _expand_dotted_keys(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ``grid.start_hz`` style keys to nested dictionaries."""
    expanded: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = str(raw_key)
        target = expanded
        parts = key.split(".")
        for part in parts[:-1]:
            nested = target.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Conflicting override path: {key!r}")
            target = nested
        target[parts[-1]] = (
            _expand_dotted_keys(value) if isinstance(value, Mapping) else value
        )
    return expanded


def _replace_nested(instance: Any, overrides: Mapping[str, Any], path: str = "") -> Any:
    """Apply validated nested replacements to frozen dataclasses."""
    if not is_dataclass(instance):
        raise TypeError(f"{path or 'value'} is not a configuration dataclass")

    valid_names = {item.name for item in fields(instance)}
    replacements: dict[str, Any] = {}
    for name, value in overrides.items():
        full_name = f"{path}.{name}" if path else name
        if name not in valid_names:
            raise ValueError(f"Unknown configuration parameter: {full_name}")
        if name == "profile":
            raise ValueError(
                "Select profile through build_config(profile=...), not overrides"
            )

        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"{full_name} must be a mapping")
            replacements[name] = _replace_nested(current, value, full_name)
        elif isinstance(current, Path):
            replacements[name] = Path(value)
        elif isinstance(current, tuple) and isinstance(value, list):
            replacements[name] = tuple(value)
        else:
            replacements[name] = value
    return replace(instance, **replacements)


def apply_config_overrides(
    config: SRSConfig,
    overrides: Mapping[str, Any] | None,
) -> SRSConfig:
    """Apply one override layer and run all dataclass validations."""
    if not overrides:
        return config
    expanded = _expand_dotted_keys(overrides)
    if "quality_factor_q" in expanded and "damping_ratio_zeta" not in expanded:
        expanded["damping_ratio_zeta"] = None
    if "damping_ratio_zeta" in expanded and "quality_factor_q" not in expanded:
        expanded["quality_factor_q"] = None
    return _replace_nested(config, expanded)


def _validate_contest_requirements(config: SRSConfig) -> None:
    """Reject attempts to alter parameters fixed by the jury."""
    expected = CONTEST_CONFIG
    locked_values = {
        "sample_rate_hz": (config.sample_rate_hz, expected.sample_rate_hz),
        "quality_factor_q": (config.quality_factor_q, expected.quality_factor_q),
        "spectrum_type": (config.spectrum_type, expected.spectrum_type),
        "response_type": (config.response_type, expected.response_type),
        "grid.source": (config.grid.source, expected.grid.source),
        "grid.start_hz": (config.grid.start_hz, expected.grid.start_hz),
        "grid.stop_hz": (config.grid.stop_hz, expected.grid.stop_hz),
        "grid.octave_fraction": (
            config.grid.octave_fraction,
            expected.grid.octave_fraction,
        ),
        "event.trigger_g": (config.event.trigger_g, expected.event.trigger_g),
        "event.trigger_mode": (
            config.event.trigger_mode,
            expected.event.trigger_mode,
        ),
        "event.frame_level_fraction_of_amax": (
            config.event.frame_level_fraction_of_amax,
            expected.event.frame_level_fraction_of_amax,
        ),
    }
    changed = [
        name for name, (actual, required) in locked_values.items() if actual != required
    ]
    if changed:
        raise ValueError(
            "Contest profile parameters are fixed; attempted changes: "
            + ", ".join(changed)
        )


def build_config(
    profile: ProfileName = "contest",
    *,
    file_overrides: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> SRSConfig:
    """Build configuration using profile -> file -> CLI priority."""
    config = get_profile_config(profile)
    config = apply_config_overrides(config, file_overrides)
    config = apply_config_overrides(config, cli_overrides)
    if profile == "contest":
        _validate_contest_requirements(config)
    return config


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON configuration file into an override mapping."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {input_path}")
    if input_path.suffix.lower() != ".json":
        raise ValueError("Configuration file must have .json extension")
    with input_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("Configuration file root must be a mapping")
    return values


def load_config(
    profile: ProfileName = "contest",
    *,
    config_file: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> SRSConfig:
    """Load a profile and optional file/CLI layers in the required order."""
    file_overrides = read_config_file(config_file) if config_file else None
    return build_config(
        profile,
        file_overrides=file_overrides,
        cli_overrides=cli_overrides,
    )
