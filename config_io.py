"""Profiles and JSON/CLI overrides: profile -> file -> command line."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from config import ComputationLimitsConfig, FrequencyGridConfig, ProfileName, SRSConfig

CONTEST_CONFIG = SRSConfig()

CUSTOM_CONFIG = SRSConfig(
    profile="custom",
    grid=FrequencyGridConfig(
        source="fractional_octave",
    ),
    limits=ComputationLimitsConfig(max_frequency_count=None),
)

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
        "event.gravity_excluded_from_trigger": (
            config.event.gravity_excluded_from_trigger,
            expected.event.gravity_excluded_from_trigger,
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
