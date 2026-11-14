"""Orchestrate CSV loading, preprocessing, detection and SRS calculation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from config import ProcessingMode, SRSConfig
from csv_io import load_acceleration_csv
from event_detection import detect_event_frame
from frequency_grid import frequencies_from_config
from models import AccelerationData, EventFrame, SRSResult, StaticCompensation
from preprocessing import apply_static_compensation, resolve_static_compensation
from smallwood import calculate_srs_xyz

PipelineStatus = Literal["validation_completed", "event_processed", "no_event"]


@dataclass(frozen=True)
class PipelineResult:
    """Complete result and traceability information for one CSV file."""

    source_data: AccelerationData
    compensation: StaticCompensation | None
    event_frame: EventFrame | None
    frame_data: AccelerationData | None
    srs_result: SRSResult | None
    status: PipelineStatus
    mode: ProcessingMode = "contest"
    warnings: tuple[str, ...] = ()

    def srs_table(self) -> np.ndarray | None:
        """Return columns [f, gx, gy, gz, g_total], or None without an event."""
        return None if self.srs_result is None else self.srs_result.as_table()


def _slice_event_data(data: AccelerationData, frame: EventFrame) -> AccelerationData:
    """Return a copy of both inclusive frame boundaries."""
    selection = slice(frame.start_index, frame.end_index + 1)
    return replace(
        data,
        time_s=data.time_s[selection].copy(),
        ax_g=data.ax_g[selection].copy(),
        ay_g=data.ay_g[selection].copy(),
        az_g=data.az_g[selection].copy(),
    )


def _validate_contest_input(config: SRSConfig) -> None:
    if not config.event.gravity_excluded_from_trigger:
        raise ValueError("Contest requires gravity_excluded_from_trigger=True")
    if config.input.gravity_compensated and config.compensation.method != "none":
        raise ValueError(
            "Input already excludes gravity; do not apply a second compensation"
        )
    if config.compensation.method == "none" and not config.input.gravity_compensated:
        raise ValueError(
            "Contest requires a 3g trigger excluding gravity. Choose a verified rest "
            "interval with --rest-start/--rest-end, or pass --input-gravity-compensated "
            "ONLY if the CSV already excludes gravity and static bias."
        )


def _calculate_srs(data: AccelerationData, config: SRSConfig) -> SRSResult:
    frequencies, warnings = frequencies_from_config(config.grid)
    limit = config.limits.max_frequency_count
    if limit is not None and frequencies.size > limit:
        raise ValueError(
            f"Grid contains {frequencies.size} frequencies, limit is {limit}"
        )
    if config.spectrum_type != "maximax":
        raise ValueError("Only Maximax SRS is implemented")
    result = calculate_srs_xyz(
        data,
        frequencies,
        float(config.quality_factor_q),
        tail_mode=config.tail_mode,
        tail_time_constants=config.tail_time_constants,
    )
    return replace(result, warnings=warnings)


def run_srs_pipeline(
    csv_path: str | Path,
    config: SRSConfig,
    *,
    mode: ProcessingMode | None = None,
) -> PipelineResult:
    """Load -> validate -> compensate -> detect -> slice -> calculate SRS."""
    selected_mode = config.processing_mode if mode is None else mode
    if selected_mode not in {"contest", "validation"}:
        raise ValueError("mode must be 'contest' or 'validation'")
    data = load_acceleration_csv(
        csv_path,
        time_unit=config.input.time_unit,
        expected_sample_rate_hz=config.sample_rate_hz,
        sample_rate_tolerance_fraction=config.limits.sample_rate_tolerance_fraction,
        interval_tolerance_fraction=config.limits.interval_tolerance_fraction,
    )
    if (
        config.limits.max_samples is not None
        and data.sample_count > config.limits.max_samples
    ):
        raise ValueError(
            f"CSV contains {data.sample_count} samples, limit is {config.limits.max_samples}"
        )
    if selected_mode == "validation":
        srs = _calculate_srs(data, config)
        return PipelineResult(
            data,
            None,
            None,
            None,
            srs,
            "validation_completed",
            mode="validation",
            warnings=data.warnings + srs.warnings,
        )
    return _run_contest(data, config)


def _run_contest(data: AccelerationData, config: SRSConfig) -> PipelineResult:
    _validate_contest_input(config)
    compensation = resolve_static_compensation(data, config.compensation)
    dynamic = (
        apply_static_compensation(data, compensation)
        if compensation is not None
        else data
    )
    frame = detect_event_frame(
        dynamic,
        trigger_g=config.event.trigger_g,
        frame_level_fraction_of_amax=config.event.frame_level_fraction_of_amax,
    )
    if frame is None:
        return PipelineResult(
            data, compensation, None, None, None, "no_event", warnings=data.warnings
        )
    # Detection always uses dynamic axes; SRS follows the explicit magnitude
    # policy, including the original raw-magnitude option.
    selected = (
        dynamic if config.compensation.magnitude_mode == "dynamic_magnitude" else data
    )
    frame_data = _slice_event_data(data, frame)
    srs = _calculate_srs(_slice_event_data(selected, frame), config)
    return PipelineResult(
        data,
        compensation,
        frame,
        frame_data,
        srs,
        "event_processed",
        warnings=data.warnings + srs.warnings,
    )
