"""End-to-end CSV -> compensation -> event -> frame -> SRS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from .accelerometer import (
        StaticCompensation,
        detect_event_frame,
        resolve_static_compensation,
    )
    from .config import ProcessingMode, SRSConfig, StaticCompensationConfig
    from .csv_io import AccelerationData, load_acceleration_csv
    from .data_models import EventFrame, SRSResult
    from .smallwood import calculate_srs_xyz
except ImportError:  # Direct execution with files in one directory.
    from accelerometer import (
        StaticCompensation,
        detect_event_frame,
        resolve_static_compensation,
    )
    from config import ProcessingMode, SRSConfig, StaticCompensationConfig
    from csv_io import AccelerationData, load_acceleration_csv
    from data_models import EventFrame, SRSResult
    from smallwood import calculate_srs_xyz


PipelineStatus = Literal["validation_completed", "event_processed", "no_event"]


@dataclass(frozen=True)
class EventAccelerationData:
    """Original acceleration channels sliced to one detected event."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    sample_rate_hz: float
    source_start_index: int
    source_end_index: int

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)


@dataclass(frozen=True)
class PipelineResult:
    """Complete result and traceability information for one CSV file."""

    source_data: AccelerationData
    compensation: StaticCompensation | None
    event_frame: EventFrame | None
    frame_data: EventAccelerationData | None
    srs_result: SRSResult | None
    status: PipelineStatus
    mode: ProcessingMode = "contest"
    warnings: tuple[str, ...] = ()

    def srs_table(self) -> np.ndarray | None:
        """Return columns [f, gx, gy, gz, g_total], or None without an event."""
        return None if self.srs_result is None else self.srs_result.as_table()


def _slice_event_data(
    data: AccelerationData,
    event_frame: EventFrame,
) -> EventAccelerationData:
    """Copy one inclusive event interval without changing physical values."""
    selection = slice(event_frame.start_index, event_frame.end_index + 1)
    return EventAccelerationData(
        time_s=data.time_s[selection].copy(),
        ax_g=data.ax_g[selection].copy(),
        ay_g=data.ay_g[selection].copy(),
        az_g=data.az_g[selection].copy(),
        sample_rate_hz=float(data.sample_rate_hz),
        source_start_index=event_frame.start_index,
        source_end_index=event_frame.end_index,
    )


def _check_pipeline_limits(data: AccelerationData, config: SRSConfig) -> None:
    max_samples = config.limits.max_samples
    if max_samples is not None and data.sample_count > max_samples:
        raise ValueError(
            f"CSV contains {data.sample_count} samples, limit is {max_samples}"
        )


def run_srs_pipeline(
    csv_path: str | Path,
    config: SRSConfig,
    *,
    mode: ProcessingMode | None = None,
) -> PipelineResult:
    """Run full-record validation or gravity-safe first-event contest mode."""
    selected_mode = config.processing_mode if mode is None else mode
    if selected_mode not in {"contest", "validation"}:
        raise ValueError("mode must be 'contest' or 'validation'")
    data = load_acceleration_csv(
        csv_path,
        time_unit=config.input.time_unit,
        expected_sample_rate_hz=config.sample_rate_hz,
        sample_rate_tolerance_fraction=(
            config.limits.sample_rate_tolerance_fraction
        ),
        interval_tolerance_fraction=config.limits.interval_tolerance_fraction,
        strict_timing=True,
        allow_legacy_two_column=True,
    )
    _check_pipeline_limits(data, config)

    if selected_mode == "validation":
        # Use ALL recorded samples, without detection or gravity subtraction.
        # The replacement also prevents calculate_srs_xyz() from estimating a
        # compensation vector from a config intended for contest mode.
        raw_config = replace(
            config,
            compensation=StaticCompensationConfig(
                method="none", magnitude_mode="raw_magnitude"
            ),
        )
        srs_result = calculate_srs_xyz(data, raw_config)
        return PipelineResult(
            source_data=data,
            compensation=None,
            event_frame=None,
            frame_data=None,
            srs_result=srs_result,
            status="validation_completed",
            mode="validation",
            warnings=data.warnings + srs_result.warnings,
        )

    if not config.event.gravity_excluded_from_trigger:
        raise ValueError(
            "Contest mode requires gravity_excluded_from_trigger=True; "
            "the 3g threshold must be applied after gravity removal"
        )

    # A gravity-free 3 g trigger is only meaningful with a verified source.
    # A rest-derived vector corrects raw samples; alternatively firmware may
    # supply already-compensated values (explicitly declared by the user).
    if config.input.gravity_compensated and config.compensation.method != "none":
        raise ValueError(
            "Input is marked as already gravity-compensated; do not apply "
            "a second confirmed-rest compensation"
        )
    if config.event.gravity_excluded_from_trigger and (
        config.compensation.method == "none"
        and not config.input.gravity_compensated
    ):
        raise ValueError(
            "Contest requires a 3g trigger excluding gravity. Choose a "
            "verified rest interval with --rest-start/--rest-end, or pass "
            "--input-gravity-compensated ONLY if the CSV already excludes "
            "gravity and static bias. Uncompensated raw data are unsafe "
            "for the specified threshold."
        )
    compensation = resolve_static_compensation(data, config.compensation)

    event_frame = detect_event_frame(
        data,
        trigger_g=config.event.trigger_g,
        frame_level_fraction_of_amax=(
            config.event.frame_level_fraction_of_amax
        ),
        compensation=compensation,
    )
    if event_frame is None:
        return PipelineResult(
            source_data=data,
            compensation=compensation,
            event_frame=None,
            frame_data=None,
            srs_result=None,
            status="no_event",
            mode="contest",
            warnings=data.warnings,
        )

    frame_data = _slice_event_data(data, event_frame)
    srs_result = calculate_srs_xyz(
        frame_data,
        config,
        compensation=compensation,
    )
    return PipelineResult(
        source_data=data,
        compensation=compensation,
        event_frame=event_frame,
        frame_data=frame_data,
        srs_result=srs_result,
        status="event_processed",
        mode="contest",
        warnings=data.warnings + srs_result.warnings,
    )
