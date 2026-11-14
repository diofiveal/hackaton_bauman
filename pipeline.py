"""End-to-end CSV -> compensation -> event -> frame -> SRS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from .accelerometer import (
        StaticCompensation,
        detect_event_frame,
        resolve_static_compensation,
    )
    from .config import SRSConfig
    from .csv_io import AccelerationData, load_acceleration_csv
    from .data_models import EventFrame, SRSResult
    from .smallwood import calculate_srs_xyz
except ImportError:  # Direct execution with files in one directory.
    from accelerometer import (
        StaticCompensation,
        detect_event_frame,
        resolve_static_compensation,
    )
    from config import SRSConfig
    from csv_io import AccelerationData, load_acceleration_csv
    from data_models import EventFrame, SRSResult
    from smallwood import calculate_srs_xyz


PipelineStatus = Literal["event_processed", "no_event"]


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
) -> PipelineResult:
    """Run the complete reference calculation for the first shock event."""
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

    compensation = resolve_static_compensation(data, config.compensation)
    if config.event.gravity_excluded_from_trigger and compensation is None:
        raise ValueError(
            "The trigger must exclude gravity, but no confirmed-rest "
            "compensation is configured"
        )

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
        warnings=data.warnings + srs_result.warnings,
    )
