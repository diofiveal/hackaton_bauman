"""Three-axis acceleration processing for the SRS pipeline.

This module adds the resultant acceleration channel

    a_total = sqrt(ax**2 + ay**2 + az**2)

without replacing the original ax, ay, and az channels.  Gravity and static
offsets are never removed automatically.  Dynamic magnitude requires an
explicit compensation vector estimated from a user-confirmed rest interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

try:
    from .data_models import EventFrame
except ImportError:  # Direct execution with files in one directory.
    from data_models import EventFrame


MagnitudeMode = Literal["raw_magnitude", "dynamic_magnitude"]


class AccelerationRecord(Protocol):
    """Minimal interface expected from csv_io.AccelerationData."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray


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
class CompensatedAxes:
    """Copies of the three axes after explicit static compensation."""

    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    compensation: StaticCompensation


@dataclass(frozen=True)
class TotalAccelerationResult:
    """Additional resultant-acceleration channel."""

    time_s: np.ndarray
    a_total_g: np.ndarray
    mode: MagnitudeMode
    compensation: StaticCompensation | None


@dataclass(frozen=True)
class PreparedAcceleration:
    """Axes and resultant selected by the compensation configuration."""

    time_s: np.ndarray
    ax_g: np.ndarray
    ay_g: np.ndarray
    az_g: np.ndarray
    a_total_g: np.ndarray
    mode: MagnitudeMode
    compensation: StaticCompensation | None


def _validated_arrays(
    data: AccelerationRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated float64 views of time and three acceleration axes."""
    time_s = np.asarray(data.time_s, dtype=np.float64)
    ax_g = np.asarray(data.ax_g, dtype=np.float64)
    ay_g = np.asarray(data.ay_g, dtype=np.float64)
    az_g = np.asarray(data.az_g, dtype=np.float64)

    arrays = (time_s, ax_g, ay_g, az_g)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("time_s, ax_g, ay_g, and az_g must be one-dimensional")
    if time_s.size < 3:
        raise ValueError("At least three acceleration samples are required")
    if not (time_s.size == ax_g.size == ay_g.size == az_g.size):
        raise ValueError("time_s, ax_g, ay_g, and az_g must have equal lengths")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Acceleration data contain NaN or infinity")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")

    return time_s, ax_g, ay_g, az_g


def estimate_static_compensation(
    data: AccelerationRecord,
    rest_start_s: float,
    rest_end_s: float,
    *,
    minimum_samples: int = 3,
    maximum_rest_std_g: float | None = None,
    maximum_rest_peak_to_peak_g: float | None = None,
    maximum_gravity_norm_deviation_g: float | None = None,
) -> StaticCompensation:
    """Estimate gravity plus sensor offset from a confirmed rest interval.

    The caller must select an interval in which the sensor is known to be
    stationary and its orientation is representative.  The function does not
    search for rest automatically.

    Rest quality is checked before a compensation vector is accepted:
    per-axis standard deviation, peak-to-peak range, and (optionally) the
    norm of the mean gravity vector (near 1 g for a calibrated sensor).
    Thresholds are engineering choices set by config.py, not jury constants.
    """
    time_s, ax_g, ay_g, az_g = _validated_arrays(data)

    if not np.isfinite(rest_start_s) or not np.isfinite(rest_end_s):
        raise ValueError("Rest interval boundaries must be finite")
    if not time_s[0] <= rest_start_s < rest_end_s <= time_s[-1]:
        raise ValueError(
            "Rest interval must satisfy "
            "time_s[0] <= rest_start_s < rest_end_s <= time_s[-1]"
        )
    if minimum_samples < 3:
        raise ValueError("minimum_samples must be at least 3")
    for name, limit in (
        ("maximum_rest_std_g", maximum_rest_std_g),
        ("maximum_rest_peak_to_peak_g", maximum_rest_peak_to_peak_g),
        ("maximum_gravity_norm_deviation_g", maximum_gravity_norm_deviation_g),
    ):
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError(f"{name} must be positive and finite")

    mask = (time_s >= rest_start_s) & (time_s <= rest_end_s)
    sample_count = int(np.count_nonzero(mask))
    if sample_count < minimum_samples:
        raise ValueError(
            f"Confirmed rest interval contains {sample_count} samples; "
            f"at least {minimum_samples} are required"
        )

    means = np.array(
        [np.mean(ax_g[mask]), np.mean(ay_g[mask]), np.mean(az_g[mask])],
        dtype=np.float64,
    )
    standard_deviations = np.array(
        [np.std(ax_g[mask]), np.std(ay_g[mask]), np.std(az_g[mask])],
        dtype=np.float64,
    )

    peak_to_peak = np.array(
        [np.ptp(ax_g[mask]), np.ptp(ay_g[mask]), np.ptp(az_g[mask])],
        dtype=np.float64,
    )
    mean_gravity_norm = float(np.linalg.norm(means))

    if (
        maximum_rest_std_g is not None
        and np.max(standard_deviations) > maximum_rest_std_g
    ):
        raise ValueError(
            "Selected rest interval is too noisy: maximum axis standard "
            f"deviation is {np.max(standard_deviations):.6g} g, limit is "
            f"{maximum_rest_std_g:.6g} g"
        )

    if (
        maximum_rest_peak_to_peak_g is not None
        and np.max(peak_to_peak) > maximum_rest_peak_to_peak_g
    ):
        raise ValueError(
            "Selected rest interval has excessive peak-to-peak variation: "
            f"{np.max(peak_to_peak):.6g} g, limit "
            f"{maximum_rest_peak_to_peak_g:.6g} g"
        )
    if (
        maximum_gravity_norm_deviation_g is not None
        and abs(mean_gravity_norm - 1.0) > maximum_gravity_norm_deviation_g
    ):
        raise ValueError(
            "Selected rest interval has implausible mean gravity magnitude: "
            f"{mean_gravity_norm:.6g} g; expected approximately 1 g "
            f"(+/-{maximum_gravity_norm_deviation_g:.6g} g). "
            "Check sensor units, calibration, orientation and rest interval."
        )

    return StaticCompensation(
        ax_g=float(means[0]),
        ay_g=float(means[1]),
        az_g=float(means[2]),
        rest_start_s=float(rest_start_s),
        rest_end_s=float(rest_end_s),
        sample_count=sample_count,
        std_ax_g=float(standard_deviations[0]),
        std_ay_g=float(standard_deviations[1]),
        std_az_g=float(standard_deviations[2]),
        max_rest_peak_to_peak_g=float(np.max(peak_to_peak)),
        gravity_vector_norm_g=mean_gravity_norm,
    )


def apply_static_compensation(
    data: AccelerationRecord,
    compensation: StaticCompensation,
) -> CompensatedAxes:
    """Subtract an explicitly supplied gravity-plus-offset vector."""
    _, ax_g, ay_g, az_g = _validated_arrays(data)
    if not np.all(np.isfinite(compensation.vector_g)):
        raise ValueError("Compensation vector contains NaN or infinity")

    return CompensatedAxes(
        ax_g=ax_g - compensation.ax_g,
        ay_g=ay_g - compensation.ay_g,
        az_g=az_g - compensation.az_g,
        compensation=compensation,
    )


def resolve_static_compensation(
    data: AccelerationRecord,
    compensation_config: object,
) -> StaticCompensation | None:
    """Create compensation only from an explicitly configured rest interval."""
    method = getattr(compensation_config, "method", "none")
    if method == "none":
        return None
    if method != "confirmed_rest":
        raise ValueError(f"Unsupported compensation method: {method!r}")

    rest_start_s = getattr(compensation_config, "rest_start_s", None)
    rest_end_s = getattr(compensation_config, "rest_end_s", None)
    if rest_start_s is None or rest_end_s is None:
        raise ValueError(
            "confirmed_rest compensation requires rest_start_s and rest_end_s"
        )
    return estimate_static_compensation(
        data,
        float(rest_start_s),
        float(rest_end_s),
        minimum_samples=int(
            getattr(compensation_config, "minimum_rest_samples", 32)
        ),
        maximum_rest_std_g=getattr(
            compensation_config, "maximum_rest_std_g", None
        ),
        maximum_rest_peak_to_peak_g=getattr(
            compensation_config, "maximum_rest_peak_to_peak_g", None
        ),
        maximum_gravity_norm_deviation_g=getattr(
            compensation_config, "maximum_gravity_norm_deviation_g", None
        ),
    )


def prepare_acceleration_channels(
    data: AccelerationRecord,
    compensation_config: object,
    *,
    compensation: StaticCompensation | None = None,
) -> PreparedAcceleration:
    """Apply the configured raw or dynamic channel-selection policy."""
    time_s, ax_g, ay_g, az_g = _validated_arrays(data)
    selected_compensation = compensation
    if selected_compensation is None:
        selected_compensation = resolve_static_compensation(
            data,
            compensation_config,
        )

    mode = getattr(compensation_config, "magnitude_mode", "raw_magnitude")
    if mode == "raw_magnitude":
        selected_ax, selected_ay, selected_az = ax_g, ay_g, az_g
        total = calculate_total_acceleration(data, mode="raw_magnitude")
    elif mode == "dynamic_magnitude":
        if selected_compensation is None:
            raise ValueError(
                "dynamic_magnitude requires confirmed static compensation"
            )
        corrected = apply_static_compensation(data, selected_compensation)
        selected_ax = corrected.ax_g
        selected_ay = corrected.ay_g
        selected_az = corrected.az_g
        total = calculate_total_acceleration(
            data,
            mode="dynamic_magnitude",
            compensation=selected_compensation,
        )
    else:
        raise ValueError(f"Unsupported magnitude mode: {mode!r}")

    return PreparedAcceleration(
        time_s=time_s.copy(),
        ax_g=np.asarray(selected_ax, dtype=np.float64).copy(),
        ay_g=np.asarray(selected_ay, dtype=np.float64).copy(),
        az_g=np.asarray(selected_az, dtype=np.float64).copy(),
        a_total_g=total.a_total_g.copy(),
        mode=mode,
        compensation=selected_compensation,
    )


def calculate_total_acceleration(
    data: AccelerationRecord,
    mode: MagnitudeMode = "raw_magnitude",
    *,
    compensation: StaticCompensation | None = None,
) -> TotalAccelerationResult:
    """Calculate raw or explicitly compensated resultant acceleration.

    ``raw_magnitude`` uses the original axes, including gravity and offsets.
    ``dynamic_magnitude`` requires ``compensation`` and subtracts its vector
    before calculating the magnitude.
    """
    time_s, ax_g, ay_g, az_g = _validated_arrays(data)

    if mode == "raw_magnitude":
        if compensation is not None:
            raise ValueError(
                "Do not pass compensation in raw_magnitude mode; use "
                "dynamic_magnitude explicitly"
            )
        selected_ax, selected_ay, selected_az = ax_g, ay_g, az_g
    elif mode == "dynamic_magnitude":
        if compensation is None:
            raise ValueError(
                "dynamic_magnitude requires a StaticCompensation created "
                "from a confirmed rest interval"
            )
        corrected = apply_static_compensation(data, compensation)
        selected_ax = corrected.ax_g
        selected_ay = corrected.ay_g
        selected_az = corrected.az_g
    else:
        raise ValueError("mode must be 'raw_magnitude' or 'dynamic_magnitude'")

    # Nested hypot is stable and avoids unnecessary overflow in squaring.
    a_total_g = np.hypot(np.hypot(selected_ax, selected_ay), selected_az)
    return TotalAccelerationResult(
        time_s=time_s.copy(),
        a_total_g=a_total_g,
        mode=mode,
        compensation=compensation,
    )


def contest_trigger_mask(
    data: AccelerationRecord,
    *,
    trigger_g: float = 3.0,
    compensation: StaticCompensation | None = None,
) -> np.ndarray:
    """Return samples exceeding the contest threshold on any single axis.

    The resultant acceleration is deliberately not used.  If compensation is
    supplied, the three compensated axes are checked; otherwise the original
    axes are checked.  The comparison follows the requirement "exceeds 3g",
    therefore it uses ``>`` rather than ``>=``.
    """
    if not np.isfinite(trigger_g) or trigger_g <= 0:
        raise ValueError("trigger_g must be positive and finite")

    _, ax_g, ay_g, az_g = _validated_arrays(data)
    if compensation is not None:
        corrected = apply_static_compensation(data, compensation)
        ax_g, ay_g, az_g = corrected.ax_g, corrected.ay_g, corrected.az_g

    return (
        (np.abs(ax_g) > trigger_g)
        | (np.abs(ay_g) > trigger_g)
        | (np.abs(az_g) > trigger_g)
    )


def first_contest_trigger_index(
    data: AccelerationRecord,
    *,
    trigger_g: float = 3.0,
    compensation: StaticCompensation | None = None,
) -> int | None:
    """Return the first threshold-crossing index, or ``None`` if absent."""
    indices = np.flatnonzero(
        contest_trigger_mask(data, trigger_g=trigger_g, compensation=compensation)
    )
    return int(indices[0]) if indices.size else None


def detect_event_frame(
    data: AccelerationRecord,
    *,
    trigger_g: float = 3.0,
    frame_level_fraction_of_amax: float = 0.25,
    compensation: StaticCompensation | None = None,
) -> EventFrame | None:
    """Find the first triggered shock and its 0.25*Amax frame boundaries.

    ``Amax`` is the maximum of ``max(|ax|, |ay|, |az|)`` after the first
    trigger.  Boundaries are the nearest crossings of the configured Amax
    fraction on the left and right of that peak.
    """
    if not np.isfinite(frame_level_fraction_of_amax) or not (
        0.0 < frame_level_fraction_of_amax <= 1.0
    ):
        raise ValueError("frame_level_fraction_of_amax must be in (0, 1]")

    _, ax_g, ay_g, az_g = _validated_arrays(data)
    if compensation is not None:
        corrected = apply_static_compensation(data, compensation)
        ax_g, ay_g, az_g = corrected.ax_g, corrected.ay_g, corrected.az_g

    axis_envelope_g = np.maximum.reduce(
        (np.abs(ax_g), np.abs(ay_g), np.abs(az_g))
    )
    trigger_indices = np.flatnonzero(axis_envelope_g > trigger_g)
    if trigger_indices.size == 0:
        return None

    trigger_index = int(trigger_indices[0])
    peak_index = trigger_index + int(np.argmax(axis_envelope_g[trigger_index:]))
    amax_g = float(axis_envelope_g[peak_index])
    frame_level_g = frame_level_fraction_of_amax * amax_g

    start_index = peak_index
    while start_index > 0 and axis_envelope_g[start_index - 1] >= frame_level_g:
        start_index -= 1

    end_index = peak_index
    last_index = axis_envelope_g.size - 1
    while end_index < last_index and axis_envelope_g[end_index + 1] >= frame_level_g:
        end_index += 1

    return EventFrame(
        start_index=start_index,
        end_index=end_index,
        amax_g=amax_g,
        status="triggered",
    )


def build_acceleration_table(
    data: AccelerationRecord,
    total: TotalAccelerationResult,
) -> np.ndarray:
    """Return columns [t, ax, ay, az, a_total] without changing the axes."""
    time_s, ax_g, ay_g, az_g = _validated_arrays(data)
    if total.a_total_g.shape != time_s.shape or not np.allclose(total.time_s, time_s):
        raise ValueError("Total-acceleration result does not match the source record")
    return np.column_stack((time_s, ax_g, ay_g, az_g, total.a_total_g))
