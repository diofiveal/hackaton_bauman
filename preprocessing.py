"""Confirmed-rest compensation. Gravity is never estimated automatically."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import StaticCompensationConfig
from models import AccelerationData, StaticCompensation
from validation import validate_acceleration


def _rest_mask(
    time_s: np.ndarray, start: float, end: float, minimum: int
) -> np.ndarray:
    """Select an inclusive, explicitly confirmed interval in input seconds."""
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("Rest interval boundaries must be finite")
    if not time_s[0] <= start < end <= time_s[-1]:
        raise ValueError(
            f"Rest interval [{start}, {end}] s is outside the recording "
            f"[{time_s[0]}, {time_s[-1]}] s. Boundaries use input timestamps "
            "converted to seconds; both interval endpoints are included."
        )
    if minimum < 3:
        raise ValueError("minimum_samples must be at least 3")
    mask = (time_s >= start) & (time_s <= end)
    count = int(np.count_nonzero(mask))
    if count < minimum:
        raise ValueError(
            f"Confirmed rest interval contains {count} samples; at least {minimum} are required"
        )
    return mask


def _check_quality_limits(
    std_limit: float | None,
    peak_to_peak_limit: float | None,
    gravity_limit: float | None,
) -> None:
    for name, limit in (
        ("maximum_rest_std_g", std_limit),
        ("maximum_rest_peak_to_peak_g", peak_to_peak_limit),
        ("maximum_gravity_norm_deviation_g", gravity_limit),
    ):
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError(f"{name} must be positive and finite")


def _check_rest_quality(
    standard_deviations: np.ndarray,
    peak_to_peak: np.ndarray,
    mean_gravity_norm: float,
    maximum_rest_std_g: float | None,
    maximum_rest_peak_to_peak_g: float | None,
    maximum_gravity_norm_deviation_g: float | None,
) -> None:
    """Reject a noisy or physically inconsistent confirmed rest interval."""
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


def estimate_static_compensation(
    data: AccelerationData,
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
    time_s, ax_g, ay_g, az_g = validate_acceleration(data)

    mask = _rest_mask(time_s, rest_start_s, rest_end_s, minimum_samples)
    sample_count = int(np.count_nonzero(mask))
    _check_quality_limits(
        maximum_rest_std_g,
        maximum_rest_peak_to_peak_g,
        maximum_gravity_norm_deviation_g,
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

    _check_rest_quality(
        standard_deviations,
        peak_to_peak,
        mean_gravity_norm,
        maximum_rest_std_g,
        maximum_rest_peak_to_peak_g,
        maximum_gravity_norm_deviation_g,
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
    data: AccelerationData, compensation: StaticCompensation
) -> AccelerationData:
    """Subtract the confirmed vector once and preserve time/measurement metadata."""
    validate_acceleration(data)
    if not np.all(np.isfinite(compensation.vector_g)):
        raise ValueError("Compensation vector contains NaN or infinity")
    return replace(
        data,
        ax_g=data.ax_g - compensation.ax_g,
        ay_g=data.ay_g - compensation.ay_g,
        az_g=data.az_g - compensation.az_g,
    )


def resolve_static_compensation(
    data: AccelerationData, config: StaticCompensationConfig
) -> StaticCompensation | None:
    """Estimate a vector only for explicitly configured confirmed rest."""
    if config.method == "none":
        return None
    if config.method != "confirmed_rest":
        raise ValueError(f"Unsupported compensation method: {config.method!r}")
    if config.rest_start_s is None or config.rest_end_s is None:
        raise ValueError("confirmed_rest requires both interval boundaries")
    return estimate_static_compensation(
        data,
        config.rest_start_s,
        config.rest_end_s,
        minimum_samples=config.minimum_rest_samples,
        maximum_rest_std_g=config.maximum_rest_std_g,
        maximum_rest_peak_to_peak_g=config.maximum_rest_peak_to_peak_g,
        maximum_gravity_norm_deviation_g=config.maximum_gravity_norm_deviation_g,
    )
