"""Validation shared by the numerical modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import AccelerationData

import numpy as np


def finite_signal(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    """Return a finite, nonempty, one-dimensional float64 signal."""
    signal = np.asarray(
        tuple(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.float64,
    )
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} contains NaN or infinity")
    return signal


def frequency_array(
    frequencies_hz: Iterable[float] | np.ndarray,
    *,
    minimum_size: int = 1,
) -> np.ndarray:
    """Validate an increasing oscillator-frequency grid."""
    frequencies = finite_signal(frequencies_hz, "frequencies_hz")
    if frequencies.size < minimum_size:
        raise ValueError(f"Frequency grid requires at least {minimum_size} values")
    if np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain positive values")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequencies_hz must be strictly increasing")
    return frequencies


def validate_acceleration(
    data: AccelerationData,
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
