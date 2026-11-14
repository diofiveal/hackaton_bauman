"""SRS calculations using one shared oscillator kernel for every channel."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from coefficients import SmallwoodCoefficients, precompute_smallwood_coefficients
from models import AccelerationData, SRSCalculationInfo, SRSResult
from oscillator import calculate_one_oscillator_maximax
from validation import finite_signal, validate_acceleration


def calculate_total_acceleration(data: AccelerationData) -> np.ndarray:
    """Return magnitude of the supplied axes (raw or already compensated)."""
    return np.hypot(np.hypot(data.ax_g, data.ay_g), data.az_g)


def _srs_from_coefficients(
    signal: np.ndarray,
    coefficients: tuple[SmallwoodCoefficients, ...],
    tail_mode: str,
    tail_time_constants: float,
) -> np.ndarray:
    return np.array(
        [
            calculate_one_oscillator_maximax(
                signal,
                row,
                tail_mode=tail_mode,
                tail_time_constants=tail_time_constants,
            )
            for row in coefficients
        ],
        dtype=np.float64,
    )


def calculate_srs_1d(
    acceleration_g: Iterable[float] | np.ndarray,
    frequencies_hz: Iterable[float] | np.ndarray,
    sample_rate_hz: float,
    q: float,
    *,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> np.ndarray:
    """Return Maximax values in the same order as frequencies_hz."""
    signal = finite_signal(acceleration_g, "acceleration_g")
    coefficients = precompute_smallwood_coefficients(frequencies_hz, sample_rate_hz, q)
    return _srs_from_coefficients(signal, coefficients, tail_mode, tail_time_constants)


def calculate_srs_xyz(
    data: AccelerationData,
    frequencies_hz: np.ndarray,
    q: float,
    *,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> SRSResult:
    """Calculate independent XYZ and additional magnitude SRS from prepared data.

    Sampling rate comes from data.sample_rate_hz. This function performs no
    compensation, event detection or file access.
    """
    validate_acceleration(data)
    channels = (data.ax_g, data.ay_g, data.az_g, calculate_total_acceleration(data))
    coefficients = precompute_smallwood_coefficients(
        frequencies_hz, data.sample_rate_hz, q
    )
    spectra = [
        _srs_from_coefficients(
            finite_signal(channel, "acceleration_g"),
            coefficients,
            tail_mode,
            tail_time_constants,
        )
        for channel in channels
    ]
    return SRSResult(
        frequencies_hz=frequencies_hz,
        gx_g=spectra[0],
        gy_g=spectra[1],
        gz_g=spectra[2],
        g_total_g=spectra[3],
        info=SRSCalculationInfo(
            sample_rate_hz=data.sample_rate_hz,
            quality_factor_q=q,
            tail_mode=tail_mode,
            tail_time_constants=tail_time_constants,
        ),
    )
