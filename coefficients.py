"""Smallwood coefficient formulas in the minus-feedback convention."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from validation import frequency_array


@dataclass(frozen=True)
class SmallwoodCoefficients:
    """Ramp-invariant coefficients for one virtual oscillator.

    The recurrence uses the case-document convention:

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]
    """

    frequency_hz: float
    sample_rate_hz: float
    quality_factor_q: float
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


def calculate_smallwood_coefficients(
    frequency_hz: float,
    sample_rate_hz: float,
    quality_factor_q: float,
) -> SmallwoodCoefficients:
    """Calculate the five coefficients from the case-document formulas.

    Returned ``a1`` and ``a2`` follow the document recurrence:

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]

    They are not converted to the CMSIS plus-feedback convention here.
    """
    parameters = np.array(
        [frequency_hz, sample_rate_hz, quality_factor_q], dtype=np.float64
    )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("frequency_hz, sample_rate_hz, and Q must be finite")
    if frequency_hz <= 0 or sample_rate_hz <= 0:
        raise ValueError("frequency_hz and sample_rate_hz must be positive")
    if quality_factor_q <= 0.5:
        raise ValueError("quality_factor_q must be greater than 0.5")

    sample_period_s = 1.0 / sample_rate_hz
    omega_n = 2.0 * np.pi * frequency_hz
    a_value = omega_n * sample_period_s / (2.0 * quality_factor_q)
    b_value = (
        omega_n * sample_period_s * np.sqrt(1.0 - 1.0 / (4.0 * quality_factor_q**2))
    )
    exponential = np.exp(-a_value)
    sinc_value = float(np.sinc(b_value / np.pi))

    b0 = 1.0 - exponential * sinc_value
    b1 = 2.0 * exponential * (sinc_value - np.cos(b_value))
    b2 = np.exp(-2.0 * a_value) - exponential * sinc_value
    a1 = -2.0 * exponential * np.cos(b_value)
    a2 = np.exp(-2.0 * a_value)

    coefficients = SmallwoodCoefficients(
        frequency_hz=float(frequency_hz),
        sample_rate_hz=float(sample_rate_hz),
        quality_factor_q=float(quality_factor_q),
        b0=float(b0),
        b1=float(b1),
        b2=float(b2),
        a1=float(a1),
        a2=float(a2),
    )
    if not np.all(np.isfinite([b0, b1, b2])) or not np.all(np.isfinite([1.0, a1, a2])):
        raise FloatingPointError(
            "Smallwood coefficient calculation produced nonfinite values"
        )
    if not np.all(np.abs(np.roots([1.0, a1, a2])) < 1.0):
        raise ValueError(
            f"Unstable oscillator coefficients for {frequency_hz:g} Hz, "
            f"fs={sample_rate_hz:g} Hz, Q={quality_factor_q:g}"
        )
    return coefficients


def precompute_smallwood_coefficients(
    frequencies_hz: Iterable[float] | np.ndarray,
    sample_rate_hz: float,
    quality_factor_q: float,
) -> tuple[SmallwoodCoefficients, ...]:
    """Return one coefficient row per validated natural frequency."""
    return tuple(
        calculate_smallwood_coefficients(
            float(frequency), sample_rate_hz, quality_factor_q
        )
        for frequency in frequency_array(frequencies_hz)
    )
