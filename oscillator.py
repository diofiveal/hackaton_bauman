"""One oscillator: recursive sample update, state, response and Maximax."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from coefficients import SmallwoodCoefficients
from validation import finite_signal


@dataclass
class SmallwoodState:
    """Mutable streaming state for one oscillator."""

    x1_g: float = 0.0
    x2_g: float = 0.0
    y1_g: float = 0.0
    y2_g: float = 0.0
    peak_abs_g: float = 0.0

    def reset(self) -> None:
        self.x1_g = 0.0
        self.x2_g = 0.0
        self.y1_g = 0.0
        self.y2_g = 0.0
        self.peak_abs_g = 0.0


def update_oscillator_state(
    sample_g: float,
    coefficients: SmallwoodCoefficients,
    state: SmallwoodState,
) -> float:
    """Process one input sample and update one oscillator state.

    The function implements exactly one recurrence step::

        y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2]
               - a1*y[k-1] - a2*y[k-2]

    The returned value is ``y[k]``.  ``state.peak_abs_g`` stores Maximax.
    """
    sample = float(sample_g)
    if not np.isfinite(sample):
        raise ValueError("sample_g must be finite")

    response_g = (
        coefficients.b0 * sample
        + coefficients.b1 * state.x1_g
        + coefficients.b2 * state.x2_g
        - coefficients.a1 * state.y1_g
        - coefficients.a2 * state.y2_g
    )
    if not np.isfinite(response_g):
        raise FloatingPointError("Oscillator response became NaN or infinity")

    state.x2_g = state.x1_g
    state.x1_g = sample
    state.y2_g = state.y1_g
    state.y1_g = float(response_g)
    state.peak_abs_g = max(state.peak_abs_g, abs(float(response_g)))
    return float(response_g)


def _tail_sample_count(
    coefficients: SmallwoodCoefficients,
    tail_mode: str,
    tail_time_constants: float,
) -> int:
    """Return zero-tail length from the damped-oscillator time constant."""
    if tail_mode == "frame_only":
        return 0
    if tail_mode != "zero_tail":
        raise ValueError("tail_mode must be 'frame_only' or 'zero_tail'")
    if not np.isfinite(tail_time_constants) or tail_time_constants <= 0.0:
        raise ValueError("zero_tail requires positive finite tail_time_constants")

    # Envelope: exp(-zeta*omega_n*t), therefore tau = Q/(pi*f_n).
    time_constant_s = coefficients.quality_factor_q / (
        np.pi * coefficients.frequency_hz
    )
    tail_duration_s = tail_time_constants * time_constant_s
    return int(np.ceil(tail_duration_s * coefficients.sample_rate_hz))


def calculate_one_oscillator(
    acceleration_g: Iterable[float] | np.ndarray,
    coefficients: SmallwoodCoefficients,
    *,
    state: SmallwoodState | None = None,
    reset_state: bool = True,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> np.ndarray:
    """Calculate the complete response of one virtual oscillator.

    A caller may provide ``state`` for streaming processing.  By default that
    state is reset before the signal is processed, so repeated calculations
    are independent.  Set ``reset_state=False`` only to continue a previous
    block of the same signal.
    """
    signal = finite_signal(acceleration_g, "acceleration_g")
    oscillator_state = state if state is not None else SmallwoodState()
    if reset_state:
        oscillator_state.reset()

    tail_count = _tail_sample_count(
        coefficients,
        tail_mode,
        tail_time_constants,
    )
    response_g = np.empty(signal.size + tail_count, dtype=np.float64)
    for index, sample_g in enumerate(signal):
        response_g[index] = update_oscillator_state(
            float(sample_g), coefficients, oscillator_state
        )
    for tail_index in range(tail_count):
        response_g[signal.size + tail_index] = update_oscillator_state(
            0.0,
            coefficients,
            oscillator_state,
        )
    return response_g


def calculate_one_oscillator_maximax(
    acceleration_g: Iterable[float] | np.ndarray,
    coefficients: SmallwoodCoefficients,
    *,
    tail_mode: str = "frame_only",
    tail_time_constants: float = 0.0,
) -> float:
    """Return ``max(abs(y))`` for one signal and one oscillator."""
    state = SmallwoodState()
    calculate_one_oscillator(
        acceleration_g,
        coefficients,
        state=state,
        reset_state=True,
        tail_mode=tail_mode,
        tail_time_constants=tail_time_constants,
    )
    return float(state.peak_abs_g)
