"""3 g per-axis trigger and inclusive frame at a fraction of Amax."""

from __future__ import annotations

import numpy as np

from models import AccelerationData, EventFrame
from validation import validate_acceleration


def detect_event_frame(
    data: AccelerationData,
    *,
    trigger_g: float = 3.0,
    frame_level_fraction_of_amax: float = 0.25,
) -> EventFrame | None:
    """Find the first triggered shock and its 0.25*Amax frame boundaries.

    ``Amax`` is the maximum of ``max(|ax|, |ay|, |az|)`` after the first
    trigger.  Boundaries are the nearest crossings of the configured Amax
    fraction on the left and right of that peak.
    """
    if not np.isfinite(trigger_g) or trigger_g <= 0:
        raise ValueError("trigger_g must be positive and finite")
    if not np.isfinite(frame_level_fraction_of_amax) or not (
        0.0 < frame_level_fraction_of_amax <= 1.0
    ):
        raise ValueError("frame_level_fraction_of_amax must be in (0, 1]")

    _, ax_g, ay_g, az_g = validate_acceleration(data)

    axis_envelope_g = np.maximum.reduce((np.abs(ax_g), np.abs(ay_g), np.abs(az_g)))
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
    )
