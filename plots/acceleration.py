"""Plot the recorded XYZ acceleration and inclusive detected frame."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import PipelineResult


def plot_acceleration_with_frame(
    result: PipelineResult,
    output_path: str | Path,
    *,
    title: str = "Acceleration signal with detected shock frame",
    dpi: int = 150,
) -> Path:
    """Plot the full raw ax/ay/az signals and highlight the selected event frame.

    The frame indices are supplied by the existing SRS pipeline: this function
    does not detect a shock or change the acceleration data. The source data
    have time in seconds and acceleration in g.
    """
    source = result.source_data
    time_s = np.asarray(source.time_s, dtype=np.float64)
    channels = (
        ("X", np.asarray(source.ax_g, dtype=np.float64), "tab:blue"),
        ("Y", np.asarray(source.ay_g, dtype=np.float64), "tab:green"),
        ("Z", np.asarray(source.az_g, dtype=np.float64), "tab:red"),
    )
    if time_s.ndim != 1 or time_s.size == 0:
        raise ValueError("Cannot plot an empty or invalid time series")
    if any(values.shape != time_s.shape for _, values, _ in channels):
        raise ValueError("Acceleration channels must match the time array")

    event = result.event_frame
    event_start_s = event_end_s = None
    if event is not None:
        if not (0 <= event.start_index <= event.end_index < time_s.size):
            raise ValueError("Event frame indices are outside the input time series")
        event_start_s = float(time_s[event.start_index])
        event_end_s = float(time_s[event.end_index])

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    try:
        for axis, (name, values_g, color) in zip(axes, channels, strict=True):
            axis.plot(
                time_s,
                values_g,
                color=color,
                linewidth=1.0,
                label=f"a{name.lower()}(t)",
            )
            axis.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.65)

            if event_start_s is not None and event_end_s is not None:
                axis.axvspan(
                    event_start_s,
                    event_end_s,
                    color="orange",
                    alpha=0.23,
                    label="Selected shock frame",
                )
                axis.axvline(
                    event_start_s, color="darkorange", linestyle="--", linewidth=1.0
                )
                axis.axvline(
                    event_end_s, color="darkorange", linestyle="--", linewidth=1.0
                )

            axis.set_ylabel(f"a{name.lower()}, g")
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.legend(loc="upper right")

        axes[-1].set_xlabel("Time, s")
        if event is None:
            detail = "No shock frame detected"
        else:
            detail = (
                f"Frame: samples {event.start_index}–{event.end_index} | "
                f"Amax = {event.amax_g:.3f} g | "
                f"t = {event_start_s:.6f}–{event_end_s:.6f} s"
            )
        fig.suptitle(f"{title}\n{detail}", fontsize=13)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        return output_path
    finally:
        plt.close(fig)
