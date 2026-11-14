"""Existing contest allowable-SRS curve and its plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def srs_limit_g(frequencies_hz: np.ndarray | float) -> np.ndarray:
    """Allowable SRS (g) from the supplied contest table.

    10..100 Hz: 10 -> 20 g, linear with log10(f).
    100..1000 Hz: constant 20 g.
    1000..2000 Hz: 20 -> 50 g, linear with frequency.

    Frequencies outside [10, 2000] return NaN: the table defines no limit
    outside that interval, so extrapolation must not be shown as a rule.
    """
    f = np.asarray(frequencies_hz, dtype=np.float64)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("Frequencies must be finite and positive")

    limit = np.full(f.shape, np.nan, dtype=np.float64)
    low = (f >= 10.0) & (f <= 100.0)
    middle = (f > 100.0) & (f <= 1000.0)
    high = (f > 1000.0) & (f <= 2000.0)

    limit[low] = 10.0 + 10.0 * np.log10(f[low] / 10.0)
    limit[middle] = 20.0
    limit[high] = 20.0 + (50.0 - 20.0) * (f[high] - 1000.0) / 1000.0
    return limit


def _limit_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Use dense samples to draw straight segments in the intended scales."""
    freq = np.concatenate(
        (
            np.geomspace(10.0, 100.0, 160),
            np.geomspace(100.0, 1000.0, 160)[1:],
            np.linspace(1000.0, 2000.0, 160)[1:],
        )
    )
    return freq, srs_limit_g(freq)


def draw_srs_limit(axis: plt.Axes) -> None:
    freq, limit = _limit_coordinates()
    axis.plot(
        freq,
        limit,
        color="black",
        linewidth=2.6,
        linestyle="-.",
        label="Allowable SRS (As)",
        zorder=15,
    )


def plot_srs_limit(output_path: str | Path, *, dpi: int = 160) -> Path:
    """Save the threshold curve without loading any SRS CSV files."""
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    try:
        draw_srs_limit(ax)
        ax.set_xscale("log")
        ax.set_xlim(10.0, 2000.0)
        ax.set_ylim(0, 55.0)
        ax.set_xticks([10.0, 100.0, 1000.0, 2000.0])
        ax.set_xticklabels(["10", "100", "1000", "2000"])
        ax.set_xlabel("Natural frequency, Hz")
        ax.set_ylabel("Allowable SRS, g")
        ax.set_title("Allowable SRS threshold (contest)")
        ax.grid(True, which="both", linestyle=":", alpha=0.45)
        ax.legend(loc="upper left")
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=dpi)
        return dest
    finally:
        plt.close(fig)
