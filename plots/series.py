"""Series-of-shocks visualizations, independent of numerical SRS calculation.

Input data already contain spectra calculated by the current pipeline.  Plotting
never interpolates, re-samples or modifies a spectrum or its frequency grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import SRSResult
from plots.limits import draw_srs_limit

CHANNELS = (
    ("X", "gx_g", "tab:blue"),
    ("Y", "gy_g", "tab:green"),
    ("Z", "gz_g", "tab:red"),
)


def _save(fig: plt.Figure, path: str | Path, dpi: int = 160) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(target, dpi=dpi)
        return target
    finally:
        plt.close(fig)


def _setup_axis(ax: plt.Axes, *, y_label: str, show_limit: bool) -> None:
    if show_limit:
        draw_srs_limit(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Natural frequency, Hz")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", linestyle=":", alpha=0.40)


def plot_series_srs(
    spectra: Mapping[str, SRSResult],
    envelope: SRSResult,
    output_path: str | Path,
    *,
    references: Mapping[str, SRSResult] | None = None,
    layout: str = "combined",
    axis: str | None = None,
    show_limit: bool = True,
) -> Path:
    """Plot individual PC SRS, optional matching reference, and XYZ envelopes.

    layout=combined: all axes on a single plot; stacked: three separate panels.
    axis=X/Y/Z: all individual spectra and envelope for just one axis.
    """
    if not spectra:
        raise ValueError("At least one SRS is required for series plots")
    if layout not in {"combined", "stacked"}:
        raise ValueError("layout must be 'combined' or 'stacked'")
    if axis is not None and axis.upper() not in {"X", "Y", "Z"}:
        raise ValueError("axis must be X, Y, Z or None")
    refs = references or {}
    selected = [ch for ch in CHANNELS if axis is None or ch[0] == axis.upper()]
    stacked = layout == "stacked" and len(selected) > 1
    if stacked:
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
        axes = [ax]
    for i, (name, attr, color) in enumerate(selected):
        ax = axes[i] if stacked else axes[0]
        for j, (shock_id, spectrum) in enumerate(spectra.items()):
            labeled = axis is not None and len(spectra) <= 8
            if labeled:
                label = f"PC shock {shock_id}"
            elif j == 0:
                label = f"{name}: PC shocks"
            else:
                label = None
            ax.plot(spectrum.frequencies_hz, getattr(spectrum, attr),
                    color=color, alpha=0.35, linewidth=1.2, label=label)
            if shock_id in refs:
                check = refs[shock_id]
                check_label = (f"Check shock {shock_id}" if labeled else
                               f"{name}: check shocks" if j == 0 else None)
                ax.plot(check.frequencies_hz, getattr(check, attr),
                        color=color, linestyle="--", alpha=0.30,
                        linewidth=1.0, label=check_label)
        ax.plot(envelope.frequencies_hz, getattr(envelope, attr),
                color=color, linewidth=2.9, zorder=8, label=f"Envelope {name}")
        _setup_axis(ax, y_label=f"SRS {name}, g" if stacked or axis else "SRS, g",
                    show_limit=show_limit and (stacked or i == len(selected)-1))
        if stacked:
            ax.legend(loc="best", fontsize=8, ncol=2)
    if not stacked:
        axes[0].legend(loc="best", fontsize=8, ncol=2 if axis is None else 3)
    axes[-1].set_xlim(left=min(10.0, float(envelope.frequencies_hz[0])),
                       right=max(2000.0, float(envelope.frequencies_hz[-1])))
    fig.suptitle(
        f"SRS series: {len(spectra)} shock(s), PC + check + envelopes"
        if axis is None else f"SRS axis {axis.upper()}: all shocks and allowable limit",
        fontsize=14,
    )
    return _save(fig, output_path)


def plot_envelope_with_limit(
    envelope: SRSResult,
    output_path: str | Path,
    *,
    layout: str = "combined",
) -> Path:
    """Plot final XYZ envelopes against the specified allowable SRS curve."""
    if layout not in {"combined", "stacked"}:
        raise ValueError("layout must be combined or stacked")
    if layout == "stacked":
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
        axes = [ax]
    for i, (name, attr, color) in enumerate(CHANNELS):
        ax = axes[i] if layout == "stacked" else axes[0]
        ax.plot(envelope.frequencies_hz, getattr(envelope, attr),
                color=color, linewidth=2.4, label=f"Envelope {name}")
        _setup_axis(ax, y_label=f"SRS {name}, g" if layout == "stacked" else "SRS, g",
                    show_limit=layout == "stacked" or i == 2)
        if layout == "stacked":
            ax.legend(loc="best", fontsize=9)
    if layout == "combined":
        axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlim(left=min(10.0, float(envelope.frequencies_hz[0])),
                       right=max(2000.0, float(envelope.frequencies_hz[-1])))
    fig.suptitle("Maximum SRS envelope vs allowable threshold")
    return _save(fig, output_path)


def plot_single_srs(spectrum: SRSResult, output_path: str | Path, *,
                    layout: str = "combined", title: str = "Calculated SRS") -> Path:
    """Plot a single shock even when no matching reference CSV is available."""
    if layout not in {"combined", "stacked"}:
        raise ValueError("layout must be combined or stacked")
    if layout == "stacked":
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        axes = [ax]
    for i, (name, attr, color) in enumerate(CHANNELS):
        ax = axes[i] if layout == "stacked" else axes[0]
        ax.plot(spectrum.frequencies_hz, getattr(spectrum, attr),
                color=color, linewidth=2.0, label=f"SRS {name}")
        _setup_axis(ax, y_label=f"SRS {name}, g" if layout == "stacked" else "SRS, g",
                    show_limit=layout == "stacked" or i == 2)
        if layout == "stacked":
            ax.legend(fontsize=8)
    if layout == "combined":
        axes[0].legend(fontsize=9)
    axes[-1].set_xlim(left=min(10.0, float(spectrum.frequencies_hz[0])),
                       right=max(2000.0, float(spectrum.frequencies_hz[-1])))
    fig.suptitle(title)
    return _save(fig, output_path)
