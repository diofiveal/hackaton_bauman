"""Overlay PC and check spectra on their original frequency grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.limits import draw_srs_limit
from srs_io import read_srs_csv


def plot_srs_comparison(
    srs_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 160,
    layout: str = "combined",
    show_limit: bool = True,
) -> Path:
    """Compare PC/check SRS and optionally overlay allowable SRS threshold.

    Combined layout shows all X/Y/Z curves on one chart; stacked layout
    shows three panels. When enabled, the contest limit is drawn on each.
    No Smallwood recalculation, interpolation, or frequency rounding occurs.
    Both series retain the frequency coordinates stored in their own CSVs.
    Accepted headers are defined by srs_io.read_srs_csv, including
    f_hz,gx_g,gy_g,gz_g and f,gx,gy,gz.
    """
    pc_path = Path(srs_path)
    check_path = Path(reference_path)
    pc = read_srs_csv(pc_path)
    check = read_srs_csv(check_path)

    pc_f = np.asarray(pc.frequencies_hz)
    check_f = np.asarray(check.frequencies_hz)

    _warn_grid_difference(pc_f, check_f)

    if layout not in {"combined", "stacked"}:
        raise ValueError("layout must be 'combined' or 'stacked'")

    channels = (
        ("X", pc.gx_g, check.gx_g, "tab:blue"),
        ("Y", pc.gy_g, check.gy_g, "tab:green"),
        ("Z", pc.gz_g, check.gz_g, "tab:red"),
    )
    if layout == "combined":
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        axes = (ax,)
    else:
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
    try:
        for channel_i, (axis_name, pc_g, check_g, color) in enumerate(channels):
            ax = axes[0] if layout == "combined" else axes[channel_i]
            _draw_axis_comparison(
                ax,
                axis_name,
                pc_g,
                check_g,
                color,
                pc_f,
                check_f,
                pc_path.name,
                check_path.name,
                layout,
            )

        if show_limit:
            for axis in axes:
                draw_srs_limit(axis)
                # On the combined plot, redraw legend after adding the limit.
                if layout == "stacked":
                    axis.legend(loc="best", fontsize=8)

        if layout == "combined":
            axes[0].set_ylabel("SRS, g")
            axes[0].legend(loc="best", fontsize=9, ncol=2)

        axes[-1].set_xlabel("Natural frequency, Hz")
        axes[-1].set_xlim(
            min(
                float(pc_f[0]), float(check_f[0]), 10.0 if show_limit else float("inf")
            ),
            max(float(pc_f[-1]), float(check_f[-1]), 2000.0 if show_limit else 0.0),
        )
        fig.suptitle("Maximax SRS: PC calculation vs check spectrum", fontsize=15)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=dpi)
        return dest
    finally:
        plt.close(fig)


def _warn_grid_difference(pc_f: np.ndarray, check_f: np.ndarray) -> None:
    # A graph can overlay close but not bit-identical grids (for example,
    # the check CSV often rounds frequencies to three decimal places).
    # Avoid pretending that a frequency-by-frequency numerical comparison
    # is valid if the grids have materially different coordinates.
    if pc_f.size == check_f.size:
        max_frequency_difference = float(np.max(np.abs(pc_f - check_f)))
        if max_frequency_difference > 1e-3:
            print(
                "WARNING: PC and check SRS have different frequency values "
                f"(maximum difference {max_frequency_difference:.6g} Hz). "
                "Both are plotted on their original grids, without interpolation."
            )
    else:
        print(
            f"WARNING: PC and check SRS have {pc_f.size} and {check_f.size} "
            "frequency points. Plots use the original grids without interpolation."
        )


def _draw_axis_comparison(
    ax: plt.Axes,
    axis_name: str,
    pc_g: np.ndarray,
    check_g: np.ndarray,
    color: str,
    pc_f: np.ndarray,
    check_f: np.ndarray,
    pc_name: str,
    check_name: str,
    layout: str,
) -> None:
    ax.plot(
        pc_f,
        pc_g,
        color=color,
        linewidth=2.0,
        label=f"{axis_name}: PC ({pc_name})",
    )
    ax.plot(
        check_f,
        check_g,
        color=color,
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        label=f"{axis_name}: check ({check_name})",
    )
    ax.set_xscale("log")
    if layout == "stacked":
        ax.set_ylabel(f"SRS {axis_name}, g")
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
