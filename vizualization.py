"""Plots three accelerometer axes and the detected shock frame with Matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Image export must also work without a desktop display (for example, in CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .pipeline import PipelineResult
except ImportError:
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
        for axis, (name, values_g, color) in zip(axes, channels):
            axis.plot(time_s, values_g, color=color, linewidth=1.0, label=f"a{name.lower()}(t)")
            axis.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.65)

            if event_start_s is not None and event_end_s is not None:
                axis.axvspan(
                    event_start_s,
                    event_end_s,
                    color="orange",
                    alpha=0.23,
                    label="Selected shock frame",
                )
                axis.axvline(event_start_s, color="darkorange", linestyle="--", linewidth=1.0)
                axis.axvline(event_end_s, color="darkorange", linestyle="--", linewidth=1.0)

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


def plot_srs_comparison(
    srs_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 160,
    layout: str = "combined",
) -> Path:
    """Compare already calculated PC and check SRS on one Matplotlib figure.

    There are three panels (X/Y/Z), each with two overlaid spectra.
    No Smallwood recalculation, interpolation, or frequency rounding occurs.
    Both series retain the frequency coordinates stored in their own CSVs.
    Accepted headers are defined by envelope.read_srs_csv, including
    f_hz,gx_g,gy_g,gz_g and f,gx,gy,gz.
    """
    try:
        from .envelope import read_srs_csv
    except ImportError:
        from envelope import read_srs_csv

    pc_path = Path(srs_path)
    check_path = Path(reference_path)
    pc = read_srs_csv(pc_path)
    check = read_srs_csv(check_path)

    pc_f = np.asarray(pc.frequencies_hz)
    check_f = np.asarray(check.frequencies_hz)

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
            ax.plot(
                pc_f, pc_g, color=color, linewidth=2.0,
                label=f"{axis_name}: PC ({pc_path.name})",
            )
            ax.plot(
                check_f, check_g, color=color, linewidth=1.8,
                linestyle="--", alpha=0.8,
                label=f"{axis_name}: check ({check_path.name})",
            )
            ax.set_xscale("log")
            if layout == "stacked":
                ax.set_ylabel(f"SRS {axis_name}, g")
                ax.legend(loc="best", fontsize=8)
            ax.grid(True, which="both", linestyle=":", alpha=0.45)

        if layout == "combined":
            axes[0].set_ylabel("SRS, g")
            axes[0].legend(loc="best", fontsize=9, ncol=2)

        axes[-1].set_xlabel("Natural frequency, Hz")
        axes[-1].set_xlim(
            min(float(pc_f[0]), float(check_f[0])),
            max(float(pc_f[-1]), float(check_f[-1])),
        )
        fig.suptitle("Maximax SRS: PC calculation vs check spectrum", fontsize=15)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=dpi)
        return dest
    finally:
        plt.close(fig)


def _comparison_cli() -> int:
    """Allow plotting two existing SRS files without running main.py."""
    import argparse
    import sys

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Overlay PC and check SRS spectra for X, Y and Z."
    )
    parser.add_argument(
        "--srs-file", type=Path, default=root / "output" / "srs_result.csv",
        help="PC SRS CSV (f_hz,gx_g,gy_g,gz_g,...)"
    )
    parser.add_argument(
        "--check-file", type=Path, default=root / "input" / "srs_check_0051.csv",
        help="Check/reference SRS CSV (f,gx,gy,gz)"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "output" / "srs_comparison.png",
        help="Output PNG with both spectra"
    )
    parser.add_argument(
        "--layout", choices=("combined", "stacked"), default="combined",
        help="combined: all six curves on one chart; stacked: one panel per axis",
    )
    args = parser.parse_args()
    try:
        result_path = plot_srs_comparison(
            args.srs_file, args.check_file, args.output, layout=args.layout
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"SRS comparison plot: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_comparison_cli())
