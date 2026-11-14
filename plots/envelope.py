"""Plot spectra together with their pointwise XYZ maxima."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from models import SRSResult


def plot_envelope(
    spectra: Sequence[SRSResult],
    envelope: SRSResult,
    source_names: Sequence[str],
    output_path: str | Path,
    *,
    dpi: int = 160,
    show: bool = False,
) -> None:
    """Display every input SRS (thin gray) and the envelope (bold color)."""
    import matplotlib.pyplot as plt

    if len(spectra) != len(source_names):
        raise ValueError("Each SRS spectrum needs a source name")
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    series = (
        ("X", "gx_g", envelope.gx_g, "tab:blue"),
        ("Y", "gy_g", envelope.gy_g, "tab:green"),
        ("Z", "gz_g", envelope.gz_g, "tab:red"),
    )
    for ax, (axis_name, value_attr, maximum, color) in zip(axes, series, strict=True):
        for i, (spectrum, name) in enumerate(zip(spectra, source_names, strict=True)):
            label = (
                name if len(spectra) <= 6 else ("Individual shocks" if i == 0 else None)
            )
            ax.plot(
                envelope.frequencies_hz,
                getattr(spectrum, value_attr),
                color="0.55",
                linewidth=1.1,
                alpha=0.6,
                label=label,
            )
        ax.plot(
            envelope.frequencies_hz,
            maximum,
            color=color,
            linewidth=2.5,
            label=f"Envelope {axis_name}",
            zorder=10,
        )
        ax.set_xscale("log")
        ax.set_ylabel(f"SRS {axis_name}, g")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Frequency, Hz")
    axes[-1].set_xlim(
        float(envelope.frequencies_hz[0]), float(envelope.frequencies_hz[-1])
    )
    fig.suptitle(f"Maximax SRS envelope: {len(spectra)} shock(s)")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi)
        if show:
            plt.show()
    finally:
        plt.close(fig)
