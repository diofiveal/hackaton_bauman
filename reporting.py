"""Human-readable calculation summary and spectrum peak values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline import PipelineResult


def spectrum_peak(frequencies_hz: np.ndarray, values_g: np.ndarray) -> dict[str, float]:
    index = int(np.argmax(values_g))
    return {
        "value_g": float(values_g[index]),
        "frequency_hz": float(frequencies_hz[index]),
    }


def print_summary(
    result: PipelineResult,
    srs_path: Path,
    metadata_path: Path,
    plot_path: Path,
) -> None:
    print("SRS pipeline completed")
    print(f"Mode: {result.mode}")
    print(f"Status: {result.status}")
    print(f"Input samples: {result.source_data.sample_count}")
    print(f"Measured sample rate: {result.source_data.sample_rate_hz:.6f} Hz")

    if result.srs_result is None:
        print("Shock event above the configured threshold was not found")
    else:
        spectrum = result.srs_result
        if result.mode == "validation":
            print(f"SRS input: FULL record ({result.source_data.sample_count} samples)")
        elif result.event_frame is not None:
            frame = result.event_frame
            print(
                f"Event frame: samples {frame.start_index}..{frame.end_index} "
                f"({frame.sample_count} samples)"
            )
            print(f"Amax: {frame.amax_g:.6f} g")
        for name, values in (
            ("gx", spectrum.gx_g),
            ("gy", spectrum.gy_g),
            ("gz", spectrum.gz_g),
            ("g_total", spectrum.g_total_g),
        ):
            peak = spectrum_peak(spectrum.frequencies_hz, values)
            print(
                f"Max {name}: {peak['value_g']:.6f} g at {peak['frequency_hz']:.6f} Hz"
            )
        print(f"SRS table: {srs_path}")

    print(f"Metadata: {metadata_path}")
    print(f"Acceleration plot: {plot_path}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
