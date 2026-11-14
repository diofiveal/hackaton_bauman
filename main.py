"""Command-line entry point for the complete SRS processing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

from vizualization import plot_acceleration_with_frame

from envelope import build_envelope_from_files
from config import load_config
from pipeline import PipelineResult, run_srs_pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_DIR / "input" / "raw_signal_0051.csv"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "examples" / "contest.json"
DEFAULT_SRS_PATH = PROJECT_DIR / "output" / "srs_result.csv"
DEFAULT_METADATA_PATH = PROJECT_DIR / "output" / "event_metadata.json"
DEFAULT_ACCELERATION_PLOT_PATH = PROJECT_DIR / "output" / "acceleration_plot.png"
DEFAULT_ENVELOPE_PATH = PROJECT_DIR / "output" / "srs_envelope.csv"
DEFAULT_ENVELOPE_PLOT_PATH = PROJECT_DIR / "output" / "srs_envelope.png"


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a shock event and calculate Maximax SRS.",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Input [t, ax, ay, az] CSV "
            "(default: input/raw_signal_0051.csv)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "contest"),
        default=None,
        help="Processing mode: validation=whole raw record; contest=detected 3g shock frame",
    )
    parser.add_argument(
        "--profile",
        choices=("contest", "custom"),
        default="contest",
        help="Base configuration profile (default: contest)",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional JSON configuration file",
    )
    parser.add_argument(
        "--rest-start",
        type=float,
        help="Confirmed rest interval start in seconds",
    )
    parser.add_argument(
        "--rest-end",
        type=float,
        help="Confirmed rest interval end in seconds",
    )
    parser.add_argument(
        "--no-compensation",
        action="store_true",
        help="Disable rest compensation (contest then needs --input-gravity-compensated)",
    )
    parser.add_argument(
        "--input-gravity-compensated",
        action="store_true",
        help="Assert CSV already excludes gravity and bias (do not use on raw data)",
    )
    parser.add_argument(
        "--tail-mode",
        choices=("frame_only", "zero_tail"),
        help="Override oscillator tail mode",
    )
    parser.add_argument(
        "--tail-time-constants",
        type=float,
        help="Decay duration in time constants for zero_tail",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SRS_PATH,
        help="Output SRS CSV path",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_ACCELERATION_PLOT_PATH,
        help="Output PNG path for acceleration visualization",
    )
    parser.add_argument(
        "--envelope-inputs",
        nargs="*",
        default=[],
        metavar="SRS_CSV",
        help=(
            "Previously saved SRS CSVs to combine with the new SRS. "
            "Accepts files and quoted glob patterns."
        ),
    )
    parser.add_argument(
        "--envelope-output",
        type=Path,
        default=DEFAULT_ENVELOPE_PATH,
        help="Output CSV for the SRS envelope",
    )
    parser.add_argument(
        "--envelope-plot",
        type=Path,
        default=DEFAULT_ENVELOPE_PLOT_PATH,
        help="Output PNG for the SRS envelope",
    )
    parser.add_argument(
        "--no-envelope",
        action="store_true",
        help="Skip automatic envelope generation",
    )
    return parser


def _command_line_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if (args.rest_start is None) != (args.rest_end is None):
        raise ValueError("--rest-start and --rest-end must be specified together")
    if args.no_compensation and args.rest_start is not None:
        raise ValueError("--no-compensation conflicts with --rest-start/--rest-end")
    if args.input_gravity_compensated and args.rest_start is not None:
        raise ValueError("Already-compensated input conflicts with rest compensation")
    if args.mode == "validation" and (
        args.rest_start is not None or args.input_gravity_compensated
    ):
        raise ValueError("Validation uses the full raw signal; no compensation flags")

    overrides: dict[str, Any] = {}
    if args.mode is not None:
        overrides["processing_mode"] = args.mode
    if args.rest_start is not None:
        overrides.update(
            {
                "compensation.magnitude_mode": "dynamic_magnitude",
                "compensation.method": "confirmed_rest",
                "compensation.rest_start_s": args.rest_start,
                "compensation.rest_end_s": args.rest_end,
            }
        )
    if args.no_compensation:
        overrides.update({
            "compensation.method": "none",
            "compensation.magnitude_mode": "raw_magnitude",
            "compensation.rest_start_s": None,
            "compensation.rest_end_s": None,
        })
    if args.input_gravity_compensated:
        overrides["input.gravity_compensated"] = True
    if args.tail_mode is not None:
        overrides["tail_mode"] = args.tail_mode
    if args.tail_time_constants is not None:
        overrides["tail_time_constants"] = args.tail_time_constants
    return overrides


def _spectrum_peak(frequencies_hz: np.ndarray, values_g: np.ndarray) -> dict[str, float]:
    index = int(np.argmax(values_g))
    return {
        "value_g": float(values_g[index]),
        "frequency_hz": float(frequencies_hz[index]),
    }


def _metadata(result: PipelineResult, input_path: Path) -> dict[str, Any]:
    document: dict[str, Any] = {
        "input_file": str(input_path),
        "status": result.status,
        "processing_mode": result.mode,
        "srs_input_sample_count": (
            result.source_data.sample_count
            if result.mode == "validation"
            else (result.frame_data.sample_count if result.frame_data is not None else 0)
        ),
        "compensation_applied": (
            result.mode == "contest" and result.compensation is not None
        ),
        "gravity_handling": (
            "raw_full_record"
            if result.mode == "validation"
            else (
                "checked_rest_vector"
                if result.compensation is not None
                else "declared_precompensated_input"
            )
        ),
        "sample_count": result.source_data.sample_count,
        "sample_rate_hz": float(result.source_data.sample_rate_hz),
        "warnings": list(result.warnings),
    }

    if result.compensation is not None:
        document["compensation"] = {
            "ax_g": result.compensation.ax_g,
            "ay_g": result.compensation.ay_g,
            "az_g": result.compensation.az_g,
            "rest_start_s": result.compensation.rest_start_s,
            "rest_end_s": result.compensation.rest_end_s,
            "sample_count": result.compensation.sample_count,
            "maximum_rest_std_g": result.compensation.maximum_rest_std_g,
            "max_rest_peak_to_peak_g": result.compensation.max_rest_peak_to_peak_g,
            "gravity_vector_norm_g": result.compensation.gravity_vector_norm_g,
        }

    if result.event_frame is not None and result.frame_data is not None:
        document["event"] = {
            "start_index": result.event_frame.start_index,
            "end_index": result.event_frame.end_index,
            "start_time_s": float(result.frame_data.time_s[0]),
            "end_time_s": float(result.frame_data.time_s[-1]),
            "sample_count": result.event_frame.sample_count,
            "amax_g": result.event_frame.amax_g,
        }

    if result.srs_result is not None:
        spectrum = result.srs_result
        document["srs"] = {
            "frequency_count": int(spectrum.frequencies_hz.size),
            "quality_factor_q": spectrum.info.quality_factor_q,
            "tail_mode": spectrum.info.tail_mode,
            "tail_time_constants": spectrum.info.tail_time_constants,
            "peaks": {
                "gx": _spectrum_peak(spectrum.frequencies_hz, spectrum.gx_g),
                "gy": _spectrum_peak(spectrum.frequencies_hz, spectrum.gy_g),
                "gz": _spectrum_peak(spectrum.frequencies_hz, spectrum.gz_g),
                "g_total": _spectrum_peak(
                    spectrum.frequencies_hz,
                    spectrum.g_total_g,
                ),
            },
        }
    return document


def _write_outputs(
    result: PipelineResult,
    input_path: Path,
    srs_path: Path,
    metadata_path: Path,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(_metadata(result, input_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table = result.srs_table()
    if table is None:
        return
    srs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        srs_path,
        table,
        delimiter=",",
        header="f_hz,gx_g,gy_g,gz_g,g_total_g",
        comments="",
        fmt="%.12g",
    )


def _print_summary(
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
            peak = _spectrum_peak(spectrum.frequencies_hz, values)
            print(
                f"Max {name}: {peak['value_g']:.6f} g "
                f"at {peak['frequency_hz']:.6f} Hz"
            )
        print(f"SRS table: {srs_path}")

    print(f"Metadata: {metadata_path}")
    print(f"Acceleration plot: {plot_path}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")


def main() -> int:
    parser = _create_argument_parser()
    args = parser.parse_args()
    try:
        config_file = args.config_file
        using_example_input = args.csv_path.resolve() == DEFAULT_INPUT_PATH.resolve()
        if (
            config_file is None
            and using_example_input
            and args.rest_start is None
            and args.rest_end is None
            and args.mode != "validation"
        ):
            config_file = DEFAULT_CONFIG_PATH

        config = load_config(
            args.profile,
            config_file=config_file,
            cli_overrides=_command_line_overrides(args),
        )
        result = run_srs_pipeline(args.csv_path, config)
        _write_outputs(
            result,
            args.csv_path,
            args.output,
            args.metadata_output,
        )
        plot_acceleration_with_frame(
            result,
            args.plot_output,
            title="Acceleration signal and detected shock frame",
        )
        _print_summary(result, args.output, args.metadata_output, args.plot_output)
        if not args.no_envelope and result.mode == "validation":
            print("Envelope skipped: validation is a full-record reference, not a shock")
        elif not args.no_envelope:
            if result.srs_result is None:
                print("Envelope skipped: no shock was detected")
            else:
                envelope = build_envelope_from_files(
                    [args.output, *args.envelope_inputs],
                    output_path=args.envelope_output,
                    plot_path=args.envelope_plot,
                )
                print(f"Envelope frequencies: {envelope.frequencies_hz.size}")
                print(f"Envelope CSV: {args.envelope_output}")
                print(f"Envelope plot: {args.envelope_plot}")
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
