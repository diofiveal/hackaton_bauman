"""Load CLI settings and orchestrate calculation, export and optional graphics."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from cli_args import create_argument_parser
from config_io import load_config
from envelope import build_envelope_from_files
from export import write_outputs
from paths import DEFAULT_CONFIG_PATH, DEFAULT_INPUT_PATH
from pipeline import PipelineResult, run_srs_pipeline
from reporting import print_summary


def command_line_overrides(args: argparse.Namespace) -> dict[str, Any]:
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
        overrides.update(
            {
                "compensation.method": "none",
                "compensation.magnitude_mode": "raw_magnitude",
                "compensation.rest_start_s": None,
                "compensation.rest_end_s": None,
            }
        )
    if args.input_gravity_compensated:
        overrides["input.gravity_compensated"] = True
    if args.tail_mode is not None:
        overrides["tail_mode"] = args.tail_mode
    if args.tail_time_constants is not None:
        overrides["tail_time_constants"] = args.tail_time_constants
    return overrides


def _write_envelope(result: PipelineResult, args: argparse.Namespace) -> None:
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


def main() -> int:
    parser = create_argument_parser()
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
            cli_overrides=command_line_overrides(args),
        )
        result = run_srs_pipeline(args.csv_path, config)
        write_outputs(
            result,
            args.csv_path,
            args.output,
            args.metadata_output,
        )
        from plots.acceleration import plot_acceleration_with_frame

        plot_acceleration_with_frame(
            result,
            args.plot_output,
            title="Acceleration signal and detected shock frame",
        )
        print_summary(result, args.output, args.metadata_output, args.plot_output)
        _write_envelope(result, args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0
