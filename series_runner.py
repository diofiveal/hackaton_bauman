"""One-command SRS processing for raw_signal_<ID>.csv shock series.

Examples:
    python series_runner.py
    python series_runner.py --ids 1 2 3
    python series_runner.py --ids 0051 0052 --mode validation

The existing pipeline/smallwood calculation is reused unchanged.  Jury/check
spectra are optional and NEVER included in the calculated envelope.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from config_io import load_config
from envelope import calculate_envelope
from export import write_outputs
from paths import PROJECT_DIR
from pipeline import run_srs_pipeline
from plots.acceleration import plot_acceleration_with_frame
from plots.comparison import plot_srs_comparison
from plots.envelope import plot_envelope
from plots.limits import plot_srs_limit
from plots.series import plot_envelope_with_limit, plot_series_srs, plot_single_srs
from srs_io import read_srs_csv, save_envelope_csv
from threshold_check import write_threshold_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-process raw_signal_<ID>.csv shocks, SRS, envelopes and plots."
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_DIR / "input",
                        help="Directory with raw_signal_<ID>.csv and srs_check_<ID>.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output" / "series",
                        help="Directory for all batch outputs")
    parser.add_argument("--ids", nargs="+", metavar="ID", help="Only selected numeric suffixes (e.g. 1 0051)")
    parser.add_argument("--signal-prefix", default="raw_signal_", help="Acceleration CSV name prefix")
    parser.add_argument("--check-prefix", default="srs_check_", help="Check SRS CSV name prefix")
    parser.add_argument("--require-check", action="store_true", help="Fail if matching check SRS CSV absent")
    parser.add_argument("--mode", choices=("validation", "contest"), default="validation",
                        help="validation=entire supplied CSV; contest=detected shock frame")
    parser.add_argument("--profile", choices=("contest", "custom"), default="contest")
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--input-gravity-compensated", action="store_true",
                        help="Contest ONLY; assert that CSV has no gravity/static bias")
    parser.add_argument("--rest-start", type=float, default=None, help="Contest rest start (seconds)")
    parser.add_argument("--rest-end", type=float, default=None, help="Contest rest end (seconds)")
    parser.add_argument("--tail-mode", choices=("frame_only", "zero_tail"), default=None)
    parser.add_argument("--tail-time-constants", type=float, default=None)
    return parser


def discover_signals(input_dir: Path, prefix: str, ids: list[str] | None) -> OrderedDict[str, Path]:
    """Natural-numeric order, but retain exact IDs including leading zeros."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not prefix or any(char in prefix for char in "*?[]/\\"):
        raise ValueError("Signal prefix must be a simple, nonempty filename prefix")
    wanted = set(ids) if ids is not None else None
    if wanted is not None and any(re.fullmatch(r"[0-9]+", i) is None for i in wanted):
        raise ValueError("--ids must contain only digits (leading zeros permitted)")
    matches: dict[str, Path] = {}
    pattern = re.compile(r"^" + re.escape(prefix) + r"([0-9]+)\.csv$", re.I)
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        suffix = match.group(1)
        if wanted is not None and suffix not in wanted:
            continue
        if suffix in matches:
            raise ValueError(f"Duplicate acceleration CSV ID {suffix!r} in {input_dir}")
        matches[suffix] = path
    if not matches:
        raise FileNotFoundError(f"No {prefix}<digits>.csv files found in {input_dir}")
    if wanted is not None and wanted - matches.keys():
        raise FileNotFoundError("Requested IDs not found: " + ", ".join(sorted(wanted - matches.keys())))
    return OrderedDict(sorted(matches.items(), key=lambda item: (int(item[0]), item[0])))


def _configuration(args: argparse.Namespace):
    if (args.rest_start is None) != (args.rest_end is None):
        raise ValueError("--rest-start and --rest-end must be specified together")
    if args.mode == "validation" and (args.rest_start is not None or args.input_gravity_compensated):
        raise ValueError("Validation uses raw full records; do not pass gravity compensation options")
    if args.input_gravity_compensated and args.rest_start is not None:
        raise ValueError("Choose precompensated input OR a confirmed rest interval, not both")
    overrides: dict[str, Any] = {"processing_mode": args.mode}
    if args.input_gravity_compensated:
        overrides["input.gravity_compensated"] = True
        overrides["compensation.method"] = "none"
    if args.rest_start is not None:
        overrides.update({
            "compensation.method": "confirmed_rest",
            "compensation.magnitude_mode": "dynamic_magnitude",
            "compensation.rest_start_s": args.rest_start,
            "compensation.rest_end_s": args.rest_end,
        })
    if args.tail_mode is not None:
        overrides["tail_mode"] = args.tail_mode
    if args.tail_time_constants is not None:
        overrides["tail_time_constants"] = args.tail_time_constants
    return load_config(args.profile, config_file=args.config_file, cli_overrides=overrides)


def _relative_comparison(pc, check, output_path: Path) -> dict[str, Any]:
    """Numerical comparison only if the two grids match within CSV rounding."""
    f = np.asarray(pc.frequencies_hz)
    cf = np.asarray(check.frequencies_hz)
    if f.shape != cf.shape or np.max(np.abs(f - cf)) > 0.001:
        return {"status": "grid_mismatch", "message": "Original grids plotted; no numerical comparison was attempted"}
    columns = [f]
    headers = ["f_hz"]
    metrics: dict[str, Any] = {"status": "compared", "max_grid_difference_hz": float(np.max(np.abs(f-cf)))}
    for axis, attr in (("x", "gx_g"), ("y", "gy_g"), ("z", "gz_g")):
        p = getattr(pc, attr)
        c = getattr(check, attr)
        # Relative percent errors only defined where reference value != 0.
        nonzero = c != 0
        rel = np.full(c.shape, np.nan)
        rel[nonzero] = 100.0 * np.abs(p[nonzero] - c[nonzero]) / np.abs(c[nonzero])
        metrics[axis] = {
            "mean_relative_error_pct": float(np.mean(rel[nonzero])) if np.any(nonzero) else None,
            "max_relative_error_pct": float(np.max(rel[nonzero])) if np.any(nonzero) else None,
            "max_absolute_error_g": float(np.max(np.abs(p-c))),
            "reference_zero_points": int(np.count_nonzero(~nonzero)),
        }
        columns.extend((p, c, rel))
        headers.extend((f"pc_{axis}_g", f"check_{axis}_g", f"relative_error_{axis}_pct"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.column_stack(columns), delimiter=",",
               header=",".join(headers), comments="", fmt="%.12g")
    return metrics


def _print_comparison(comparison: dict[str, Any], label: str) -> None:
    """Print PC/check deviations; a zero reference has no relative error."""
    print(f"{label}: PC/check SRS deviation")
    if comparison.get("status") != "compared":
        print("  Numerical comparison unavailable: frequency grids differ")
        return
    print("  Axis     Max absolute error (g)     Max relative error (%)")
    for axis in ("x", "y", "z"):
        metrics = comparison[axis]
        relative = metrics["max_relative_error_pct"]
        relative_text = "N/A (zero reference)" if relative is None else f"{relative:.8f}"
        print(f"  {axis.upper():<8} {metrics['max_absolute_error_g']:<26.9f} {relative_text}")
        if metrics["reference_zero_points"]:
            print(f"    Relative error excludes {metrics['reference_zero_points']} zero reference points")


def _series_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [row["comparison"] for row in records
                   if row.get("comparison", {}).get("status") == "compared"]
    if not comparisons:
        return {"status": "not_compared", "compared_records": 0}
    result: dict[str, Any] = {"status": "compared", "compared_records": len(comparisons)}
    for axis in ("x", "y", "z"):
        relative = [item[axis]["max_relative_error_pct"] for item in comparisons
                    if item[axis]["max_relative_error_pct"] is not None]
        result[axis] = {
            "max_absolute_error_g": max(item[axis]["max_absolute_error_g"] for item in comparisons),
            "max_relative_error_pct": max(relative) if relative else None,
            "reference_zero_points": sum(item[axis]["reference_zero_points"] for item in comparisons),
        }
    return result


def run_series(args: argparse.Namespace) -> dict[str, Any]:
    config = _configuration(args)
    signals = discover_signals(args.input_dir, args.signal_prefix, args.ids)
    if not args.check_prefix or any(c in args.check_prefix for c in "*?[]/\\"):
        raise ValueError("Check prefix must be a simple nonempty filename prefix")
    # Resolve matching checks before computing, so strict mode fails early.
    checks: dict[str, Path] = {}
    for shock_id in signals:
        candidate = args.input_dir / f"{args.check_prefix}{shock_id}.csv"
        if candidate.is_file():
            checks[shock_id] = candidate
        elif args.require_check:
            raise FileNotFoundError(f"Missing matching reference SRS: {candidate}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pc_spectra = OrderedDict()
    check_spectra = OrderedDict()
    records: list[dict[str, Any]] = []
    for shock_id, input_path in signals.items():
        dest = output / f"shock_{shock_id}"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{shock_id}] Processing {input_path.name}", flush=True)
        result = run_srs_pipeline(input_path, config)
        pc_path = dest / "srs_result.csv"
        write_outputs(result, input_path, pc_path, dest / "event_metadata.json")
        plot_acceleration_with_frame(result, dest / "acceleration_plot.png")
        row: dict[str, Any] = {
            "id": shock_id, "input_file": str(input_path),
            "mode": result.mode, "status": result.status,
            "samples": result.source_data.sample_count,
            "srs_input_samples": (result.source_data.sample_count if result.mode == "validation" else
                                  result.frame_data.sample_count if result.frame_data is not None else 0),
            "warnings": list(result.warnings),
            "reference_file": str(checks[shock_id]) if shock_id in checks else None,
        }
        if result.srs_result is None:
            print(f"[{shock_id}] No SRS; excluded from series envelope", flush=True)
            records.append(row)
            continue
        spectrum = result.srs_result
        pc_spectra[shock_id] = spectrum
        plot_single_srs(spectrum, dest / "srs_calculated_all_curves.png",
                        layout="combined", title=f"Calculated SRS shock {shock_id}")
        plot_single_srs(spectrum, dest / "srs_calculated_stacked.png",
                        layout="stacked", title=f"Calculated SRS shock {shock_id}")
        # Output matching the onboard/jury CSV header, in addition to main's full CSV.
        np.savetxt(dest / "srs_jury.csv", spectrum.as_table()[:, :4],
                   delimiter=",", header="f,gx,gy,gz", comments="", fmt="%.12g")
        if shock_id in checks:
            check = read_srs_csv(checks[shock_id])
            check_spectra[shock_id] = check
            for layout in ("combined", "stacked"):
                plot_srs_comparison(pc_path, checks[shock_id],
                                    dest / f"srs_comparison_{layout}.png", layout=layout,
                                    show_limit=True)
            row["comparison"] = _relative_comparison(spectrum, check, dest / "comparison_errors.csv")
            _print_comparison(row["comparison"], f"[{shock_id}]")
        records.append(row)
    if not pc_spectra:
        raise ValueError("No SRS could be calculated for any input; envelope cannot be built")
    envelope = calculate_envelope(list(pc_spectra.values()))
    save_envelope_csv(envelope, output / "srs_envelope.csv")
    plot_envelope(list(pc_spectra.values()), envelope,
                  [f"shock_{name}" for name in pc_spectra],
                  output / "srs_envelope_individual.png")
    plot_series_srs(pc_spectra, envelope, output / "series_all_curves.png",
                    references=check_spectra, layout="combined")
    plot_series_srs(pc_spectra, envelope, output / "series_stacked.png",
                    references=check_spectra, layout="stacked")
    for axis in ("X", "Y", "Z"):
        plot_series_srs(pc_spectra, envelope, output / f"series_axis_{axis.lower()}.png",
                        references=check_spectra, axis=axis)
    plot_envelope_with_limit(envelope, output / "series_envelope_vs_limit.png")
    plot_envelope_with_limit(envelope, output / "series_envelope_stacked.png", layout="stacked")
    plot_srs_limit(output / "srs_limit.png")
    threshold = write_threshold_reports(envelope, output, source_file=output / "srs_envelope.csv")
    comparison_summary = _series_comparison(records)
    summary: dict[str, Any] = {
        "mode": args.mode, "profile": args.profile, "input_dir": str(args.input_dir),
        "output_dir": str(output), "requested_files": len(signals),
        "calculated_spectra": len(pc_spectra), "reference_spectra": len(check_spectra),
        "frequency_points": int(envelope.frequencies_hz.size),
        "included_ids": list(pc_spectra.keys()),
        "threshold": threshold, "shocks": records,
        "comparison_summary": comparison_summary,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed: {len(pc_spectra)}/{len(signals)} SRS; {len(check_spectra)} check spectra")
    if comparison_summary["status"] == "compared":
        _print_comparison(comparison_summary, "SERIES MAXIMUM")
    else:
        print("PC/check deviations unavailable: no successfully compared reference spectra")
    print(f"Envelope CSV: {output / 'srs_envelope.csv'}")
    print(f"Critical SRS threshold: {threshold['status']} "
          f"({threshold['frequencies_with_any_exceedance']} frequencies exceeded)")
    for axis_name, axis_info in threshold["axes"].items():
        print(f"  {axis_name.upper()}: {axis_info['exceedance_count']} frequency points exceeded; "
              f"max excess {axis_info['max_excess_g']:.6g} g at "
              f"{axis_info['frequency_at_max_excess_hz']} Hz")
    if threshold["frequency_points_outside_limit"]:
        print(f"WARNING: {threshold['frequency_points_outside_limit']} frequencies are "
              "outside the 10..2000 Hz specified limit (not evaluated)")
    print(f"All plots and reports: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_series(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
