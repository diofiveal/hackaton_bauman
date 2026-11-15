"""Numerical acceptance check of an SRS envelope against the contest As curve.

The limit is defined ONLY on 10..2000 Hz. Each point and each of the X/Y/Z
axes is checked independently; values above the limit are violations. Points
outside the defined frequency interval are marked NOT_EVALUATED, not PASS.

Run directly on an already calculated envelope:
    python threshold_check.py output/series/srs_envelope.csv
    python threshold_check.py output/series/srs_envelope.csv --output-dir output/series

Called automatically by series_runner.py after the series envelope is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from models import SRSResult
from plots.limits import srs_limit_g
from srs_io import read_srs_csv


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Per-point numerical results and a JSON-serializable summary."""

    frequencies_hz: np.ndarray
    limit_g: np.ndarray
    evaluated: np.ndarray
    values_g: dict[str, np.ndarray]
    margin_g: dict[str, np.ndarray]
    excess_g: dict[str, np.ndarray]
    ratio: dict[str, np.ndarray]
    excess_percent: dict[str, np.ndarray]
    exceeds: dict[str, np.ndarray]
    summary: dict[str, Any]


def evaluate_srs_threshold(envelope: SRSResult) -> ThresholdEvaluation:
    """Evaluate As(f), exactly at each original frequency (no interpolation).

    margin = As - SRS (negative means exceedance)
    excess = max(SRS - As, 0) in g
    excess_percent = 100 * excess / As
    ratio = SRS / As

    A spectrum equal to the limit is acceptable; strictly above is a violation.
    Out-of-range points have NaN metrics and are excluded from decisions.
    """
    f = np.asarray(envelope.frequencies_hz, dtype=np.float64)
    if f.ndim != 1 or not f.size or not np.all(np.isfinite(f)):
        raise ValueError("SRS frequency grid must be a nonempty finite 1-D array")
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("SRS frequencies must be positive and strictly increasing")

    limit = srs_limit_g(f)
    evaluated = np.isfinite(limit)
    if not np.any(evaluated):
        raise ValueError("The spectrum contains no frequencies within the specified 10..2000 Hz limit range")

    values: dict[str, np.ndarray] = {}
    margins: dict[str, np.ndarray] = {}
    excesses: dict[str, np.ndarray] = {}
    ratios: dict[str, np.ndarray] = {}
    percentages: dict[str, np.ndarray] = {}
    exceeded: dict[str, np.ndarray] = {}
    axes: dict[str, Any] = {}

    for axis, attr in (("x", "gx_g"), ("y", "gy_g"), ("z", "gz_g")):
        arr = np.asarray(getattr(envelope, attr), dtype=np.float64)
        if arr.shape != f.shape or np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"Invalid {axis.upper()} SRS values")
        values[axis] = arr.copy()

        margin = np.full(f.shape, np.nan)
        excess = np.full(f.shape, np.nan)
        ratio = np.full(f.shape, np.nan)
        pct = np.full(f.shape, np.nan)
        margin[evaluated] = limit[evaluated] - arr[evaluated]
        excess[evaluated] = np.maximum(-margin[evaluated], 0.0)
        ratio[evaluated] = arr[evaluated] / limit[evaluated]
        pct[evaluated] = 100.0 * excess[evaluated] / limit[evaluated]
        flags = evaluated & (arr > limit)

        margins[axis] = margin
        excesses[axis] = excess
        ratios[axis] = ratio
        percentages[axis] = pct
        exceeded[axis] = flags

        valid_indices = np.flatnonzero(evaluated)
        peak_ratio_idx = int(valid_indices[np.argmax(ratio[evaluated])])
        violation_indices = np.flatnonzero(flags)
        worst_idx = (
            int(violation_indices[np.argmax(excess[violation_indices])])
            if violation_indices.size else None
        )
        worst_percent_idx = (
            int(violation_indices[np.argmax(pct[violation_indices])])
            if violation_indices.size else None
        )
        axes[axis] = {
            "exceeded": bool(violation_indices.size),
            "exceedance_count": int(violation_indices.size),
            "first_exceedance_frequency_hz": float(f[violation_indices[0]]) if violation_indices.size else None,
            "last_exceedance_frequency_hz": float(f[violation_indices[-1]]) if violation_indices.size else None,
            "max_ratio": float(ratio[peak_ratio_idx]),
            "frequency_at_max_ratio_hz": float(f[peak_ratio_idx]),
            "max_excess_g": float(excess[worst_idx]) if worst_idx is not None else 0.0,
            "max_excess_percent": float(pct[worst_percent_idx]) if worst_percent_idx is not None else 0.0,
            "frequency_at_max_excess_hz": float(f[worst_idx]) if worst_idx is not None else None,
            "frequency_at_max_excess_percent_hz": float(f[worst_percent_idx]) if worst_percent_idx is not None else None,
            "max_srs_within_limit_range_g": float(np.max(arr[evaluated])),
        }

    exceeds_any = exceeded["x"] | exceeded["y"] | exceeded["z"]
    any_exceedance = bool(np.any(exceeds_any))
    full_coverage = bool(np.all(evaluated))
    if any_exceedance:
        status = "EXCEEDED"
    elif full_coverage:
        status = "WITHIN_LIMIT"
    else:
        status = "WITHIN_LIMIT_IN_EVALUATED_RANGE"

    summary = {
        "status": status,
        "frequency_range_hz": [10.0, 2000.0],
        "limit_rule": {
            "10_to_100_hz": "10 + 10*log10(f/10) g",
            "100_to_1000_hz": "20 g",
            "1000_to_2000_hz": "20 + 30*(f-1000)/1000 g",
            "exceedance_condition": "SRS > As (equality is allowed)",
        },
        "frequency_points_total": int(f.size),
        "frequency_points_checked": int(np.count_nonzero(evaluated)),
        "frequency_points_outside_limit": int(np.count_nonzero(~evaluated)),
        "full_frequency_coverage": full_coverage,
        "outside_limit_frequencies_hz": [float(v) for v in f[~evaluated]],
        "frequencies_with_any_exceedance": int(np.count_nonzero(exceeds_any)),
        "any_exceedance": any_exceedance,
        "axes": axes,
        "out_of_range_policy": "NOT_EVALUATED: no defined threshold outside 10..2000 Hz",
    }
    return ThresholdEvaluation(
        frequencies_hz=f.copy(), limit_g=limit, evaluated=evaluated,
        values_g=values, margin_g=margins, excess_g=excesses,
        ratio=ratios, excess_percent=percentages,
        exceeds=exceeded, summary=summary,
    )


def write_threshold_reports(
    envelope: SRSResult,
    output_dir: str | Path,
    *,
    source_file: str | Path | None = None,
) -> dict[str, Any]:
    """Write detailed threshold_report.csv and threshold_report.json.

    Retains the prior CSV columns (f_hz, limit_g, envelope_*, ratio_*,
    exceeds_*, exceeds_any) while appending additional numerical diagnostics.
    """
    result = evaluate_srs_threshold(envelope)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    headings = [
        "f_hz", "limit_g", "envelope_x_g", "envelope_y_g", "envelope_z_g",
        "ratio_x", "ratio_y", "ratio_z",
        "exceeds_x", "exceeds_y", "exceeds_z", "exceeds_any",
        "margin_x_g", "margin_y_g", "margin_z_g",
        "excess_x_g", "excess_y_g", "excess_z_g",
        "excess_x_pct", "excess_y_pct", "excess_z_pct", "status",
    ]
    axes = ("x", "y", "z")

    def fmt(n: float) -> str:
        return f"{float(n):.17g}"

    with (dest / "threshold_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        for i, f in enumerate(result.frequencies_hz):
            inside = bool(result.evaluated[i])
            triggered = inside and any(result.exceeds[a][i] for a in axes)
            row: list[Any] = [fmt(f), fmt(result.limit_g[i]) if inside else ""]
            row.extend(fmt(result.values_g[a][i]) for a in axes)
            row.extend(fmt(result.ratio[a][i]) if inside else "" for a in axes)
            row.extend(int(result.exceeds[a][i]) if inside else "" for a in axes)
            row.append(int(triggered) if inside else "")
            row.extend(fmt(result.margin_g[a][i]) if inside else "" for a in axes)
            row.extend(fmt(result.excess_g[a][i]) if inside else "" for a in axes)
            row.extend(fmt(result.excess_percent[a][i]) if inside else "" for a in axes)
            row.append("NOT_EVALUATED" if not inside else ("EXCEEDED" if triggered else "OK"))
            writer.writerow(row)

    summary = dict(result.summary)
    if source_file is not None:
        summary["source_file"] = str(source_file)
    (dest / "threshold_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a ready SRS envelope against contest As(f).")
    parser.add_argument("envelope_csv", type=Path, help="CSV with f,ax,ay,az or compatible SRS columns")
    parser.add_argument("--output-dir", type=Path, help="Destination (default: envelope CSV directory)")
    parser.add_argument("--fail-on-exceed", action="store_true", help="Return exit code 1 if As is exceeded")
    args = parser.parse_args(argv)
    try:
        spectrum = read_srs_csv(args.envelope_csv)
        out = args.output_dir or args.envelope_csv.parent
        report = write_threshold_reports(spectrum, out, source_file=args.envelope_csv)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Critical SRS threshold check: {report['status']}")
    print(f"Frequency points evaluated: {report['frequency_points_checked']} / {report['frequency_points_total']}")
    for name, values in report["axes"].items():
        print(
            f"  {name.upper()}: exceeded at {values['exceedance_count']} frequencies; "
            f"max SRS/limit={values['max_ratio']:.4f}; "
            f"max excess={values['max_excess_g']:.4f} g "
            f"at {values['frequency_at_max_excess_hz']} Hz"
        )
    if report["frequency_points_outside_limit"]:
        print("WARNING: Frequencies outside 10..2000 Hz were not evaluated")
    print(f"CSV report: {out / 'threshold_report.csv'}")
    print(f"JSON report: {out / 'threshold_report.json'}")
    if args.fail_on_exceed and report["any_exceedance"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
