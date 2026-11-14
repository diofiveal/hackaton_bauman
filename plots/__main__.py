"""Command: python -m plots --srs-file ... --check-file ..."""

from __future__ import annotations

from pathlib import Path

from paths import PROJECT_DIR
from plots.comparison import plot_srs_comparison
from plots.limits import plot_srs_limit


def main() -> int:
    """Allow plotting two existing SRS files without running main.py."""
    import argparse
    import sys

    root = PROJECT_DIR
    parser = argparse.ArgumentParser(
        description="Overlay PC and check SRS spectra for X, Y and Z."
    )
    parser.add_argument(
        "--srs-file",
        type=Path,
        default=root / "output" / "srs_result.csv",
        help="PC SRS CSV (f_hz,gx_g,gy_g,gz_g,...)",
    )
    parser.add_argument(
        "--check-file",
        type=Path,
        default=root / "input" / "srs_check_0051.csv",
        help="Check/reference SRS CSV (f,gx,gy,gz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "srs_comparison.png",
        help="Output PNG with both spectra",
    )
    parser.add_argument(
        "--layout",
        choices=("combined", "stacked"),
        default="combined",
        help="combined: all six curves on one chart; stacked: one panel per axis",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Hide the allowable SRS threshold on the comparison plot",
    )
    parser.add_argument(
        "--limit-only",
        action="store_true",
        help="Plot only the allowable SRS threshold, without reading any CSV",
    )
    args = parser.parse_args()
    try:
        if args.limit_only:
            result_path = plot_srs_limit(args.output)
        else:
            result_path = plot_srs_comparison(
                args.srs_file,
                args.check_file,
                args.output,
                layout=args.layout,
                show_limit=not args.no_limit,
            )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"SRS plot: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
