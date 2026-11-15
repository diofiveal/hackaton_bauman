# Batch SRS pipeline (refactored project)

This version uses `cli.py` and `pipeline.py` for a single record and the
`plots/` package for all visualizations. The new **`series_runner.py`**
orchestrates the same calculation for many shocks. It does **not** implement
BIN conversion: convert your STM32 BIN files into CSV using your colleague's
converter before running the series.

## One-command Windows CMD launch

Open **Command Prompt** in this project folder:

```bat
run_series.bat
```

Process only selected files (IDs preserve leading zeroes):

```bat
run_series.bat --ids 1 2 3
run_series.bat --ids 0051 0052
```

Choose paths (quotes are required when Windows paths contain spaces):

```bat
run_series.bat --input-dir "C:\Data\Shock Records" --output-dir "C:\Data\SRS Results"
```

In Git Bash/Linux/macOS:

```bash
bash run_series.sh --ids 1 2 3
```

Install dependencies once:

```bat
python -m pip install -r requirements.txt
```

**CMD runs `.bat`, not a Bash `.sh` directly.** The `.sh` is provided for Bash.

## Input naming

Put the converted CSVs in `input/` (or pass `--input-dir`):

```text
input/
    raw_signal_1.csv       # columns t;ax;ay;az or t,ax,ay,az
    raw_signal_2.csv
    raw_signal_0051.csv
    srs_check_1.csv        # optional check: f;gx;gy;gz or f,gx,gy,gz
    srs_check_2.csv
    srs_check_0051.csv
```

The numerical suffix must match *exactly*, including leading zeros.
`raw_signal_51.csv` is **not** matched to `srs_check_0051.csv`.
When a check file is absent, the runner still computes that PC spectrum and
includes it in the envelope; only the pairwise comparison is omitted. Add
`--require-check` to require every pair.

Custom filename prefixes are supported through `--signal-prefix` and
`--check-prefix`.

## Modes

By default the runner uses `--mode validation`: the entire existing CSV record
is used to calculate SRS without estimating/removing gravity or detecting an
event. This reproduces the check CSVs in the provided sample dataset. **It does
not prove correct contest event detection.**

`--mode contest` selects the event frame and requires gravity-free inputs.
For already gravity-compensated CSVs (this must be true, not merely presumed):

```bat
run_series.bat --mode contest --input-gravity-compensated
```

Alternatively, pass `--rest-start` and `--rest-end` in seconds for a genuinely
confirmed rest interval that is present in **every** processed record. The
pipeline validates rest quality and rejects inadequate data. If the intervals
vary per record, run them separately or prepare calibrated input CSVs first.

Optional common SRS parameters: `--tail-mode`, `--tail-time-constants`,
`--config-file`, `--profile`.

## Outputs

All outputs go to `output/series/` unless `--output-dir` overrides it:

```text
output/series/
    srs_envelope.csv             # f,ax,ay,az (pointwise PC maximum, all shocks)
    series_all_curves.png        # PC + reference shocks + XYZ envelope + limit
    series_stacked.png           # X/Y/Z in separate panels + limit
    series_axis_x.png           # all X shock curves, envelope, limit
    series_axis_y.png
    series_axis_z.png
    series_envelope_vs_limit.png
    series_envelope_stacked.png
    srs_envelope_individual.png # old plot format, input spectra + envelope
    srs_limit.png
    threshold_report.csv
    threshold_report.json
    run_summary.json
    shock_1/
        acceleration_plot.png
        srs_result.csv          # original f_hz,gx_g,gy_g,gz_g,g_total_g
        srs_jury.csv            # additional f,gx,gy,gz
        event_metadata.json
        srs_calculated_all_curves.png
        srs_calculated_stacked.png
        srs_comparison_combined.png  # when matching check exists
        srs_comparison_stacked.png   # when matching check exists
        comparison_errors.csv        # only when frequency grid is compatible
    shock_2/
    ...
```

The envelope is computed only from **PC SRS** spectra, not check spectra. It
requires a common calculation frequency grid (no interpolation or silent
frequency rounding). The threshold curve is defined from 10 to 2000 Hz only;
frequency points above 2000 Hz are excluded from threshold decisions.

`threshold_report.csv` provides the envelope, limit, ratios and exceedances
by frequency and axis. `threshold_report.json` provides totals. At exact
threshold equality, no exceedance is reported; strictly greater means exceed.

## Verification

```bat
python -m unittest discover -s tests -v
```

This pipeline intentionally leaves existing `main.py`, `cli.py`, `pipeline.py`,
`smallwood.py`, `envelope.py` and existing plots unchanged. The new batch
runner builds on their public functions.
