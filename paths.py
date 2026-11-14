"""Default project paths, independent of the current working directory."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_DIR / "input" / "raw_signal_0051.csv"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "examples" / "contest.json"
DEFAULT_SRS_PATH = PROJECT_DIR / "output" / "srs_result.csv"
DEFAULT_METADATA_PATH = PROJECT_DIR / "output" / "event_metadata.json"
DEFAULT_ACCELERATION_PLOT_PATH = PROJECT_DIR / "output" / "acceleration_plot.png"
DEFAULT_ENVELOPE_PATH = PROJECT_DIR / "output" / "srs_envelope.csv"
DEFAULT_ENVELOPE_PLOT_PATH = PROJECT_DIR / "output" / "srs_envelope.png"
