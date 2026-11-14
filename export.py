"""Serialize pipeline metadata and the calculated SRS CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import AccelerationData, EventFrame, SRSResult, StaticCompensation
from pipeline import PipelineResult
from reporting import spectrum_peak
from srs_io import save_srs_csv


def build_metadata(result: PipelineResult, input_path: Path) -> dict[str, Any]:
    if result.mode == "validation":
        input_count = result.source_data.sample_count
        gravity_handling = "raw_full_record"
    else:
        input_count = result.frame_data.sample_count if result.frame_data else 0
        gravity_handling = (
            "checked_rest_vector"
            if result.compensation
            else "declared_precompensated_input"
        )
    document: dict[str, Any] = {
        "input_file": str(input_path),
        "status": result.status,
        "processing_mode": result.mode,
        "srs_input_sample_count": input_count,
        "compensation_applied": (
            result.mode == "contest" and result.compensation is not None
        ),
        "gravity_handling": gravity_handling,
        "sample_count": result.source_data.sample_count,
        "sample_rate_hz": float(result.source_data.sample_rate_hz),
        "warnings": list(result.warnings),
    }

    if result.compensation is not None:
        document["compensation"] = _compensation_metadata(result.compensation)
    if result.event_frame is not None and result.frame_data is not None:
        document["event"] = _event_metadata(result.event_frame, result.frame_data)
    if result.srs_result is not None:
        document["srs"] = _srs_metadata(result.srs_result)
    return document


def write_outputs(
    result: PipelineResult,
    input_path: Path,
    srs_path: Path,
    metadata_path: Path,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(build_metadata(result, input_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if result.srs_result is not None:
        save_srs_csv(result.srs_result, srs_path)


def _compensation_metadata(compensation: StaticCompensation) -> dict[str, Any]:
    return {
        "ax_g": compensation.ax_g,
        "ay_g": compensation.ay_g,
        "az_g": compensation.az_g,
        "rest_start_s": compensation.rest_start_s,
        "rest_end_s": compensation.rest_end_s,
        "sample_count": compensation.sample_count,
        "maximum_rest_std_g": compensation.maximum_rest_std_g,
        "max_rest_peak_to_peak_g": compensation.max_rest_peak_to_peak_g,
        "gravity_vector_norm_g": compensation.gravity_vector_norm_g,
    }


def _event_metadata(frame: EventFrame, data: AccelerationData) -> dict[str, Any]:
    return {
        "start_index": frame.start_index,
        "end_index": frame.end_index,
        "start_time_s": float(data.time_s[0]),
        "end_time_s": float(data.time_s[-1]),
        "sample_count": frame.sample_count,
        "amax_g": frame.amax_g,
    }


def _srs_metadata(spectrum: SRSResult) -> dict[str, Any]:
    if spectrum.info is None:
        raise ValueError("Calculated SRS must include its calculation settings")
    return {
        "frequency_count": int(spectrum.frequencies_hz.size),
        "quality_factor_q": spectrum.info.quality_factor_q,
        "tail_mode": spectrum.info.tail_mode,
        "tail_time_constants": spectrum.info.tail_time_constants,
        "peaks": {
            "gx": spectrum_peak(spectrum.frequencies_hz, spectrum.gx_g),
            "gy": spectrum_peak(spectrum.frequencies_hz, spectrum.gy_g),
            "gz": spectrum_peak(spectrum.frequencies_hz, spectrum.gz_g),
            "g_total": spectrum_peak(
                spectrum.frequencies_hz,
                spectrum.g_total_g,
            ),
        },
    }
