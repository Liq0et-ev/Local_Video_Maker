"""Highlight finder: the analysis half of Dinamic_Control_YouTube_Videos / extract_highlights.py.

Returns the (start, end) intervals of the most dynamic moments; cutting and
export are handled by the shared ffmpeg layer so reframing to 9:16 happens in one pass.
"""
import os
from dataclasses import dataclass

import numpy as np

from .audio_features import extract_audio_features
from .detectors import detect_multimodal_attention, detect_surprisal, detect_weighted_heuristic
from .dsp_pipeline import (
    adaptive_threshold_fast,
    minmax_norm,
    resample_to_common_axis,
    smooth_sg,
)
from .temporal import build_extraction_map
from .video_features import extract_video_features

VARIANTS = {
    "heuristic": ("Weighted Heuristic (A)", detect_weighted_heuristic),
    "surprisal": ("Information-Theoretic Surprisal (B)", detect_surprisal),
    "attention": ("Multimodal Attention (C)", detect_multimodal_attention),
}


@dataclass
class HighlightResult:
    intervals: list[tuple[float, float]]
    clip_duration: float
    source_duration: float
    diagnostics_png: str | None = None


def find_highlights(
    video_path: str,
    variant: str = "heuristic",
    target_sec: float = 60.0,
    buffer_sec: float = 5.0,
    merge_gap_sec: float = 3.0,
    threshold_k: float = 1.5,
    sample_fps: float = 2.0,
    sg_window: int = 15,
    diagnostics_dir: str | None = None,
) -> HighlightResult:
    vid_ts, motion_raw, flow_raw, _fps, total_dur = extract_video_features(video_path, sample_fps=sample_fps)
    aud_ts, rms_raw, flux_raw = extract_audio_features(video_path)

    dt = 1.0 / sample_fps
    common_ts = np.arange(0, total_dur, dt)

    def prep(ts, sig):
        return smooth_sg(minmax_norm(resample_to_common_axis(ts, sig, common_ts)), window=sg_window)

    motion, flow = prep(vid_ts, motion_raw), prep(vid_ts, flow_raw)
    rms, flux = prep(aud_ts, rms_raw), prep(aud_ts, flux_raw)

    variant_name, detect = VARIANTS[variant]
    mask = detect(motion, flow, rms, flux, dt=dt, threshold_k=threshold_k)

    intervals, clip_dur, raw_intervals = build_extraction_map(
        mask, common_ts, total_dur,
        buffer_sec=buffer_sec, merge_gap_sec=merge_gap_sec, target_sec=target_sec,
    )

    png = None
    if diagnostics_dir:
        from .diagnostics import plot_diagnostics  # matplotlib is only needed for the plot

        composite = 0.30 * motion + 0.30 * flow + 0.20 * rms + 0.20 * flux
        threshold = adaptive_threshold_fast(composite, window_sec=30.0, dt=dt, k=threshold_k)
        os.makedirs(diagnostics_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        png = os.path.join(diagnostics_dir, f"{base}_diagnostics.png")
        plot_diagnostics(
            timestamps=common_ts, motion=motion, flow=flow, rms=rms, flux=flux,
            composite=composite, threshold=threshold, mask=mask,
            intervals=intervals, raw_intervals=raw_intervals,
            video_duration=total_dur, output_path=png, variant_name=variant_name,
        )

    return HighlightResult(intervals=intervals, clip_duration=clip_dur,
                           source_duration=total_dur, diagnostics_png=png)
