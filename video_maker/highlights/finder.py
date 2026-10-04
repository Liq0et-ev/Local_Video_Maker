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

DEFAULT_WEIGHTS = {"motion": 0.30, "flow": 0.30, "rms": 0.20, "flux": 0.20}


@dataclass
class HighlightResult:
    intervals: list[tuple[float, float]]
    clip_duration: float
    source_duration: float
    diagnostics_png: str | None = None
    raw_segments: int = 0


def normalise_weights(weights: dict | None) -> dict:
    """Weights are relative: scale them to sum to 1 (all-zero input falls back to the defaults)."""
    w = {k: max(0.0, float((weights or DEFAULT_WEIGHTS).get(k, 0.0))) for k in DEFAULT_WEIGHTS}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else dict(DEFAULT_WEIGHTS)


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
    weights: dict | None = None,
    analysis_window_sec: float = 30.0,
    min_activity_sec: float = 1.0,
    median_sec: float = 3.0,
    min_segment_sec: float = 0.0,
    max_segments: int = 0,
    selection: str = "proportional",
    fill_to_target: bool = False,
) -> HighlightResult:
    w = normalise_weights(weights)
    vid_ts, motion_raw, flow_raw, _fps, total_dur = extract_video_features(video_path, sample_fps=sample_fps)
    aud_ts, rms_raw, flux_raw = extract_audio_features(video_path)

    dt = 1.0 / sample_fps
    common_ts = np.arange(0, total_dur, dt)

    def prep(ts, sig):
        return smooth_sg(minmax_norm(resample_to_common_axis(ts, sig, common_ts)), window=sg_window)

    motion, flow = prep(vid_ts, motion_raw), prep(vid_ts, flow_raw)
    rms, flux = prep(aud_ts, rms_raw), prep(aud_ts, flux_raw)

    variant_name, detect = VARIANTS[variant]
    common = dict(dt=dt, threshold_k=threshold_k, window_sec=analysis_window_sec,
                  min_duration_sec=min_activity_sec)
    if variant == "heuristic":
        common["weights"] = w
    mask = detect(motion, flow, rms, flux, **common)

    # Weighted activity curve: ranks segments and feeds the diagnostics plot
    composite = w["motion"] * motion + w["flow"] * flow + w["rms"] * rms + w["flux"] * flux

    intervals, clip_dur, raw_intervals = build_extraction_map(
        mask, common_ts, total_dur,
        buffer_sec=buffer_sec, merge_gap_sec=merge_gap_sec, target_sec=target_sec,
        median_kernel_sec=median_sec, scores=composite, min_segment_sec=min_segment_sec,
        max_segments=max_segments, selection=selection, fill_to_target=fill_to_target,
    )

    png = None
    if diagnostics_dir:
        from .diagnostics import plot_diagnostics  # matplotlib is only needed for the plot

        threshold = adaptive_threshold_fast(composite, window_sec=analysis_window_sec, dt=dt, k=threshold_k)
        os.makedirs(diagnostics_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        png = os.path.join(diagnostics_dir, f"{base}_diagnostics.png")
        plot_diagnostics(
            timestamps=common_ts, motion=motion, flow=flow, rms=rms, flux=flux,
            composite=composite, threshold=threshold, mask=mask,
            intervals=intervals, raw_intervals=raw_intervals,
            video_duration=total_dur, output_path=png, variant_name=variant_name,
        )

    return HighlightResult(intervals=intervals, clip_duration=clip_dur, source_duration=total_dur,
                           diagnostics_png=png, raw_segments=len(raw_intervals))
