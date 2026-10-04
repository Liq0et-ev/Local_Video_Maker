"""End-to-end pipeline: long video -> finished YouTube Shorts.

    analyse / plan  ->  cut + reframe to 9:16  ->  subtitles  ->  background music

Mode "highlight" (Dinamic_Control_YouTube_Videos) builds ONE short from the most
dynamic moments. Mode "slice" (One_Minute_Videos) turns the whole video into
sequential random-length shorts. Subtitles come from Semi_Final_Video_Redactor /
Youtube_AutoTitles_Creator, music from Background_Audio.
"""
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audio_mix
from .config import PipelineOptions
from .ffmpeg_utils import concat_clips, crossfade_clips, cut_clip, extract_wav, probe, require_ffmpeg
from .slicer import compute_cut_points
from .snap import loudness_envelope, snap_cut_points, snap_intervals
from .transition_catalog import CATALOG
from .transitions import TransitionPlan, plan_transitions

ProgressCallback = Callable[[float, str], None]


class Cancelled(Exception):
    """Raised between stages when the user cancels a job."""


@dataclass
class PipelineResult:
    outputs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diagnostics_png: Path | None = None
    transitions: list[str] = field(default_factory=list)   # transitions used between highlight segments


def _clean_name(path: Path) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)
    return keep.strip("_")[:50] or "video"


def run_pipeline(
    source: Path,
    options: PipelineOptions,
    output_dir: Path,
    work_dir: Path,
    music_dir: Path,
    whisper_model: str = "medium",
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    confirm_transitions: Callable[[TransitionPlan], TransitionPlan] | None = None,
) -> PipelineResult:
    options.validate()
    require_ffmpeg()
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    report = progress or (lambda pct, msg: None)

    def checkpoint() -> None:
        if should_cancel and should_cancel():
            raise Cancelled()

    info = probe(source)
    if not info["has_video"] or info["duration"] < 1.0:
        raise ValueError("The file has no usable video stream.")

    name = _clean_name(source)
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=work_dir))
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()
    enc = options.encode_settings()
    env_cache: dict = {}

    def envelope(force: bool = False):
        """Loudness envelope (snapping cuts to pauses, judging transitions); computed once, lazily."""
        if options.snap_sec <= 0 and not force:
            return None
        if "env" not in env_cache:
            report(1, "Finding quiet moments for clean cuts...")
            env_cache["env"] = loudness_envelope(source)
        return env_cache["env"]

    try:
        # ---- 1. Plan + cut + reframe --------------------------------------
        base_clips: list[Path] = []
        if options.mode == "highlight":
            report(2, "Analysing motion and audio dynamics...")
            from .highlights.finder import find_highlights

            hl = find_highlights(
                str(source), variant=options.variant, target_sec=options.target_sec,
                buffer_sec=options.buffer_sec, merge_gap_sec=options.merge_gap_sec,
                threshold_k=options.threshold_k, sample_fps=options.sample_fps,
                sg_window=options.smoothing_window, diagnostics_dir=str(out_dir),
                weights={"motion": options.weight_motion, "flow": options.weight_flow,
                         "rms": options.weight_loudness, "flux": options.weight_audio_change},
                analysis_window_sec=options.analysis_window_sec, min_activity_sec=options.min_activity_sec,
                median_sec=options.median_sec, min_segment_sec=options.min_segment_sec,
                max_segments=options.max_segments, selection=options.selection,
                fill_to_target=options.fill_to_target,
            )
            checkpoint()
            if hl.diagnostics_png:
                result.diagnostics_png = Path(hl.diagnostics_png)
            intervals = hl.intervals
            if not intervals:
                raise ValueError("No dynamic moments found. Try a lower sensitivity threshold.")
            if envelope() is not None:
                intervals = snap_intervals(intervals, envelope(), options.snap_sec, info["duration"],
                                           min_len=max(1.0, options.min_segment_sec / 2))
            total_len = sum(e - s for s, e in intervals)
            result.notes.append(
                f"Highlight: {len(intervals)} segment(s), {total_len:.0f}s from {hl.source_duration:.0f}s"
            )
            if total_len < options.target_sec * 0.5:
                result.notes.append(
                    f"Highlight is much shorter than the {options.target_sec:.0f}s target; "
                    "lower the sensitivity threshold or use Slice mode for more material."
                )

            plan = None
            if len(intervals) > 1 and options.transition != "cut":
                report(14, "Choosing transitions between segments...")
                plan = plan_transitions(source, intervals, options, envelope(force=True))
                if options.transition == "ask" and confirm_transitions is not None:
                    plan = confirm_transitions(plan)  # the user may accept or edit the proposal
                    checkpoint()
                result.transitions = [f"{CATALOG[c.key].ru} ({c.seconds:.2f}s)" for c in plan.choices]

            parts = []
            for i, (s, e) in enumerate(intervals):
                report(15 + 20 * i / len(intervals), f"Cutting segment {i + 1}/{len(intervals)}...")
                part = tmp / f"seg_{i:03d}.mp4"
                cut_clip(source, s, e, part, options.vertical, enc)
                parts.append(part)
                checkpoint()
            base = tmp / "highlight.mp4"
            if len(parts) == 1:
                shutil.move(parts[0], base)
            else:
                report(36, f"Joining {len(parts)} segments...")
                joined = plan is not None and crossfade_clips(parts, base, plan.joins(), enc)
                if not joined:
                    if plan is not None:
                        result.notes.append("Segments too short for transitions; joined with hard cuts.")
                        result.transitions = []
                    concat_clips(parts, base, tmp / "concat.txt")
            base_clips.append(base)
        else:
            cuts = compute_cut_points(info["duration"], options.min_clip_sec, options.max_clip_sec)
            if len(cuts) > options.max_clips:
                result.notes.append(f"Source gives {len(cuts)} clips; processing the first {options.max_clips}.")
                cuts = cuts[: options.max_clips]
            if len(cuts) > 1 and envelope() is not None:
                cuts = snap_cut_points(cuts, envelope(), options.snap_sec, min_len=options.min_clip_sec)
            for i, (s, e) in enumerate(cuts, 1):
                report(5 + 25 * i / len(cuts), f"Cutting clip {i}/{len(cuts)}...")
                part = tmp / f"part_{i:03d}.mp4"
                cut_clip(source, s, e, part, options.vertical, enc)
                base_clips.append(part)
                checkpoint()

        # ---- 2/3. Subtitles + music per clip -------------------------------
        total = len(base_clips)
        for idx, clip in enumerate(base_clips, 1):
            lo = 35 + 60 * (idx - 1) / total
            span = 60 / total
            stem = clip.stem
            current = clip

            if options.subtitles and not probe(current)["has_audio"]:
                result.notes.append(f"Clip {idx}: no audio track, subtitles skipped.")
            elif options.subtitles:
                report(lo, f"[{idx}/{total}] Transcribing speech...")
                from .subtitles.burn import burn_subtitles, resolve_style
                from .subtitles.transcriber import get_transcriber

                wav = tmp / f"{stem}.wav"
                extract_wav(current, wav)
                segments = get_transcriber(whisper_model).transcribe(str(wav), language=options.language)
                checkpoint()
                if any(s["words"] for s in segments):
                    report(lo + span * 0.4, f"[{idx}/{total}] Rendering subtitles...")
                    subbed = tmp / f"{stem}_sub.mp4"
                    burn_subtitles(current, subbed, segments, resolve_style(options.brand_kit), enc)
                    current = subbed
                else:
                    result.notes.append(f"Clip {idx}: no speech detected, subtitles skipped.")
                checkpoint()

            if options.music:
                report(lo + span * 0.85, f"[{idx}/{total}] Mixing background music...")
                mixed = tmp / f"{stem}_music.mp4"
                if audio_mix.add_background_music(current, mixed, music_dir, options.music_volume, enc):
                    current = mixed
                else:
                    result.notes.append("No music tracks found in the music folder; music skipped.")
                checkpoint()

            final = out_dir / (f"{name}_short.mp4" if options.mode == "highlight" else f"{name}_short_{idx:02d}.mp4")
            shutil.move(current, final)
            result.outputs.append(final)

        report(100, "Done")
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
