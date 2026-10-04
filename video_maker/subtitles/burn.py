"""Burn word-highlighted subtitles into a video (ported from Semi_Final_Video_Redactor/subtitles/processor.py)."""
from pathlib import Path

import numpy as np
from PIL import Image

try:  # moviepy 1.x
    from moviepy.editor import VideoFileClip
    MOVIEPY_V2 = False
except ModuleNotFoundError:  # moviepy 2.x
    from moviepy import VideoFileClip
    MOVIEPY_V2 = True

from .brand_kits import PRESETS
from .renderer import SubtitleRenderer

DEFAULT_STYLE = PRESETS["viral_yellow"]


def resolve_style(brand_kit: str | None) -> dict:
    return dict(PRESETS.get(brand_kit or "", DEFAULT_STYLE))


def burn_subtitles(input_path: Path, output_path: Path, segments: list[dict], style: dict) -> None:
    """Render subtitles frame by frame; audio is carried over from the input."""
    clip = VideoFileClip(str(input_path))
    renderer = SubtitleRenderer(width=int(clip.w), height=int(clip.h), style=style)

    def active_segment(t):
        for seg in segments:
            if seg["start"] <= t <= seg["end"]:
                return seg
        return None

    def make_frame(get_frame, t):
        frame = get_frame(t)
        seg = active_segment(t)
        if seg is None:
            return frame
        return np.array(renderer.render_frame(seg, t, Image.fromarray(frame)))

    if MOVIEPY_V2:
        out = clip.transform(make_frame)
    else:
        out = clip.fl(make_frame, apply_to=["mask"])

    kwargs = dict(codec="libx264", audio_codec="aac", fps=clip.fps or 30, threads=4,
                  ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if not MOVIEPY_V2:
        kwargs["preset"] = "medium"
    out.write_videofile(str(output_path), logger=None, **kwargs)
    clip.close()
    out.close()
