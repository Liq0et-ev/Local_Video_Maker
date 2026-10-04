"""Thin ffmpeg/ffprobe helpers shared by every pipeline stage."""
import json
import shutil
import subprocess
from pathlib import Path

from .config import SHORTS_HEIGHT, SHORTS_WIDTH, EncodeSettings

MAX_FPS = 60.0


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(f"{tool} not found on PATH. Install FFmpeg first.")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-1500:]
        raise FFmpegError(f"{cmd[0]} failed ({proc.returncode}):\n{tail}")


def _fps_expr(raw: str | None) -> str:
    """ffprobe '30000/1001' -> a usable ffmpeg fps expression (capped, never 0)."""
    try:
        num, _, den = (raw or "").partition("/")
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return "30"
    if value < 5:
        return "30"
    return "60" if value > MAX_FPS else (raw or "30")


def probe(path: str | Path) -> dict:
    """Duration (s), stream presence, size and frame rate of a media file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.decode(errors='replace').strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    fps_raw = (video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else None
    return {
        "duration": float(info["format"].get("duration", 0.0)),
        "has_video": video is not None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
        "fps_expr": _fps_expr(fps_raw),
    }


def vertical_filter(mode: str) -> str | None:
    """ffmpeg filter chain producing a 1080x1920 frame, or None to keep the source."""
    w, h = SHORTS_WIDTH, SHORTS_HEIGHT
    if mode == "off":
        return None
    if mode == "crop":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1"
    return (
        f"split[a][b];"
        f"[a]scale={w}:{h}:force_original_aspect_ratio=increase:flags=bilinear,crop={w}:{h},boxblur=25:5[bg];"
        f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def video_filter_chain(mode: str, enc: EncodeSettings, source_fps: str) -> str:
    """Reframe -> constant frame rate -> optional sharpening, as one filter graph."""
    parts = []
    vf = vertical_filter(mode)
    if vf:
        parts.append(vf)
    # Keep the source rate by default: forcing e.g. 30 fps on 24/25/60 fps video causes judder.
    parts.append(f"fps={enc.fps:g}" if enc.fps else f"fps={source_fps}")
    if enc.sharpen > 0:
        parts.append(f"unsharp=5:5:{enc.sharpen:.2f}:5:5:0.0")
    return ",".join(parts)


def encode_args(enc: EncodeSettings, with_audio: bool) -> list[str]:
    args = ["-c:v", "libx264", "-preset", enc.preset, "-crf", str(enc.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if with_audio:
        args += ["-c:a", "aac", "-b:a", enc.audio_bitrate, "-ar", "44100", "-ac", "2"]
    return args


def cut_clip(src: str | Path, start: float, end: float, dst: str | Path, vertical: str = "blur",
             enc: EncodeSettings | None = None) -> None:
    """Frame-accurate cut re-encoded to uniform H.264/AAC, optionally reframed to 9:16.

    Every clip of one source gets identical parameters, so they can be concatenated or cross-faded.
    """
    enc = enc or EncodeSettings()
    info = probe(src)
    duration = end - start
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
           "-filter_complex", f"[0:v]{video_filter_chain(vertical, enc, info['fps_expr'])}[v]", "-map", "[v]"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0"]
        fade = enc.audio_fade_sec
        if fade > 0 and duration > 4 * fade:  # tiny fades remove the click at every cut
            cmd += ["-af", f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={duration - fade:.3f}:d={fade:.3f}"]
    cmd += encode_args(enc, info["has_audio"]) + [str(dst)]
    run(cmd)


def concat_clips(parts: list[Path], dst: str | Path, list_file: Path) -> None:
    """Hard-cut concatenation of clips produced by cut_clip, without re-encoding."""
    list_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", "-movflags", "+faststart", str(dst)])


def crossfade_clips(parts: list[Path], dst: str | Path, joins: list[tuple[str, float]],
                    enc: EncodeSettings) -> bool:
    """Join clips with a video transition (xfade effect) and an audio cross-fade at every join.

    `joins[k]` = (xfade effect, seconds) between parts[k] and parts[k + 1]. Every overlap is clamped so
    that it fits inside both neighbouring clips. Returns False when the clips are too short for
    transitions; the caller then falls back to hard cuts.
    """
    if len(parts) < 2 or len(joins) != len(parts) - 1:
        return False
    infos = [probe(p) for p in parts]
    durations = [i["duration"] for i in infos]
    with_audio = all(i["has_audio"] for i in infos)

    overlaps: list[float] = []
    clean_prev = durations[0]  # length of the previous clip not yet used by its own incoming transition
    for k, (_, seconds) in enumerate(joins):
        d = min(seconds, 0.45 * clean_prev, 0.45 * durations[k + 1])
        if d < 0.03:
            return False
        overlaps.append(d)
        clean_prev = durations[k + 1] - d

    graph = []
    v_prev, a_prev, total = "[0:v]", "[0:a]", durations[0]
    for k, ((effect, _), d) in enumerate(zip(joins, overlaps), start=1):
        graph.append(f"{v_prev}[{k}:v]xfade=transition={effect}:duration={d:.3f}:offset={total - d:.3f}[v{k}]")
        v_prev = f"[v{k}]"
        if with_audio:
            graph.append(f"{a_prev}[{k}:a]acrossfade=d={d:.3f}[a{k}]")
            a_prev = f"[a{k}]"
        total += durations[k] - d

    cmd = ["ffmpeg", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", ";".join(graph), "-map", v_prev]
    if with_audio:
        cmd += ["-map", a_prev]
    cmd += encode_args(enc, with_audio) + [str(dst)]
    run(cmd)
    return True


def extract_wav(src: str | Path, dst: str | Path, sr: int = 16000) -> None:
    run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), str(dst)])
