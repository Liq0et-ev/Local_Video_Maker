"""Thin ffmpeg/ffprobe helpers shared by every pipeline stage."""
import json
import shutil
import subprocess
from pathlib import Path

from .config import SHORTS_HEIGHT, SHORTS_WIDTH


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


def probe(path: str | Path) -> dict:
    """Return duration (s) and whether the file has audio/video streams."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr.decode(errors='replace').strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    return {
        "duration": float(info["format"].get("duration", 0.0)),
        "has_video": video is not None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
    }


def vertical_filter(mode: str) -> str | None:
    """ffmpeg filter chain producing a 1080x1920 frame, or None to keep the source."""
    w, h = SHORTS_WIDTH, SHORTS_HEIGHT
    if mode == "off":
        return None
    if mode == "crop":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
    return (
        f"split[a][b];"
        f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bg];"
        f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def cut_clip(src: str | Path, start: float, end: float, dst: str | Path, vertical: str = "blur") -> None:
    """Frame-accurate cut re-encoded to uniform H.264/AAC, optionally reframed to 9:16.

    Re-encoding (instead of stream copy) keeps cuts exact and makes every segment
    share identical codec parameters so they can be concatenated safely.
    """
    has_audio = probe(src)["has_audio"]
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}"]
    vf = vertical_filter(vertical)
    if vf:
        cmd += ["-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]"]
    else:
        cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", "30", "-movflags", "+faststart", str(dst)]
    run(cmd)


def concat_clips(parts: list[Path], dst: str | Path, list_file: Path) -> None:
    """Concatenate clips produced by cut_clip (identical encoding) without re-encoding."""
    list_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", "-movflags", "+faststart", str(dst)])


def extract_wav(src: str | Path, dst: str | Path, sr: int = 16000) -> None:
    run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), str(dst)])
