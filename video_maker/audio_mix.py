"""Background music mixing (ported from Background_Audio, reimplemented on pure ffmpeg).

Random tracks are chained until the video duration is covered, scaled to
`volume`, and mixed with the original audio. The video stream is not re-encoded.
"""
import random
from pathlib import Path

from .config import AUDIO_EXTENSIONS, EncodeSettings
from .ffmpeg_utils import probe, run


def list_tracks(music_dir: Path) -> list[Path]:
    if not music_dir.is_dir():
        return []
    return sorted(p for p in music_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def pick_tracks(tracks: list[Path], duration: float, rng: random.Random | None = None) -> list[Path]:
    """Randomly chain tracks (avoiding an immediate repeat) until `duration` is covered."""
    rng = rng or random
    chosen: list[Path] = []
    covered = 0.0
    for _ in range(200):
        if covered >= duration:
            break
        pool = [t for t in tracks if not chosen or t != chosen[-1]] or tracks
        track = rng.choice(pool)
        length = probe(track)["duration"]
        if length <= 0:
            continue
        chosen.append(track)
        covered += length
    return chosen


def add_background_music(video: Path, dst: Path, music_dir: Path, volume: float = 0.2,
                         enc: EncodeSettings | None = None) -> bool:
    """Mix music into `video`, writing `dst`. Returns False if no music is available."""
    enc = enc or EncodeSettings()
    tracks = list_tracks(music_dir)
    if not tracks:
        return False
    info = probe(video)
    chain = pick_tracks(tracks, info["duration"])
    if not chain:
        return False

    cmd = ["ffmpeg", "-y", "-i", str(video)]
    for t in chain:
        cmd += ["-i", str(t)]
    n = len(chain)
    prep = "".join(f"[{i + 1}:a]aresample=44100,aformat=channel_layouts=stereo[m{i}];" for i in range(n))
    joined = "".join(f"[m{i}]" for i in range(n))
    fade_start = max(info["duration"] - 1.5, 0)
    bg = (f"{joined}concat=n={n}:v=0:a=1,atrim=0:{info['duration']:.3f},"
          f"afade=t=out:st={fade_start:.3f}:d=1.5,volume={volume:.3f}[bg]")
    if info["has_audio"]:
        mix = ";[0:a]aresample=44100[orig];[orig][bg]amix=inputs=2:duration=first:normalize=0[aout]"
    else:
        mix = ";[bg]anull[aout]"
    cmd += ["-filter_complex", prep + bg + mix, "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", enc.audio_bitrate, "-shortest",
            "-movflags", "+faststart", str(dst)]
    run(cmd)
    return True
