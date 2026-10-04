import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    """Tiny synthetic media: a 16:9 video with audio, a silent 9:16 video, two music tracks."""
    d = tmp_path_factory.mktemp("media")
    wide = d / "wide.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi",
            "-i", "sine=frequency=220:sample_rate=44100", "-t", "30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(wide))
    silent = d / "silent_vertical.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30", "-t", "12",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(silent))
    music = d / "music"
    music.mkdir()
    for name, freq, dur in (("a.mp3", 440, 7), ("b.wav", 330, 5)):
        _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=44100", "-t", str(dur), str(music / name))
    (music / "notes.txt").write_text("not audio")
    return {"wide": wide, "silent": silent, "music": music}


@pytest.fixture(scope="session")
def activity_video(tmp_path_factory) -> Path:
    """120 s clip: calm blue background with three busy+loud bursts (20-30 s, 60-70 s, 100-110 s)."""
    out = tmp_path_factory.mktemp("activity") / "activity.mp4"
    busy = "between(t,20,30)+between(t,60,70)+between(t,100,110)"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x303060:s=640x360:r=30", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30",
            "-f", "lavfi", "-i", "sine=f=220:sample_rate=44100", "-t", "120",
            "-filter_complex",
            f"[0:v][1:v]overlay=enable='{busy}'[v];[2:a]volume='if({busy},1,0.03)':eval=frame[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(out))
    return out


@pytest.fixture(scope="session")
def fps_videos(tmp_path_factory) -> dict[str, Path]:
    """Short clips with unusual frame rates, to prove the source rate is preserved."""
    d = tmp_path_factory.mktemp("fps")
    out = {}
    for name, rate in (("fps24", "24"), ("fps25", "25"), ("fps2997", "30000/1001"), ("fps50", "50")):
        p = d / f"{name}.mp4"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=320x180:rate={rate}", "-f", "lavfi",
                "-i", "sine=frequency=300:sample_rate=44100", "-t", "6", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(p))
        out[name] = p
    return out
