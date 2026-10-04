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
