"""Local configuration: folders and pipeline options come from config.ini (no Telegram, no .env)."""
import configparser
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = ROOT / "config.ini"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}

SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
# Pixel margins that avoid the YouTube Shorts UI overlay (like/share/subscribe)
SAFE_ZONE_LEFT = 60
SAFE_ZONE_RIGHT = 60


@dataclass
class PipelineOptions:
    """Everything a single run can tweak."""

    # "highlight": one clip built from the most dynamic moments (Dinamic_Control_YouTube_Videos)
    # "slice":     whole video cut into sequential random-length clips (One_Minute_Videos)
    mode: str = "highlight"
    # highlight mode
    variant: str = "heuristic"      # heuristic | surprisal | attention
    target_sec: float = 60.0
    threshold_k: float = 1.5
    buffer_sec: float = 5.0
    merge_gap_sec: float = 3.0
    # slice mode
    min_clip_sec: float = 45.0
    max_clip_sec: float = 60.0
    max_clips: int = 10              # safety cap per video
    # formatting
    vertical: str = "blur"           # blur (full frame over blurred bg) | crop (fill 9:16) | off
    # subtitles
    subtitles: bool = True
    language: str | None = None      # None = auto-detect (en/ru)
    brand_kit: str = "viral_yellow"
    # music
    music: bool = True
    music_volume: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PipelineOptions":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def validate(self) -> None:
        if self.mode not in ("highlight", "slice"):
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.variant not in ("heuristic", "surprisal", "attention"):
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.vertical not in ("blur", "crop", "off"):
            raise ValueError(f"Unknown vertical mode: {self.vertical}")
        if not 0.0 <= self.music_volume <= 1.0:
            raise ValueError("music_volume must be within 0..1")
        if self.min_clip_sec >= self.max_clip_sec:
            raise ValueError("min_clip_sec must be less than max_clip_sec")
        if self.max_clips < 1:
            raise ValueError("max_clips must be at least 1")
        if self.target_sec <= 0:
            raise ValueError("target_sec must be positive")


@dataclass
class LocalConfig:
    input_dir: Path
    music_dir: Path
    output_dir: Path
    work_dir: Path
    whisper_model: str
    options: PipelineOptions

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)


def _path(raw: str, base: Path) -> Path:
    p = Path(raw.strip().strip('"').strip("'")).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def _opt_float(value: str) -> float:
    return float(value.replace(",", "."))


def load_config(path: str | Path | None = None) -> LocalConfig:
    """Read config.ini. Relative folders are resolved against the config file's own folder."""
    cfg_file = Path(path) if path else DEFAULT_CONFIG_FILE
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    cp.read(cfg_file, encoding="utf-8")
    base = cfg_file.resolve().parent

    def get(section: str, key: str, default: str) -> str:
        return cp.get(section, key, fallback=default).strip()

    def flag(section: str, key: str, default: bool) -> bool:
        return cp.getboolean(section, key, fallback=default)

    d = PipelineOptions()
    lang = get("subtitles", "language", "auto").lower()
    options = PipelineOptions(
        mode=get("pipeline", "mode", d.mode).lower(),
        variant=get("pipeline", "variant", d.variant).lower(),
        target_sec=_opt_float(get("pipeline", "target_seconds", str(d.target_sec))),
        threshold_k=_opt_float(get("pipeline", "threshold", str(d.threshold_k))),
        buffer_sec=_opt_float(get("pipeline", "buffer_seconds", str(d.buffer_sec))),
        merge_gap_sec=_opt_float(get("pipeline", "merge_gap_seconds", str(d.merge_gap_sec))),
        min_clip_sec=_opt_float(get("pipeline", "min_clip_seconds", str(d.min_clip_sec))),
        max_clip_sec=_opt_float(get("pipeline", "max_clip_seconds", str(d.max_clip_sec))),
        max_clips=int(get("pipeline", "max_clips", str(d.max_clips))),
        vertical=get("pipeline", "vertical", d.vertical).lower(),
        subtitles=flag("subtitles", "enabled", d.subtitles),
        language=None if lang in ("", "auto", "none") else lang,
        brand_kit=get("subtitles", "brand_kit", d.brand_kit),
        music=flag("music", "enabled", d.music),
        music_volume=_opt_float(get("music", "volume", str(d.music_volume))),
    )
    options.validate()
    return LocalConfig(
        input_dir=_path(get("paths", "input_dir", "input"), base),
        music_dir=_path(get("paths", "music_dir", "music"), base),
        output_dir=_path(get("paths", "output_dir", "output"), base),
        work_dir=_path(get("paths", "work_dir", "work"), base),
        whisper_model=get("subtitles", "whisper_model", "medium"),
        options=options,
    )
