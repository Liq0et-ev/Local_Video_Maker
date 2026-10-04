"""Local configuration: folders and pipeline options come from config.ini (no Telegram, no .env)."""
import configparser
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .transition_catalog import KEYS as TRANSITION_KEYS
from .transition_catalog import MODES as TRANSITION_MODES
from .transition_catalog import STYLES as TRANSITION_STYLES

ROOT =Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = ROOT / "config.ini"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}

SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
# Pixel margins that avoid the YouTube Shorts UI overlay (like/share/subscribe)
SAFE_ZONE_LEFT = 60
SAFE_ZONE_RIGHT = 60


X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
# quality profile -> (x264 crf, x264 preset)
QUALITY_PROFILES = {
    "draft": (26, "veryfast"),
    "standard": (21, "fast"),
    "high": (18, "medium"),
    "max": (15, "slow"),
}


@dataclass(frozen=True)
class EncodeSettings:
    """Concrete encoder parameters shared by every ffmpeg / moviepy export step."""

    crf: int = 18
    preset: str = "medium"
    fps: float | None = None       # None = keep the source frame rate
    sharpen: float = 0.0
    audio_bitrate_kbps: int = 192
    audio_fade_sec: float = 0.03

    @property
    def audio_bitrate(self) -> str:
        return f"{self.audio_bitrate_kbps}k"


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

    # --- highlight detection tuning ---------------------------------------
    # relative importance of the four analysed signals (normalised internally)
    weight_motion: float = 0.30        # frame-to-frame picture change
    weight_flow: float = 0.30          # optical flow = amount of movement
    weight_loudness: float = 0.20      # speech / sound level
    weight_audio_change: float = 0.20  # sudden changes in the sound (hits, cheers, music drops)
    analysis_window_sec: float = 30.0  # window used to judge what is "normal" for this video
    min_activity_sec: float = 1.0      # activity shorter than this is ignored
    smoothing_window: int = 15         # signal smoothing (odd number, bigger = calmer curve)
    median_sec: float = 3.0            # bigger = fewer, longer, steadier segments
    sample_fps: float = 2.0            # analysis frames per second (higher = slower, more precise)

    # --- how the highlight is assembled -----------------------------------
    min_segment_sec: float = 4.0       # drop segments shorter than this (removes flicker)
    max_segments: int = 6              # at most N segments in one highlight (0 = unlimited)
    selection: str = "best"            # best = strongest segments first | proportional = trim all
    fill_to_target: bool = True        # add context around the moments if total < target_seconds

    # --- cuts between frames ------------------------------------------------
    snap_sec: float = 1.0              # move every cut to the nearest pause within +-N s (0 = off)
    # Transition between highlight segments:
    #   ask  = the program analyses every join, proposes a transition and waits for your OK (terminal);
    #          without a terminal it behaves like "auto"
    #   auto = same analysis, applied without asking
    #   or one fixed transition for all joins: cut, fade, fadeblack, wipeleft, pushup, zoom, ...
    transition: str = "ask"
    transition_style: str = "auto"     # tempo of the transitions: auto | calm | neutral | dynamic
    transition_sec: float | None = None  # None = the program picks 0.15-0.9 s per join
    audio_fade_ms: int = 30            # tiny fade at each cut: no clicks

    # --- video quality ------------------------------------------------------
    quality: str = "high"              # draft | standard | high | max (sets crf + preset)
    crf: int | None = None             # 0-51, lower = better quality and bigger file (None = from quality)
    preset: str | None = None          # x264 speed preset (None = from quality)
    fps: float | None = None           # None = keep the source frame rate (no judder)
    sharpen: float = 0.0               # 0 = off, 0.3-0.8 gentle, up to 2
    audio_bitrate_kbps: int = 192
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
        weights = (self.weight_motion, self.weight_flow, self.weight_loudness, self.weight_audio_change)
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("weights must be >= 0 and at least one must be above 0")
        if self.smoothing_window < 5:
            raise ValueError("smoothing_window must be at least 5")
        if self.sample_fps <= 0 or self.sample_fps > 30:
            raise ValueError("sample_fps must be within (0, 30]")
        for name in ("analysis_window_sec", "median_sec", "min_activity_sec", "min_segment_sec", "snap_sec"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.snap_sec > 5:
            raise ValueError("snap_sec must be at most 5")
        if self.max_segments < 0:
            raise ValueError("max_segments must be 0 (unlimited) or more")
        if self.selection not in ("best", "proportional"):
            raise ValueError(f"Unknown selection: {self.selection}")
        if self.transition not in TRANSITION_MODES + TRANSITION_KEYS:
            raise ValueError(f"Unknown transition: {self.transition} "
                             f"(use {', '.join(TRANSITION_MODES + TRANSITION_KEYS)})")
        if self.transition_style not in TRANSITION_STYLES:
            raise ValueError(f"Unknown transition_style: {self.transition_style} (use {', '.join(TRANSITION_STYLES)})")
        if self.transition_sec is not None and not 0.05 <= self.transition_sec <= 2:
            raise ValueError("transition_sec must be within 0.05..2 (or empty to let the program choose)")
        if not 0 <= self.audio_fade_ms <= 500:
            raise ValueError("audio_fade_ms must be within 0..500")
        if self.quality not in QUALITY_PROFILES:
            raise ValueError(f"Unknown quality: {self.quality} (use {', '.join(QUALITY_PROFILES)})")
        if self.crf is not None and not 0 <= self.crf <= 51:
            raise ValueError("crf must be within 0..51")
        if self.preset is not None and self.preset not in X264_PRESETS:
            raise ValueError(f"Unknown preset: {self.preset} (use {', '.join(X264_PRESETS)})")
        if self.fps is not None and not 10 <= self.fps <= 120:
            raise ValueError("fps must be within 10..120 (or empty to keep the source rate)")
        if not 0 <= self.sharpen <= 2:
            raise ValueError("sharpen must be within 0..2")
        if not 64 <= self.audio_bitrate_kbps <= 512:
            raise ValueError("audio_bitrate_kbps must be within 64..512")

    def encode_settings(self) -> "EncodeSettings":
        """Resolve quality profile + explicit overrides into concrete encoder settings."""
        profile_crf, profile_preset = QUALITY_PROFILES[self.quality]
        return EncodeSettings(
            crf=self.crf if self.crf is not None else profile_crf,
            preset=self.preset or profile_preset,
            fps=self.fps,
            sharpen=self.sharpen,
            audio_bitrate_kbps=self.audio_bitrate_kbps,
            audio_fade_sec=self.audio_fade_ms / 1000.0,
        )


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


def _auto(value: str) -> str | None:
    """Empty / auto / source / none  ->  None (let the program choose)."""
    v = value.strip().lower()
    return None if v in ("", "auto", "source", "none", "original") else v


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
        weight_motion=_opt_float(get("highlight", "weight_motion", str(d.weight_motion))),
        weight_flow=_opt_float(get("highlight", "weight_flow", str(d.weight_flow))),
        weight_loudness=_opt_float(get("highlight", "weight_loudness", str(d.weight_loudness))),
        weight_audio_change=_opt_float(get("highlight", "weight_audio_change", str(d.weight_audio_change))),
        analysis_window_sec=_opt_float(get("highlight", "analysis_window_seconds", str(d.analysis_window_sec))),
        min_activity_sec=_opt_float(get("highlight", "min_activity_seconds", str(d.min_activity_sec))),
        smoothing_window=int(get("highlight", "smoothing_window", str(d.smoothing_window))),
        median_sec=_opt_float(get("highlight", "median_seconds", str(d.median_sec))),
        sample_fps=_opt_float(get("highlight", "sample_fps", str(d.sample_fps))),
        min_segment_sec=_opt_float(get("highlight", "min_segment_seconds", str(d.min_segment_sec))),
        max_segments=int(get("highlight", "max_segments", str(d.max_segments))),
        selection=get("highlight", "selection", d.selection).lower(),
        fill_to_target=flag("highlight", "fill_to_target", d.fill_to_target),
        snap_sec=_opt_float(get("cuts", "snap_seconds", str(d.snap_sec))),
        transition=get("cuts", "transition", d.transition).lower(),
        transition_style=get("cuts", "transition_style", d.transition_style).lower(),
        transition_sec=(_opt_float(v) if (v := _auto(get("cuts", "transition_seconds", ""))) is not None else None),
        audio_fade_ms=int(get("cuts", "audio_fade_ms", str(d.audio_fade_ms))),
        quality=get("video", "quality", d.quality).lower(),
        crf=int(v) if (v := _auto(get("video", "crf", ""))) is not None else None,
        preset=_auto(get("video", "preset", "")),
        fps=_opt_float(v) if (v := _auto(get("video", "fps", ""))) is not None else None,
        sharpen=_opt_float(get("video", "sharpen", str(d.sharpen))),
        audio_bitrate_kbps=int(get("video", "audio_bitrate_kbps", str(d.audio_bitrate_kbps))),
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
