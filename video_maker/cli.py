"""Command line / interactive entry point.

    python main.py                       # interactive: numbered list, pick videos
    python main.py --case all            # every video in input_dir
    python main.py --case 1,3-5 --mode slice --no-music
"""
import argparse
import dataclasses
import sys
from pathlib import Path

from .batch import ProcessingLog, discover_videos, parse_selection, run_batch
from .config import (
    DEFAULT_CONFIG_FILE,
    QUALITY_PROFILES,
    TRANSITION_KEYS,
    TRANSITION_MODES,
    TRANSITION_STYLES,
    X264_PRESETS,
    LocalConfig,
    load_config,
)
from .ffmpeg_utils import FFmpegError, probe, require_ffmpeg
from .subtitles.brand_kits import PRESETS
from .transitions import TransitionPlan, apply_user_edits, format_catalog, format_plan


# (flag, PipelineOptions field, type, choices, help)
TUNING_FLAGS = [
    ("--buffer", "buffer_sec", float, None, "highlight: safety seconds kept after each event"),
    ("--merge-gap", "merge_gap_sec", float, None, "highlight: merge segments closer than this"),
    ("--analysis-window", "analysis_window_sec", float, None, "seconds used to judge what is 'normal' for the video"),
    ("--min-activity", "min_activity_sec", float, None, "ignore activity shorter than this"),
    ("--smoothing-window", "smoothing_window", int, None, "signal smoothing, odd number >= 5"),
    ("--median", "median_sec", float, None, "bigger = fewer, longer, steadier segments"),
    ("--sample-fps", "sample_fps", float, None, "analysis frames per second"),
    ("--min-segment", "min_segment_sec", float, None, "drop highlight segments shorter than this"),
    ("--max-segments", "max_segments", int, None, "at most N segments (0 = unlimited)"),
    ("--selection", "selection", str, ["best", "proportional"], "how to reach the target length"),
    ("--snap", "snap_sec", float, None, "snap cuts to the nearest pause within +-N seconds (0 = off)"),
    ("--transition", "transition", str, list(TRANSITION_MODES + TRANSITION_KEYS),
     "ask = propose and confirm, auto = decide silently, or one fixed transition"),
    ("--transition-style", "transition_style", str, list(TRANSITION_STYLES), "tempo of the transitions"),
    ("--transition-sec", "transition_sec", float, None, "force one transition duration"),
    ("--audio-fade-ms", "audio_fade_ms", int, None, "tiny audio fade at each cut (anti-click)"),
    ("--quality", "quality", str, list(QUALITY_PROFILES), "draft | standard | high | max"),
    ("--crf", "crf", int, None, "x264 quality 0-51 (lower = better, bigger)"),
    ("--preset", "preset", str, list(X264_PRESETS), "x264 speed preset"),
    ("--fps", "fps", float, None, "force a frame rate (default: keep the source rate)"),
    ("--sharpen", "sharpen", float, None, "0 = off, 0.3-0.8 gentle, up to 2"),
    ("--audio-bitrate", "audio_bitrate_kbps", int, None, "audio bitrate in kbit/s"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert long videos from a local folder into YouTube Shorts (no Telegram).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Case selection (--case):
  all        every video in the input folder
  3          only video #3
  1,3,5      videos #1, #3 and #5
  2-5        videos #2 through #5
  (omitted)  interactive: shows the list and asks

All settings live in config.ini; every flag below overrides it for this run only.
""",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="path to config.ini")
    p.add_argument("--case", help="which videos to process: all | 3 | 1,3,5 | 2-5")
    p.add_argument("--force", action="store_true", help="re-process videos already in the processing log")
    p.add_argument("--list", action="store_true", dest="list_only", help="only list videos and exit")
    p.add_argument("--ask-folders", action="store_true",
                   help="ask for the input / music / output folders at start (Enter keeps the current one)")

    g = p.add_argument_group("folders (override config.ini)")
    g.add_argument("--input-dir")
    g.add_argument("--music-dir")
    g.add_argument("--output-dir")
    g.add_argument("--work-dir")

    o = p.add_argument_group("pipeline (override config.ini)")
    o.add_argument("--mode", choices=["highlight", "slice"])
    o.add_argument("--variant", choices=["heuristic", "surprisal", "attention"])
    o.add_argument("--target", type=float, help="highlight length in seconds")
    o.add_argument("--threshold", type=float, help="highlight sensitivity k (lower = more segments)")
    o.add_argument("--min", type=float, dest="min_clip", help="slice: minimum clip seconds")
    o.add_argument("--max", type=float, dest="max_clip", help="slice: maximum clip seconds")
    o.add_argument("--max-clips", type=int, help="slice: cap on clips per video")
    o.add_argument("--vertical", choices=["blur", "crop", "off"])
    o.add_argument("--kit", choices=sorted(PRESETS), help="subtitle style")
    o.add_argument("--lang", choices=["en", "ru"], help="force subtitle language (default: auto)")
    o.add_argument("--whisper", help="Whisper model: tiny | base | small | medium | large")
    o.add_argument("--music-volume", type=float)
    o.add_argument("--no-subtitles", action="store_true")
    o.add_argument("--no-music", action="store_true")

    q = p.add_argument_group("highlight tuning, cuts, transitions and video quality (override config.ini)")
    for flag, dest, typ, choices, help_text in TUNING_FLAGS:
        q.add_argument(flag, dest=dest, type=typ, choices=choices, help=help_text)
    q.add_argument("--weights", help="signal weights 'motion,flow,loudness,audio_change', e.g. 0.4,0.3,0.2,0.1")
    q.add_argument("--no-fill", action="store_true", help="do not pad a too-short highlight with context")
    return p


def parse_weights(text: str) -> tuple[float, float, float, float]:
    try:
        values = [float(x) for x in text.replace(";", ",").split(",")]
    except ValueError:
        values = []
    if len(values) != 4:
        raise ValueError("--weights needs four numbers: motion,flow,loudness,audio_change")
    return values[0], values[1], values[2], values[3]


def resolve_path(value: str | Path, base: Path) -> Path:
    """Expand ~ and quotes; relative paths are taken from `base`."""
    p = Path(str(value).strip().strip('"').strip("'")).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


FOLDER_KEYS = ("input_dir", "music_dir", "output_dir", "work_dir")


def apply_overrides(cfg: LocalConfig, a: argparse.Namespace, folders: dict | None = None,
                    folders_base: Path | None = None) -> LocalConfig:
    """Apply overrides on top of config.ini. Precedence: command-line flag > `folders` (main.py) > config.ini.

    Flag paths are relative to the current directory; `folders` paths are relative to `folders_base`
    (the folder containing main.py).
    """
    cwd = Path.cwd()
    base = folders_base or cwd
    chosen: dict[str, Path] = {}
    for key in FOLDER_KEYS:
        flag = getattr(a, key, None)
        preset = (folders or {}).get(key)
        if flag:
            chosen[key] = resolve_path(flag, cwd)
        elif preset:
            chosen[key] = resolve_path(preset, base)

    opts = dataclasses.replace(
        cfg.options,
        **{k: v for k, v in {
            "mode": a.mode, "variant": a.variant, "target_sec": a.target, "threshold_k": a.threshold,
            "min_clip_sec": a.min_clip, "max_clip_sec": a.max_clip, "max_clips": a.max_clips,
            "vertical": a.vertical, "brand_kit": a.kit, "language": a.lang, "music_volume": a.music_volume,
        }.items() if v is not None},
    )
    extra = {dest: getattr(a, dest) for _, dest, *_ in TUNING_FLAGS if getattr(a, dest, None) is not None}
    opts = dataclasses.replace(opts, **extra)
    if getattr(a, "weights", None):
        wm, wf, wl, wc = parse_weights(a.weights)
        opts = dataclasses.replace(opts, weight_motion=wm, weight_flow=wf, weight_loudness=wl,
                                   weight_audio_change=wc)
    if getattr(a, "no_fill", False):
        opts.fill_to_target = False
    if a.no_subtitles:
        opts.subtitles = False
    if a.no_music:
        opts.music = False
    opts.validate()
    return LocalConfig(
        input_dir=chosen.get("input_dir", cfg.input_dir),
        music_dir=chosen.get("music_dir", cfg.music_dir),
        output_dir=chosen.get("output_dir", cfg.output_dir),
        work_dir=chosen.get("work_dir", cfg.work_dir),
        whisper_model=a.whisper or cfg.whisper_model,
        options=opts,
    )


def _is_interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def _ask(prompt: str) -> str | None:
    """input() that returns None on Ctrl+C / closed stdin."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def ask_folders(cfg: LocalConfig) -> LocalConfig:
    """Let the user type the three folders; Enter keeps the shown default."""
    print("\n  Folders (press Enter to keep the value in brackets):")
    values = {}
    for key, label in (("input_dir", "Folder with videos"), ("music_dir", "Folder with music"),
                       ("output_dir", "Folder for finished shorts")):
        current = getattr(cfg, key)
        answer = _ask(f"  {label} [{current}]: ")
        values[key] = resolve_path(answer, Path.cwd()) if answer else current
    return dataclasses.replace(cfg, **values)


def find_videos(cfg: LocalConfig, interactive: bool) -> tuple[LocalConfig, list[Path] | None]:
    """Scan input_dir. When it is missing or empty and we can talk to the user, offer to type another folder.

    Returns (config, videos) where videos is None if the folder does not exist, [] if it has no videos.
    """
    while True:
        try:
            videos: list[Path] | None = discover_videos(cfg.input_dir)
        except FileNotFoundError:
            videos = None
        if videos:
            return cfg, videos
        print(f"\n  The input folder {cfg.input_dir} "
              f"{'does not exist' if videos is None else 'contains no videos'}.")
        if not interactive:
            return cfg, videos
        answer = _ask("  Enter another folder with videos (Enter to quit): ")
        if not answer:
            return cfg, videos
        cfg = dataclasses.replace(cfg, input_dir=resolve_path(answer, Path.cwd()))


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def print_list(videos: list[Path], cfg: LocalConfig) -> None:
    log = ProcessingLog(cfg.output_dir)
    print(f"\n  Input : {cfg.input_dir}")
    print(f"  Music : {cfg.music_dir}")
    print(f"  Output: {cfg.output_dir}")
    o = cfg.options
    print(f"  Mode  : {o.mode} | 9:16 {o.vertical} | subtitles {'on' if o.subtitles else 'off'}"
          f" | music {'on' if o.music else 'off'}\n")
    print(f"  {'#':>3}  {'File':<46} {'Length':>8}  Status")
    print(f"  {'-' * 3}  {'-' * 46} {'-' * 8}  {'-' * 8}")
    for i, v in enumerate(videos, 1):
        try:
            length = fmt_duration(probe(v)["duration"])
        except FFmpegError:
            length = "???"
        name = v.name if len(v.name) <= 46 else v.name[:43] + "..."
        status = "done" if log.is_done(v, o) else ""
        print(f"  {i:>3}  {name:<46} {length:>8}  {status}")
    print()


def ask_selection(videos: list[Path]) -> list[int]:
    print("  Select videos:  'all'  |  '3'  |  '1,3,5'  |  '2-5'  |  'q' to quit\n")
    while True:
        try:
            choice = input("  Your selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return []
        if choice.lower() in ("q", "quit", "exit"):
            return []
        picked = parse_selection(choice, len(videos))
        if picked:
            return picked
        print("  No valid selection, try again.")


def confirm_transitions(plan: TransitionPlan) -> TransitionPlan:
    """Show the proposed transitions and let the user accept (Enter) or change them."""
    print("\n" + format_plan(plan))
    print("\n  Enter = принять  |  2=zoom  |  2=alt (другой вариант)  |  3=pushup:0.3 (с длительностью)"
          "  |  all=fade  |  list = все переходы")
    while True:
        answer = _ask("  Ваш выбор: ")
        if answer is None or answer.lower() in ("", "ok", "y", "yes", "да", "д"):
            return plan
        if answer.lower() in ("list", "l", "список"):
            print("\n" + format_catalog())
            continue
        plan, errors = apply_user_edits(plan, answer)
        for problem in errors:
            print(f"   ! {problem}")
        print("\n" + format_plan(plan))
        print("  Enter = принять, или введите ещё правки.")


def main(argv: list[str] | None = None, folders: dict | None = None, folders_base: Path | None = None,
         ask_for_folders: bool = False) -> int:
    """`folders` / `folders_base` come from the constants at the top of main.py."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)

    try:
        require_ffmpeg()
        cfg = apply_overrides(load_config(args.config), args, folders, folders_base)
    except (FileNotFoundError, FFmpegError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    interactive = _is_interactive() and args.case is None and not args.list_only
    if interactive and (args.ask_folders or ask_for_folders):
        cfg = ask_folders(cfg)
    cfg, videos = find_videos(cfg, interactive)
    if videos is None:
        print("ERROR: set input_dir in main.py, config.ini or with --input-dir.", file=sys.stderr)
        return 2
    if not videos:
        print("  Put videos into that folder or point to another one (main.py, config.ini, --input-dir).")
        return 0
    print_list(videos, cfg)
    if args.list_only:
        return 0

    if args.case is not None:
        picked = parse_selection(args.case, len(videos))
        if not picked:
            print(f"ERROR: --case '{args.case}' matches no video (1-{len(videos)}).", file=sys.stderr)
            return 2
    else:
        picked = ask_selection(videos)
        if not picked:
            print("  Nothing selected.")
            return 0

    chosen = [videos[i] for i in picked]
    print(f"  Processing {len(chosen)} video(s)...\n")

    last = {"msg": ""}

    def show(n: int, total: int, pct: float, msg: str) -> None:
        line = f"    [{n}/{total}] {pct:3.0f}%  {msg}"
        if line != last["msg"]:
            print(line)
            last["msg"] = line

    # In a terminal the program proposes transitions and waits for your OK; otherwise it decides itself.
    confirm = confirm_transitions if _is_interactive() else None
    try:
        results = run_batch(chosen, cfg, force=args.force, progress=show, confirm_transitions=confirm)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return 130

    print("\n" + "=" * 64)
    done = [r for r in results if r.status == "done"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]
    for r in results:
        extra = f"{len(r.outputs)} short(s), {r.seconds:.0f}s" if r.status == "done" else (r.error or "")[-80:]
        print(f"  {r.status:<9} {r.video.name[:40]:<40} {extra}")
    print(f"\n  Done: {len(done)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    print(f"  Shorts saved in: {cfg.output_dir}")
    print("=" * 64)
    return 1 if failed else 0
