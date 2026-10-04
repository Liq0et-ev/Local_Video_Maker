"""Folder scanning, case selection and batch processing with a persistent processing log.

Ported from the selection logic of One_Minute_Videos / Background_Audio (numbered list, "all",
"1,3,5", "2-5") and the JSON processing log idea of Dinamic_Control_YouTube_Videos.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import VIDEO_EXTENSIONS, LocalConfig, PipelineOptions
from .pipeline import Cancelled, run_pipeline

LOG_NAME = "processing_log.json"


def discover_videos(input_dir: Path) -> list[Path]:
    """Video files directly inside input_dir, sorted by name (case-insensitive)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def parse_selection(text: str, total: int) -> list[int]:
    """'all' | '3' | '1,3,5' | '2-5' | '1,3-5,8'  ->  sorted 0-based indices. Invalid parts are ignored."""
    text = text.strip().lower()
    if text in ("all", "a", "*"):
        return list(range(total))
    picked: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = sorted((int(a), int(b)))
                picked.update(n - 1 for n in range(lo, hi + 1) if 1 <= n <= total)
            else:
                n = int(part)
                if 1 <= n <= total:
                    picked.add(n - 1)
        except ValueError:
            continue
    return sorted(picked)


class ProcessingLog:
    """JSON file in the output folder remembering which videos were already converted."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / LOG_NAME
        self._data: dict[str, dict] = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}  # a corrupt log must never block processing

    @staticmethod
    def key(video: Path, options: PipelineOptions) -> str:
        st = video.stat()
        return f"{video.name}|{st.st_size}|{st.st_mtime_ns}|{options.mode}"

    def is_done(self, video: Path, options: PipelineOptions) -> bool:
        entry = self._data.get(self.key(video, options))
        return bool(entry) and all(Path(o).exists() for o in entry.get("outputs", []))

    def record(self, video: Path, options: PipelineOptions, outputs: list[Path]) -> None:
        self._data[self.key(video, options)] = {
            "outputs": [str(o) for o in outputs],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class BatchItem:
    video: Path
    status: str            # done | skipped | failed | cancelled
    outputs: list[Path]
    seconds: float = 0.0
    error: str | None = None
    notes: list[str] | None = None
    transitions: list[str] | None = None


def run_batch(
    videos: list[Path],
    cfg: LocalConfig,
    force: bool = False,
    log_fn: Callable[[str], None] = print,
    progress: Callable[[int, int, float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    confirm_transitions: Callable | None = None,
) -> list[BatchItem]:
    """Convert each video in turn. One failing video never stops the rest."""
    cfg.ensure_dirs()
    log = ProcessingLog(cfg.output_dir)
    results: list[BatchItem] = []

    for n, video in enumerate(videos, 1):
        if should_cancel and should_cancel():
            results.append(BatchItem(video, "cancelled", []))
            break
        if not force and log.is_done(video, cfg.options):
            log_fn(f"[{n}/{len(videos)}] {video.name}: already processed, skipping (use --force to redo)")
            results.append(BatchItem(video, "skipped", []))
            continue

        log_fn(f"[{n}/{len(videos)}] {video.name}")
        t0 = time.time()

        def on_progress(pct: float, msg: str, _n=n) -> None:
            if progress:
                progress(_n, len(videos), pct, msg)

        try:
            res = run_pipeline(
                video, cfg.options, cfg.output_dir, cfg.work_dir, cfg.music_dir,
                whisper_model=cfg.whisper_model, progress=on_progress, should_cancel=should_cancel,
                confirm_transitions=confirm_transitions,
            )
        except Cancelled:
            results.append(BatchItem(video, "cancelled", [], time.time() - t0))
            break
        except Exception as exc:  # noqa: BLE001 - report and continue with the next video
            log_fn(f"    FAILED: {str(exc).strip()[-400:]}")
            results.append(BatchItem(video, "failed", [], time.time() - t0, error=str(exc)))
            continue

        log.record(video, cfg.options, res.outputs)
        for note in res.notes:
            log_fn(f"    note: {note}")
        if res.transitions:
            log_fn(f"    transitions: {', '.join(res.transitions)}")
        for out in res.outputs:
            log_fn(f"    -> {out}")
        results.append(BatchItem(video, "done", res.outputs, time.time() - t0, notes=res.notes,
                                 transitions=res.transitions))

    return results
