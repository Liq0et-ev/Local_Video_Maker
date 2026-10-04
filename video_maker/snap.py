"""Snap cut points to quiet moments so shorts do not start or end in the middle of a word.

The loudness envelope comes straight from ffmpeg (8 kHz mono), so this works in both modes
(highlight and slice) and costs a few seconds even for long videos.
"""
import subprocess
from pathlib import Path

import numpy as np

Envelope = tuple[np.ndarray, np.ndarray]  # (frame centre times, RMS loudness)


def loudness_envelope(path: str | Path, frame_sec: float = 0.05, sr: int = 8000) -> Envelope | None:
    """Short-term loudness of the audio track, or None when the file has no usable audio."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return None
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    n = max(1, int(sr * frame_sec))
    frames = len(samples) // n
    if frames < 3:
        return None
    view = samples[: frames * n].reshape(frames, n)
    rms = np.empty(frames, dtype=np.float32)
    step = 20000  # process in chunks: long videos must not blow up memory
    for i in range(0, frames, step):
        chunk = view[i:i + step].astype(np.float32) / 32768.0
        rms[i:i + step] = np.sqrt((chunk ** 2).mean(axis=1))
    rms = np.convolve(rms, np.ones(3) / 3, mode="same")
    times = (np.arange(frames) + 0.5) * frame_sec
    return times, rms


def snap_time(t: float, env: Envelope | None, window: float, lo: float = 0.0, hi: float = float("inf")) -> float:
    """Move t to the quietest nearby moment inside [t-window, t+window] and [lo, hi].

    A small distance penalty prefers the closest pause when several are equally quiet.
    """
    if env is None or window <= 0:
        return t
    times, rms = env
    a, b = max(lo, t - window), min(hi, t + window)
    if b <= a:
        return t
    inside = (times >= a) & (times <= b)
    if not inside.any():
        return t
    seg, tt = rms[inside], times[inside]
    cost = seg / (float(seg.max()) + 1e-9) + 0.15 * np.abs(tt - t) / window
    return float(tt[int(np.argmin(cost))])


def snap_intervals(intervals: list[tuple[float, float]], env: Envelope | None, window: float,
                   video_duration: float, min_len: float = 1.0) -> list[tuple[float, float]]:
    """Snap both edges of every highlight segment; keep order, drop slivers, merge touching ones."""
    if env is None or window <= 0 or not intervals:
        return list(intervals)
    ordered = sorted(intervals)
    snapped: list[tuple[float, float]] = []
    prev_end = 0.0
    for k, (s, e) in enumerate(ordered):
        s2 = snap_time(s, env, window, lo=prev_end, hi=e - min_len)
        next_start = ordered[k + 1][0] if k + 1 < len(ordered) else video_duration
        e2 = snap_time(e, env, window, lo=s2 + min_len, hi=min(video_duration, next_start))
        if e2 - s2 >= min_len:
            snapped.append((s2, e2))
            prev_end = e2
    merged: list[tuple[float, float]] = []
    for s, e in snapped:
        if merged and s - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged or list(intervals)


def snap_cut_points(cuts: list[tuple[float, float]], env: Envelope | None, window: float,
                    min_len: float) -> list[tuple[float, float]]:
    """Snap the shared boundaries of contiguous slices; every clip keeps at least min_len seconds."""
    if env is None or window <= 0 or len(cuts) < 2:
        return list(cuts)
    original = [c[1] for c in cuts[:-1]]
    total = cuts[-1][1]
    bounds: list[float] = []
    prev = cuts[0][0]
    for i, b in enumerate(original):
        nxt = original[i + 1] if i + 1 < len(original) else total
        lo, hi = prev + min_len, nxt - min_len
        nb = snap_time(b, env, window, lo=lo, hi=hi) if lo <= hi else b
        bounds.append(nb)
        prev = nb
    starts = [cuts[0][0], *bounds]
    ends = [*bounds, total]
    return list(zip(starts, ends))
