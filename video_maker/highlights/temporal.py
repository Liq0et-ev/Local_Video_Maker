"""
Temporal Management Module
===========================
Converts a boolean activity mask into concrete time intervals,
then applies:

  1. +5-second safety buffer after each event
  2. Interval merging for close-proximity segments
  3. Target clip-length adjustment (~60 s where feasible)
  4. Median-filter post-processing for segment coherence
"""

import numpy as np
from scipy.ndimage import median_filter


def median_smooth_mask(mask: np.ndarray, kernel_sec: float = 3.0,
                       dt: float = 1.0) -> np.ndarray:
    """
    Apply a 1-D median filter to the boolean mask to remove
    isolated spikes and fill tiny gaps (temporal coherence).
    """
    kernel = max(3, int(kernel_sec / dt))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = median_filter(mask.astype(np.float64), size=kernel)
    return smoothed > 0.5


def mask_to_intervals(mask: np.ndarray, timestamps: np.ndarray):
    """
    Convert a boolean mask to a list of (start_sec, end_sec) tuples.
    """
    intervals = []
    in_seg = False
    start = 0.0
    for i, active in enumerate(mask):
        if active and not in_seg:
            start = timestamps[i]
            in_seg = True
        elif not active and in_seg:
            intervals.append((start, timestamps[i]))
            in_seg = False
    if in_seg:
        intervals.append((start, timestamps[-1]))
    return intervals


def add_post_buffer(intervals, buffer_sec: float = 5.0,
                    video_duration: float = float("inf")):
    """
    Extend each interval by *buffer_sec* after its end to capture the
    tail of events (e.g. crowd reaction after a goal).
    """
    buffered = []
    for s, e in intervals:
        new_end = min(e + buffer_sec, video_duration)
        buffered.append((s, new_end))
    return buffered


def merge_intervals(intervals, gap_sec: float = 3.0):
    """
    Merge overlapping or nearly-adjacent intervals (gap <= gap_sec).
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + gap_sec:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def enforce_target_length(intervals, target_sec: float = 60.0,
                          video_duration: float = float("inf")):
    """
    Adjust intervals to hit a target total clip length.

    Strategy:
      - If total is already <= target_sec:  keep as-is.
      - If total > target_sec: trim longest segments proportionally.
      - If total < target_sec * 0.5: expand buffers slightly.

    Returns the (possibly modified) interval list and total duration.
    """
    if not intervals:
        return intervals, 0.0

    total = sum(e - s for s, e in intervals)

    if total <= target_sec:
        return intervals, total

    # --- Proportional trimming ---
    ratio = target_sec / total
    trimmed = []
    for s, e in intervals:
        dur = e - s
        new_dur = dur * ratio
        # Trim from the end (keep the event onset intact)
        trimmed.append((s, min(s + new_dur, video_duration)))

    new_total = sum(e - s for s, e in trimmed)
    return trimmed, new_total


def drop_short_segments(intervals, min_sec: float):
    """Remove segments shorter than min_sec (tiny clips feel like flicker)."""
    return [(s, e) for s, e in intervals if e - s >= min_sec]


def segment_scores(intervals, timestamps: np.ndarray, scores: np.ndarray):
    """Mean activity score inside each interval (used to rank segments)."""
    out = []
    for s, e in intervals:
        inside = (timestamps >= s) & (timestamps < e)
        out.append(float(scores[inside].mean()) if inside.any() else 0.0)
    return out


def top_segments(intervals, seg_scores, max_segments: int):
    """Keep the max_segments highest-scoring segments, in chronological order."""
    if not max_segments or len(intervals) <= max_segments:
        return list(intervals)
    keep = sorted(sorted(range(len(intervals)), key=lambda i: seg_scores[i], reverse=True)[:max_segments])
    return [intervals[i] for i in keep]


def select_best_segments(intervals, seg_scores, target_sec: float, max_segments: int, min_sec: float):
    """Fill the target length with the STRONGEST segments instead of chopping every segment short.

    Segments are taken best-first while they fit; the one that no longer fits is trimmed from its
    end (the event onset is kept) if enough room is left. Output is chronological.
    """
    order = sorted(range(len(intervals)), key=lambda i: seg_scores[i], reverse=True)
    chosen, total = [], 0.0
    for i in order:
        if max_segments and len(chosen) >= max_segments:
            break
        s, e = intervals[i]
        dur = e - s
        if total + dur <= target_sec:
            chosen.append((s, e))
            total += dur
        else:
            room = target_sec - total
            if room >= max(min_sec, 1.0):
                chosen.append((s, s + room))
                total += room
    if not chosen and order:  # target shorter than every segment: keep the best one, trimmed
        s, e = intervals[order[0]]
        chosen, total = [(s, min(e, s + target_sec))], min(e - s, target_sec)
    chosen.sort()
    return chosen, total


def expand_to_target(intervals, target_sec: float, video_duration: float):
    """If the detected moments are shorter than the target, add context before and after them.

    Segments grow evenly on both sides until the target is reached, a neighbour is touched
    (then they merge) or the video borders stop them.
    """
    iv = sorted(intervals)
    for _ in range(30):
        total = sum(e - s for s, e in iv)
        need = target_sec - total
        if need < 0.25 or not iv:
            break
        per_side = need / (2 * len(iv))
        grown = []
        for k, (s, e) in enumerate(iv):
            lo = iv[k - 1][1] if k > 0 else 0.0
            hi = iv[k + 1][0] if k + 1 < len(iv) else video_duration
            grown.append((max(lo, s - per_side), min(hi, e + per_side)))
        merged = [grown[0]]
        for s, e in grown[1:]:
            if s <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        if sum(e - s for s, e in merged) <= total + 1e-3:
            break  # no room left to grow
        iv = merged
    return iv, sum(e - s for s, e in iv)


def build_extraction_map(mask: np.ndarray,
                         timestamps: np.ndarray,
                         video_duration: float,
                         buffer_sec: float = 5.0,
                         merge_gap_sec: float = 3.0,
                         target_sec: float = 60.0,
                         median_kernel_sec: float = 3.0,
                         scores: np.ndarray | None = None,
                         min_segment_sec: float = 0.0,
                         max_segments: int = 0,
                         selection: str = "proportional",
                         fill_to_target: bool = False):
    """
    Full temporal pipeline: mask -> final clip intervals.

    Defaults reproduce the original behaviour. New knobs:
      min_segment_sec  drop merged segments shorter than this (fewer flickery micro-cuts)
      max_segments     cap on the number of segments (0 = unlimited), best ones are kept
      selection        "proportional" (trim every segment) or "best" (strongest segments first;
                       needs `scores`, otherwise segment length is used for ranking)
      fill_to_target   add context around the segments if the total is below target_sec

    Returns
    -------
    intervals   : list of (start, end) tuples in seconds
    total_dur   : total clip duration
    raw_intervals : intervals before target-length adjustment
    """
    dt = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 1.0

    # 1. Median-filter the mask for coherence
    mask = median_smooth_mask(mask, kernel_sec=median_kernel_sec, dt=dt)

    # 2. Convert to intervals, add the post-event buffer, merge close segments
    intervals = mask_to_intervals(mask, timestamps)
    intervals = add_post_buffer(intervals, buffer_sec=buffer_sec, video_duration=video_duration)
    intervals = merge_intervals(intervals, gap_sec=merge_gap_sec)

    # 3. Drop micro-segments (but never everything: keep the longest one)
    if min_segment_sec > 0 and intervals:
        kept = drop_short_segments(intervals, min_segment_sec)
        intervals = kept or [max(intervals, key=lambda iv: iv[1] - iv[0])]
    raw_intervals = list(intervals)
    if not intervals:
        return [], 0.0, raw_intervals

    # 4. Adjust toward the target length
    if scores is not None:
        seg_sc = segment_scores(intervals, timestamps, scores)
    else:
        seg_sc = [e - s for s, e in intervals]
    if selection == "best":
        intervals, total_dur = select_best_segments(
            intervals, seg_sc, target_sec, max_segments, min_segment_sec)
    else:
        intervals = top_segments(intervals, seg_sc, max_segments)
        intervals, total_dur = enforce_target_length(
            intervals, target_sec=target_sec, video_duration=video_duration)

    # 5. Pad with context when the detected moments are too short
    if fill_to_target and total_dur < target_sec:
        intervals, total_dur = expand_to_target(intervals, target_sec, video_duration)

    return intervals, total_dur, raw_intervals
