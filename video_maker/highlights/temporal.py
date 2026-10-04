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


def build_extraction_map(mask: np.ndarray,
                         timestamps: np.ndarray,
                         video_duration: float,
                         buffer_sec: float = 5.0,
                         merge_gap_sec: float = 3.0,
                         target_sec: float = 60.0,
                         median_kernel_sec: float = 3.0):
    """
    Full temporal pipeline: mask -> final clip intervals.

    Returns
    -------
    intervals   : list of (start, end) tuples in seconds
    total_dur   : total clip duration
    raw_intervals : intervals before target-length adjustment
    """
    dt = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 1.0

    # 1. Median-filter the mask for coherence
    mask = median_smooth_mask(mask, kernel_sec=median_kernel_sec, dt=dt)

    # 2. Convert to intervals
    intervals = mask_to_intervals(mask, timestamps)

    # 3. Add +5 s safety buffer
    intervals = add_post_buffer(intervals, buffer_sec=buffer_sec,
                                video_duration=video_duration)

    # 4. Merge close segments
    intervals = merge_intervals(intervals, gap_sec=merge_gap_sec)
    raw_intervals = list(intervals)

    # 5. Adjust toward target length
    intervals, total_dur = enforce_target_length(
        intervals, target_sec=target_sec, video_duration=video_duration
    )

    return intervals, total_dur, raw_intervals
