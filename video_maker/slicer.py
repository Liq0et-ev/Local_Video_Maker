"""Sequential random-length slicing (ported from One_Minute_Videos).

Pure planning logic: no gaps, no overlaps, a short tail is merged into the previous clip.
"""
import random


def compute_cut_points(total: float, min_sec: float, max_sec: float,
                       rng: random.Random | None = None) -> list[tuple[float, float]]:
    rng = rng or random
    cuts: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < total:
        remaining = total - cursor
        if remaining < min_sec:
            if cuts:
                cuts[-1] = (cuts[-1][0], total)
            else:
                cuts.append((0.0, total))
            break
        length = rng.uniform(min_sec, max_sec)
        if remaining - length < min_sec:
            cuts.append((cursor, total))
            break
        cuts.append((cursor, cursor + length))
        cursor += length
    return cuts
