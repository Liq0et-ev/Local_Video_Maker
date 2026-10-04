"""
DSP Pipeline Module
====================
Signal conditioning layer that sits between raw features and the
detection logic.  Responsibilities:

1. Resample all signals to a common time axis.
2. Normalise (Min-Max **and** Z-score variants).
3. Smooth with a Savitzky-Golay filter (preserves peaks better than
   a moving average).
4. Compute 1st and 2nd derivatives for onset / offset detection.
5. Provide a sliding-window adaptive threshold based on rolling
   statistics (the "memory" that adapts to each video's noise floor).
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


# ------------------------------------------------------------------ #
#  Resampling
# ------------------------------------------------------------------ #
def resample_to_common_axis(timestamps, signal, common_ts):
    """Linearly interpolate *signal* onto *common_ts*."""
    if len(timestamps) < 2:
        return np.zeros_like(common_ts)
    f = interp1d(timestamps, signal, kind="linear",
                 bounds_error=False, fill_value=0.0)
    return f(common_ts)


# ------------------------------------------------------------------ #
#  Normalisation
# ------------------------------------------------------------------ #
def minmax_norm(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1].  Returns zeros if range is zero."""
    lo, hi = x.min(), x.max()
    rng = hi - lo
    if rng < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / rng


def zscore_norm(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


# ------------------------------------------------------------------ #
#  Smoothing
# ------------------------------------------------------------------ #
def smooth_sg(x: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay filter.

    Why SG?  Unlike a moving average it preserves the shape and height
    of sharp peaks while still suppressing high-frequency noise.
    The window length and polynomial order control the trade-off.

    Parameters
    ----------
    window : int   - must be odd and > polyorder
    polyorder : int
    """
    if len(x) < window:
        window = max(5, len(x) // 2 * 2 + 1)  # ensure odd
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1
    return savgol_filter(x, window_length=window, polyorder=polyorder)


# ------------------------------------------------------------------ #
#  Derivatives
# ------------------------------------------------------------------ #
def derivatives(x: np.ndarray, dt: float = 1.0):
    """
    Compute first and second derivatives using central differences.

    Returns
    -------
    dx  : 1st derivative (velocity of change)
    ddx : 2nd derivative (acceleration - marks *onset* of events)
    """
    dx = np.gradient(x, dt)
    ddx = np.gradient(dx, dt)
    return dx, ddx


# ------------------------------------------------------------------ #
#  Adaptive threshold (sliding window memory)
# ------------------------------------------------------------------ #
def adaptive_threshold(signal: np.ndarray,
                       window_sec: float = 30.0,
                       dt: float = 1.0,
                       k: float = 1.5) -> np.ndarray:
    """
    Sliding-window threshold = rolling_mean + k * rolling_std.

    This lets the system adapt to each video's specific noise floor.
    Quiet sections get a lower threshold; noisy sections get a higher
    one - preventing both missed events and false positives.

    Parameters
    ----------
    signal     : 1-D array (already normalised / smoothed).
    window_sec : sliding window width in seconds.
    dt         : time step between samples.
    k          : multiplier on the standard deviation (sensitivity).

    Returns
    -------
    threshold : 1-D array, same length as *signal*.
    """
    win = max(3, int(window_sec / dt))
    n = len(signal)
    threshold = np.empty(n)

    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        chunk = signal[lo:hi]
        threshold[i] = chunk.mean() + k * chunk.std()

    return threshold


def adaptive_threshold_fast(signal: np.ndarray,
                            window_sec: float = 30.0,
                            dt: float = 1.0,
                            k: float = 1.5) -> np.ndarray:
    """Vectorised rolling-window threshold using cumulative sums."""
    win = max(3, int(window_sec / dt))
    n = len(signal)

    # Pad signal for edge handling
    pad = win // 2
    padded = np.pad(signal, pad, mode="edge")

    # Cumulative sum trick for rolling mean
    cs = np.cumsum(padded)
    cs = np.insert(cs, 0, 0.0)
    rolling_mean = (cs[win:] - cs[:-win]) / win

    # Rolling std via E[x^2] - E[x]^2
    cs2 = np.cumsum(padded ** 2)
    cs2 = np.insert(cs2, 0, 0.0)
    rolling_sq_mean = (cs2[win:] - cs2[:-win]) / win
    rolling_var = np.maximum(rolling_sq_mean - rolling_mean ** 2, 0.0)
    rolling_std = np.sqrt(rolling_var)

    # Trim to original length
    rolling_mean = rolling_mean[:n]
    rolling_std = rolling_std[:n]

    return rolling_mean + k * rolling_std
