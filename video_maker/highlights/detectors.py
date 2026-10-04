"""
Dynamics Detection Variants
============================
Three strategies to decide *which moments are dynamic*:

  (A) Weighted Heuristic   - simple, interpretable, fast
  (B) Information-Theoretic Surprisal - principled, adaptive
  (C) Multimodal Attention  - learns relative channel importance

All three accept the same normalised + smoothed signals and return
a boolean mask over the common time axis.

-----------------------------------------------------------------
**Recommendation for unstructured video (see §7 Justify):**

Variant (A) Weighted Heuristic is the best default because:
  1. It needs no training data - critical when every video has a
     different structure.
  2. The weights are transparent and tuneable by non-experts.
  3. It performs well across diverse content (lectures, sports,
     vlogs) because the per-channel adaptive threshold already
     handles domain shift.
  4. Surprisal (B) can over-trigger on stylistic edits (jump cuts),
     while Attention (C) needs a learned prior that may not
     generalise.

Variants B and C are provided for experimentation and for users
who want to fine-tune on a specific genre.
-----------------------------------------------------------------
"""

import numpy as np
from .dsp_pipeline import adaptive_threshold_fast


# ================================================================== #
#  (A)  Weighted Heuristic
# ================================================================== #
def detect_weighted_heuristic(
    motion: np.ndarray,
    flow: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    dt: float,
    weights: dict | None = None,
    threshold_k: float = 1.5,
    window_sec: float = 30.0,
    min_duration_sec: float = 1.0,
) -> np.ndarray:
    """
    Combine channels with fixed weights, then compare to an adaptive
    threshold.

    Parameters
    ----------
    weights : dict with keys 'motion', 'flow', 'rms', 'flux'.
              Default: {0.30, 0.30, 0.20, 0.20}.
    threshold_k : how many σ above the rolling mean counts as dynamic.
    min_duration_sec : minimum consecutive duration to count.

    Returns
    -------
    mask : bool array - True where activity exceeds threshold.
    """
    w = weights or {"motion": 0.30, "flow": 0.30, "rms": 0.20, "flux": 0.20}
    composite = (
        w["motion"] * motion
        + w["flow"] * flow
        + w["rms"] * rms
        + w["flux"] * flux
    )

    thresh = adaptive_threshold_fast(composite, window_sec=window_sec,
                                     dt=dt, k=threshold_k)
    raw_mask = composite > thresh

    # Enforce minimum activity duration
    return _enforce_min_duration(raw_mask, dt, min_duration_sec)


# ================================================================== #
#  (B)  Information-Theoretic Surprisal
# ================================================================== #
def detect_surprisal(
    motion: np.ndarray,
    flow: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    dt: float,
    threshold_k: float = 1.8,
    window_sec: float = 30.0,
    min_duration_sec: float = 1.0,
) -> np.ndarray:
    """
    Model each channel as a local Gaussian; *surprisal* = −log p(x).
    High surprisal means the current value is unlikely given recent
    history -> something interesting is happening.

    Surprisal for a Gaussian:
        S(x) = 0.5 * ((x - μ) / σ)² + log(σ) + const

    We drop constants and sum across channels.
    """
    channels = [motion, flow, rms, flux]
    win = max(3, int(window_sec / dt))
    n = len(motion)
    surprisal = np.zeros(n)

    for ch in channels:
        for i in range(n):
            lo = max(0, i - win // 2)
            hi = min(n, i + win // 2 + 1)
            chunk = ch[lo:hi]
            mu, sigma = chunk.mean(), chunk.std()
            if sigma < 1e-12:
                sigma = 1e-12
            z = (ch[i] - mu) / sigma
            surprisal[i] += 0.5 * z ** 2 + np.log(sigma)

    # Normalise surprisal itself and apply adaptive threshold
    s_min, s_max = surprisal.min(), surprisal.max()
    if (s_max - s_min) > 1e-12:
        surprisal = (surprisal - s_min) / (s_max - s_min)

    thresh = adaptive_threshold_fast(surprisal, window_sec=window_sec,
                                     dt=dt, k=threshold_k)
    raw_mask = surprisal > thresh
    return _enforce_min_duration(raw_mask, dt, min_duration_sec)


# ================================================================== #
#  (C)  Multimodal Attention
# ================================================================== #
def detect_multimodal_attention(
    motion: np.ndarray,
    flow: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    dt: float,
    threshold_k: float = 1.5,
    window_sec: float = 30.0,
    min_duration_sec: float = 1.0,
) -> np.ndarray:
    """
    Data-driven channel weighting: at each time step, the channel
    whose local derivative (rate of change) is highest receives the
    most weight.  This mimics human attention - we look at whatever
    is changing most right now.

    attention_i(t) = softmax( |d/dt channel_i(t)| )
    composite(t) = Σ  attention_i(t) * channel_i(t)
    """
    channels = np.stack([motion, flow, rms, flux])  # (4, T)

    # Per-channel absolute gradient
    grads = np.abs(np.gradient(channels, dt, axis=1))  # (4, T)

    # Softmax across channels at each time step (temperature = 1)
    # Shift for numerical stability
    grads_shifted = grads - grads.max(axis=0, keepdims=True)
    exp_g = np.exp(grads_shifted)
    attention = exp_g / (exp_g.sum(axis=0, keepdims=True) + 1e-12)  # (4, T)

    composite = (attention * channels).sum(axis=0)  # (T,)

    # Normalise
    c_min, c_max = composite.min(), composite.max()
    if (c_max - c_min) > 1e-12:
        composite = (composite - c_min) / (c_max - c_min)

    thresh = adaptive_threshold_fast(composite, window_sec=window_sec,
                                     dt=dt, k=threshold_k)
    raw_mask = composite > thresh
    return _enforce_min_duration(raw_mask, dt, min_duration_sec)


# ================================================================== #
#  Helpers
# ================================================================== #
def _enforce_min_duration(mask: np.ndarray, dt: float,
                          min_sec: float) -> np.ndarray:
    """Remove active regions shorter than *min_sec*."""
    min_samples = max(1, int(min_sec / dt))
    out = mask.copy()
    in_run = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_run:
            start = i
            in_run = True
        elif not mask[i] and in_run:
            if (i - start) < min_samples:
                out[start:i] = False
            in_run = False
    # Handle run at end
    if in_run and (len(mask) - start) < min_samples:
        out[start:] = False
    return out
