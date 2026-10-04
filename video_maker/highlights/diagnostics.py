"""
Diagnostic Plotting Module
============================
Produces a multi-panel Matplotlib figure showing:

  Panel 1 - Normalised signals (motion, flow, RMS, flux)
  Panel 2 - Composite signal with adaptive threshold + detected peaks
  Panel 3 - Extraction map (which time regions will be clipped)

Saved as a PNG alongside the output clips.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_diagnostics(
    timestamps: np.ndarray,
    motion: np.ndarray,
    flow: np.ndarray,
    rms: np.ndarray,
    flux: np.ndarray,
    composite: np.ndarray,
    threshold: np.ndarray,
    mask: np.ndarray,
    intervals: list,
    raw_intervals: list,
    video_duration: float,
    output_path: str,
    variant_name: str = "Weighted Heuristic",
):
    """Generate and save the 3-panel diagnostic figure."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Dynamic Highlight Extraction - {variant_name}",
                 fontsize=14, fontweight="bold")

    t = timestamps

    # ---- Panel 1: Raw normalised signals ----
    ax = axes[0]
    ax.plot(t, motion, label="Motion Intensity", alpha=0.8, linewidth=0.8)
    ax.plot(t, flow, label="Optical Flow", alpha=0.8, linewidth=0.8)
    ax.plot(t, rms, label="RMS Energy", alpha=0.8, linewidth=0.8)
    ax.plot(t, flux, label="Spectral Flux", alpha=0.8, linewidth=0.8)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Multimodal Feature Signals (normalised & smoothed)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Composite + threshold + peaks ----
    ax = axes[1]
    ax.plot(t, composite, label="Composite Signal", color="steelblue",
            linewidth=1.0)
    ax.plot(t, threshold, label="Adaptive Threshold", color="crimson",
            linewidth=1.0, linestyle="--")

    # Mark detected peak regions
    peak_indices = np.where(mask)[0]
    if len(peak_indices) > 0:
        ax.fill_between(t, 0, composite, where=mask,
                        color="orange", alpha=0.3, label="Detected Peaks")

    ax.set_ylabel("Composite Value")
    ax.set_title("Composite Signal vs Adaptive Threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Extraction map ----
    ax = axes[2]
    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Extraction Map (green = selected for clip)")

    # Background: full video as grey
    ax.axhspan(0, 1, color="#e0e0e0", zorder=0)

    # Raw intervals (before trimming) as light green
    for s, e in raw_intervals:
        ax.add_patch(Rectangle((s, 0.0), e - s, 1.0,
                               facecolor="#a5d6a7", edgecolor="none",
                               alpha=0.5, zorder=1))

    # Final intervals as solid green
    for s, e in intervals:
        ax.add_patch(Rectangle((s, 0.0), e - s, 1.0,
                               facecolor="#2e7d32", edgecolor="white",
                               linewidth=0.5, alpha=0.85, zorder=2))

    total = sum(e - s for s, e in intervals)
    ax.text(video_duration * 0.02, 0.5,
            f"Clip total: {total:.1f}s  |  Segments: {len(intervals)}",
            fontsize=10, va="center", color="white", fontweight="bold",
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8))

    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Diagnostic figure saved -> {output_path}")
