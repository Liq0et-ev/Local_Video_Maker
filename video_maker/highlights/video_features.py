"""
Video Feature Extraction Module
================================
Extracts visual dynamics from video frames using:
  - Frame differencing (motion intensity)
  - Dense optical flow (Farneback method)

Outputs a per-second feature vector that quantifies how much visual
change is happening at each moment.
"""

import cv2
import numpy as np


def extract_video_features(video_path: str, sample_fps: float = 2.0):
    """
    Analyse a video file and return two time-aligned 1-D signals:

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    sample_fps : float
        How many frames per second to actually analyse.  Lower values
        speed up processing; 2 fps is a good balance for 30-min videos.

    Returns
    -------
    timestamps : np.ndarray   - time in seconds for each sample
    motion     : np.ndarray   - normalised frame-difference magnitude
    flow       : np.ndarray   - mean optical-flow magnitude
    native_fps : float        - original FPS of the video
    total_dur  : float        - total duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / native_fps if native_fps > 0 else 0.0

    # How many native frames to skip between samples
    skip = max(1, int(round(native_fps / sample_fps)))

    prev_gray = None
    timestamps, motion_vals, flow_vals = [], [], []
    frame_idx = 0

    print(f"[video] Analysing {total_dur:.1f}s video at ~{sample_fps} sample-fps "
          f"(every {skip} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Down-scale for speed (max 480px wide)
            h, w = gray.shape
            if w > 480:
                scale = 480 / w
                gray = cv2.resize(gray, (480, int(h * scale)))

            t = frame_idx / native_fps
            timestamps.append(t)

            if prev_gray is not None:
                # --- Motion intensity (absolute frame difference) ---
                diff = cv2.absdiff(gray, prev_gray)
                motion_vals.append(np.mean(diff))

                # --- Dense optical flow (Farneback) ---
                flow_map = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, _ = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
                flow_vals.append(np.mean(mag))
            else:
                motion_vals.append(0.0)
                flow_vals.append(0.0)

            prev_gray = gray

        frame_idx += 1

        # Progress indicator every ~60 seconds of video
        if frame_idx % (int(native_fps) * 60) == 0:
            pct = frame_idx / total_frames * 100
            print(f"  ... {pct:.0f}% processed")

    cap.release()
    print(f"[video] Done - {len(timestamps)} samples extracted.")

    return (
        np.array(timestamps),
        np.array(motion_vals, dtype=np.float64),
        np.array(flow_vals, dtype=np.float64),
        native_fps,
        total_dur,
    )
