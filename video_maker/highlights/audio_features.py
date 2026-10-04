"""
Audio Feature Extraction Module
================================
Extracts auditory dynamics using Librosa:
  - RMS energy       - overall loudness envelope
  - Spectral flux    - rate of spectral change (timbral surprise)

Returns time-aligned 1-D signals at a configurable hop rate.
"""

import subprocess
import os
import tempfile

import numpy as np
import librosa


def extract_audio_features(video_path: str, sr: int = 22050, hop_length: int = 512):
    """
    Extract audio-domain features from a video file.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    sr : int
        Target sample rate for audio analysis.
    hop_length : int
        Librosa hop length (samples).  Controls time resolution.

    Returns
    -------
    timestamps   : np.ndarray - centre time of each analysis frame
    rms_energy   : np.ndarray - root-mean-square energy
    spectral_flux: np.ndarray - frame-to-frame spectral difference
    """
    # --- Step 1: demux audio to a temp WAV via ffmpeg ---
    fd, tmp_wav = tempfile.mkstemp(prefix="_dyn_audio_", suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
        tmp_wav,
    ]
    print("[audio] Extracting audio track with ffmpeg...")
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Step 2: load waveform ---
    try:
        if proc.returncode != 0 or os.path.getsize(tmp_wav) < 100:
            # No audio stream: treat as silence so video-only analysis still works
            print("[audio] No usable audio track - using silence.")
            y = np.zeros(sr, dtype=np.float32)
        else:
            y, _ = librosa.load(tmp_wav, sr=sr, mono=True)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    duration = len(y) / sr
    print(f"[audio] Loaded {duration:.1f}s of audio at {sr} Hz.")

    # --- Step 3: RMS energy ---
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # --- Step 4: Spectral flux (onset strength as proxy) ---
    # Spectral flux measures how rapidly the spectrum changes between
    # adjacent frames - a strong indicator of new auditory events.
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    # Pad to match rms length (diff loses one frame)
    flux = np.concatenate([[0.0], flux])

    # Ensure equal lengths
    min_len = min(len(rms), len(flux))
    rms = rms[:min_len]
    flux = flux[:min_len]

    timestamps = librosa.frames_to_time(
        np.arange(min_len), sr=sr, hop_length=hop_length
    )

    print(f"[audio] Done - {min_len} frames extracted.")
    return timestamps, rms.astype(np.float64), flux.astype(np.float64)
