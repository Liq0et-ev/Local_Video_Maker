"""
Speech-to-Text engine using OpenAI Whisper.
Supports English and Russian with word-level timestamps.
"""

import whisper
import torch


class Transcriber:

    def __init__(self, model_size="medium"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading Whisper '{model_size}' on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print("  Model ready.")

    def detect_language(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected = max(probs, key=probs.get)
        if detected not in ("en", "ru"):
            print(f"  Detected '{detected}', falling back to 'en'.")
            return "en"
        return detected

    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio and return list of segments.

        Each segment: {
            "text": str,
            "start": float,
            "end": float,
            "language": str,
            "words": [{"word": str, "start": float, "end": float}, ...]
        }
        """
        if language is None:
            language = self.detect_language(audio_path)
            print(f"  Detected language: {language}")

        print(f"  Transcribing ({language})...")
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        for seg in result["segments"]:
            words = []
            if "words" in seg:
                for w in seg["words"]:
                    words.append({
                        "word": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })
            segments.append({
                "text": seg["text"].strip(),
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "language": language,
                "words": words,
            })

        total_words = sum(len(s["words"]) for s in segments)
        print(f"  Done: {len(segments)} segments, {total_words} words.")
        return segments

    def check_audio_quality(self, segments):
        """Flag timestamps where audio may be unclear."""
        warnings = []
        for seg in segments:
            for i, word in enumerate(seg["words"]):
                duration = word["end"] - word["start"]
                if duration < 0.05 and len(word["word"]) > 3:
                    warnings.append(
                        f"  [{word['start']:.1f}s] Word '{word['word']}' has "
                        f"very short duration ({duration:.3f}s) -- may be misheard"
                    )
                if i > 0:
                    gap = word["start"] - seg["words"][i - 1]["end"]
                    if gap > 2.0:
                        warnings.append(
                            f"  [{word['start']:.1f}s] Large gap ({gap:.1f}s) before "
                            f"'{word['word']}' -- audio may be unclear"
                        )
        return warnings


_CACHE: dict[str, "Transcriber"] = {}


def get_transcriber(model_size: str = "medium") -> "Transcriber":
    """Load each Whisper model once per process (loading takes seconds to minutes)."""
    if model_size not in _CACHE:
        _CACHE.clear()  # keep at most one model in memory
        _CACHE[model_size] = Transcriber(model_size)
    return _CACHE[model_size]
