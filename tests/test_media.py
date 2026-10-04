import random

import pytest

from video_maker import audio_mix
from video_maker.config import PipelineOptions
from video_maker.ffmpeg_utils import concat_clips, cut_clip, probe
from video_maker.pipeline import Cancelled, run_pipeline


@pytest.mark.parametrize("mode,expected", [("blur", (1080, 1920)), ("crop", (1080, 1920)), ("off", (640, 360))])
def test_cut_clip_reframing(media, tmp_path, mode, expected):
    out = tmp_path / "c.mp4"
    cut_clip(media["wide"], 2.0, 6.0, out, mode)
    info = probe(out)
    assert (info["width"], info["height"]) == expected
    assert info["duration"] == pytest.approx(4.0, abs=0.2)
    assert info["has_audio"]


def test_cut_clip_handles_source_without_audio(media, tmp_path):
    out = tmp_path / "c.mp4"
    cut_clip(media["silent"], 0, 5, out, "blur")
    info = probe(out)
    assert info["has_video"] and not info["has_audio"]


def test_concat_of_cut_clips_keeps_total_duration(media, tmp_path):
    parts = []
    for i, (s, e) in enumerate([(0, 3), (10, 14)]):
        p = tmp_path / f"p{i}.mp4"
        cut_clip(media["wide"], s, e, p, "blur")
        parts.append(p)
    out = tmp_path / "joined.mp4"
    concat_clips(parts, out, tmp_path / "list.txt")
    assert probe(out)["duration"] == pytest.approx(7.0, abs=0.3)


def test_list_tracks_filters_by_extension(media):
    names = sorted(p.name for p in audio_mix.list_tracks(media["music"]))
    assert names == ["a.mp3", "b.wav"]
    assert audio_mix.list_tracks(media["music"] / "missing") == []


def test_pick_tracks_covers_duration_without_immediate_repeat(media):
    tracks = audio_mix.list_tracks(media["music"])
    chain = audio_mix.pick_tracks(tracks, 40, random.Random(1))
    assert sum(probe(t)["duration"] for t in chain) >= 40
    assert all(a != b for a, b in zip(chain, chain[1:]))


def test_music_mix_keeps_video_and_audio_length(media, tmp_path):
    out = tmp_path / "mixed.mp4"
    assert audio_mix.add_background_music(media["wide"], out, media["music"], 0.3)
    a, b = probe(media["wide"]), probe(out)
    assert b["has_audio"] and b["has_video"]
    assert b["duration"] == pytest.approx(a["duration"], abs=0.5)


def test_music_mix_on_silent_video_adds_audio(media, tmp_path):
    out = tmp_path / "mixed.mp4"
    assert audio_mix.add_background_music(media["silent"], out, media["music"], 0.3)
    assert probe(out)["has_audio"]


def test_music_mix_without_tracks_returns_false(media, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert audio_mix.add_background_music(media["wide"], tmp_path / "x.mp4", empty, 0.3) is False


def _run(media, tmp_path, **opts):
    progress = []
    res = run_pipeline(
        media["wide"], PipelineOptions(subtitles=False, **opts),
        tmp_path / "out", tmp_path / "work", media["music"],
        progress=lambda p, m: progress.append(p),
    )
    return res, progress


def test_pipeline_slice_mode_makes_vertical_shorts_and_cleans_workdir(media, tmp_path):
    random.seed(7)                                       # the slice plan is random: make the test deterministic
    res, progress = _run(media, tmp_path, mode="slice", min_clip_sec=8, max_clip_sec=12, max_clips=2, music=False)
    assert len(res.outputs) == 2                         # a 30 s source gives >= 2 clips, the cap keeps two
    for out in res.outputs:
        info = probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)
    assert progress[-1] == 100 and progress == sorted(progress)
    assert list((tmp_path / "work").iterdir()) == []      # temp files removed


def test_pipeline_slice_with_music_and_without_subtitles(media, tmp_path):
    res, _ = _run(media, tmp_path, mode="slice", min_clip_sec=10, max_clip_sec=14, max_clips=1, music=True)
    assert len(res.outputs) == 1 and probe(res.outputs[0])["has_audio"]


def test_pipeline_notes_missing_music(media, tmp_path):
    empty = tmp_path / "nomusic"
    empty.mkdir()
    res = run_pipeline(media["wide"], PipelineOptions(mode="slice", min_clip_sec=10, max_clip_sec=14,
                       max_clips=1, subtitles=False, music=True),
                       tmp_path / "out", tmp_path / "work", empty)
    assert any("No music" in n for n in res.notes)


def test_pipeline_skips_subtitles_for_silent_clip(media, tmp_path):
    res = run_pipeline(media["silent"], PipelineOptions(mode="slice", min_clip_sec=5, max_clip_sec=8,
                       max_clips=1, subtitles=True, music=False),
                       tmp_path / "out", tmp_path / "work", media["music"])
    assert any("no audio track" in n for n in res.notes)
    assert len(res.outputs) == 1


def test_pipeline_cancel_stops_and_cleans_up(media, tmp_path):
    with pytest.raises(Cancelled):
        run_pipeline(media["wide"], PipelineOptions(mode="slice", min_clip_sec=8, max_clip_sec=12,
                     subtitles=False, music=False), tmp_path / "out", tmp_path / "work", media["music"],
                     should_cancel=lambda: True)
    assert list((tmp_path / "work").iterdir()) == []


def test_pipeline_rejects_missing_and_invalid(media, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "nope.mp4", PipelineOptions(), tmp_path / "o", tmp_path / "w", media["music"])
    with pytest.raises(ValueError):
        run_pipeline(media["wide"], PipelineOptions(mode="bad"), tmp_path / "o", tmp_path / "w", media["music"])


def test_pipeline_highlight_mode_end_to_end(media, tmp_path):
    res = run_pipeline(media["wide"], PipelineOptions(mode="highlight", target_sec=10, subtitles=False, music=False,
                       threshold_k=0.5), tmp_path / "out", tmp_path / "work", media["music"])
    assert len(res.outputs) == 1
    assert probe(res.outputs[0])["height"] == 1920
    assert res.diagnostics_png and res.diagnostics_png.exists()
