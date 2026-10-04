import numpy as np
import pytest

from video_maker.config import QUALITY_PROFILES, EncodeSettings, PipelineOptions, load_config
from video_maker.ffmpeg_utils import _fps_expr, cut_clip, probe, video_filter_chain
from video_maker.highlights.finder import normalise_weights
from video_maker.highlights.temporal import (
    build_extraction_map,
    drop_short_segments,
    expand_to_target,
    segment_scores,
    select_best_segments,
    top_segments,
)
from video_maker.snap import loudness_envelope, snap_cut_points, snap_intervals, snap_time
from video_maker.subtitles.burn import burn_subtitles, resolve_style


# ------------------------------------------------------------------ segment shaping (temporal.py)
def test_defaults_of_build_extraction_map_keep_the_original_behaviour():
    ts = np.arange(0, 100, 0.5)
    mask = np.zeros(len(ts), bool)
    mask[20:40] = True                                  # 10-20 s
    mask[100:120] = True                                # 50-60 s
    intervals, total, raw = build_extraction_map(mask, ts, 100.0, buffer_sec=2, merge_gap_sec=3, target_sec=60)
    assert raw == intervals and len(intervals) == 2 and total == pytest.approx(sum(e - s for s, e in intervals))


def test_min_segment_drops_flicker_but_never_everything():
    assert drop_short_segments([(0, 2), (10, 20)], 4) == [(10, 20)]
    ts = np.arange(0, 60, 0.5)
    mask = np.zeros(len(ts), bool)
    mask[10:14] = True                                  # a single 2 s blip
    intervals, _, _ = build_extraction_map(mask, ts, 60.0, buffer_sec=0, merge_gap_sec=0, target_sec=60,
                                           median_kernel_sec=1.0, min_segment_sec=30)
    assert len(intervals) == 1                          # kept: the longest one is better than nothing


def test_select_best_prefers_strong_segments_and_stays_chronological():
    ivs = [(0, 10), (20, 30), (40, 50), (60, 70)]
    scores = [0.2, 0.9, 0.3, 0.8]
    chosen, total = select_best_segments(ivs, scores, target_sec=20, max_segments=0, min_sec=2)
    assert chosen == [(20, 30), (60, 70)] and total == 20


def test_select_best_trims_only_the_last_segment_and_respects_the_cap():
    ivs = [(0, 10), (20, 30), (40, 50)]
    chosen, total = select_best_segments(ivs, [0.9, 0.8, 0.7], target_sec=25, max_segments=0, min_sec=2)
    assert chosen == [(0, 10), (20, 30), (40, 45)] and total == 25     # trimmed from the end, onset kept
    chosen, _ = select_best_segments(ivs, [0.9, 0.8, 0.7], target_sec=100, max_segments=2, min_sec=2)
    assert chosen == [(0, 10), (20, 30)]


def test_select_best_with_target_shorter_than_any_segment():
    chosen, total = select_best_segments([(0, 30)], [0.5], target_sec=10, max_segments=0, min_sec=20)
    assert chosen == [(0, 10)] and total == 10


def test_top_segments_and_scores():
    ivs = [(0, 5), (10, 15), (20, 25)]
    assert top_segments(ivs, [0.1, 0.9, 0.5], 2) == [(10, 15), (20, 25)]
    assert top_segments(ivs, [0.1, 0.9, 0.5], 0) == ivs
    ts = np.arange(0, 30, 1.0)
    sc = np.where((ts >= 10) & (ts < 15), 1.0, 0.1)
    s = segment_scores(ivs, ts, sc)
    assert s[1] == pytest.approx(1.0) and s[0] == pytest.approx(0.1)


def test_expand_to_target_adds_context_within_the_video():
    ivs, total = expand_to_target([(40, 48)], 30, 100)
    assert total == pytest.approx(30, abs=0.3) and ivs[0][0] < 40 and ivs[0][1] > 48
    ivs, total = expand_to_target([(0, 5)], 30, 100)                    # video border stops growth on one side
    assert ivs[0][0] == 0 and total == pytest.approx(30, abs=0.3)
    ivs, total = expand_to_target([(0, 20), (24, 40)], 100, 50)         # whole video is shorter than the target
    assert ivs == [(0, 50)] and total == 50
    assert expand_to_target([], 30, 100) == ([], 0.0)


def test_build_extraction_map_best_mode_with_cap_and_fill():
    ts = np.arange(0, 200, 0.5)
    mask = np.zeros(len(ts), bool)
    for a, b in ((20, 30), (70, 85), (120, 128), (170, 180)):
        mask[int(a * 2):int(b * 2)] = True
    scores = np.where(mask, 0.5, 0.0)
    scores[int(70 * 2):int(85 * 2)] = 0.9
    ivs, total, raw = build_extraction_map(mask, ts, 200.0, buffer_sec=1, merge_gap_sec=1, target_sec=40,
                                           scores=scores, min_segment_sec=5, max_segments=2, selection="best")
    assert len(ivs) <= 2 and total <= 40 + 1e-6 and len(raw) == 4
    assert any(s <= 70 <= e for s, e in ivs)                              # the strongest burst survives
    filled, ftotal, _ = build_extraction_map(mask, ts, 200.0, buffer_sec=0, merge_gap_sec=0, target_sec=60,
                                             scores=scores, selection="best", max_segments=1, fill_to_target=True)
    assert ftotal == pytest.approx(60, abs=0.5) and len(filled) == 1


# ------------------------------------------------------------------ weights
def test_weights_are_normalised_and_safe():
    w = normalise_weights({"motion": 2, "flow": 2, "rms": 0, "flux": 0})
    assert w == {"motion": 0.5, "flow": 0.5, "rms": 0.0, "flux": 0.0}
    assert normalise_weights({"motion": 0, "flow": 0, "rms": 0, "flux": 0}) == normalise_weights(None)
    assert sum(normalise_weights({"motion": -1, "flow": 3, "rms": 1, "flux": 0}).values()) == pytest.approx(1)


# ------------------------------------------------------------------ snapping cuts to pauses
def fake_env(silence=(3.0, 3.6), seconds=10.0):
    times = np.arange(0.025, seconds, 0.05)
    rms = np.full(len(times), 0.3, dtype=np.float32)
    rms[(times >= silence[0]) & (times <= silence[1])] = 0.001
    return times, rms


def test_snap_time_moves_into_the_pause_and_obeys_limits():
    env = fake_env()
    assert 3.0 <= snap_time(2.5, env, 1.5) <= 3.6
    assert snap_time(6.0, env, 1.5) == pytest.approx(6.0, abs=0.06)         # nothing quiet nearby: stays put
    assert snap_time(2.5, env, 0.2) == pytest.approx(2.5, abs=0.06)         # the pause lies outside the window
    assert snap_time(2.5, env, 1.5, lo=2.6, hi=2.9) >= 2.6                  # bounds win
    assert snap_time(2.5, None, 1.5) == 2.5 and snap_time(2.5, env, 0) == 2.5
    assert snap_time(2.5, env, 1.5, lo=4, hi=3) == 2.5                      # empty range: unchanged


def test_snap_time_prefers_the_closest_of_equally_quiet_moments():
    times = np.arange(0.025, 10, 0.05)
    rms = np.full(len(times), 0.3, dtype=np.float32)
    rms[(times > 2.0) & (times < 2.3)] = 0.0
    rms[(times > 4.4) & (times < 4.7)] = 0.0                                # this pause is closer to t=3.5
    assert 4.4 <= snap_time(3.5, (times, rms), 1.5) <= 4.7


def test_snap_intervals_keeps_order_min_length_and_no_overlap():
    env = fake_env(seconds=100)
    ivs = [(10.0, 20.0), (22.0, 30.0), (60.0, 75.0)]
    out = snap_intervals(ivs, env, 1.5, 100.0, min_len=2.0)
    assert out == sorted(out) and all(e - s >= 2.0 for s, e in out)
    assert all(a[1] <= b[0] for a, b in zip(out, out[1:]))
    assert snap_intervals(ivs, None, 1.5, 100.0) == ivs and snap_intervals([], env, 1.5, 100.0) == []
    assert snap_intervals(ivs, env, 0, 100.0) == ivs


def test_snap_cut_points_stay_contiguous_and_keep_minimum_length():
    env = fake_env(silence=(44.0, 44.6), seconds=200)
    cuts = [(0.0, 45.0), (45.0, 100.0), (100.0, 150.0), (150.0, 200.0)]
    out = snap_cut_points(cuts, env, 2.0, min_len=40)
    assert out[0][0] == 0 and out[-1][1] == 200
    assert all(a[1] == b[0] for a, b in zip(out, out[1:]))
    assert all(e - s >= 40 - 1e-6 for s, e in out)
    assert 44.0 <= out[0][1] <= 44.6                                          # moved into the pause
    assert snap_cut_points(cuts[:1], env, 2.0, 40) == cuts[:1]


def test_loudness_envelope_sees_a_real_pause(tmp_path):
    import subprocess

    wav = tmp_path / "gap.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=300:sample_rate=16000",
                    "-t", "8", "-af", "volume='if(between(t,3,3.6),0,1)':eval=frame", str(wav)], check=True)
    env = loudness_envelope(wav)
    assert env is not None
    snapped = snap_time(2.4, env, 1.5)
    assert 3.0 <= snapped <= 3.6
    assert loudness_envelope(tmp_path / "missing.wav") is None


def test_loudness_envelope_of_silent_video_is_none_or_flat(media):
    assert loudness_envelope(media["silent"]) is None                       # no audio stream at all


# ------------------------------------------------------------------ video quality
def test_quality_profiles_and_overrides():
    assert PipelineOptions(quality="draft").encode_settings().crf == QUALITY_PROFILES["draft"][0]
    e = PipelineOptions(quality="max", crf=12, preset="veryfast", fps=24, sharpen=0.5,
                        audio_bitrate_kbps=256, audio_fade_ms=50).encode_settings()
    assert (e.crf, e.preset, e.fps, e.sharpen, e.audio_bitrate, e.audio_fade_sec) == (12, "veryfast", 24, 0.5, "256k", 0.05)
    crfs = [QUALITY_PROFILES[q][0] for q in ("draft", "standard", "high", "max")]
    assert crfs == sorted(crfs, reverse=True)                               # better profile -> lower crf


@pytest.mark.parametrize("kw", [
    dict(quality="ultra"), dict(crf=60), dict(crf=-1), dict(preset="warp"), dict(fps=5), dict(fps=500),
    dict(sharpen=3), dict(audio_bitrate_kbps=8), dict(snap_sec=9), dict(snap_sec=-1), dict(transition="wobble"),
    dict(transition_style="wild"), dict(transition_sec=5), dict(transition_sec=0), dict(selection="random"),
    dict(max_segments=-1), dict(smoothing_window=3), dict(sample_fps=0), dict(sample_fps=60), dict(audio_fade_ms=900),
    dict(weight_motion=-1), dict(weight_motion=0, weight_flow=0, weight_loudness=0, weight_audio_change=0),
    dict(median_sec=-1), dict(min_segment_sec=-2),
])
def test_new_options_are_validated(kw):
    with pytest.raises(ValueError):
        PipelineOptions(**kw).validate()


def test_new_options_have_sane_defaults():
    PipelineOptions().validate()
    o = PipelineOptions()
    assert o.transition == "ask" and o.fps is None and o.quality == "high" and o.snap_sec > 0


def test_ini_parses_all_new_sections(tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text("""
[highlight]
weight_motion = 0.5
weight_flow = 0,1          # decimal comma is accepted
weight_loudness = 0.2
weight_audio_change = 0.2
analysis_window_seconds = 45
min_activity_seconds = 2
smoothing_window = 21
median_seconds = 5
sample_fps = 3
min_segment_seconds = 6
max_segments = 3
selection = proportional
fill_to_target = no
[cuts]
snap_seconds = 0.5
audio_fade_ms = 10
transition = Pixelize
transition_style = DYNAMIC
transition_seconds = 0.45
[video]
quality = max
crf = 14
preset = slow
fps = 30
sharpen = 0.6
audio_bitrate_kbps = 256
""", encoding="utf-8")
    o = load_config(ini).options
    assert (o.weight_motion, o.weight_flow, o.smoothing_window, o.max_segments) == (0.5, 0.1, 21, 3)
    assert (o.selection, o.fill_to_target, o.snap_sec, o.audio_fade_ms) == ("proportional", False, 0.5, 10)
    assert (o.transition, o.transition_style, o.transition_sec) == ("pixelize", "dynamic", 0.45)
    assert (o.quality, o.crf, o.preset, o.fps, o.sharpen, o.audio_bitrate_kbps) == ("max", 14, "slow", 30.0, 0.6, 256)


@pytest.mark.parametrize("body", [
    "[video]\ncrf = abc\n", "[video]\nquality = bogus\n", "[cuts]\ntransition = spin\n", "[highlight]\nselection = x\n",
    "[video]\nfps = 3\n", "[cuts]\ntransition_seconds = 9\n",
])
def test_ini_rejects_bad_new_values(tmp_path, body):
    ini = tmp_path / "c.ini"
    ini.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(ini)


def test_ini_auto_values_mean_none(tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text("[video]\ncrf = auto\npreset = auto\nfps = source\n[cuts]\ntransition_seconds = auto\n", encoding="utf-8")
    o = load_config(ini).options
    assert (o.crf, o.preset, o.fps, o.transition_sec) == (None, None, None, None)


# ------------------------------------------------------------------ no judder: the source frame rate survives
@pytest.mark.parametrize("raw,expected", [("30000/1001", "30000/1001"), ("25/1", "25/1"), ("24/1", "24/1"),
                                          ("120/1", "60"), ("0/0", "30"), ("1/1", "30"), (None, "30"), ("abc", "30")])
def test_fps_expression(raw, expected):
    assert _fps_expr(raw) == expected


@pytest.mark.parametrize("name,expected", [("fps24", 24.0), ("fps25", 25.0), ("fps2997", 30000 / 1001), ("fps50", 50.0)])
def test_cut_clip_keeps_the_source_frame_rate(fps_videos, tmp_path, name, expected):
    out = tmp_path / "o.mp4"
    cut_clip(fps_videos[name], 0, 3, out, "off", EncodeSettings(crf=28, preset="ultrafast"))
    num, den = probe_rate(out)
    assert num / den == pytest.approx(expected, rel=0.002)


def probe_rate(path):
    import json
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    num, den = json.loads(out)["streams"][0]["avg_frame_rate"].split("/")
    return float(num), float(den)


def test_forced_fps_overrides_the_source(fps_videos, tmp_path):
    out = tmp_path / "o.mp4"
    cut_clip(fps_videos["fps25"], 0, 3, out, "off", EncodeSettings(crf=28, preset="ultrafast", fps=30))
    num, den = probe_rate(out)
    assert num / den == pytest.approx(30.0, rel=0.002)


def test_filter_chain_contents():
    chain = video_filter_chain("blur", EncodeSettings(sharpen=0.5), "25/1")
    assert "fps=25/1" in chain and "unsharp=5:5:0.50" in chain and "lanczos" in chain
    assert "unsharp" not in video_filter_chain("crop", EncodeSettings(), "25/1")
    assert video_filter_chain("off", EncodeSettings(fps=24), "25/1") == "fps=24"


# ------------------------------------------------------------------ encoder settings really reach the file
def test_crf_changes_file_size_and_all_quality_options_run(media, tmp_path):
    sizes = {}
    for crf in (16, 40):
        out = tmp_path / f"crf{crf}.mp4"
        cut_clip(media["wide"], 0, 8, out, "crop", EncodeSettings(crf=crf, preset="ultrafast"))
        sizes[crf] = out.stat().st_size
    assert sizes[16] > sizes[40] * 1.5
    sharp = tmp_path / "sharp.mp4"
    cut_clip(media["wide"], 0, 3, sharp, "blur", EncodeSettings(crf=23, preset="ultrafast", sharpen=1.0,
                                                               audio_fade_sec=0.1, audio_bitrate_kbps=128))
    assert probe(sharp)["has_audio"] and probe(sharp)["height"] == 1920


def test_audio_fades_are_skipped_for_tiny_clips(media, tmp_path):
    out = tmp_path / "t.mp4"
    cut_clip(media["wide"], 0, 0.1, out, "off", EncodeSettings(crf=28, preset="ultrafast", audio_fade_sec=0.05))
    assert probe(out)["has_video"]


def test_subtitle_rendering_honours_the_chosen_quality(media, tmp_path):
    enc_hi, enc_lo = EncodeSettings(crf=14, preset="ultrafast"), EncodeSettings(crf=38, preset="ultrafast")
    src = tmp_path / "src.mp4"
    cut_clip(media["wide"], 0, 4, src, "crop", EncodeSettings(crf=20, preset="ultrafast"))
    words = [{"word": w, "start": 0.2 + i * 0.5, "end": 0.65 + i * 0.5} for i, w in enumerate("quality should be kept".split())]
    segs = [{"text": "x", "start": 0.0, "end": 4.0, "language": "en", "words": words}]
    hi, lo = tmp_path / "hi.mp4", tmp_path / "lo.mp4"
    burn_subtitles(src, hi, segs, resolve_style("viral_yellow"), enc_hi)
    burn_subtitles(src, lo, segs, resolve_style("viral_yellow"), enc_lo)
    assert hi.stat().st_size > lo.stat().st_size * 1.5
    assert probe(hi)["has_audio"] and probe(hi)["height"] == 1920
