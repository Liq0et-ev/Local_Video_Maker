import pytest

from video_maker.config import EncodeSettings, PipelineOptions
from video_maker.ffmpeg_utils import crossfade_clips, cut_clip, probe
from video_maker.pipeline import run_pipeline
from video_maker.transition_catalog import CATALOG, KEYS
from video_maker.transitions import (
    CUT_SECONDS,
    apply_user_edits,
    available_xfade,
    detect_style,
    format_catalog,
    format_plan,
    pick_duration,
    plan_from_features,
    plan_transitions,
    rank_transitions,
    usable_keys,
)


def feat(**kw) -> dict:
    base = {"gap": 10.0, "sim": 0.5, "luma_a": 0.5, "luma_b": 0.5, "energy_a": 0.5, "energy_b": 0.5,
            "loud_a": 0.3, "loud_b": 0.3}
    base.update(kw)
    return base


def top(f, style="neutral") -> str:
    return rank_transitions(f, style)[0][0]


# ------------------------------------------------------------------ catalogue
def test_every_catalogue_effect_is_supported_by_local_ffmpeg():
    have = available_xfade()
    assert "fade" in have
    assert set(usable_keys()) == set(KEYS)           # nothing silently disappears on this machine
    assert CATALOG["cut"].xfade is None
    assert all(CATALOG[k].xfade in have for k in KEYS if k != "cut")


def test_catalogue_entries_are_documented():
    for t in CATALOG.values():
        assert t.ru and t.analog and t.when and t.family
    assert "pushup" in format_catalog() and "fadeblack" in format_catalog()


# ------------------------------------------------------------------ rules of the advisor
def test_calm_similar_shots_get_a_soft_transition():
    assert top(feat(sim=0.9, energy_a=0.1, energy_b=0.1)) in ("fade", "blur", "dissolve")


def test_continuous_moments_get_gentle_join_not_an_effect():
    assert top(feat(gap=0.5, sim=0.8)) in ("fade", "cut", "dissolve")


def test_energy_drop_gets_a_pause_through_black():
    assert top(feat(energy_a=0.8, energy_b=0.1)) == "fadeblack"


def test_energy_rise_gets_an_impact_in_dynamic_style_but_not_in_calm_style():
    f = feat(gap=3, energy_a=0.2, energy_b=0.8)
    assert top(f, "dynamic") == "zoom"
    assert not CATALOG[top(f, "calm")].flashy


def test_busy_scene_change_gets_wipe_or_push():
    assert top(feat(energy_a=0.8, energy_b=0.8, sim=0.2), "dynamic") in ("wipeleft", "pushup")


def test_dark_shots_fade_through_black_and_never_flash_white():
    f = feat(luma_a=0.1, luma_b=0.1, sim=0.4)
    ranked = {k: s for k, s, _ in rank_transitions(f, "neutral")}
    assert top(f) == "fadeblack" and ranked["fadewhite"] < 0


def test_similar_shots_do_not_get_arbitrary_wipes():
    ranked = {k: s for k, s, _ in rank_transitions(feat(sim=0.9, gap=4), "neutral")}
    assert ranked["wipeleft"] < ranked["fade"]


def test_every_ranking_covers_all_effects_and_explains_itself():
    ranked = rank_transitions(feat(), "neutral")
    assert {k for k, _, _ in ranked} == set(KEYS)
    assert [s for _, s, _ in ranked] == sorted((s for _, s, _ in ranked), reverse=True)


def test_duration_follows_tempo_and_protects_speech():
    calm = pick_duration("fade", feat(energy_a=0.1, energy_b=0.1), "calm", None)
    fast = pick_duration("fade", feat(energy_a=0.9, energy_b=0.9), "dynamic", None)
    assert fast < calm
    assert pick_duration("fade", feat(loud_a=0.9, loud_b=0.9), "calm", None) <= 0.25
    assert pick_duration("cut", feat(), "calm", None) == CUT_SECONDS
    assert pick_duration("fade", feat(), "calm", 0.5) == 0.5            # explicit value wins
    assert 0.15 <= pick_duration("dissolve", feat(), "dynamic", None) <= 0.9


def test_style_detection_from_overall_energy():
    assert detect_style([feat(energy_a=0.8, energy_b=0.7)] * 3) == "dynamic"
    assert detect_style([feat(energy_a=0.1, energy_b=0.1)] * 3) == "calm"
    assert detect_style([feat(energy_a=0.45, energy_b=0.4)] * 3) == "neutral"
    assert detect_style([]) == "neutral"


# ------------------------------------------------------------------ variety and pacing
def _joins(n):
    return [((i * 20.0, i * 20.0 + 10.0), (i * 20.0 + 20.0, i * 20.0 + 30.0)) for i in range(n)]


@pytest.mark.parametrize("style", ["calm", "neutral", "dynamic"])
def test_plan_never_repeats_an_effect_and_rations_flashy_ones(style):
    n = 8
    busy = [feat(energy_a=0.9, energy_b=0.9, sim=0.2, gap=12)] * n
    plan = plan_from_features(_joins(n), busy, PipelineOptions(transition_style=style))
    keys = [c.key for c in plan.choices]
    assert len(keys) == n
    assert all(a != b for a, b in zip(keys, keys[1:]))
    flashy = [CATALOG[k].flashy for k in keys]
    assert not any(a and b for a, b in zip(flashy, flashy[1:]))          # never two attention-grabbers in a row
    assert sum(flashy) <= {"calm": 0, "neutral": 3, "dynamic": 4}[style]
    assert all(c.reason and 0.15 <= c.seconds <= 0.9 for c in plan.choices)
    assert all(len(c.alternatives) <= 3 and c.key not in c.alternatives for c in plan.choices)


def test_plan_describes_each_join_with_source_times():
    plan = plan_from_features(_joins(2), [feat(), feat()], PipelineOptions())
    assert [(c.at_a, c.at_b) for c in plan.choices] == [(10.0, 20.0), (30.0, 40.0)]
    text = format_plan(plan)
    assert "0:10 -> 0:20" in text and CATALOG[plan.choices[0].key].ru in text


def test_fixed_transition_skips_analysis_and_applies_everywhere():
    plan = plan_transitions("does-not-exist.mp4", [(0, 5), (10, 15), (20, 25)],
                            PipelineOptions(transition="zoom", transition_sec=0.5))
    assert [c.key for c in plan.choices] == ["zoom", "zoom"] and plan.style == "manual"
    assert all(c.seconds == 0.5 for c in plan.choices)
    cut = plan_transitions("x.mp4", [(0, 5), (10, 15)], PipelineOptions(transition="cut"))
    assert cut.choices[0].seconds == CUT_SECONDS and cut.joins() == [("fade", CUT_SECONDS)]


def test_plan_on_a_real_video_measures_every_join(media):
    ivs = [(0, 4), (8, 12), (16, 20), (24, 28)]
    plan = plan_transitions(media["wide"], ivs, PipelineOptions(transition="auto"))
    assert len(plan.choices) == 3
    for c in plan.choices:
        assert c.key in KEYS and c.reason
        assert set(c.features) >= {"gap", "sim", "luma_a", "luma_b", "energy_a", "energy_b"}
        assert -1.0 <= c.features["sim"] <= 1.0 and 0.0 <= c.features["luma_a"] <= 1.0


# ------------------------------------------------------------------ user edits
def make_plan(n=3):
    return plan_from_features(_joins(n), [feat()] * n, PipelineOptions(transition_style="neutral"))


def test_user_can_set_one_join_all_joins_and_duration():
    plan = make_plan(3)
    plan, errors = apply_user_edits(plan, "2=zoom")
    assert errors == [] and plan.choices[1].key == "zoom" and plan.choices[1].reason == "выбрано пользователем"
    plan, errors = apply_user_edits(plan, "all=fadeblack, 3=pushup:0.6")
    assert errors == [] and [c.key for c in plan.choices] == ["fadeblack", "fadeblack", "pushup"]
    assert plan.choices[2].seconds == 0.6
    plan, _ = apply_user_edits(plan, "1=cut")
    assert plan.choices[0].seconds == CUT_SECONDS
    plan, _ = apply_user_edits(plan, "1=fade")
    assert plan.choices[0].seconds > CUT_SECONDS                         # leaving "cut" restores a real duration


def test_user_can_cycle_through_alternatives():
    plan = make_plan(1)
    first = plan.choices[0].key
    seen = {first}
    for _ in range(4):
        plan, errors = apply_user_edits(plan, "1=alt")
        assert errors == []
        seen.add(plan.choices[0].key)
    assert len(seen) >= 3 and first in seen


@pytest.mark.parametrize("text,fragment", [
    ("9=fade", "нет стыка"), ("2=bogus", "неизвестный переход"), ("fade", "ожидается"), ("1=", "ожидается"),
    ("1=fade:abc", "длительность"), ("1=fade:9", "длительность"), ("x=fade", "нет стыка"),
])
def test_bad_user_input_is_reported_and_changes_nothing(text, fragment):
    plan = make_plan(2)
    before = [(c.key, c.seconds) for c in plan.choices]
    plan, errors = apply_user_edits(plan, text)
    assert errors and fragment in errors[0]
    assert [(c.key, c.seconds) for c in plan.choices] == before


def test_valid_parts_are_applied_even_if_another_part_is_wrong():
    plan, errors = apply_user_edits(make_plan(2), "1=zoom, 7=fade, 2=blur")
    assert len(errors) == 1 and [c.key for c in plan.choices] == ["zoom", "blur"]


# ------------------------------------------------------------------ rendering joins with ffmpeg
@pytest.fixture(scope="module")
def clips(media, tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    out = []
    for i, start in enumerate((0, 6, 12, 18)):
        p = d / f"c{i}.mp4"
        cut_clip(media["wide"], start, start + 3.0, p, "off", EncodeSettings(crf=28, preset="ultrafast"))
        out.append(p)
    return out


@pytest.mark.parametrize("key", [k for k in KEYS if k != "cut"])
def test_every_catalogue_effect_renders(clips, tmp_path, key):
    out = tmp_path / "j.mp4"
    assert crossfade_clips(clips[:2], out, [(CATALOG[key].xfade, 0.4)], EncodeSettings(crf=28, preset="ultrafast"))
    info = probe(out)
    assert info["has_video"] and info["has_audio"] and (info["width"], info["height"]) == (640, 360)
    assert info["duration"] == pytest.approx(3.0 + 3.0 - 0.4, abs=0.25)


def test_mixed_joins_have_exactly_the_expected_length_and_stay_in_sync(clips, tmp_path):
    out = tmp_path / "mix.mp4"
    joins = [("wipeleft", 0.4), ("zoomin", 0.3), ("fadeblack", 0.5)]
    assert crossfade_clips(clips, out, joins, EncodeSettings(crf=28, preset="ultrafast"))
    info = probe(out)
    assert info["duration"] == pytest.approx(4 * 3.0 - 1.2, abs=0.3)
    assert info["has_audio"]


def test_too_short_clips_or_wrong_join_count_fall_back(clips, tmp_path, media):
    enc = EncodeSettings(crf=28, preset="ultrafast")
    assert crossfade_clips(clips[:2], tmp_path / "a.mp4", [], enc) is False            # join count mismatch
    assert crossfade_clips(clips[:1], tmp_path / "a.mp4", [], enc) is False
    tiny = tmp_path / "tiny.mp4"
    cut_clip(media["wide"], 0, 0.02, tiny, "off", enc)
    assert crossfade_clips([tiny, clips[0]], tmp_path / "b.mp4", [("fade", 0.5)], enc) is False


def test_overlaps_are_clamped_so_no_clip_is_eaten(clips, tmp_path):
    out = tmp_path / "big.mp4"
    assert crossfade_clips(clips[:3], out, [("fade", 2.0), ("fade", 2.0)], EncodeSettings(crf=28, preset="ultrafast"))
    info = probe(out)
    assert 3 * 3.0 - 2 * 1.35 - 0.3 <= info["duration"] <= 3 * 3.0 - 2 * 0.5   # overlap never exceeds 45 % of a clip


def test_clips_without_audio_are_joined_too(media, tmp_path):
    enc = EncodeSettings(crf=28, preset="ultrafast")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    cut_clip(media["silent"], 0, 4, a, "off", enc)
    cut_clip(media["silent"], 4, 8, b, "off", enc)
    out = tmp_path / "v.mp4"
    assert crossfade_clips([a, b], out, [("pixelize", 0.4)], enc)
    assert probe(out)["has_video"] and not probe(out)["has_audio"]


# ------------------------------------------------------------------ the whole pipeline
def _opts(**kw) -> PipelineOptions:
    base = dict(mode="highlight", threshold_k=0.3, analysis_window_sec=120, buffer_sec=1.0, target_sec=40,
                min_segment_sec=4, max_segments=3, fill_to_target=False, subtitles=False, music=False,
                quality="draft")
    base.update(kw)
    return PipelineOptions(**base)


def test_highlight_pipeline_auto_transitions(activity_video, tmp_path):
    res = run_pipeline(activity_video, _opts(transition="auto"), tmp_path / "o", tmp_path / "w", tmp_path / "m")
    assert len(res.outputs) == 1
    assert len(res.transitions) == 2                                   # three bursts -> two joins
    assert all("(" in t and "s)" in t for t in res.transitions)
    info = probe(res.outputs[0])
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    assert 30 <= info["duration"] <= 42


def test_highlight_pipeline_asks_the_user_and_obeys_the_answer(activity_video, tmp_path):
    seen = []

    def confirm(plan):
        seen.append(len(plan.choices))
        apply_user_edits(plan, "all=fadeblack:0.5")
        return plan

    res = run_pipeline(activity_video, _opts(transition="ask"), tmp_path / "o", tmp_path / "w", tmp_path / "m",
                       confirm_transitions=confirm)
    assert seen == [2]
    assert all("Исчезание через чёрное" in t and "0.50s" in t for t in res.transitions)


def test_ask_mode_without_a_terminal_decides_by_itself(activity_video, tmp_path):
    res = run_pipeline(activity_video, _opts(transition="ask"), tmp_path / "o", tmp_path / "w", tmp_path / "m")
    assert len(res.transitions) == 2


def test_cut_transition_means_hard_cuts_and_no_proposal(activity_video, tmp_path):
    asked = []
    res = run_pipeline(activity_video, _opts(transition="cut"), tmp_path / "o", tmp_path / "w", tmp_path / "m",
                       confirm_transitions=lambda plan: asked.append(plan) or plan)
    assert asked == [] and res.transitions == []
    assert len(res.outputs) == 1


def test_fixed_transition_is_used_for_every_join(activity_video, tmp_path):
    res = run_pipeline(activity_video, _opts(transition="wipeup", transition_sec=0.4),
                       tmp_path / "o", tmp_path / "w", tmp_path / "m")
    assert res.transitions == ["Затирание вверх (0.40s)"] * 2


def test_plan_values_are_plain_python_numbers(activity_video):
    plan = plan_transitions(activity_video, [(18.5, 31.5), (59.0, 72.5)], PipelineOptions(transition="auto"))
    assert all(isinstance(c.seconds, float) and isinstance(c.key, str) for c in plan.choices)
    assert all(type(v) is float for v in plan.choices[0].features.values())
