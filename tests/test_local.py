import shutil
from pathlib import Path

import pytest

from video_maker.batch import ProcessingLog, discover_videos, parse_selection, run_batch
from video_maker.cli import apply_overrides, build_parser, main
from video_maker.config import LocalConfig, PipelineOptions, load_config


# ------------------------------------------------------------------ config.ini
def write_ini(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_config_ini_is_valid_and_matches_defaults():
    cfg = load_config()
    assert cfg.options == PipelineOptions()
    assert cfg.whisper_model == "medium"
    assert cfg.input_dir.name == "input" and cfg.output_dir.name == "output"


def test_paths_resolve_relative_to_config_file_and_absolute_kept(tmp_path):
    abs_dir = tmp_path / "abs_music"
    (tmp_path / "cfg").mkdir()
    ini = write_ini(tmp_path / "cfg" / "c.ini",
                    f"[paths]\ninput_dir = vids   # comment\nmusic_dir = {abs_dir}\noutput_dir = ~/out_x\n")
    cfg = load_config(ini)
    assert cfg.input_dir == (tmp_path / "cfg" / "vids").resolve()
    assert cfg.music_dir == abs_dir.resolve()
    assert cfg.output_dir == (Path.home() / "out_x").resolve()


def test_values_inline_comments_percent_and_decimal_comma(tmp_path):
    ini = write_ini(tmp_path / "c.ini", """
[pipeline]
mode = SLICE            # case-insensitive
min_clip_seconds = 20,5
max_clip_seconds = 30
[subtitles]
enabled = no
language = RU
[music]
enabled = false
volume = 0,35
[paths]
input_dir = 100%_real
""")
    cfg = load_config(ini)
    o = cfg.options
    assert o.mode == "slice" and o.min_clip_sec == 20.5 and o.subtitles is False
    assert o.language == "ru" and o.music is False and o.music_volume == 0.35
    assert cfg.input_dir.name == "100%_real"


def test_missing_sections_fall_back_to_defaults(tmp_path):
    cfg = load_config(write_ini(tmp_path / "c.ini", ""))
    assert cfg.options == PipelineOptions()
    assert cfg.input_dir == (tmp_path / "input").resolve()


@pytest.mark.parametrize("body", [
    "[pipeline]\nmode = turbo\n", "[music]\nvolume = 2\n",
    "[pipeline]\nmin_clip_seconds = 90\nmax_clip_seconds = 60\n", "[pipeline]\nmax_clips = 0\n",
    "[pipeline]\ntarget_seconds = abc\n",
])
def test_invalid_values_raise_value_error(tmp_path, body):
    with pytest.raises(ValueError):
        load_config(write_ini(tmp_path / "c.ini", body))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.ini")


# ------------------------------------------------------------------ selection
@pytest.mark.parametrize("text,total,expected", [
    ("all", 4, [0, 1, 2, 3]), ("*", 2, [0, 1]), ("3", 5, [2]), ("1,3,5", 5, [0, 2, 4]),
    ("2-4", 5, [1, 2, 3]), ("4-2", 5, [1, 2, 3]), ("1,3-4,8", 5, [0, 2, 3]), (" 2 , 2 ", 3, [1]),
    ("0", 3, []), ("9", 3, []), ("x,2", 3, [1]), ("", 3, []), ("a-b", 3, []),
])
def test_parse_selection(text, total, expected):
    assert parse_selection(text, total) == expected


# ------------------------------------------------------------------ discovery
def test_discover_videos_filters_and_sorts(tmp_path):
    for n in ("b.mp4", "A.MKV", "c.txt", "d.mov"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "z.mp4").write_bytes(b"x")
    assert [p.name for p in discover_videos(tmp_path)] == ["A.MKV", "b.mp4", "d.mov"]
    with pytest.raises(FileNotFoundError):
        discover_videos(tmp_path / "missing")


# ------------------------------------------------------------------ processing log
def test_processing_log_roundtrip_mode_awareness_and_corruption(tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"123")
    out = tmp_path / "o.mp4"; out.write_bytes(b"x")
    log = ProcessingLog(tmp_path)
    hl, sl = PipelineOptions(mode="highlight"), PipelineOptions(mode="slice")
    assert not log.is_done(video, hl)
    log.record(video, hl, [out])
    assert ProcessingLog(tmp_path).is_done(video, hl)          # persisted
    assert not ProcessingLog(tmp_path).is_done(video, sl)      # other mode is a new job
    out.unlink()
    assert not ProcessingLog(tmp_path).is_done(video, hl)      # result deleted -> redo
    video.write_bytes(b"1234")
    out.write_bytes(b"x")
    assert not ProcessingLog(tmp_path).is_done(video, hl)      # source changed -> redo
    (tmp_path / "processing_log.json").write_text("{not json")
    assert not ProcessingLog(tmp_path).is_done(video, hl)      # corrupt log never blocks


# ------------------------------------------------------------------ batch (real ffmpeg)
def make_cfg(tmp_path, media, **opts) -> LocalConfig:
    inp = tmp_path / "in"; inp.mkdir(exist_ok=True)
    return LocalConfig(
        input_dir=inp, music_dir=media["music"], output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        whisper_model="tiny",
        options=PipelineOptions(mode="slice", min_clip_sec=8, max_clip_sec=12, max_clips=1,
                                subtitles=False, music=False, **opts),
    )


def test_batch_processes_skips_and_forces(media, tmp_path):
    cfg = make_cfg(tmp_path, media)
    shutil.copy(media["wide"], cfg.input_dir / "one.mp4")
    videos = discover_videos(cfg.input_dir)
    msgs = []
    first = run_batch(videos, cfg, log_fn=msgs.append)
    assert [r.status for r in first] == ["done"] and first[0].outputs[0].exists()
    second = run_batch(videos, cfg, log_fn=msgs.append)
    assert [r.status for r in second] == ["skipped"]
    third = run_batch(videos, cfg, force=True, log_fn=msgs.append)
    assert [r.status for r in third] == ["done"]
    assert list(cfg.work_dir.iterdir()) == []


def test_batch_continues_after_a_broken_video_and_can_cancel(media, tmp_path):
    cfg = make_cfg(tmp_path, media)
    (cfg.input_dir / "a_broken.mp4").write_bytes(b"this is not a video")
    shutil.copy(media["wide"], cfg.input_dir / "b_good.mp4")
    results = run_batch(discover_videos(cfg.input_dir), cfg, log_fn=lambda m: None)
    assert [(r.video.name, r.status) for r in results] == [("a_broken.mp4", "failed"), ("b_good.mp4", "done")]
    assert results[0].error
    cancelled = run_batch(discover_videos(cfg.input_dir), cfg, force=True, log_fn=lambda m: None,
                          should_cancel=lambda: True)
    assert [r.status for r in cancelled] == ["cancelled"]


# ------------------------------------------------------------------ CLI
def test_cli_overrides_beat_config_ini(tmp_path):
    cfg = load_config()
    a = build_parser().parse_args(["--mode", "slice", "--no-music", "--min", "10", "--max", "20",
                                   "--kit", "neon_blue", "--lang", "ru", "--output-dir", "res", "--whisper", "base"])
    new = apply_overrides(cfg, a)
    o = new.options
    assert (o.mode, o.music, o.min_clip_sec, o.max_clip_sec, o.brand_kit, o.language) == ("slice", False, 10, 20, "neon_blue", "ru")
    assert new.output_dir == (Path.cwd() / "res").resolve() and new.whisper_model == "base"
    assert new.input_dir == cfg.input_dir and cfg.options.mode == "highlight"   # original untouched


def test_cli_rejects_invalid_override_combination():
    a = build_parser().parse_args(["--min", "100", "--max", "50"])
    with pytest.raises(ValueError):
        apply_overrides(load_config(), a)


def _ini(tmp_path, media, extra="") -> Path:
    inp = tmp_path / "in"; inp.mkdir(exist_ok=True)
    return write_ini(tmp_path / "c.ini", f"""
[paths]
input_dir = {inp}
music_dir = {media['music']}
output_dir = {tmp_path / 'out'}
work_dir = {tmp_path / 'work'}
[pipeline]
mode = slice
min_clip_seconds = 8
max_clip_seconds = 12
max_clips = 1
[subtitles]
enabled = false
[music]
enabled = false
{extra}
""")


def test_cli_end_to_end_list_case_and_exit_codes(media, tmp_path, capsys):
    ini = _ini(tmp_path, media)
    assert main(["--config", str(ini)]) == 0                       # empty input folder is not an error
    shutil.copy(media["wide"], tmp_path / "in" / "clip.mp4")
    assert main(["--config", str(ini), "--list"]) == 0
    assert "clip.mp4" in capsys.readouterr().out
    assert main(["--config", str(ini), "--case", "7"]) == 2        # nothing matches
    assert main(["--config", str(ini), "--case", "all"]) == 0
    assert list((tmp_path / "out" / "clip").glob("*_short_01.mp4"))
    assert main(["--config", str(ini), "--case", "1"]) == 0        # skipped, still success
    assert "skipped" in capsys.readouterr().out
    assert main(["--config", str(tmp_path / "missing.ini")]) == 2


def test_cli_interactive_selection(media, tmp_path, monkeypatch):
    ini = _ini(tmp_path, media)
    shutil.copy(media["wide"], tmp_path / "in" / "clip.mp4")
    answers = iter(["zzz", "1"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert main(["--config", str(ini)]) == 0
    assert list((tmp_path / "out" / "clip").glob("*.mp4"))
    monkeypatch.setattr("builtins.input", lambda _="": "q")
    assert main(["--config", str(ini), "--force"]) == 0


def test_cli_failed_video_gives_nonzero_exit(media, tmp_path):
    ini = _ini(tmp_path, media)
    (tmp_path / "in" / "bad.mp4").write_bytes(b"nope")
    assert main(["--config", str(ini), "--case", "all"]) == 1


# ------------------------------------------------------------------ folders given by the user
def _args(*flags):
    return build_parser().parse_args(list(flags))


def test_folder_precedence_flag_beats_main_py_beats_config_ini(tmp_path):
    cfg = load_config()
    base = tmp_path / "proj"
    folders = {"input_dir": "vids", "music_dir": str(tmp_path / "abs_music"), "output_dir": "", "work_dir": None}
    new = apply_overrides(cfg, _args(), folders, base)
    assert new.input_dir == (base / "vids").resolve()              # relative -> main.py folder
    assert new.music_dir == (tmp_path / "abs_music").resolve()
    assert new.output_dir == cfg.output_dir and new.work_dir == cfg.work_dir   # empty -> config.ini
    flagged = apply_overrides(cfg, _args("--input-dir", "from_flag", "--output-dir", str(tmp_path / "o")),
                              folders, base)
    assert flagged.input_dir == (Path.cwd() / "from_flag").resolve()           # flag wins, relative to cwd
    assert flagged.output_dir == (tmp_path / "o").resolve()
    assert flagged.music_dir == (tmp_path / "abs_music").resolve()


def test_main_py_folders_are_used_end_to_end(media, tmp_path):
    ini = _ini(tmp_path, media)
    other_in = tmp_path / "my_videos"; other_in.mkdir()
    shutil.copy(media["wide"], other_in / "mine.mp4")
    code = main(["--config", str(ini), "--case", "all"],
                folders={"input_dir": str(other_in), "output_dir": str(tmp_path / "my_shorts")})
    assert code == 0
    assert list((tmp_path / "my_shorts" / "mine").glob("*_short_01.mp4"))


def test_missing_input_folder_non_interactive_is_an_error_but_empty_is_not(media, tmp_path):
    ini = _ini(tmp_path, media)
    assert main(["--config", str(ini), "--input-dir", str(tmp_path / "nope"), "--case", "all"]) == 2
    assert main(["--config", str(ini), "--case", "all"]) == 0


def test_interactive_prompt_offers_another_input_folder(media, tmp_path, monkeypatch):
    ini = _ini(tmp_path, media)                                    # configured input folder is empty
    good = tmp_path / "typed"; good.mkdir()
    shutil.copy(media["wide"], good / "typed.mp4")
    monkeypatch.setattr("video_maker.cli._is_interactive", lambda: True)
    answers = iter([str(tmp_path / "does_not_exist"), f'"{good}"', "1"])   # bad path, quoted good path, selection
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert main(["--config", str(ini)]) == 0
    assert list((tmp_path / "out" / "typed").glob("*.mp4"))


def test_interactive_prompt_enter_quits_cleanly(media, tmp_path, monkeypatch):
    ini = _ini(tmp_path, media)
    monkeypatch.setattr("video_maker.cli._is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _="": "")
    assert main(["--config", str(ini)]) == 0


def test_ask_folders_keeps_defaults_on_enter_and_takes_typed_values(media, tmp_path, monkeypatch):
    ini = _ini(tmp_path, media)
    shutil.copy(media["wide"], tmp_path / "in" / "x.mp4")
    new_out = tmp_path / "typed_out"
    monkeypatch.setattr("video_maker.cli._is_interactive", lambda: True)
    answers = iter(["", "", str(new_out), "1"])                    # keep input, keep music, new output, pick #1
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert main(["--config", str(ini), "--ask-folders"]) == 0
    assert list((new_out / "x").glob("*.mp4")) and not (tmp_path / "out" / "x").exists()


# ------------------------------------------------------------------ tuning flags and transition proposal
def test_all_tuning_flags_reach_the_options():
    a = build_parser().parse_args([
        "--weights", "0.5,0.2,0.2,0.1", "--transition", "zoom", "--transition-style", "dynamic",
        "--transition-sec", "0.4", "--quality", "max", "--crf", "15", "--preset", "slow", "--fps", "30",
        "--sharpen", "0.4", "--snap", "0", "--max-segments", "3", "--min-segment", "5",
        "--selection", "proportional", "--no-fill", "--analysis-window", "60", "--min-activity", "2",
        "--smoothing-window", "21", "--median", "4", "--sample-fps", "3", "--audio-fade-ms", "10",
        "--audio-bitrate", "256",
    ])
    o = apply_overrides(load_config(), a).options
    assert (o.weight_motion, o.weight_flow, o.weight_loudness, o.weight_audio_change) == (0.5, 0.2, 0.2, 0.1)
    assert (o.transition, o.transition_style, o.transition_sec) == ("zoom", "dynamic", 0.4)
    assert (o.quality, o.crf, o.preset, o.fps, o.sharpen) == ("max", 15, "slow", 30.0, 0.4)
    assert (o.snap_sec, o.max_segments, o.min_segment_sec, o.selection, o.fill_to_target) == (0.0, 3, 5.0, "proportional", False)
    assert (o.analysis_window_sec, o.min_activity_sec, o.smoothing_window, o.median_sec) == (60, 2, 21, 4)
    assert (o.sample_fps, o.audio_fade_ms, o.audio_bitrate_kbps) == (3.0, 10, 256)


@pytest.mark.parametrize("flags", [["--weights", "1,2,3"], ["--weights", "a,b,c,d"], ["--weights", "0,0,0,0"],
                                   ["--crf", "99"], ["--sharpen", "5"]])
def test_bad_tuning_values_are_rejected(flags):
    with pytest.raises(ValueError):
        apply_overrides(load_config(), build_parser().parse_args(flags))


def test_unknown_choices_are_rejected_by_argparse():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--transition", "wobble"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--quality", "ultra"])


def test_confirm_dialog_applies_edits_reports_errors_and_lists_catalogue(monkeypatch, capsys):
    from video_maker.cli import confirm_transitions
    from video_maker.transitions import plan_from_features

    feats = [{"gap": 10.0, "sim": 0.5, "luma_a": 0.5, "luma_b": 0.5, "energy_a": 0.5, "energy_b": 0.5,
              "loud_a": 0.3, "loud_b": 0.3}] * 2
    joins = [((0, 10), (20, 30)), ((20, 30), (40, 50))]
    plan = plan_from_features(joins, feats, PipelineOptions(transition_style="neutral"))
    answers = iter(["bogus", "9=fade", "list", "1=zoom, 2=fadeblack:0.5", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    result = confirm_transitions(plan)
    out = capsys.readouterr().out
    assert [c.key for c in result.choices] == ["zoom", "fadeblack"] and result.choices[1].seconds == 0.5
    assert "ожидается" in out and "нет стыка" in out and "Доступные переходы" in out and "Предлагаемые переходы" in out
    # Ctrl+C / closed input accepts the proposal untouched
    plan2 = plan_from_features(joins, feats, PipelineOptions(transition_style="neutral"))
    before = [c.key for c in plan2.choices]
    monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(EOFError()))
    assert [c.key for c in confirm_transitions(plan2).choices] == before


def test_cli_proposes_transitions_and_uses_the_answer(activity_video, tmp_path, monkeypatch, capsys):
    inp = tmp_path / "in"; inp.mkdir()
    shutil.copy(activity_video, inp / "act.mp4")
    ini = write_ini(tmp_path / "c.ini", f"""
[paths]
input_dir = {inp}
output_dir = {tmp_path / 'out'}
work_dir = {tmp_path / 'work'}
music_dir = {tmp_path / 'music'}
[pipeline]
mode = highlight
threshold = 0.3
target_seconds = 40
buffer_seconds = 1
[highlight]
analysis_window_seconds = 120
min_segment_seconds = 4
max_segments = 3
fill_to_target = false
[subtitles]
enabled = false
[music]
enabled = false
[video]
quality = draft
""")
    monkeypatch.setattr("video_maker.cli._is_interactive", lambda: True)
    answers = iter(["all=pushup:0.4", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert main(["--config", str(ini), "--case", "all"]) == 0
    out = capsys.readouterr().out
    assert "Предлагаемые переходы" in out
    assert out.count("Вытеснение вверх (0.40s)") >= 1 and "transitions:" in out
    assert list((tmp_path / "out" / "act").glob("*_short.mp4"))
