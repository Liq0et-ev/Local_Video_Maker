import random

import pytest

from video_maker.config import PipelineOptions
from video_maker.slicer import compute_cut_points


@pytest.mark.parametrize("total", [3, 44, 45, 46, 90, 91, 200, 3600, 7261.5])
@pytest.mark.parametrize("seed", range(5))
def test_cuts_are_contiguous_and_respect_minimum(total, seed):
    cuts = compute_cut_points(total, 45, 120, random.Random(seed))
    assert cuts[0][0] == 0.0
    assert cuts[-1][1] == pytest.approx(total)
    for (_, e1), (s2, _) in zip(cuts, cuts[1:]):
        assert e1 == pytest.approx(s2)          # no gaps, no overlaps
    for s, e in cuts:
        assert e > s
        if total >= 45:
            assert e - s >= 45 - 1e-6            # tail is merged, never left as a stub


def test_short_video_is_single_clip():
    assert compute_cut_points(20, 45, 120) == [(0.0, 20)]


def test_same_seed_is_reproducible():
    a = compute_cut_points(1000, 45, 120, random.Random(7))
    b = compute_cut_points(1000, 45, 120, random.Random(7))
    assert a == b


def test_options_roundtrip_and_unknown_keys_ignored():
    o = PipelineOptions(mode="slice", music=False, brand_kit="neon_blue")
    restored = PipelineOptions.from_dict({**o.to_dict(), "removed_in_future": 1})
    assert restored == o
    assert PipelineOptions.from_dict(None) == PipelineOptions()


@pytest.mark.parametrize("kwargs", [
    {"mode": "nope"}, {"variant": "x"}, {"vertical": "stretch"},
    {"music_volume": 1.5}, {"min_clip_sec": 60, "max_clip_sec": 60},
])
def test_options_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        PipelineOptions(**kwargs).validate()


def test_default_options_are_valid():
    PipelineOptions().validate()
