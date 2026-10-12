"""Tests for frame parsing and the time-blocked re-split."""

import numpy as np
import pytest
import yaml

from porkqc.data.audit import frame_number
from porkqc.data.resplit import (
    assign_time_blocks,
    collect_frames,
    materialise_split,
    nearest_other_gap,
    scene_of,
)


@pytest.mark.parametrize("name, expected", [
    ("frame_with_red_2247_jpg.rf.4485d2fc.jpg", 2247),
    ("frame_with_red_12_jpg", 12),
    ("frame_7_png.rf.x.jpg", 7),
    ("tray_photo_jpg.rf.x.jpg", None),
    ("plain.jpg", None),
])
def test_frame_number(name, expected):
    assert frame_number(name) == expected


def test_nearest_other_gap():
    frames = np.array([0, 10, 12, 30, 31])
    splits = np.array(["train", "test", "train", "test", "valid"])
    np.testing.assert_array_equal(nearest_other_gap(frames, splits, "test", ["train"]), [2, 18])
    np.testing.assert_array_equal(nearest_other_gap(frames, splits, "test", ["valid"]), [21, 1])
    assert np.isinf(nearest_other_gap(frames, splits, "test", ["nothing"])).all()


def _frames(n=1000, step=1):
    return {f"f{i}": i for i in range(0, n, step)}


def test_blocks_are_whole_and_fractions_roughly_met():
    frames = _frames()
    a = assign_time_blocks(frames, block_size=100, fractions=(0.7, 0.15, 0.15), gap=0, seed=1)
    by_block = {}
    for name, s in a.items():
        by_block.setdefault(frames[name] // 100, set()).add(s)
    assert all(len(v) == 1 for v in by_block.values())        # no block split across sets
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "valid", "test")}
    assert counts["train"] >= 600 and counts["valid"] >= 100 and counts["test"] >= 100


def test_purge_gap_separates_splits():
    frames = _frames()
    gap = 15
    a = assign_time_blocks(frames, block_size=100, gap=gap, seed=3)
    kept = [(frames[n], s) for n, s in a.items() if s is not None]
    f = np.array([k[0] for k in kept]); s = np.array([k[1] for k in kept])
    for t in ("valid", "test"):
        others = [o for o in ("train", "valid", "test") if o != t]
        assert nearest_other_gap(f, s, t, others).min() > gap
    assert any(v is None for v in a.values())          # something was purged
    # purging removes training frames first: evaluation blocks keep their full size
    full = assign_time_blocks(frames, block_size=100, gap=0, seed=3)
    assert sum(v == "test" for v in a.values()) == sum(v == "test" for v in full.values())


def test_deterministic_and_seed_dependent():
    frames = _frames()
    assert assign_time_blocks(frames, 100, seed=5) == assign_time_blocks(frames, 100, seed=5)
    assert assign_time_blocks(frames, 100, seed=5) != assign_time_blocks(frames, 100, seed=6)


@pytest.mark.parametrize("kw", [dict(block_size=0), dict(block_size=10, gap=-1),
                                dict(block_size=10, fractions=(0.5, 0.5))])
def test_bad_arguments(kw):
    with pytest.raises(ValueError):
        assign_time_blocks(_frames(100), **kw)


def _scene(seed):
    """Smooth random 'photo' 720x540."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((720, 540), Image.Resampling.BICUBIC)


@pytest.fixture
def framed_dataset(tmp_path):
    """Roboflow-style export.

    - frame 1: three colour-augmented copies of ONE scene in train
    - frame 500: two DIFFERENT scenes that share the name (as seen in the real data)
    - frame 2 in valid, frame 900 in test, and one name without a frame number
    """
    pytest.importorskip("PIL")
    from PIL import ImageEnhance
    root = tmp_path / "ds"
    base1, a500, b500 = _scene(1), _scene(2), _scene(3)
    layout = {
        "train": [("frame_with_red_1_jpg.rf.a", base1),
                  ("frame_with_red_1_jpg.rf.b", ImageEnhance.Color(base1).enhance(1.29)),
                  ("frame_with_red_1_jpg.rf.c", ImageEnhance.Brightness(base1).enhance(0.9)),
                  ("frame_with_red_500_jpg.rf.d", a500),
                  ("frame_with_red_500_jpg.rf.e", b500)],
        "valid": [("frame_with_red_2_jpg.rf.f", _scene(4))],
        "test": [("frame_with_red_900_jpg.rf.g", _scene(5)), ("odd_name_jpg.rf.h", _scene(6))],
    }
    for split, items in layout.items():
        for sub in ("images", "labels"):
            (root / split / sub).mkdir(parents=True)
        for n, im in items:
            im.save(root / split / "images" / f"{n}.jpg", quality=95)
            (root / split / "labels" / f"{n}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return root


def test_collect_frames_separates_same_named_different_images(framed_dataset):
    frames, files, unparsed = collect_frames(framed_dataset)
    assert frames == {"frame_with_red_1_jpg": 1, "frame_with_red_2_jpg": 2, "frame_with_red_500_jpg": 500,
                      "frame_with_red_500_jpg#1": 500, "frame_with_red_900_jpg": 900}
    assert len(files["frame_with_red_1_jpg"]) == 3          # augmented copies stay together
    assert len(files["frame_with_red_500_jpg"]) == 1        # different content -> separate scenes
    assert unparsed == ["odd_name_jpg.rf.h.jpg"]
    assert scene_of(files)["frame_with_red_500_jpg.rf.e.jpg"] == "frame_with_red_500_jpg#1"


def test_collect_frames_by_name_only(framed_dataset):
    frames, files, _ = collect_frames(framed_dataset, separate_distinct=False)
    assert len(files["frame_with_red_500_jpg"]) == 2 and "frame_with_red_500_jpg#1" not in files


def test_materialise_split_links_and_dedupes_eval_copies(framed_dataset, tmp_path):
    frames, files, _ = collect_frames(framed_dataset)
    assignment = {"frame_with_red_1_jpg": "test", "frame_with_red_2_jpg": None,
                  "frame_with_red_500_jpg": "train", "frame_with_red_500_jpg#1": "valid",
                  "frame_with_red_900_jpg": "valid"}
    out = tmp_path / "new"
    counts = materialise_split(framed_dataset, out, assignment, ["a"], files=files)
    assert counts == {"train": 1, "valid": 2, "test": 1}   # 3 copies of frame 1 -> 1 in test
    linked = list((out / "test" / "images").iterdir())
    assert len(linked) == 1 and linked[0].is_symlink()
    assert (out / "test" / "labels" / f"{linked[0].stem}.txt").read_text().startswith("0 ")
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["a"] and data["val"].endswith("valid/images")
    # re-running replaces links instead of failing
    assert materialise_split(framed_dataset, out, assignment, ["a"], files=files) == counts
