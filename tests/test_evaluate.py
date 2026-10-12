"""Tests for dataset-level evaluation and bootstrap on small hand-built splits."""

import numpy as np
import pytest

from porkqc.metrics.evaluate import (
    bootstrap,
    counts_at_confidence,
    match_dataset,
    summarise,
)

NAMES = ["a", "b", "c"]


def img(file, boxes):
    return {"file": file, "width": 100, "height": 100, "boxes": boxes}


def pimg(file, pred):
    return {"file": file, "width": 100, "height": 100, "pred": pred}


@pytest.fixture
def split():
    """Class 'a': 2 GT, found once at high confidence plus one FP.
    Class 'b': 1 GT, found. Class 'c': no GT anywhere (AP undefined).
    img3 has no GT (a 'good' tray) but one false positive."""
    gt = {"images": [
        img("img1_jpg.rf.x.jpg", [[0, 10, 10, 30, 30], [1, 50, 50, 70, 70]]),
        img("img2_jpg.rf.y.jpg", [[0, 10, 10, 30, 30]]),
        img("img3_jpg.rf.z.jpg", []),
    ]}
    pred = {"images": [
        pimg("img1_jpg.rf.x.jpg", [[0, 0.9, 10, 10, 30, 30], [1, 0.8, 50, 50, 70, 70]]),
        pimg("img3_jpg.rf.z.jpg", [[0, 0.6, 0, 0, 20, 20]]),
        # img2: no predictions at all (missed defect)
    ]}
    return gt, pred


def test_match_dataset_structure(split):
    gt, pred = split
    m = match_dataset(gt, pred, n_classes=3)
    assert [x.file for x in m] == [e["file"] for e in gt["images"]]
    assert m[0].tp.shape == (2, 10) and m[0].tp.all()   # perfect boxes: TP at all IoUs
    assert m[1].tp.shape == (0, 10)                       # no predictions
    assert m[2].n_gt.tolist() == [0, 0, 0]
    assert m[0].group == "img1_jpg"


def test_summary_values(split):
    gt, pred = split
    s = summarise(match_dataset(gt, pred, 3), NAMES)
    a, b, c = s["per_class"]
    # class a ranked: TP (0.9), FP (0.6); n_gt = 2 -> recall .5 at precision 1 -> exact AP 0.5
    assert a["ap50"] == pytest.approx(0.5) and a["n_gt"] == 2 and a["n_pred"] == 2
    assert b["ap50"] == pytest.approx(1.0)
    assert c["ap50"] is None and c["n_gt"] == 0
    assert s["map50"] == pytest.approx(0.75)  # mean over classes that have GT
    assert s["map50_95"] == pytest.approx(0.75)


def test_counts_at_confidence(split):
    gt, pred = split
    m = match_dataset(gt, pred, 3)
    c = counts_at_confidence(m, 3, conf=0.7)
    assert c["tp"].tolist() == [1, 1, 0]
    assert c["fp"].tolist() == [0, 0, 0]       # the 0.6 FP is below threshold
    assert c["fn"].tolist() == [1, 0, 0]
    c2 = counts_at_confidence(m, 3, conf=0.5)
    assert c2["fp"].tolist() == [1, 0, 0]
    assert c2["precision"][0] == pytest.approx(0.5)
    assert np.isnan(c2["recall"][2])


def test_size_mismatch_detected(split):
    gt, pred = split
    pred["images"][0]["width"] = 200
    with pytest.raises(ValueError, match="size"):
        match_dataset(gt, pred, 3)


def test_unknown_predicted_image(split):
    gt, pred = split
    pred["images"].append(pimg("ghost.jpg", []))
    with pytest.raises(ValueError, match="not in the ground truth"):
        match_dataset(gt, pred, 3)


def test_bootstrap_is_reproducible_and_brackets_point(split):
    gt, pred = split
    m = match_dataset(gt, pred, 3)
    stat = lambda mm: summarise(mm, NAMES)["map50"]  # noqa: E731
    r1 = bootstrap(m, stat, n_boot=200, seed=3)
    r2 = bootstrap(m, stat, n_boot=200, seed=3)
    assert r1 == r2
    assert r1["low"] <= r1["point"] <= r1["high"]
    assert r1["n_boot"] == 200


def test_grouped_bootstrap_keeps_groups_together():
    # Two copies of one photo + one other photo: grouped resampling has 2 units.
    gt = {"images": [img("p_jpg.rf.1.jpg", [[0, 0, 0, 10, 10]]),
                     img("p_jpg.rf.2.jpg", [[0, 0, 0, 10, 10]]),
                     img("q_jpg.rf.3.jpg", [[0, 0, 0, 10, 10]])]}
    pred = {"images": [pimg("p_jpg.rf.1.jpg", [[0, 0.9, 0, 0, 10, 10]]),
                       pimg("p_jpg.rf.2.jpg", [[0, 0.9, 0, 0, 10, 10]])]}
    m = match_dataset(gt, pred, 1)
    n_images = lambda mm: float(len(mm))  # noqa: E731
    r = bootstrap(m, n_images, n_boot=300, seed=0, grouped=True)
    # resample size is 2 groups -> 2, 3 or 4 images; never 1 (copies stay together)
    assert 2.0 <= r["low"] and r["high"] <= 4.0
