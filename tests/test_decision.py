"""Tests for the tray-level decision layer."""

import numpy as np
import pytest

from porkqc.decision import DecisionCosts, choose_thresholds, decide, evaluate_decisions, image_scores


def test_decide_bands():
    out = decide(np.array([0.05, 0.3, 0.6, 0.9]), t_review=0.2, t_fail=0.6)
    assert out.tolist() == ["PASS", "REVIEW", "FAIL", "FAIL"]
    assert decide(np.array([0.5]), 0.5, 0.5).tolist() == ["FAIL"]   # no review band


@pytest.mark.parametrize("tr, tf", [(0.6, 0.2), (-0.1, 0.5), (0.2, 1.1)])
def test_decide_rejects_bad_thresholds(tr, tf):
    with pytest.raises(ValueError):
        decide(np.array([0.5]), tr, tf)


def test_evaluate_counts_and_cost():
    d = np.array(["PASS", "PASS", "REVIEW", "FAIL", "FAIL"])
    y = np.array([True, False, True, False, True])
    r = evaluate_decisions(d, y, DecisionCosts(10, 1, 0.5))
    assert (r["misses"], r["false_rejects"], r["reviews"]) == (1, 1, 1)
    assert r["cost_per_image"] == pytest.approx((10 + 1 + 0.5) / 5)
    assert r["miss_rate"] == pytest.approx(1 / 3) and r["false_reject_rate"] == pytest.approx(1 / 2)
    assert r["auto_fail_precision"] == pytest.approx(0.5)


def test_choose_thresholds_perfectly_separable():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    r = choose_thresholds(scores, y)
    assert r["cost_per_image"] == 0 and r["misses"] == 0 and r["reviews"] == 0
    assert 0.2 < r["t_fail"] <= 0.8


def test_choose_thresholds_uses_review_band_when_scores_overlap():
    # Clean and defective scores overlap in the middle: with misses 10x as costly
    # as false rejects and reviews cheap, the optimum sends the overlap to review.
    scores = np.array([0.05, 0.4, 0.5, 0.45, 0.55, 0.95])
    y = np.array([False, False, True, True, False, True])
    r = choose_thresholds(scores, y, DecisionCosts(10, 1, 0.25))
    assert r["misses"] == 0 and r["reviews"] >= 1
    assert r["cost_per_image"] < evaluate_decisions(decide(scores, 0.5, 0.5), y)["cost_per_image"]


def test_image_scores_class_filter():
    gt = {"images": [{"file": "a", "boxes": [[1, 0, 0, 1, 1]]}, {"file": "b", "boxes": [[3, 0, 0, 1, 1]]}]}
    pred = {"images": [{"file": "a", "pred": [[1, 0.7, 0, 0, 1, 1], [3, 0.9, 0, 0, 1, 1]]}, {"file": "b", "pred": []}]}
    s, y, f = image_scores(gt, pred)
    assert s.tolist() == [0.9, 0.0] and y.tolist() == [True, True] and f == ["a", "b"]
    s, y, _ = image_scores(gt, pred, classes=[1])
    assert s.tolist() == [0.7, 0.0] and y.tolist() == [True, False]
