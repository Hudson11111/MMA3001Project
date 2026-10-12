"""Tests for matching and average precision — hand-computed cases."""

import numpy as np
import pytest

from porkqc.metrics.ap import (
    ap_coco101,
    ap_exact,
    ap_ultralytics,
    ap_voc11,
    average_precision,
    interpolated_precision,
    pr_curve,
    precision_envelope,
)
from porkqc.metrics.matching import match_image

T05 = np.array([0.5])


def box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------- matching

def test_single_match_depends_on_threshold():
    gt = np.array([box(0, 0)])
    pred = np.array([[2, 0, 12, 10]])  # IoU = 80 / 120 = 0.667
    out = match_image([0], [0.9], pred, [0], gt, np.array([0.5, 0.75]))
    assert out.tolist() == [[True, False]]


def test_duplicate_detection_only_highest_confidence_counts():
    gt = np.array([box(0, 0)])
    preds = np.array([box(0, 0), box(0, 0)])
    out = match_image([0, 0], [0.3, 0.9], preds, [0], gt, T05)
    assert out[:, 0].tolist() == [False, True]  # the 0.9 one claims the box


def test_wrong_class_is_false_positive():
    out = match_image([1], [0.9], np.array([box(0, 0)]), [0], np.array([box(0, 0)]), T05)
    assert not out.any()


def test_empty_inputs():
    assert match_image([], [], np.zeros((0, 4)), [0], np.array([box(0, 0)]), T05).shape == (0, 1)
    assert not match_image([0], [0.5], np.array([box(0, 0)]), [], np.zeros((0, 4)), T05).any()


def test_confidence_and_ultralytics_rules_can_differ():
    # GT A at x=0, GT B at x=20. p1 (0.9) overlaps only B strongly; p2 (0.8) overlaps
    # B best but also A above threshold.
    A, B = box(0, 0, 20, 10), box(20, 0, 20, 10)
    gt = np.array([A, B])
    p1 = box(22, 0, 20, 10)          # IoU(B) = 18*10/(400-180)... high
    p2 = [8, 0, 36, 10]              # spans part of A and most of B
    preds = np.array([p1, p2])
    iou_tp_conf = match_image([0, 0], [0.9, 0.8], preds, [0, 0], gt, np.array([0.3]), rule="confidence")
    iou_tp_ul = match_image([0, 0], [0.9, 0.8], preds, [0, 0], gt, np.array([0.3]), rule="ultralytics")
    # Sanity: p2 overlaps both boxes above 0.3, with B the better match.
    from porkqc.metrics.iou import box_iou
    m = box_iou(gt, preds)
    assert m[1, 1] > m[0, 1] >= 0.3 and m[1, 0] > m[1, 1]
    # Confidence rule: p1 takes B, p2 falls back to A -> 2 TP.
    assert iou_tp_conf[:, 0].tolist() == [True, True]
    # Ultralytics rule: p2 keeps only its best pair (B), which p1 owns -> 1 TP.
    assert iou_tp_ul[:, 0].sum() == 1


def test_match_rejects_bad_input():
    with pytest.raises(ValueError):
        match_image([0, 1], [0.5], np.zeros((2, 4)), [], np.zeros((0, 4)), T05)
    with pytest.raises(ValueError):
        match_image([0], [0.5], np.array([box(0, 0)]), [0], np.array([box(0, 0)]), T05, rule="hungarian")


# ---------------------------------------------------------------- PR curve and AP

def test_pr_curve_known_values():
    r, p = pr_curve([1, 0, 1], [0.9, 0.8, 0.7], n_gt=2)
    np.testing.assert_allclose(r, [0.5, 0.5, 1.0])
    np.testing.assert_allclose(p, [1.0, 0.5, 2 / 3])


def test_pr_curve_sorts_by_score():
    r, p = pr_curve([1, 0, 1], [0.7, 0.8, 0.9], n_gt=2)  # ranked: TP, FP, TP
    np.testing.assert_allclose(r, [0.5, 0.5, 1.0])


def test_pr_curve_errors():
    with pytest.raises(ValueError):
        pr_curve([1], [0.5], n_gt=0)
    with pytest.raises(ValueError):
        pr_curve([1, 1], [0.5, 0.4], n_gt=1)


def test_envelope_is_running_max_from_right():
    np.testing.assert_allclose(precision_envelope([1.0, 0.5, 2 / 3, 0.4]), [1.0, 2 / 3, 2 / 3, 0.4])


def test_interpolated_precision_step_function():
    r, p = [0.5, 0.5, 1.0], [1.0, 0.5, 2 / 3]
    q = interpolated_precision(r, p, np.array([0.0, 0.5, 0.51, 1.0]))
    np.testing.assert_allclose(q, [1.0, 1.0, 2 / 3, 2 / 3])
    # beyond the maximum recall reached, precision is 0
    assert interpolated_precision([0.5], [1.0], np.array([0.6]))[0] == 0.0


def test_ap_rules_hand_computed():
    r, p = [0.5, 0.5, 1.0], [1.0, 0.5, 2 / 3]
    assert ap_exact(r, p) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)          # 0.8333
    assert ap_coco101(r, p) == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)  # 0.8350
    assert ap_voc11(r, p) == pytest.approx((6 * 1 + 5 * 2 / 3) / 11)       # 0.8485


def test_ap_when_recall_never_reaches_one():
    r, p = [0.5], [1.0]  # found half the boxes, perfectly
    assert ap_exact(r, p) == pytest.approx(0.5)
    assert ap_coco101(r, p) == pytest.approx(51 / 101)
    assert ap_voc11(r, p) == pytest.approx(6 / 11)


def test_perfect_detector():
    r, p = pr_curve([1, 1, 1, 1], [0.9, 0.8, 0.7, 0.6], n_gt=4)
    for method in ("exact", "coco101", "voc11"):
        assert average_precision(r, p, method) == pytest.approx(1.0)
    # Ultralytics' rule caps at 0.995: its sentinel drops precision to 0 at recall 1,
    # and the trapezoid over the last 0.01-wide panel averages 1 and 0
    # (the familiar "mAP50 = 0.995" in YOLO training logs). See VERIFICATION_LOG V6.
    assert average_precision(r, p, "ultralytics") == pytest.approx(0.995)


def test_empty_curve_gives_zero():
    assert ap_exact([], []) == 0.0 and ap_coco101([], []) == 0.0


def test_ultralytics_rule_close_to_exact_on_dense_curve():
    rng = np.random.default_rng(1)
    tp = rng.random(2000) < 0.6
    r, p = pr_curve(tp, rng.random(2000), n_gt=int(tp.sum()) + 100)
    assert abs(ap_ultralytics(r, p) - ap_exact(r, p)) < 0.01


def test_unknown_method():
    with pytest.raises(ValueError):
        average_precision([1.0], [1.0], "simpsons")
