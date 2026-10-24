"""Tests for porkqc.metrics.iou — hand-computed values plus invariants."""

import numpy as np
import pytest

from porkqc.metrics.iou import box_area, box_iou

# ---------- positive cases: values computed by hand ----------


def test_identical_boxes_iou_one():
    b = np.array([[10, 20, 50, 80]])
    assert box_iou(b, b)[0, 0] == pytest.approx(1.0)


def test_disjoint_boxes_iou_zero():
    a = np.array([[0, 0, 1, 1]])
    b = np.array([[2, 2, 3, 3]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_touching_boxes_iou_zero():
    a = np.array([[0, 0, 1, 1]])
    b = np.array([[1, 0, 2, 1]])  # share an edge only
    assert box_iou(a, b)[0, 0] == 0.0


def test_partial_overlap_known_value():
    # intersection = 1 x 2 = 2; union = 4 + 4 - 2 = 6 -> IoU = 1/3
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[1, 0, 3, 2]])
    assert box_iou(a, b)[0, 0] == pytest.approx(1 / 3)


def test_contained_box():
    # inner area 1, outer area 16 -> IoU = 1/16
    outer = np.array([[0, 0, 4, 4]])
    inner = np.array([[1, 1, 2, 2]])
    assert box_iou(outer, inner)[0, 0] == pytest.approx(1 / 16)


def test_pairwise_shape_and_symmetry():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, size=(7, 2))
    wh = rng.uniform(1, 30, size=(7, 2))
    a = np.hstack([xy, xy + wh])
    xy2 = rng.uniform(0, 100, size=(4, 2))
    b = np.hstack([xy2, xy2 + rng.uniform(1, 30, size=(4, 2))])
    m = box_iou(a, b)
    assert m.shape == (7, 4)
    np.testing.assert_allclose(m, box_iou(b, a).T)
    assert np.all((m >= 0) & (m <= 1))


def test_degenerate_boxes_give_zero_not_nan():
    p = np.array([[5, 5, 5, 5]])  # zero-area box
    out = box_iou(p, p)
    assert out[0, 0] == 0.0 and not np.isnan(out[0, 0])


def test_empty_inputs():
    assert box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_box_area():
    np.testing.assert_allclose(box_area(np.array([[0, 0, 2, 3], [1, 1, 1, 4]])), [6, 0])


# ---------- negative cases ----------


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros(4), np.zeros((1, 4, 1))])
def test_rejects_wrong_shape(bad):
    with pytest.raises(ValueError):
        box_iou(bad, np.zeros((1, 4)))


def test_rejects_inverted_corners():
    with pytest.raises(ValueError, match="x2 < x1"):
        box_iou(np.array([[5, 0, 1, 1]]), np.array([[0, 0, 1, 1]]))


def test_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        box_area(np.array([[0, 0, np.inf, 1]]))
