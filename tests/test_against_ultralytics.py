"""
Verification against a trusted implementation: on identical inputs, our
"ultralytics" matching rule and AP rule must reproduce Ultralytics' own
functions. Skipped automatically where Ultralytics (and torch) are not
installed, e.g. in CI; run in Colab or a full environment.
"""

from types import SimpleNamespace

import numpy as np
import pytest

ul_metrics = pytest.importorskip("ultralytics.utils.metrics")
torch = pytest.importorskip("torch")
from ultralytics.engine.validator import BaseValidator  # noqa: E402

from porkqc.metrics.ap import ap_ultralytics, pr_curve  # noqa: E402
from porkqc.metrics.evaluate import COCO_IOU_THRESHOLDS, ImageMatches, ap_table  # noqa: E402
from porkqc.metrics.iou import box_iou  # noqa: E402
from porkqc.metrics.matching import match_image  # noqa: E402


def random_scene(rng, n_gt=6, n_pred=15, n_classes=3, size=200.0):
    """Ground truth plus noisy, partly duplicated, partly spurious predictions."""
    xy = rng.uniform(0, size - 40, (n_gt, 2))
    wh = rng.uniform(10, 40, (n_gt, 2))
    gt_boxes = np.hstack([xy, xy + wh])
    gt_cls = rng.integers(0, n_classes, n_gt)
    src = rng.integers(0, n_gt, n_pred)
    jitter = rng.normal(0, 4, (n_pred, 4))
    pred_boxes = gt_boxes[src] + jitter
    spurious = rng.random(n_pred) < 0.3
    pred_boxes[spurious] = np.hstack([rng.uniform(0, size - 40, (spurious.sum(), 2))] * 2) + [0, 0, 20, 20]
    pred_boxes[:, 2:] = np.maximum(pred_boxes[:, 2:], pred_boxes[:, :2] + 1)
    pred_cls = np.where(rng.random(n_pred) < 0.85, gt_cls[src], rng.integers(0, n_classes, n_pred))
    scores = rng.random(n_pred)
    order = np.argsort(-scores)  # Ultralytics receives predictions sorted by confidence
    return gt_cls, gt_boxes, pred_cls[order], scores[order], pred_boxes[order]


@pytest.mark.parametrize("seed", range(20))
def test_matching_reproduces_ultralytics(seed):
    rng = np.random.default_rng(seed)
    gt_cls, gt_boxes, pred_cls, scores, pred_boxes = random_scene(rng)
    ours = match_image(
        pred_cls, scores, pred_boxes, gt_cls, gt_boxes, COCO_IOU_THRESHOLDS, rule="ultralytics"
    )
    iou = torch.from_numpy(box_iou(gt_boxes, pred_boxes))
    fake_validator = SimpleNamespace(iouv=torch.from_numpy(COCO_IOU_THRESHOLDS))
    theirs = BaseValidator.match_predictions(
        fake_validator, torch.from_numpy(pred_cls), torch.from_numpy(gt_cls), iou
    ).numpy()
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("seed", range(20))
def test_compute_ap_reproduces_ultralytics(seed):
    rng = np.random.default_rng(100 + seed)
    m = int(rng.integers(1, 60))
    tp = rng.random(m) < rng.uniform(0.2, 0.9)
    n_gt = int(tp.sum()) + int(rng.integers(0, 10)) or 1
    r, p = pr_curve(tp, rng.random(m), n_gt)
    theirs, _, _ = ul_metrics.compute_ap(r, p)
    assert ap_ultralytics(r, p) == pytest.approx(theirs, abs=1e-12)


def test_ap_per_class_reproduces_ultralytics_end_to_end():
    """Many images -> our ap_table vs Ultralytics' ap_per_class on the same TP matrix."""
    rng = np.random.default_rng(7)
    n_classes = 3
    matches, tps, confs, pcls, tcls = [], [], [], [], []
    for i in range(30):
        gt_cls, gt_boxes, pred_cls, scores, pred_boxes = random_scene(rng, n_classes=n_classes)
        tp = match_image(
            pred_cls, scores, pred_boxes, gt_cls, gt_boxes, COCO_IOU_THRESHOLDS, rule="ultralytics"
        )
        matches.append(
            ImageMatches(f"{i}.jpg", str(i), pred_cls, scores, tp, np.bincount(gt_cls, minlength=n_classes))
        )
        tps.append(tp)
        confs.append(scores)
        pcls.append(pred_cls)
        tcls.append(gt_cls)
    ours = ap_table(matches, n_classes, method="ultralytics")
    out = ul_metrics.ap_per_class(
        np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls)
    )
    theirs, classes = out[5], out[6]
    np.testing.assert_allclose(ours[classes], theirs, atol=1e-9)
