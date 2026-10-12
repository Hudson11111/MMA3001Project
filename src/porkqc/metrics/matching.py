"""
Matching predictions to ground-truth boxes (true / false positives).

A prediction is a **true positive** at IoU threshold ``t`` if it can be paired
with a ground-truth box of the same class with IoU ≥ ``t`` that no other
prediction has already claimed. Every unpaired prediction is a false positive;
every unpaired ground-truth box is a false negative (a missed defect).

The pairing rule is a design choice (DECISION_LOG D-019):

``"confidence"`` (default, COCO/VOC convention)
    Predictions are processed from highest to lowest confidence; each takes the
    still-unclaimed ground-truth box it overlaps most. This mirrors how an
    operator would read the detector's output: most confident first.

``"ultralytics"``
    Reproduces ``match_predictions`` in Ultralytics 8.4.175: candidate pairs are
    sorted by IoU, each prediction keeps its best pair, then each ground-truth
    box keeps the earliest prediction (input order). Used only to verify that
    our pipeline reproduces Ultralytics' numbers on identical inputs.
"""

from __future__ import annotations

import numpy as np

from .iou import box_iou

__all__ = ["match_image", "MATCH_RULES"]

MATCH_RULES = ("confidence", "ultralytics")


def match_image(
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    pred_boxes: np.ndarray,
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresholds: np.ndarray,
    rule: str = "confidence",
) -> np.ndarray:
    """
    Flag each prediction in one image as a true positive at each IoU threshold.

    Parameters
    ----------
    pred_classes : numpy.ndarray
        Shape ``(M,)``, integer class ids of predictions.
    pred_scores : numpy.ndarray
        Shape ``(M,)``, confidences.
    pred_boxes : numpy.ndarray
        Shape ``(M, 4)``, ``[x1, y1, x2, y2]``.
    gt_classes : numpy.ndarray
        Shape ``(N,)``, integer class ids of ground-truth boxes.
    gt_boxes : numpy.ndarray
        Shape ``(N, 4)``, same units as ``pred_boxes``.
    iou_thresholds : numpy.ndarray
        Shape ``(T,)``, e.g. ``[0.5]`` or ``numpy.linspace(0.5, 0.95, 10)``.
    rule : {"confidence", "ultralytics"}, optional
        Pairing rule (see module docstring).

    Returns
    -------
    numpy.ndarray
        Shape ``(M, T)``, boolean; entry ``[i, k]`` is True if prediction ``i``
        is a true positive at threshold ``iou_thresholds[k]``. Row order follows
        the input order.

    Raises
    ------
    ValueError
        If array lengths disagree or ``rule`` is unknown.
    """
    pred_classes = np.asarray(pred_classes).astype(int).ravel()
    pred_scores = np.asarray(pred_scores, dtype=np.float64).ravel()
    gt_classes = np.asarray(gt_classes).astype(int).ravel()
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    thresholds = np.atleast_1d(np.asarray(iou_thresholds, dtype=np.float64))
    m, n = len(pred_classes), len(gt_classes)
    if len(pred_scores) != m or len(pred_boxes) != m:
        raise ValueError("pred_classes, pred_scores and pred_boxes must have the same length.")
    if len(gt_boxes) != n:
        raise ValueError("gt_classes and gt_boxes must have the same length.")
    if rule not in MATCH_RULES:
        raise ValueError(f"rule must be one of {MATCH_RULES}, got {rule!r}.")

    correct = np.zeros((m, len(thresholds)), dtype=bool)
    if m == 0 or n == 0:
        return correct

    # IoU with wrong-class pairs zeroed: shape (N, M), ground truth in rows.
    iou = box_iou(gt_boxes, pred_boxes) * (gt_classes[:, None] == pred_classes[None, :])

    if rule == "confidence":
        order = np.argsort(-pred_scores, kind="stable")
        for k, t in enumerate(thresholds):
            claimed = np.zeros(n, dtype=bool)
            for i in order:
                cand = np.where(~claimed & (iou[:, i] >= t) & (iou[:, i] > 0))[0]
                if cand.size:
                    j = cand[np.argmax(iou[cand, i])]
                    claimed[j] = True
                    correct[i, k] = True
    else:  # "ultralytics" — transcription of the library's numpy logic
        for k, t in enumerate(thresholds):
            matches = np.array(np.nonzero(iou >= t)).T  # rows: (gt index, pred index)
            matches = matches[iou[matches[:, 0], matches[:, 1]] > 0]
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), k] = True
    return correct
