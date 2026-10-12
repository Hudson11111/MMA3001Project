"""
Intersection over Union (IoU) for axis-aligned bounding boxes.

IoU is the overlap measure used to decide whether a predicted box "matches" a
ground-truth box (e.g. IoU >= 0.5 for mAP50). Every detection metric in this
project is built on it, so it is implemented here explicitly and unit-tested
rather than taken on trust from a library.

For boxes A and B::

    IoU(A, B) = area(A ∩ B) / area(A ∪ B)
              = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))
"""

from __future__ import annotations

import numpy as np

__all__ = ["box_area", "box_iou"]


def _as_boxes(boxes: np.ndarray, name: str) -> np.ndarray:
    """Validate and convert an ``(N, 4)`` ``[x1, y1, x2, y2]`` array."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4) as [x1, y1, x2, y2], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr[:, 2] < arr[:, 0]) or np.any(arr[:, 3] < arr[:, 1]):
        raise ValueError(f"{name} has boxes with x2 < x1 or y2 < y1.")
    return arr


def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Area of each box.

    Parameters
    ----------
    boxes : numpy.ndarray
        Shape ``(N, 4)``, ``[x1, y1, x2, y2]`` with ``x2 >= x1`` and ``y2 >= y1``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)``, areas in squared input units (pixels² for pixel boxes).

    Raises
    ------
    ValueError
        If the array has the wrong shape, non-finite values, or inverted corners.
    """
    b = _as_boxes(boxes, "boxes")
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of boxes.

    Parameters
    ----------
    boxes1 : numpy.ndarray
        Shape ``(N, 4)``, ``[x1, y1, x2, y2]``.
    boxes2 : numpy.ndarray
        Shape ``(M, 4)``, same convention and units as ``boxes1``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, M)``; entry ``[i, j]`` is IoU(boxes1[i], boxes2[j]) in [0, 1].
        Pairs whose union has zero area (two degenerate boxes) get IoU 0 rather
        than NaN.

    Raises
    ------
    ValueError
        If either input has the wrong shape, non-finite values, or inverted corners.

    Notes
    -----
    Vectorised with broadcasting: memory and work scale as O(N·M), which is
    negligible for the tens of boxes per image in this dataset.
    """
    a = _as_boxes(boxes1, "boxes1")
    b = _as_boxes(boxes2, "boxes2")

    # Intersection rectangle for every pair, via broadcasting (N,1) against (1,M).
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou
