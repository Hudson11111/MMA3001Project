r"""
Precision–recall curves and average precision (AP) as a numerical integral.

Ranking a class's predictions by confidence and walking down the list gives, after
each prediction, a recall :math:`r_k` (fraction of ground-truth boxes found) and a
precision :math:`p_k` (fraction of predictions so far that were correct).
Average precision is the area under the precision–recall curve,

.. math::  \mathrm{AP} = \int_0^1 p_{\mathrm{interp}}(r)\,dr,
           \qquad p_{\mathrm{interp}}(r) = \max_{k:\,r_k \ge r} p_k ,

where the *interpolated precision* :math:`p_{\mathrm{interp}}` (the "precision
envelope") is the best precision achievable at recall :math:`r` or higher, and is
zero beyond the largest recall reached. It is a non-increasing **step function**:
piecewise constant with jumps at the recall values. Its integral can therefore be
computed exactly (:func:`ap_exact`), which makes AP an ideal test case for
numerical integration: libraries instead use sampled rules —

* :func:`ap_coco101` — mean of :math:`p_{\mathrm{interp}}` at 101 recall points
  (pycocotools / COCO),
* :func:`ap_voc11` — mean at 11 points (PASCAL VOC 2007),
* :func:`ap_ultralytics` — trapezoid rule on 101 samples of a *linearly
  interpolated* envelope (Ultralytics 8.4.175 ``compute_ap``).

These are different quadrature rules applied to a discontinuous integrand, so
they give different numbers for the same predictions. Notebook 05 measures how
different, and why the classical error theory of Newton–Cotes rules (which
assumes a smooth integrand) does not apply.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pr_curve",
    "precision_envelope",
    "interpolated_precision",
    "ap_exact",
    "ap_coco101",
    "ap_voc11",
    "ap_ultralytics",
    "average_precision",
    "AP_METHODS",
]

# numpy >= 2.0 renamed trapz to trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def pr_curve(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Precision–recall points for one class.

    Parameters
    ----------
    tp : numpy.ndarray
        Shape ``(M,)``, boolean/0-1: whether each prediction is a true positive.
    scores : numpy.ndarray
        Shape ``(M,)``, confidences used to rank the predictions.
    n_gt : int
        Number of ground-truth boxes of this class (must be ≥ 1).

    Returns
    -------
    recall : numpy.ndarray
        Shape ``(M,)``, non-decreasing, in [0, 1].
    precision : numpy.ndarray
        Shape ``(M,)``, in [0, 1].

    Raises
    ------
    ValueError
        If ``n_gt < 1`` (recall undefined), lengths differ, or there are more
        true positives than ground-truth boxes.
    """
    tp = np.asarray(tp, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(tp) != len(scores):
        raise ValueError("tp and scores must have the same length.")
    if n_gt < 1:
        raise ValueError("n_gt must be >= 1; recall is undefined without ground truth.")
    if tp.sum() > n_gt:
        raise ValueError(f"{int(tp.sum())} true positives but only {n_gt} ground-truth boxes.")
    order = np.argsort(-scores, kind="stable")
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(1.0 - tp[order])
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1.0)
    return recall, precision


def precision_envelope(precision: np.ndarray) -> np.ndarray:
    """
    Running maximum of precision taken from the right (non-increasing).

    Parameters
    ----------
    precision : numpy.ndarray
        Shape ``(M,)``, precision values in ranked order.

    Returns
    -------
    numpy.ndarray
        Shape ``(M,)``; element ``k`` is ``max(precision[k:])``.
    """
    p = np.asarray(precision, dtype=np.float64)
    return np.maximum.accumulate(p[::-1])[::-1] if p.size else p


def interpolated_precision(recall: np.ndarray, precision: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Evaluate the step function :math:`p_{interp}(r)` at arbitrary recall values.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.
    r : array_like
        Query recall values in [0, 1].

    Returns
    -------
    numpy.ndarray
        Same shape as ``r``: the maximum precision over all points with recall
        ≥ r, or 0 where no point reaches that recall.
    """
    recall = np.asarray(recall, dtype=np.float64)
    env = precision_envelope(precision)
    r = np.asarray(r, dtype=np.float64)
    idx = np.searchsorted(recall, r, side="left")  # first point with recall >= r
    out = np.zeros_like(r)
    ok = idx < len(recall)
    out[ok] = env[idx[ok]]
    return out


def ap_exact(recall: np.ndarray, precision: np.ndarray) -> float:
    r"""
    Exact area under the interpolated-precision step function.

    .. math:: \mathrm{AP} = \sum_k (r_k - r_{k-1})\, \hat p_k, \quad r_0 = 0,

    with :math:`\hat p_k` the envelope value. Equivalent to PASCAL VOC 2010+
    "all points" AP and Ultralytics' ``method="continuous"``.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.

    Returns
    -------
    float
        AP in [0, 1] (0 for an empty curve).
    """
    recall = np.asarray(recall, dtype=np.float64)
    if recall.size == 0:
        return 0.0
    env = precision_envelope(precision)
    widths = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(widths * env))


def _ap_sampled(recall: np.ndarray, precision: np.ndarray, n_points: int) -> float:
    """Mean of interpolated precision at ``n_points`` equally spaced recalls in [0, 1]."""
    if np.asarray(recall).size == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, n_points)
    return float(np.mean(interpolated_precision(recall, precision, thresholds)))


def ap_coco101(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    COCO-style AP: mean interpolated precision at recall = 0, 0.01, …, 1.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.

    Returns
    -------
    float
        AP in [0, 1].
    """
    return _ap_sampled(recall, precision, 101)


def ap_voc11(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    PASCAL VOC 2007 AP: mean interpolated precision at recall = 0, 0.1, …, 1.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.

    Returns
    -------
    float
        AP in [0, 1].
    """
    return _ap_sampled(recall, precision, 11)


def ap_ultralytics(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Ultralytics 8.4.175 AP (``compute_ap`` with ``method="interp"``).

    Adds sentinel points (recall 0 with precision 1; a drop to precision 0 at
    the last recall; recall 1 with precision 0), takes the precision envelope,
    linearly interpolates it at 101 recall values and integrates with the
    trapezoid rule.

    A side effect of the quadrature: a perfect detector scores **0.995**, not 1,
    because the interpolated envelope falls to 0 at recall 1 and the trapezoid
    over the final 0.01-wide panel averages 1 and 0.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.

    Returns
    -------
    float
        AP in [0, 1].
    """
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(_trapezoid(np.interp(x, mrec, mpre), x))


#: Available AP rules for :func:`average_precision`.
AP_METHODS = {
    "exact": ap_exact,
    "coco101": ap_coco101,
    "voc11": ap_voc11,
    "ultralytics": ap_ultralytics,
}


def average_precision(recall: np.ndarray, precision: np.ndarray, method: str = "exact") -> float:
    """
    Average precision with a chosen integration rule.

    Parameters
    ----------
    recall, precision : numpy.ndarray
        PR points from :func:`pr_curve`.
    method : {"exact", "coco101", "voc11", "ultralytics"}, optional
        Integration rule (default: exact step-function area).

    Returns
    -------
    float
        AP in [0, 1].

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    try:
        fn = AP_METHODS[method]
    except KeyError:
        raise ValueError(f"method must be one of {sorted(AP_METHODS)}, got {method!r}.") from None
    return fn(recall, precision)
