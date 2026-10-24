"""
Tray-level decision layer: detections → FAIL / REVIEW / PASS.

The detector outputs boxes with confidences; the production line needs one
action per image. The rule (DECISION_LOG D-023, D-026):

* image score ``s`` = highest confidence of any predicted defect box;
* ``s ≥ t_fail``            → **FAIL** (reject the tray automatically);
* ``t_review ≤ s < t_fail`` → **REVIEW** (route to a human inspector);
* ``s < t_review``          → **PASS** ("no defect found").

Thresholds are chosen on VALIDATION data by minimising the expected cost per
image, with costs from DECISION_LOG D-007:

* a defective tray passed (miss) costs ``c_miss`` (default 10),
* a good tray rejected (false reject) costs ``c_false_reject`` (default 1),
* any tray sent to review costs ``c_review`` (default 0.25: a few seconds of an
  inspector's time; the inspector is assumed to decide correctly).

The search is an exhaustive grid over threshold pairs — the cost is a step
function of the thresholds (it only changes when a threshold crosses an image
score), so gradient-based optimisation is not applicable, and with a few
hundred images the full grid is cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .predictions import pred_arrays

__all__ = ["image_scores", "decide", "DecisionCosts", "evaluate_decisions", "choose_thresholds"]

PASS, REVIEW, FAIL = "PASS", "REVIEW", "FAIL"


def image_scores(
    gt: dict, pred: dict, classes: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    One defect score and one ground-truth flag per image.

    Parameters
    ----------
    gt : dict
        Ground-truth structure (:mod:`porkqc.predictions`).
    pred : dict
        Prediction structure for the same images.
    classes : list of int, optional
        Only these class ids count, for both predictions and ground truth
        (default: all classes).

    Returns
    -------
    scores : numpy.ndarray
        Shape ``(N,)``: highest predicted confidence per image (0 if none).
    defective : numpy.ndarray
        Shape ``(N,)``, bool: image has at least one labelled defect.
    files : list of str
        Image names, in ground-truth order.

    Raises
    ------
    KeyError
        If an image in the ground truth has no entry in the predictions.
    """
    by_file = {e["file"]: e for e in pred["images"]}
    scores, defective, files = [], [], []
    for g in gt["images"]:
        c, s, _ = pred_arrays(by_file[g["file"]])
        gcls = np.asarray([b[0] for b in g["boxes"]], dtype=int)
        if classes is not None:
            s = s[np.isin(c, classes)]
            gcls = gcls[np.isin(gcls, classes)]
        scores.append(float(s.max()) if s.size else 0.0)
        defective.append(gcls.size > 0)
        files.append(g["file"])
    return np.asarray(scores), np.asarray(defective, dtype=bool), files


def decide(scores: np.ndarray, t_review: float, t_fail: float) -> np.ndarray:
    """
    Apply the two-threshold rule.

    Parameters
    ----------
    scores : numpy.ndarray
        Image defect scores in [0, 1].
    t_review, t_fail : float
        Thresholds with ``0 ≤ t_review ≤ t_fail ≤ 1``. ``t_review == t_fail``
        disables the REVIEW band.

    Returns
    -------
    numpy.ndarray
        Array of ``"PASS"``, ``"REVIEW"`` or ``"FAIL"``.

    Raises
    ------
    ValueError
        If the thresholds are out of order or outside [0, 1].
    """
    if not (0.0 <= t_review <= t_fail <= 1.0):
        raise ValueError(f"Need 0 <= t_review <= t_fail <= 1, got {t_review}, {t_fail}.")
    s = np.asarray(scores, dtype=float)
    out = np.full(s.shape, PASS, dtype=object)
    out[s >= t_review] = REVIEW
    out[s >= t_fail] = FAIL
    return out


@dataclass(frozen=True)
class DecisionCosts:
    """Relative costs of each outcome (DECISION_LOG D-007)."""

    c_miss: float = 10.0
    c_false_reject: float = 1.0
    c_review: float = 0.25


def evaluate_decisions(
    decisions: np.ndarray, defective: np.ndarray, costs: DecisionCosts | None = None
) -> dict:
    """
    Count outcomes and the mean cost per image.

    Parameters
    ----------
    decisions : numpy.ndarray
        Output of :func:`decide`.
    defective : numpy.ndarray
        Ground-truth flags.
    costs : DecisionCosts, optional
        Outcome costs (default: :class:`DecisionCosts`, i.e. 10 : 1 : 0.25).

    Returns
    -------
    dict
        ``n``, ``n_defective``, ``misses`` (defective → PASS), ``false_rejects``
        (clean → FAIL), ``reviews``, ``miss_rate`` (misses / defective),
        ``false_reject_rate`` (false rejects / clean), ``review_rate``,
        ``auto_fail_precision`` (defective share of FAIL), ``cost_per_image``.
    """
    costs = DecisionCosts() if costs is None else costs
    d = np.asarray(decisions)
    y = np.asarray(defective, dtype=bool)
    misses = int(np.sum((d == PASS) & y))
    false_rejects = int(np.sum((d == FAIL) & ~y))
    reviews = int(np.sum(d == REVIEW))
    n, n_def = len(d), int(y.sum())
    n_fail = int(np.sum(d == FAIL))
    cost = (costs.c_miss * misses + costs.c_false_reject * false_rejects + costs.c_review * reviews) / max(
        n, 1
    )
    return {
        "n": n,
        "n_defective": n_def,
        "misses": misses,
        "false_rejects": false_rejects,
        "reviews": reviews,
        "miss_rate": misses / n_def if n_def else float("nan"),
        "false_reject_rate": false_rejects / (n - n_def) if n > n_def else float("nan"),
        "review_rate": reviews / n if n else float("nan"),
        "auto_fail_precision": float(np.sum((d == FAIL) & y)) / n_fail if n_fail else float("nan"),
        "cost_per_image": cost,
    }


def choose_thresholds(
    scores: np.ndarray,
    defective: np.ndarray,
    costs: DecisionCosts | None = None,
    grid: np.ndarray | None = None,
) -> dict:
    """
    Thresholds minimising mean cost per image (exhaustive grid search).

    Ties are broken towards *fewer reviews*, then *higher thresholds* (less
    automatic rejection), so the choice is deterministic.

    Parameters
    ----------
    scores, defective : numpy.ndarray
        From :func:`image_scores` on VALIDATION data.
    costs : DecisionCosts, optional
        Outcome costs (default: :class:`DecisionCosts`, i.e. 10 : 1 : 0.25).
    grid : numpy.ndarray, optional
        Candidate threshold values (default 0.00, 0.01, …, 1.00).

    Returns
    -------
    dict
        ``t_review``, ``t_fail``, ``costs`` and the outcome counts from
        :func:`evaluate_decisions` at the optimum.
    """
    costs = DecisionCosts() if costs is None else costs
    grid = np.round(np.linspace(0, 1, 101), 2) if grid is None else np.asarray(grid, dtype=float)
    best_key, best = None, None
    for i, tr in enumerate(grid):
        for tf in grid[i:]:
            r = evaluate_decisions(decide(scores, tr, tf), defective, costs)
            key = (round(r["cost_per_image"], 12), r["reviews"], -tf, -tr)
            if best_key is None or key < best_key:
                best_key, best = key, {"t_review": float(tr), "t_fail": float(tf), **r}
    best["costs"] = asdict(costs)
    return best
