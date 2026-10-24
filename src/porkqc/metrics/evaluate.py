"""
Dataset-level detection evaluation from ground-truth and prediction files.

Pipeline::

    ground-truth JSON + prediction JSON
        └─ match_dataset()   per-image true/false positives at each IoU threshold
             ├─ ap_table()   per-class AP (any integration rule)  → mAP50, mAP50-95
             ├─ counts_at_confidence()  TP / FP / FN at an operating threshold
             └─ bootstrap()  confidence interval by resampling images

Matching is done once per image; every later step only re-aggregates those
results, so bootstrapping (thousands of re-evaluations) stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..data.audit import original_stem
from ..predictions import gt_arrays, pred_arrays
from .ap import average_precision, pr_curve
from .matching import match_image

__all__ = [
    "COCO_IOU_THRESHOLDS",
    "ImageMatches",
    "match_dataset",
    "class_pr_inputs",
    "ap_table",
    "summarise",
    "counts_at_confidence",
    "bootstrap",
    "paired_bootstrap",
]

#: IoU thresholds 0.50, 0.55, …, 0.95 used for mAP50-95 (COCO convention).
COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class ImageMatches:
    """
    Matching result for one image.

    Attributes
    ----------
    file : str
        Image file name.
    group : str
        Original-image stem (Roboflow copies of one photo share it); used for
        grouped bootstrapping.
    classes : numpy.ndarray
        Shape ``(M,)``, predicted class ids.
    scores : numpy.ndarray
        Shape ``(M,)``, confidences.
    tp : numpy.ndarray
        Shape ``(M, T)``, True where the prediction is a true positive at each
        IoU threshold.
    n_gt : numpy.ndarray
        Shape ``(n_classes,)``, ground-truth box count per class.
    """

    file: str
    group: str
    classes: np.ndarray
    scores: np.ndarray
    tp: np.ndarray
    n_gt: np.ndarray


def match_dataset(
    gt: dict,
    pred: dict,
    n_classes: int,
    iou_thresholds: np.ndarray = COCO_IOU_THRESHOLDS,
    rule: str = "confidence",
) -> list[ImageMatches]:
    """
    Match predictions to ground truth for every image of a split.

    Parameters
    ----------
    gt : dict
        Ground-truth structure (:mod:`porkqc.predictions`).
    pred : dict
        Prediction structure for the same split.
    n_classes : int
        Number of classes.
    iou_thresholds : numpy.ndarray, optional
        IoU thresholds (default 0.50:0.05:0.95).
    rule : {"confidence", "ultralytics"}, optional
        Matching rule (:func:`porkqc.metrics.matching.match_image`).

    Returns
    -------
    list of ImageMatches
        One entry per ground-truth image, in ground-truth order. Images with no
        predictions get empty arrays; images with no ground truth still count
        (their predictions are false positives).

    Raises
    ------
    ValueError
        If the prediction file contains images absent from the ground truth, or
        an image's width/height differ between the files (a coordinate-frame
        mismatch would silently ruin every IoU).
    """
    pred_by_file = {e["file"]: e for e in pred["images"]}
    gt_files = {e["file"] for e in gt["images"]}
    extra = set(pred_by_file) - gt_files
    if extra:
        raise ValueError(
            f"{len(extra)} predicted images are not in the ground truth, e.g. {sorted(extra)[:3]}"
        )

    thresholds = np.atleast_1d(np.asarray(iou_thresholds, dtype=np.float64))
    out = []
    for g in gt["images"]:
        gc, gb = gt_arrays(g)
        p = pred_by_file.get(g["file"], {"pred": []})
        if "width" in p and (p["width"], p["height"]) != (g["width"], g["height"]):
            raise ValueError(
                f"{g['file']}: size {p['width']}x{p['height']} in predictions vs "
                f"{g['width']}x{g['height']} in ground truth."
            )
        pc, ps, pb = pred_arrays(p)
        tp = match_image(pc, ps, pb, gc, gb, thresholds, rule=rule)
        out.append(
            ImageMatches(
                file=g["file"],
                group=original_stem(g["file"]),
                classes=pc,
                scores=ps,
                tp=tp,
                n_gt=np.bincount(gc, minlength=n_classes)[:n_classes],
            )
        )
    return out


def class_pr_inputs(matches: Sequence[ImageMatches], cls: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pool one class's predictions over images.

    Parameters
    ----------
    matches : sequence of ImageMatches
        Output of :func:`match_dataset` (or a bootstrap resample of it).
    cls : int
        Class id.

    Returns
    -------
    tp : numpy.ndarray
        Shape ``(K, T)`` true-positive flags of that class's predictions.
    scores : numpy.ndarray
        Shape ``(K,)`` their confidences.
    n_gt : int
        Total ground-truth boxes of that class.
    """
    tps = [m.tp[m.classes == cls] for m in matches]
    scs = [m.scores[m.classes == cls] for m in matches]
    t = matches[0].tp.shape[1] if matches else 1
    tp = np.concatenate(tps) if tps else np.zeros((0, t), dtype=bool)
    scores = np.concatenate(scs) if scs else np.zeros(0)
    n_gt = int(sum(m.n_gt[cls] for m in matches))
    return tp.reshape(-1, t), scores, n_gt


def ap_table(matches: Sequence[ImageMatches], n_classes: int, method: str = "exact") -> np.ndarray:
    """
    Per-class AP at every IoU threshold.

    Parameters
    ----------
    matches : sequence of ImageMatches
        Output of :func:`match_dataset`.
    n_classes : int
        Number of classes.
    method : str, optional
        AP integration rule (:func:`porkqc.metrics.ap.average_precision`).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_classes, T)``. NaN for classes with no ground truth in these
        images (AP undefined); 0 for classes with ground truth but no predictions.
    """
    t = matches[0].tp.shape[1] if matches else 1
    table = np.full((n_classes, t), np.nan)
    for c in range(n_classes):
        tp, scores, n_gt = class_pr_inputs(matches, c)
        if n_gt == 0:
            continue
        for k in range(t):
            if len(scores) == 0:
                table[c, k] = 0.0
            else:
                r, p = pr_curve(tp[:, k], scores, n_gt)
                table[c, k] = average_precision(r, p, method)
    return table


def summarise(
    matches: Sequence[ImageMatches],
    class_names: Sequence[str],
    method: str = "exact",
) -> dict:
    """
    Headline metrics: mAP50, mAP50-95 and a per-class breakdown.

    Parameters
    ----------
    matches : sequence of ImageMatches
        Output of :func:`match_dataset` with the 10 COCO IoU thresholds
        (index 0 must be IoU 0.5).
    class_names : sequence of str
        Ordered class names.
    method : str, optional
        AP integration rule.

    Returns
    -------
    dict
        ``map50``, ``map50_95`` (means over classes that have ground truth),
        ``method``, ``n_images`` and ``per_class`` (list of dicts with ``class``,
        ``n_gt``, ``n_pred``, ``ap50``, ``ap50_95``; APs are None when undefined).
    """
    n_classes = len(class_names)
    table = ap_table(matches, n_classes, method)
    per_class_5095 = table.mean(axis=1)  # NaN rows stay NaN
    per_class = []
    for c, name in enumerate(class_names):
        n_gt = int(sum(m.n_gt[c] for m in matches))
        n_pred = int(sum(int((m.classes == c).sum()) for m in matches))
        defined = not np.isnan(table[c, 0])
        per_class.append(
            {
                "class": name,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "ap50": float(table[c, 0]) if defined else None,
                "ap50_95": float(per_class_5095[c]) if defined else None,
            }
        )
    have = ~np.isnan(table[:, 0])
    return {
        "method": method,
        "n_images": len(matches),
        "map50": float(np.mean(table[have, 0])) if have.any() else float("nan"),
        "map50_95": float(np.mean(per_class_5095[have])) if have.any() else float("nan"),
        "per_class": per_class,
    }


def counts_at_confidence(
    matches: Sequence[ImageMatches], n_classes: int, conf: float, iou_index: int = 0
) -> dict[str, np.ndarray]:
    """
    True positives, false positives and false negatives at one operating point.

    Parameters
    ----------
    matches : sequence of ImageMatches
        Output of :func:`match_dataset`.
    n_classes : int
        Number of classes.
    conf : float
        Keep predictions with confidence ≥ ``conf``.
    iou_index : int, optional
        Column of the IoU thresholds to use (0 → IoU 0.5).

    Returns
    -------
    dict of str to numpy.ndarray
        ``tp``, ``fp``, ``fn`` (each shape ``(n_classes,)``, integers),
        ``precision`` and ``recall`` (NaN where undefined).

    Notes
    -----
    Matching was done on all predictions. Removing low-confidence predictions
    afterwards does not change which of the remaining ones are true positives
    under the confidence-ordered rule, because higher-confidence predictions
    are matched first.
    """
    tp = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    n_gt = np.zeros(n_classes, dtype=int)
    for m in matches:
        keep = m.scores >= conf
        hit = m.tp[keep, iou_index]
        cls = m.classes[keep]
        tp += np.bincount(cls[hit], minlength=n_classes)[:n_classes]
        fp += np.bincount(cls[~hit], minlength=n_classes)[:n_classes]
        n_gt += m.n_gt
    fn = n_gt - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(n_gt > 0, tp / n_gt, np.nan)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def bootstrap(
    matches: Sequence[ImageMatches],
    statistic: Callable[[list[ImageMatches]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    grouped: bool = False,
) -> dict:
    """
    Percentile bootstrap confidence interval for any evaluation statistic.

    Images (or groups of near-identical images) are the independent units and
    are resampled with replacement; the statistic is recomputed each time.

    Parameters
    ----------
    matches : sequence of ImageMatches
        Output of :func:`match_dataset`.
    statistic : callable
        Maps a list of ImageMatches to a float, e.g.
        ``lambda m: summarise(m, names)["map50"]``. May return NaN (e.g. a
        resample without any instance of a rare class); NaNs are dropped and
        counted.
    n_boot : int, optional
        Number of resamples (default 1000).
    ci : float, optional
        Confidence level (default 0.95).
    seed : int, optional
        Random seed for reproducibility.
    grouped : bool, optional
        Resample whole groups (``ImageMatches.group``, the original-image stem)
        instead of single images. Use when images are not independent, e.g.
        augmented copies or near-duplicate video frames.

    Returns
    -------
    dict
        ``point`` (statistic on the original data), ``low``, ``high`` (interval
        bounds), ``std`` (bootstrap standard error), ``n_boot`` and ``n_nan``.
    """
    rng = np.random.default_rng(seed)
    matches = list(matches)
    if grouped:
        groups: dict[str, list[ImageMatches]] = {}
        for m in matches:
            groups.setdefault(m.group, []).append(m)
        units = list(groups.values())
    else:
        units = [[m] for m in matches]

    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        samples[b] = statistic([m for i in idx for m in units[i]])
    good = samples[~np.isnan(samples)]
    alpha = (1 - ci) / 2
    return {
        "point": float(statistic(matches)),
        "low": float(np.quantile(good, alpha)) if good.size else float("nan"),
        "high": float(np.quantile(good, 1 - alpha)) if good.size else float("nan"),
        "std": float(good.std(ddof=1)) if good.size > 1 else float("nan"),
        "n_boot": n_boot,
        "n_nan": int(np.isnan(samples).sum()),
    }


def paired_bootstrap(
    matches_a: Sequence[ImageMatches],
    matches_b: Sequence[ImageMatches],
    statistic: Callable[[list[ImageMatches]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Bootstrap confidence interval for the *difference* A − B on the same images.

    Two models evaluated on identical images are correlated: an image that is hard
    for one is usually hard for the other. Resampling the same units for both
    models keeps that pairing, so the interval for the difference is much
    narrower (and more honest) than comparing two separate intervals.

    Parameters
    ----------
    matches_a, matches_b : sequence of ImageMatches
        Results of two models on the same images (same files, any order).
        Each image's ``group`` defines the resampling unit (e.g. a 50-frame run).
    statistic : callable
        Maps a list of ImageMatches to a float (e.g. mAP50).
    n_boot, ci, seed
        As in :func:`bootstrap`.

    Returns
    -------
    dict
        ``point`` (A − B on the original data), ``low``, ``high``, ``std``,
        ``p_le_0`` (share of resamples with A − B ≤ 0, a one-sided bootstrap
        p-value for "A is better"), ``n_boot``, ``n_nan``.

    Raises
    ------
    ValueError
        If the two inputs do not cover the same image files.
    """
    a_by = {m.file: m for m in matches_a}
    b_by = {m.file: m for m in matches_b}
    if set(a_by) != set(b_by):
        raise ValueError("paired_bootstrap needs both models evaluated on the same images.")
    groups: dict[str, list[str]] = {}
    for f, m in a_by.items():
        groups.setdefault(m.group, []).append(f)
    units = list(groups.values())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        files = [f for i in idx for f in units[i]]
        diffs[k] = statistic([a_by[f] for f in files]) - statistic([b_by[f] for f in files])
    good = diffs[~np.isnan(diffs)]
    alpha = (1 - ci) / 2
    return {
        "point": float(statistic(list(a_by.values())) - statistic([b_by[f] for f in a_by])),
        "low": float(np.quantile(good, alpha)) if good.size else float("nan"),
        "high": float(np.quantile(good, 1 - alpha)) if good.size else float("nan"),
        "std": float(good.std(ddof=1)) if good.size > 1 else float("nan"),
        "p_le_0": float(np.mean(good <= 0)) if good.size else float("nan"),
        "n_boot": n_boot,
        "n_nan": int(np.isnan(diffs).sum()),
    }
