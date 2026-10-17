"""
Hand-crafted image features for the scikit-learn baseline (WP4, DECISION_LOG D-006).

The baseline answers "do we even need a deep detector?": a classical pipeline
(`StandardScaler` → decision tree / RBF SVM / MLP) predicts, per image, whether a
defect of each headline class is present. It cannot localise defects, so it is
compared with the detector at the *image level* (the level of the PASS/FAIL
decision).

Features (all cheap, interpretable, no learning):

* **HSV colour histogram** (8 hue × 4 saturation × 4 value = 128 bins,
  normalised): meat, film glare and conveyor have distinct colours.
* **Grey thumbnail** (16 × 12 = 192 values, scaled to [0, 1]): coarse layout —
  where the tray and meat are.
* **Edge statistics** (4 values): mean gradient magnitude overall and in the
  bottom/right halves, plus the share of strong edges — wrinkles and lifted
  film add edges.

Total: 324 features per image.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "N_FEATURES", "feature_names", "image_features", "image_level_ap", "CANDIDATES",
    "make_pipeline", "decision_scores", "select_on_validation",
]

N_HSV = (8, 4, 4)
THUMB = (16, 12)
N_FEATURES = int(np.prod(N_HSV)) + THUMB[0] * THUMB[1] + 4


def feature_names() -> list[str]:
    """
    Names of the 324 features, in order.

    Returns
    -------
    list of str
        ``hsv_h{i}_s{j}_v{k}``, ``thumb_{row}_{col}``, then four ``edge_*`` names.
    """
    names = [f"hsv_h{h}_s{s}_v{v}" for h in range(N_HSV[0]) for s in range(N_HSV[1]) for v in range(N_HSV[2])]
    names += [f"thumb_{r}_{c}" for r in range(THUMB[1]) for c in range(THUMB[0])]
    names += ["edge_mean", "edge_mean_bottom", "edge_mean_right", "edge_strong_share"]
    return names


def image_features(image) -> np.ndarray:
    """
    Feature vector of one image.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image or numpy.ndarray
        RGB image (file or ``(H, W, 3)`` uint8 array).

    Returns
    -------
    numpy.ndarray
        Shape ``(324,)``, float32.

    Raises
    ------
    ValueError
        If an array input is not ``(H, W, 3)``.
    """
    from PIL import Image

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {image.shape}.")
        im = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        im = image.convert("RGB")
    else:
        with Image.open(image) as f:
            im = f.convert("RGB")

    hsv = np.asarray(im.convert("HSV"), dtype=np.int32)
    idx = [np.minimum(hsv[..., k] * n // 256, n - 1) for k, n in enumerate(N_HSV)]
    flat = (idx[0] * N_HSV[1] + idx[1]) * N_HSV[2] + idx[2]
    hist = np.bincount(flat.ravel(), minlength=int(np.prod(N_HSV))).astype(np.float64)
    hist /= hist.sum()

    grey = im.convert("L")
    thumb = np.asarray(grey.resize(THUMB, Image.Resampling.BOX), dtype=np.float64).ravel() / 255.0

    g = np.asarray(grey.resize((180, 135), Image.Resampling.BOX), dtype=np.float64) / 255.0
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    h, w = mag.shape
    edges = np.array([mag.mean(), mag[h // 2:].mean(), mag[:, w // 2:].mean(), (mag > 0.1).mean()])
    return np.concatenate([hist, thumb, edges]).astype(np.float32)


def image_level_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Average precision of an image-level ranking (exact step-function area).

    Uses the same integration rule as the detector metrics
    (:func:`porkqc.metrics.ap.ap_exact`), so detector and baseline are scored
    on an identical scale.

    Parameters
    ----------
    y_true : numpy.ndarray
        Shape ``(N,)``, 1 if the image contains the defect.
    scores : numpy.ndarray
        Shape ``(N,)``, higher = more likely defective.

    Returns
    -------
    float
        AP in [0, 1]; NaN if there is no positive image.
    """
    from ..metrics.ap import ap_exact, pr_curve

    y = np.asarray(y_true).astype(bool)
    if y.sum() == 0:
        return float("nan")
    r, p = pr_curve(y.astype(float), np.asarray(scores, dtype=float), int(y.sum()))
    return ap_exact(r, p)


#: Candidate models and hyperparameters, chosen on VALIDATION data only.
CANDIDATES = {
    "decision tree": ("tree", [{"max_depth": d, "min_samples_leaf": l} for d in (3, 5, 8, None) for l in (1, 5, 20)]),
    "SVM (RBF)": ("svm", [{"C": c, "gamma": g} for c in (0.3, 1, 3, 10) for g in ("scale", 0.001, 0.01)]),
    "MLP": ("mlp", [{"hidden_layer_sizes": h, "alpha": a} for h in ((64,), (128, 64)) for a in (1e-4, 1e-2, 1.0)]),
}


def make_pipeline(kind: str, params: dict, seed: int = 0):
    """
    ``StandardScaler`` → classifier pipeline.

    Scaling matters for the SVM (RBF distances) and the MLP (optimisation);
    trees are scale-invariant, but use the same pipeline for uniformity.

    Parameters
    ----------
    kind : {"tree", "svm", "mlp"}
        Model family.
    params : dict
        Hyperparameters for that family.
    seed : int, optional
        Random seed (tree tie-breaks, MLP initialisation).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.

    Raises
    ------
    ValueError
        For an unknown ``kind``.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    if kind == "tree":
        clf = DecisionTreeClassifier(random_state=seed, class_weight="balanced", **params)
    elif kind == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", **params)
    elif kind == "mlp":
        # early stopping on an internal 10% split; patience 30 epochs so small problems are not stopped prematurely
        clf = MLPClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model kind {kind!r}.")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def decision_scores(pipe, X: np.ndarray) -> np.ndarray:
    """
    Continuous defect score from a fitted pipeline, for ranking images.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Fitted pipeline from :func:`make_pipeline`.
    X : numpy.ndarray
        Features, shape ``(N, 324)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)``: the SVM margin (``decision_function``) where available,
        otherwise the positive-class probability. Only the ordering matters
        for average precision, so the two are interchangeable here.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba") and not hasattr(clf, "decision_function"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, 1]


def select_on_validation(X_tr, y_tr, X_va, y_va, candidates: dict | None = None, seed: int = 0) -> list[dict]:
    """
    Fit every candidate on TRAIN, score image-level AP on VALIDATION, keep the best per family.

    Parameters
    ----------
    X_tr, y_tr : numpy.ndarray
        Training features ``(N, 324)`` and binary labels ``(N,)``.
    X_va, y_va : numpy.ndarray
        Validation features and labels.
    candidates : dict, optional
        Family name → (kind, list of parameter dicts); default :data:`CANDIDATES`.
    seed : int, optional
        Random seed.

    Returns
    -------
    list of dict
        One entry per family: ``family``, ``kind``, ``params``, ``valid_ap``,
        ``fit_seconds`` and the fitted ``pipeline``.
    """
    import time

    out = []
    for family, (kind, grid) in (candidates or CANDIDATES).items():
        best = None
        for params in grid:
            pipe = make_pipeline(kind, params, seed)
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            ap = image_level_ap(y_va, decision_scores(pipe, X_va))
            if best is None or ap > best["valid_ap"]:
                best = {"family": family, "kind": kind, "params": params, "valid_ap": ap,
                        "fit_seconds": fit_s, "pipeline": pipe}
        out.append(best)
    return out
