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

__all__ = ["N_FEATURES", "feature_names", "image_features"]

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
