"""
Controlled image perturbations for robustness / sensitivity testing (WP7).

Each perturbation models a plausible change on a real line:

* ``exposure`` — lighting drift or camera exposure change (multiply intensities),
* ``blur`` — motion blur or focus loss (Gaussian blur, sigma in pixels),
* ``noise`` — sensor noise in low light (additive Gaussian, sigma in grey levels),
* ``jpeg`` — stronger image compression (JPEG quality 1–95).

Labels are unchanged by all of them (no geometric change), so the same ground
truth applies and accuracy can be compared directly with the clean images.
"""

from __future__ import annotations

import io

import numpy as np

__all__ = ["PERTURBATIONS", "perturb"]

#: Default levels used in notebook 04 (mild → strong).
PERTURBATIONS = {
    "exposure": [0.5, 0.75, 1.25, 1.5],
    "blur": [1.0, 2.0, 4.0],
    "noise": [10.0, 25.0],
    "jpeg": [30, 10],
}


def perturb(image: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    """
    Apply one perturbation to an RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        ``(H, W, 3)`` uint8 RGB image.
    kind : {"exposure", "blur", "noise", "jpeg"}
        Perturbation type.
    level : float
        Exposure factor (> 0), blur sigma (≥ 0, pixels), noise sigma (≥ 0,
        grey levels) or JPEG quality (1–95).
    seed : int, optional
        Seed for the noise perturbation (reproducible).

    Returns
    -------
    numpy.ndarray
        Perturbed ``(H, W, 3)`` uint8 image (values clipped to 0–255).

    Raises
    ------
    TypeError
        If the image is not uint8.
    ValueError
        For a wrong shape, unknown kind or invalid level.
    """
    from PIL import Image, ImageFilter

    img = np.asarray(image)
    if img.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {img.dtype}.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {img.shape}.")
    if kind == "exposure":
        if level <= 0:
            raise ValueError("exposure factor must be > 0.")
        return np.clip(img.astype(np.float32) * level, 0, 255).round().astype(np.uint8)
    if kind == "blur":
        if level < 0:
            raise ValueError("blur sigma must be >= 0.")
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=level)))
    if kind == "noise":
        if level < 0:
            raise ValueError("noise sigma must be >= 0.")
        rng = np.random.default_rng(seed)
        return (
            np.clip(img.astype(np.float32) + rng.normal(0, level, img.shape), 0, 255).round().astype(np.uint8)
        )
    if kind == "jpeg":
        q = int(level)
        if not 1 <= q <= 95:
            raise ValueError("JPEG quality must be in 1..95.")
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))
    raise ValueError(f"Unknown perturbation {kind!r}; choose from {sorted(PERTURBATIONS)}.")
