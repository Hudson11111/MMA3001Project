"""
Near-duplicate image detection with a perceptual difference hash (dHash).

Why: the images look like frames from conveyor video. If near-identical frames
land in both the training and the test split, the test score measures memory,
not generalisation (data leakage). This module quantifies that risk.

Method (dHash, 64 bits):

1. convert to greyscale and shrink to 9 × 8 pixels,
2. for each row, record whether each pixel is brighter than its right neighbour
   → 8 × 8 = 64 bits.

The hash depends on the *pattern of brightness gradients*, so it is largely
unchanged by global brightness/saturation changes (Roboflow's augmentations) but
changes when the scene content changes. Similarity between two images is the
**Hamming distance** between their hashes (number of differing bits, 0–64).

A threshold is not assumed: the notebook calibrates it from a *positive
control* (augmented copies of the same photo, which must count as duplicates)
and a *negative control* (pairs of different photos).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "dhash", "hash_images", "hamming_distance", "paired_hamming", "nearest_neighbours", "group_by_threshold",
    "grey_thumbnail", "thumbnail_correlation",
]

# popcount lookup table for one byte
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(image, hash_size: int = 8) -> np.ndarray:
    """
    Difference hash of one image.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image or numpy.ndarray
        Image file or image data (``(H, W)`` or ``(H, W, 3)`` uint8 array).
    hash_size : int, optional
        Hash is ``hash_size**2`` bits (default 8 → 64 bits).

    Returns
    -------
    numpy.ndarray
        Packed hash, shape ``(hash_size**2 // 8,)``, dtype ``uint8``.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            grey = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    elif isinstance(image, np.ndarray):
        grey = Image.fromarray(image).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    else:
        grey = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(grey, dtype=np.int16)
    bits = px[:, 1:] > px[:, :-1]
    return np.packbits(bits.ravel())


def hash_images(paths: Iterable[str | Path], hash_size: int = 8) -> np.ndarray:
    """
    dHash of many images.

    Parameters
    ----------
    paths : iterable of path-like
        Image files.
    hash_size : int, optional
        See :func:`dhash`.

    Returns
    -------
    numpy.ndarray
        Shape ``(K, hash_size**2 // 8)``, ``uint8`` packed hashes, in input order.
    """
    hashes = [dhash(p, hash_size) for p in paths]
    width = hash_size * hash_size // 8
    return np.vstack(hashes) if hashes else np.zeros((0, width), dtype=np.uint8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise Hamming distances between two sets of packed hashes.

    Parameters
    ----------
    a : numpy.ndarray
        Shape ``(K, B)`` packed hashes (uint8).
    b : numpy.ndarray
        Shape ``(L, B)`` packed hashes.

    Returns
    -------
    numpy.ndarray
        Shape ``(K, L)``, number of differing bits.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.uint8))
    b = np.atleast_2d(np.asarray(b, dtype=np.uint8))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Hash widths differ: {a.shape[1]} vs {b.shape[1]} bytes.")
    return _POPCOUNT[a[:, None, :] ^ b[None, :, :]].sum(axis=2, dtype=np.int32)


def paired_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hamming distance between aligned pairs of hashes (row ``i`` of ``a`` with row ``i`` of ``b``).

    Parameters
    ----------
    a, b : numpy.ndarray
        Shape ``(K, B)`` packed hashes.

    Returns
    -------
    numpy.ndarray
        Shape ``(K,)``, differing bits per pair.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.uint8))
    b = np.atleast_2d(np.asarray(b, dtype=np.uint8))
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} vs {b.shape}.")
    return _POPCOUNT[a ^ b].sum(axis=1, dtype=np.int32)


def nearest_neighbours(
    query: np.ndarray, reference: np.ndarray, same_set: bool = False, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each query hash, the closest reference hash.

    Parameters
    ----------
    query : numpy.ndarray
        Shape ``(K, B)`` packed hashes.
    reference : numpy.ndarray
        Shape ``(L, B)`` packed hashes.
    same_set : bool, optional
        If True, ``query`` and ``reference`` are the same array and each image's
        match with itself is ignored.
    chunk : int, optional
        Query rows processed at once (bounds memory to about ``chunk × L × B``
        bytes).

    Returns
    -------
    distance : numpy.ndarray
        Shape ``(K,)``, Hamming distance to the nearest reference.
    index : numpy.ndarray
        Shape ``(K,)``, index of that reference.
    """
    k = len(query)
    dist = np.empty(k, dtype=np.int32)
    idx = np.empty(k, dtype=np.int64)
    for start in range(0, k, chunk):
        d = hamming_distance(query[start:start + chunk], reference)
        if same_set:
            rows = np.arange(d.shape[0])
            d[rows, start + rows] = np.iinfo(np.int32).max
        idx[start:start + chunk] = d.argmin(axis=1)
        dist[start:start + chunk] = d.min(axis=1)
    return dist, idx


def group_by_threshold(hashes: np.ndarray, threshold: int, labels: Sequence[str] | None = None) -> list[int]:
    """
    Cluster images whose hashes are within ``threshold`` bits (single linkage).

    Uses union–find: any chain of pairs closer than the threshold ends up in one
    group. Groups can be used for a grouped train/test split or a grouped
    bootstrap, so that near-duplicates never count as independent evidence.

    Parameters
    ----------
    hashes : numpy.ndarray
        Shape ``(K, B)`` packed hashes.
    threshold : int
        Pairs with Hamming distance ≤ threshold are linked.
    labels : sequence of str, optional
        Extra pre-existing links: images with equal labels (e.g. the same
        Roboflow original stem) are always grouped together.

    Returns
    -------
    list of int
        Group id for each image (0 .. n_groups-1, in order of first appearance).
    """
    k = len(hashes)
    parent = list(range(k))

    def find(i: int) -> int:
        """Root of i's set, with path halving (keeps the trees shallow)."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        """Merge the sets containing i and j."""
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for start in range(0, k, 512):
        d = hamming_distance(hashes[start:start + 512], hashes)
        for r, c in zip(*np.nonzero(d <= threshold)):
            if start + r < c:
                union(start + r, int(c))
    if labels is not None:
        first: dict[str, int] = {}
        for i, lab in enumerate(labels):
            if lab in first:
                union(first[lab], i)
            else:
                first[lab] = i

    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(k)]


def grey_thumbnail(image, size: tuple[int, int] = (64, 48)) -> np.ndarray:
    """
    Standardised greyscale thumbnail used to recognise copies of the *same image*.

    The image is converted to greyscale, box-downsampled to ``size`` (W, H) and
    standardised to zero mean and unit variance. Standardising removes global
    brightness and contrast changes, so a colour-augmented copy (Roboflow's
    saturation ±29 % / exposure ±10 %) gives an almost identical vector, while a
    different frame does not.

    Parameters
    ----------
    image : str, pathlib.Path, PIL.Image.Image or numpy.ndarray
        Image file or data.
    size : tuple of int, optional
        Thumbnail width and height (default 64 × 48, the image aspect ratio).

    Returns
    -------
    numpy.ndarray
        Shape ``(W*H,)``, float64, mean 0 and standard deviation 1 (all zeros
        for a constant image).
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            g = im.convert("L").resize(size, Image.Resampling.BOX)
    elif isinstance(image, np.ndarray):
        g = Image.fromarray(image).convert("L").resize(size, Image.Resampling.BOX)
    else:
        g = image.convert("L").resize(size, Image.Resampling.BOX)
    v = np.asarray(g, dtype=np.float64).ravel()
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def thumbnail_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation of two standardised thumbnails.

    Parameters
    ----------
    a, b : numpy.ndarray
        Outputs of :func:`grey_thumbnail` with the same size.

    Returns
    -------
    float
        In [-1, 1]; ≈ 1 for colour-augmented copies of one image.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Thumbnail shapes differ: {a.shape} vs {b.shape}.")
    return float(np.mean(a * b))
