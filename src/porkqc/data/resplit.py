"""
Time-blocked re-split of the dataset (DECISION_LOG D-008).

Problem: the images are frames of conveyor video and Roboflow assigned frames to
train / valid / test at random, so the same tray appears in several splits
(e.g. frame 2247 in VALID and frame 2248 in TEST). Scores on such a test split
measure recognition of trays already seen, not generalisation to new trays.

Remedy used in time-series validation: split by **contiguous blocks of time**
rather than by individual samples, and **purge** samples that lie within a gap of
a block boundary belonging to a different split, so that no tray can straddle
two splits.

Procedure (:func:`assign_time_blocks`):

1. block id = frame index // ``block_size``;
2. shuffle the block ids with a fixed seed and fill train, valid, test in turn
   until each reaches its target fraction of original images;
3. purge: an image within ``gap`` frames of an image from another split is
   dropped — from the train side where possible, so evaluation sets keep their size.

The unit of assignment is a **scene** (one original image and its augmented
copies). Scenes are found by :func:`collect_frames`; file names alone are not
enough because different frames in this dataset can share a name.

:func:`materialise_split` then builds a YOLO-format folder for the new split
using symbolic links (no copying). Every augmented copy of a training scene is
kept for training; VALID/TEST get a single copy per scene so that no image is
counted twice in an evaluation.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from .audit import frame_number, list_images, original_stem

__all__ = ["SPLIT_NAMES", "nearest_other_gap", "assign_time_blocks", "collect_frames", "scene_of", "materialise_split"]

SPLIT_NAMES = ("train", "valid", "test")


def nearest_other_gap(frames: np.ndarray, splits: np.ndarray, target: str, other: Sequence[str]) -> np.ndarray:
    """
    For each image in split ``target``, the frame distance to the nearest image in ``other`` splits.

    Parameters
    ----------
    frames : numpy.ndarray
        Shape ``(K,)``, frame index of every image.
    splits : numpy.ndarray
        Shape ``(K,)``, split name of every image.
    target : str
        Split whose images are measured (e.g. ``"test"``).
    other : sequence of str
        Splits to search (e.g. ``["train"]``).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_target,)``, non-negative integer gaps (in frames), in the
        order the target images appear; ``inf`` if ``other`` is empty.
    """
    frames = np.asarray(frames)
    splits = np.asarray(splits)
    q = frames[splits == target]
    ref = np.sort(frames[np.isin(splits, list(other))])
    if ref.size == 0:
        return np.full(q.shape, np.inf)
    pos = np.searchsorted(ref, q)
    left = np.abs(q - ref[np.clip(pos - 1, 0, ref.size - 1)])
    right = np.abs(ref[np.clip(pos, 0, ref.size - 1)] - q)
    return np.minimum(left, right)


def assign_time_blocks(
    frames: Mapping[str, int],
    block_size: int,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    gap: int = 0,
    seed: int = 0,
) -> dict[str, str | None]:
    """
    Assign original images to train/valid/test by contiguous blocks of frames.

    Parameters
    ----------
    frames : mapping of str to int
        Original image stem → frame index.
    block_size : int
        Frames per block (> 0). Should exceed how long one tray stays in view.
    fractions : sequence of 3 floats, optional
        Target share of originals for train, valid, test (summing to 1).
    gap : int, optional
        Purge distance in frames (≥ 0): no two kept images from different
        splits are closer than ``gap + 1`` frames.
    seed : int, optional
        Seed for the block shuffle.

    Returns
    -------
    dict of str to (str or None)
        Original stem → ``"train"``, ``"valid"``, ``"test"`` or None (purged).

    Raises
    ------
    ValueError
        For a non-positive block size, negative gap, or fractions that do not
        have three entries summing to 1.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if gap < 0:
        raise ValueError("gap must be >= 0.")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three numbers summing to 1.")

    names = sorted(frames)
    f = np.array([frames[n] for n in names])
    blocks = f // block_size
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    counts = Counter(blocks.tolist())
    total = len(names)
    targets = np.cumsum(fractions) * total
    block_split: dict[int, str] = {}
    filled = 0
    for b in unique_blocks:
        k = int(np.searchsorted(targets, filled, side="right"))
        block_split[int(b)] = SPLIT_NAMES[min(k, 2)]
        filled += counts[int(b)]
    split = np.array([block_split[int(b)] for b in blocks], dtype=object)

    if gap > 0:
        # Purge: walk splits in priority order (test kept first, then valid, then train).
        keep = np.ones(total, dtype=bool)
        for s, others in (("valid", ("test",)), ("train", ("valid", "test"))):
            idx = np.where(split == s)[0]
            ref = f[(split != s) & np.isin(split, others) & keep]
            if ref.size == 0 or idx.size == 0:
                continue
            ref = np.sort(ref)
            pos = np.searchsorted(ref, f[idx])
            d = np.minimum(np.abs(f[idx] - ref[np.clip(pos - 1, 0, ref.size - 1)]),
                           np.abs(ref[np.clip(pos, 0, ref.size - 1)] - f[idx]))
            keep[idx[d <= gap]] = False
        split = np.where(keep, split, None)

    return {n: (None if s is None else str(s)) for n, s in zip(names, split)}


def collect_frames(
    dataset_root: str | Path, separate_distinct: bool = True, copy_threshold: float = 0.99
) -> tuple[dict[str, int], dict[str, list[Path]], list[str]]:
    """
    Gather every original image ("scene") in a Roboflow export, across its splits.

    Roboflow names every file ``<original>_jpg.rf.<hash>.jpg``, and augmented
    copies of one image share ``<original>``. **But the name is not unique per
    image in this dataset**: different frames (apparently from different videos
    numbered the same way) can share a name, e.g. two different trays both named
    ``frame_with_red_992``. With ``separate_distinct=True`` the files sharing a
    name are therefore grouped by content: files whose standardised grey
    thumbnails correlate ≥ ``copy_threshold`` are copies of one scene; others
    become separate scenes ``<name>#1``, ``<name>#2``, …

    Parameters
    ----------
    dataset_root : str or pathlib.Path
        Folder with ``train/``, ``valid/``, ``test/`` (each ``images/`` + ``labels/``).
    separate_distinct : bool, optional
        Split same-named files by content (default True). False groups purely by name.
    copy_threshold : float, optional
        Correlation above which two files are copies of the same image. Colour
        augmentation gives ≥ 0.999; different frames of this dataset ≤ ~0.93.

    Returns
    -------
    frames : dict of str to int
        Scene id → frame index (scenes without a frame number are omitted).
    files : dict of str to list of pathlib.Path
        Scene id → its image files (augmented copies included), sorted.
    unparsed : list of str
        Image names whose frame number could not be read.
    """
    from .duplicates import grey_thumbnail, thumbnail_correlation

    root = Path(dataset_root)
    by_name: dict[str, list[Path]] = {}
    unparsed: list[str] = []
    for split in SPLIT_NAMES:
        d = root / split / "images"
        if not d.is_dir():
            continue
        for img in list_images(d):
            if frame_number(img) is None:
                unparsed.append(img.name)
                continue
            by_name.setdefault(original_stem(img), []).append(img)

    frames: dict[str, int] = {}
    files: dict[str, list[Path]] = {}
    for name, paths in sorted(by_name.items()):
        paths = sorted(paths)
        groups: list[list[Path]] = [paths]
        if separate_distinct and len(paths) > 1:
            groups, reps = [], []
            for p in paths:
                t = grey_thumbnail(p)
                for g, r in zip(groups, reps):
                    if thumbnail_correlation(t, r) >= copy_threshold:
                        g.append(p)
                        break
                else:
                    groups.append([p])
                    reps.append(t)
        for k, g in enumerate(groups):
            scene = name if k == 0 else f"{name}#{k}"
            frames[scene] = frame_number(g[0])
            files[scene] = g
    return frames, files, unparsed


def scene_of(files: Mapping[str, Sequence[Path]]) -> dict[str, str]:
    """
    Invert a scene → files mapping to file name → scene id.

    Parameters
    ----------
    files : mapping of str to sequence of pathlib.Path
        From :func:`collect_frames`.

    Returns
    -------
    dict of str to str
        Image file name → scene id.
    """
    return {p.name: scene for scene, paths in files.items() for p in paths}


def materialise_split(
    dataset_root: str | Path,
    out_root: str | Path,
    assignment: Mapping[str, str | None],
    class_names: Sequence[str],
    files: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, int]:
    """
    Build a YOLO-format dataset for a new assignment using symbolic links.

    Parameters
    ----------
    dataset_root : str or pathlib.Path
        Original Roboflow export.
    out_root : str or pathlib.Path
        New dataset folder (created; existing links are replaced).
    assignment : mapping of str to (str or None)
        Scene id → split, from :func:`assign_time_blocks`.
    class_names : sequence of str
        Written to the new ``data.yaml``.
    files : mapping of str to sequence of pathlib.Path, optional
        Scene id → files from :func:`collect_frames` (recomputed if omitted;
        pass it to avoid re-reading every image).

    Returns
    -------
    dict of str to int
        Number of images linked per split.
    """
    out_root = Path(out_root)
    if files is None:
        _, files, _ = collect_frames(dataset_root)
    counts = {s: 0 for s in SPLIT_NAMES}
    for s in SPLIT_NAMES:
        for sub in ("images", "labels"):
            (out_root / s / sub).mkdir(parents=True, exist_ok=True)
    for stem, split in assignment.items():
        if split is None or stem not in files:
            continue
        chosen = files[stem] if split == "train" else files[stem][:1]
        for img in chosen:
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            for src, dst in ((img, out_root / split / "images" / img.name),
                             (label, out_root / split / "labels" / label.name)):
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src.resolve(), dst)
            counts[split] += 1
    data = {
        "path": str(out_root.resolve()),
        "train": str((out_root / "train" / "images").resolve()),
        "val": str((out_root / "valid" / "images").resolve()),
        "test": str((out_root / "test" / "images").resolve()),
        "nc": len(class_names),
        "names": list(class_names),
    }
    with (out_root / "data.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return counts
