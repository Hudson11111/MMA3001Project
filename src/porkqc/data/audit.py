"""
Dataset audit: what is actually in each split?

Answers the questions the Roboflow web page cannot:
how many objects of each class there are, how many images have no defects
("null" images — needed to validate a PASS decision), whether label files are
missing or malformed, and how many *original* images sit behind the augmented
training copies.

This module only reads file names and label files; image-content checks are in
:mod:`porkqc.data.duplicates` and :mod:`porkqc.data.resplit`.
"""

from __future__ import annotations

import random
import re
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .labels import LabelFormatError, read_label_file

__all__ = [
    "IMAGE_EXTENSIONS",
    "SplitSummary",
    "original_stem",
    "frame_number",
    "list_images",
    "summarise_split",
    "box_table",
    "image_classes",
    "summarise_dataset",
    "make_dev_sample",
]

#: File extensions treated as images (lower-case).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def original_stem(filename: str | Path) -> str:
    """
    Recover the original image name from a Roboflow export file name.

    Roboflow names exported files ``<original>_<ext>.rf.<hash>.<ext>``; every
    augmented copy of one source image shares the part before ``.rf.``. Grouping
    by this stem lets us count originals and keep all copies of an image in the
    same split (avoiding leakage).

    Parameters
    ----------
    filename : str or pathlib.Path
        File name or path, e.g. ``"tray_012_jpg.rf.9f8e7d.jpg"``.

    Returns
    -------
    str
        ``"tray_012_jpg"`` for the example above; for names without ``.rf.`` the
        plain stem (``"tray_012"`` for ``"tray_012.jpg"``).
    """
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.", 1)[0]
    return Path(name).stem


_FRAME_RE = re.compile(r"(\d+)_(?:jpe?g|png|bmp)$", re.IGNORECASE)


def frame_number(filename: str | Path) -> int | None:
    """
    Video frame index encoded in a file name, if any.

    This dataset's images are frames from conveyor video named like
    ``frame_with_red_2247_jpg.rf.<hash>.jpg``; the number before the original
    extension is the frame index. Frame order is the natural *time* axis, so it
    defines which images show the same tray (neighbouring frames).

    Parameters
    ----------
    filename : str or pathlib.Path
        Roboflow export name or original stem.

    Returns
    -------
    int or None
        The frame index, or None if the name contains no trailing frame number.
    """
    m = _FRAME_RE.search(original_stem(filename))
    return int(m.group(1)) if m else None


def list_images(images_dir: str | Path) -> list[Path]:
    """
    List image files in a folder, sorted by name for reproducibility.

    Parameters
    ----------
    images_dir : str or pathlib.Path
        Folder to scan (not recursive).

    Returns
    -------
    list of pathlib.Path
        Files whose lower-cased suffix is in :data:`IMAGE_EXTENSIONS`.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


@dataclass
class SplitSummary:
    """
    Counts describing one dataset split.

    Attributes
    ----------
    split : str
        Split name (``"train"``, ``"valid"`` or ``"test"``).
    n_images : int
        Number of image files.
    n_unique_originals : int
        Number of distinct original images (see :func:`original_stem`).
    n_missing_labels : int
        Images with no matching ``.txt`` label file.
    n_null_images : int
        Images whose label file exists but is empty (no defects labelled).
    n_invalid_label_files : int
        Label files that failed validation (details in ``issues``).
    n_instances : int
        Total labelled objects in valid label files.
    instances_per_class : list of int
        Object count per class id.
    images_per_class : list of int
        Number of images containing at least one object of each class.
    boxes_per_image_max : int
        Largest number of objects in a single image.
    issues : list of str
        Human-readable problems found (missing/invalid labels), capped in length.
    """

    split: str
    n_images: int = 0
    n_unique_originals: int = 0
    n_missing_labels: int = 0
    n_null_images: int = 0
    n_invalid_label_files: int = 0
    n_instances: int = 0
    instances_per_class: list[int] = field(default_factory=list)
    images_per_class: list[int] = field(default_factory=list)
    boxes_per_image_max: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Return the summary as a plain dictionary.

        Returns
        -------
        dict
            All attributes, JSON-serialisable (used for ``audit_summary.json``).
        """
        return asdict(self)


def summarise_split(
    split_dir: str | Path, n_classes: int, split: str | None = None, max_issues: int = 50
) -> SplitSummary:
    """
    Count images, labels, nulls and per-class objects in one split folder.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Folder containing ``images/`` and ``labels/`` sub-folders.
    n_classes : int
        Number of classes (used to validate class ids).
    split : str, optional
        Name to record; defaults to the folder name.
    max_issues : int, optional
        Maximum number of issue messages kept (counts are always complete).

    Returns
    -------
    SplitSummary
        Counts for this split. Malformed label files are counted and described
        in ``issues`` rather than raising, so one bad file cannot hide the rest
        of the audit.

    Raises
    ------
    FileNotFoundError
        If ``split_dir/images`` does not exist.
    """
    split_dir = Path(split_dir)
    images = list_images(split_dir / "images")
    labels_dir = split_dir / "labels"

    s = SplitSummary(
        split=split or split_dir.name,
        instances_per_class=[0] * n_classes,
        images_per_class=[0] * n_classes,
    )
    s.n_images = len(images)
    s.n_unique_originals = len({original_stem(p) for p in images})

    def note(msg: str) -> None:
        """Keep an issue message, up to ``max_issues`` of them."""
        if len(s.issues) < max_issues:
            s.issues.append(msg)

    for img in images:
        label_path = labels_dir / f"{img.stem}.txt"
        if not label_path.is_file():
            s.n_missing_labels += 1
            note(f"missing label: {label_path.name}")
            continue
        try:
            labels = read_label_file(label_path, n_classes)
        except LabelFormatError as exc:
            s.n_invalid_label_files += 1
            note(f"invalid label: {exc}")
            continue
        if labels.shape[0] == 0:
            s.n_null_images += 1
            continue
        classes = labels[:, 0].astype(int)
        s.n_instances += len(classes)
        s.boxes_per_image_max = max(s.boxes_per_image_max, len(classes))
        for c in classes:
            s.instances_per_class[c] += 1
        for c in set(classes.tolist()):
            s.images_per_class[c] += 1
    return s


def box_table(split_dir: str | Path, n_classes: int, split: str | None = None) -> list[dict]:
    """
    One row per labelled box, for box-size/position statistics.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Folder with ``images/`` and ``labels/``.
    n_classes : int
        Number of classes.
    split : str, optional
        Name recorded in each row (defaults to the folder name).

    Returns
    -------
    list of dict
        Keys: ``split``, ``file``, ``original`` (Roboflow original stem),
        ``class_id``, ``cx``, ``cy``, ``w``, ``h`` (normalised to [0, 1]),
        ``area_frac`` (w·h, fraction of the image) and ``aspect`` (w/h in
        normalised units). Images with missing or invalid labels are skipped
        (they are reported by :func:`summarise_split`).
    """
    split_dir = Path(split_dir)
    rows = []
    for img in list_images(split_dir / "images"):
        try:
            labels = read_label_file(split_dir / "labels" / f"{img.stem}.txt", n_classes)
        except (FileNotFoundError, LabelFormatError):
            continue
        for c, cx, cy, w, h in labels:
            rows.append(
                {
                    "split": split or split_dir.name,
                    "file": img.name,
                    "original": original_stem(img),
                    "class_id": int(c),
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "area_frac": float(w * h),
                    "aspect": float(w / h),
                }
            )
    return rows


def image_classes(split_dir: str | Path, n_classes: int) -> dict[str, list[int]]:
    """
    Sorted list of distinct classes present in each image of a split.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Folder with ``images/`` and ``labels/``.
    n_classes : int
        Number of classes.

    Returns
    -------
    dict of str to list of int
        Image file name → classes present (empty list for a defect-free image).
        Images with missing/invalid labels are omitted.
    """
    split_dir = Path(split_dir)
    out = {}
    for img in list_images(split_dir / "images"):
        try:
            labels = read_label_file(split_dir / "labels" / f"{img.stem}.txt", n_classes)
        except (FileNotFoundError, LabelFormatError):
            continue
        out[img.name] = sorted(set(labels[:, 0].astype(int).tolist()))
    return out


def summarise_dataset(
    root: str | Path, n_classes: int, splits: Sequence[str] = ("train", "valid", "test")
) -> dict[str, SplitSummary]:
    """
    Run :func:`summarise_split` on every split of a dataset.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root containing one folder per split.
    n_classes : int
        Number of classes.
    splits : sequence of str, optional
        Split folder names to include.

    Returns
    -------
    dict of str to SplitSummary
        One summary per split, in the order given.
    """
    root = Path(root)
    return {name: summarise_split(root / name, n_classes, split=name) for name in splits}


def make_dev_sample(
    root: str | Path,
    out_zip: str | Path,
    class_names: Sequence[str],
    n_per_split: dict[str, int] | None = None,
    seed: int = 0,
) -> Path:
    """
    Write a small, reproducible subset of the dataset to a zip file.

    Used to give a development environment without Roboflow access (e.g. the AI
    assistant's workspace) real images to build and test code against. The
    training sample first takes one image containing each class (where
    possible) so that rare classes are represented, then fills up at random.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root with ``train/``, ``valid/``, ``test/`` split folders.
    out_zip : str or pathlib.Path
        Output zip path.
    class_names : sequence of str
        Ordered class names (written into the sample's ``data.yaml``).
    n_per_split : dict of str to int, optional
        Images per split. Default ``{"train": 40, "valid": 20, "test": 20}``.
    seed : int, optional
        Random seed; the same seed gives the same sample.

    Returns
    -------
    pathlib.Path
        Path to the written zip. Inside: ``dev_sample/data.yaml`` and
        ``dev_sample/<split>/{images,labels}/``.
    """
    root, out_zip = Path(root), Path(out_zip)
    n_per_split = n_per_split or {"train": 40, "valid": 20, "test": 20}
    n_classes = len(class_names)
    rng = random.Random(seed)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for split, n_wanted in n_per_split.items():
            images = list_images(root / split / "images")
            rng.shuffle(images)

            # Classes present in each image (skip images with unusable labels).
            classes_in: dict[Path, set[int]] = {}
            for img in images:
                lp = root / split / "labels" / f"{img.stem}.txt"
                try:
                    classes_in[img] = set(read_label_file(lp, n_classes)[:, 0].astype(int).tolist())
                except (FileNotFoundError, LabelFormatError):
                    continue

            chosen: list[Path] = []
            for c in range(n_classes):  # one example of each class first
                for img, cls in classes_in.items():
                    if c in cls and img not in chosen:
                        chosen.append(img)
                        break
            for img in classes_in:  # then fill up at random
                if len(chosen) >= n_wanted:
                    break
                if img not in chosen:
                    chosen.append(img)

            for img in chosen[:n_wanted]:
                zf.write(img, f"dev_sample/{split}/images/{img.name}")
                zf.write(
                    root / split / "labels" / f"{img.stem}.txt", f"dev_sample/{split}/labels/{img.stem}.txt"
                )

        data_yaml = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": n_classes,
            "names": list(class_names),
        }
        zf.writestr("dev_sample/data.yaml", yaml.safe_dump(data_yaml, sort_keys=False))
    return out_zip
