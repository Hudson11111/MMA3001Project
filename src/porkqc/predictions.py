"""
Ground-truth and prediction files used for evaluation.

The GPU notebook writes two kinds of small JSON file; every metric in the project
is then computed from these files on CPU (DECISION_LOG D-018). Keeping the
format explicit makes the evaluation reproducible without re-running a model.

Ground-truth file (one per split)::

    {"split": "valid", "class_names": [...],
     "images": [{"file": "a.jpg", "width": 720, "height": 540,
                 "boxes": [[class_id, x1, y1, x2, y2], ...]}, ...]}

Prediction file (one per model, split and inference setting)::

    {"model": "yolo11n_640", "split": "valid", "settings": {...},
     "images": [{"file": "a.jpg", "width": 720, "height": 540,
                 "speed_ms": {"preprocess": .., "inference": .., "postprocess": ..},
                 "pred": [[class_id, confidence, x1, y1, x2, y2], ...]}, ...]}

Coordinates are pixels in the original image (origin top-left). Images with no
objects are kept with an empty list — they matter for false positives and for
the tray-level PASS decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data.audit import list_images
from .data.labels import read_label_file, yolo_to_xyxy

__all__ = [
    "export_ground_truth",
    "save_json",
    "load_json",
    "gt_arrays",
    "pred_arrays",
]


def export_ground_truth(split_dir: str | Path, class_names: Sequence[str], split: str | None = None) -> dict:
    """
    Convert one split's YOLO labels into a ground-truth dictionary.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Folder with ``images/`` and ``labels/``.
    class_names : sequence of str
        Ordered class names (stored in the output for traceability).
    split : str, optional
        Split name to record; defaults to the folder name.

    Returns
    -------
    dict
        Ground-truth structure described in the module docstring. Image sizes
        are read from the image files (Pillow), not assumed.

    Raises
    ------
    FileNotFoundError
        If an image has no label file (the audit should have caught this).
    porkqc.data.labels.LabelFormatError
        If a label file is malformed.
    """
    from PIL import Image  # local import: only needed here

    split_dir = Path(split_dir)
    images = []
    for img in list_images(split_dir / "images"):
        with Image.open(img) as im:
            width, height = im.size
        labels = read_label_file(split_dir / "labels" / f"{img.stem}.txt", len(class_names))
        xyxy = yolo_to_xyxy(labels, width, height)
        boxes = [[int(c), *(round(float(v), 2) for v in b)] for c, b in zip(labels[:, 0], xyxy)]
        images.append({"file": img.name, "width": width, "height": height, "boxes": boxes})
    return {"split": split or split_dir.name, "class_names": list(class_names), "images": images}


def save_json(obj: Any, path: str | Path, indent: int | None = 1) -> Path:
    """
    Write an object as JSON, creating parent folders.

    Parameters
    ----------
    obj : Any
        JSON-serialisable object.
    path : str or pathlib.Path
        Output file.
    indent : int or None, optional
        Indentation for readability (default 1). Use ``None`` for large
        prediction files: one line per number would make them several times
        larger.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
    return path


def load_json(path: str | Path) -> Any:
    """
    Read a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read.

    Returns
    -------
    Any
        Parsed contents.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def gt_arrays(image_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Ground-truth boxes of one image as arrays.

    Parameters
    ----------
    image_entry : dict
        One element of a ground-truth file's ``images`` list.

    Returns
    -------
    classes : numpy.ndarray
        Shape ``(N,)``, integer class ids.
    boxes : numpy.ndarray
        Shape ``(N, 4)``, ``[x1, y1, x2, y2]`` pixels.
    """
    rows = np.asarray(image_entry.get("boxes", []), dtype=np.float64).reshape(-1, 5)
    return rows[:, 0].astype(int), rows[:, 1:5]


def pred_arrays(image_entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predictions of one image as arrays.

    Parameters
    ----------
    image_entry : dict
        One element of a prediction file's ``images`` list.

    Returns
    -------
    classes : numpy.ndarray
        Shape ``(M,)``, integer class ids.
    scores : numpy.ndarray
        Shape ``(M,)``, confidences in [0, 1].
    boxes : numpy.ndarray
        Shape ``(M, 4)``, ``[x1, y1, x2, y2]`` pixels.
    """
    rows = np.asarray(image_entry.get("pred", []), dtype=np.float64).reshape(-1, 6)
    return rows[:, 0].astype(int), rows[:, 1], rows[:, 2:6]
