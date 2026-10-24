"""
Reading and validating YOLO-format bounding-box labels.

Format (one ``.txt`` file per image, one line per object)::

    class_id  cx  cy  w  h

* ``class_id`` — integer index into the class-name list (0-based).
* ``cx, cy`` — box centre, normalised by image width / height to [0, 1].
* ``w, h`` — box width / height, normalised the same way, in (0, 1].

An empty file means the image contains no labelled objects.

Validation is strict on purpose: a malformed label silently accepted here would
quietly distort every downstream metric, whereas an exception names the exact
file and line to fix.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = [
    "LabelFormatError",
    "parse_label_line",
    "read_label_file",
    "yolo_to_xyxy",
]

#: Default tolerance on normalised coordinates. Roboflow writes ~6 decimal
#: places, so values like 1.0000004 can appear for boxes touching the edge.
DEFAULT_TOL = 1e-3


class LabelFormatError(ValueError):
    """Raised when a YOLO label line or file violates the expected format."""


def parse_label_line(line: str, n_classes: int, tol: float = DEFAULT_TOL) -> np.ndarray:
    """
    Parse and validate one YOLO label line.

    Parameters
    ----------
    line : str
        A single non-blank line, e.g. ``"3 0.51 0.42 0.20 0.11"``.
    n_classes : int
        Number of classes; valid class ids are ``0 .. n_classes - 1``.
    tol : float, optional
        Allowed overshoot of normalised coordinates beyond [0, 1].

    Returns
    -------
    numpy.ndarray
        Shape ``(5,)``, dtype ``float64``: ``[class_id, cx, cy, w, h]``.

    Raises
    ------
    LabelFormatError
        If the line does not have exactly 5 fields, a field is not numeric or not
        finite, the class id is not an integer in range, the size is not
        positive, or the box extends outside the image by more than ``tol``.
    """
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}.")

    fields = line.split()
    if len(fields) != 5:
        raise LabelFormatError(
            f"Expected 5 fields 'class cx cy w h', got {len(fields)}: {line!r}. "
            "(More than 5 fields usually means polygon/segmentation labels.)"
        )
    try:
        values = [float(f) for f in fields]
    except ValueError as exc:
        raise LabelFormatError(f"Non-numeric field in label line: {line!r}") from exc
    if not all(math.isfinite(v) for v in values):
        raise LabelFormatError(f"Non-finite value in label line: {line!r}")

    cls, cx, cy, w, h = values
    if not cls.is_integer() or not (0 <= cls < n_classes):
        raise LabelFormatError(f"Class id must be an integer in [0, {n_classes - 1}], got {fields[0]!r}.")
    if w <= 0 or h <= 0:
        raise LabelFormatError(f"Box width and height must be positive: {line!r}")

    # Check box edges, not only the centre: a centre inside the image can still
    # describe a box that hangs off the edge.
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    if min(x1, y1) < -tol or max(x2, y2) > 1 + tol:
        raise LabelFormatError(f"Box extends outside the normalised image [0, 1] (tol={tol}): {line!r}")
    return np.array(values, dtype=np.float64)


def read_label_file(path: str | Path, n_classes: int, tol: float = DEFAULT_TOL) -> np.ndarray:
    """
    Read and validate a YOLO label file.

    Blank lines are ignored. An empty file is valid and means "no objects".

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.txt`` label file.
    n_classes : int
        Number of classes (see :func:`parse_label_line`).
    tol : float, optional
        Coordinate tolerance (see :func:`parse_label_line`).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 5)``, dtype ``float64``, rows ``[class_id, cx, cy, w, h]``.
        ``N = 0`` for an empty file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    LabelFormatError
        If any line is invalid; the message includes the file and line number.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(parse_label_line(line, n_classes, tol))
            except LabelFormatError as exc:
                raise LabelFormatError(f"{path}:{line_no}: {exc}") from exc

    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.vstack(rows)


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert normalised YOLO boxes to pixel corner coordinates.

    Parameters
    ----------
    boxes : numpy.ndarray
        Shape ``(N, 4)`` as ``[cx, cy, w, h]`` or ``(N, 5)`` as
        ``[class_id, cx, cy, w, h]`` (the class column is dropped).
        Values normalised to [0, 1].
    width, height : int
        Image size in pixels (e.g. 720 and 540 for this dataset).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 4)``, dtype ``float64``: ``[x1, y1, x2, y2]`` in pixels,
        origin at the top-left, x to the right, y downwards.

    Raises
    ------
    ValueError
        If ``boxes`` is not 2-D with 4 or 5 columns, or the image size is not
        positive.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] not in (4, 5):
        raise ValueError(f"boxes must have shape (N, 4) or (N, 5), got {boxes.shape}.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got width={width}, height={height}.")

    cx, cy, w, h = (boxes[:, -4:]).T
    out = np.empty((boxes.shape[0], 4), dtype=np.float64)
    out[:, 0] = (cx - w / 2) * width
    out[:, 1] = (cy - h / 2) * height
    out[:, 2] = (cx + w / 2) * width
    out[:, 3] = (cy + h / 2) * height
    return out
