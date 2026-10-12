"""
Shared pytest fixtures.

Tests never need the real dataset or a GPU: fixtures build tiny synthetic YOLO
datasets on disk whose correct audit results are known by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

N_CLASSES = 5

# (file name, label file content or None for "no label file")
# Known by construction (train includes EXTRA_TRAIN):
#   train: 5 images, 4 unique originals (two Roboflow copies of 'a_jpg'),
#          1 null image, 1 missing label, instances per class [2, 0, 1, 0, 1]
#   valid: 2 images, 1 invalid label file (class id 7), 1 image with class 3
#   test:  1 image with class 1
SYNTHETIC = {
    "train": [
        ("a_jpg.rf.111.jpg", "0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"),
        ("a_jpg.rf.222.jpg", "0 0.5 0.5 0.2 0.2\n"),
        ("b_jpg.rf.333.jpg", ""),  # null image: no defects
        ("c_jpg.rf.444.jpg", None),  # missing label file
    ],
    "valid": [
        ("d_jpg.rf.555.jpg", "7 0.5 0.5 0.2 0.2\n"),  # invalid class id
        ("e_jpg.rf.666.jpg", "3 0.5 0.5 0.2 0.2\n"),
    ],
}
EXTRA_TRAIN = [("f_jpg.rf.777.jpg", "4 0.1 0.1 0.1 0.1\n")]


def _write_split(root: Path, split: str, items) -> None:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for name, content in items:
        (root / split / "images" / name).write_bytes(b"\xff\xd8fake-jpeg")
        if content is not None:
            stem = Path(name).stem
            (root / split / "labels" / f"{stem}.txt").write_text(content)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a tiny Roboflow-style YOLO dataset and return its root folder."""
    root = tmp_path / "ds"
    _write_split(root, "train", SYNTHETIC["train"] + EXTRA_TRAIN)
    _write_split(root, "valid", SYNTHETIC["valid"])
    _write_split(root, "test", [("g_jpg.rf.888.jpg", "1 0.5 0.5 0.4 0.4\n")])
    return root
