"""
Configuration loading and consistency checks.

Experiment settings live in YAML files under ``configs/`` so that every run is
described by a file in version control rather than by values typed into a
notebook cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

__all__ = ["load_yaml", "check_class_names"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file into a dictionary.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict
        Parsed contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or its top level is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(data).__name__}.")
    return data


def check_class_names(found: Sequence[str], expected: Sequence[str]) -> None:
    """
    Check that class names (and their order) match what the code expects.

    Class ids in YOLO label files are positions in this list, so a silent
    reordering would corrupt every metric. This check fails loudly instead.

    Parameters
    ----------
    found : sequence of str
        Names read from the dataset's ``data.yaml`` (``names`` field).
    expected : sequence of str
        Names the project code assumes, e.g. :data:`porkqc.CLASS_NAMES`.

    Raises
    ------
    ValueError
        If the lists differ in length, content or order.
    """
    found, expected = list(found), list(expected)
    if found != expected:
        raise ValueError(
            "Class names in data.yaml do not match the expected list.\n"
            f"  found:    {found}\n  expected: {expected}\n"
            "Update porkqc.CLASS_NAMES and configs/dataset.yaml only after checking "
            "that label class ids still mean the same thing."
        )
