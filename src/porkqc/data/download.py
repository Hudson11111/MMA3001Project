"""
Downloading the Roboflow dataset and preparing its ``data.yaml``.

The dataset is not stored in the repository (size, licence attribution, and so
anyone can reproduce the download). It is fetched from Roboflow Universe with
the ``roboflow`` package using the user's own free API key.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from ..env import get_secret

__all__ = ["download_dataset", "fix_data_yaml", "read_class_names", "SPLIT_DIRS"]

#: Mapping from Ultralytics ``data.yaml`` keys to Roboflow split folder names.
SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid", "test": "test"}


def download_dataset(
    workspace: str,
    project: str,
    version: int,
    location: str | Path,
    export_format: str = "yolov8",
    api_key: str | None = None,
    overwrite: bool = False,
    retries: int = 6,
    wait_seconds: float = 10.0,
) -> Path:
    """
    Download one version of a Roboflow (Universe) dataset.

    Behaviour:

    * ``location/data.yaml`` already exists and ``overwrite`` is False → nothing
      is downloaded; the folder is returned.
    * ``location`` exists but has no ``data.yaml`` (empty or a broken earlier
      attempt) → downloaded with ``overwrite=True``. This matters because the
      ``roboflow`` package (1.7.1) silently skips the download whenever the
      target folder exists, even if it is empty (VERIFICATION_LOG V9).
    * If Roboflow reports that the export is still being generated, wait and
      retry.

    Parameters
    ----------
    workspace : str
        Roboflow workspace slug, e.g. ``"hello-2aqe0"``.
    project : str
        Project slug, e.g. ``"pork-rasher-error-packaging"``.
    version : int
        Dataset version number (this project uses 4).
    location : str or pathlib.Path
        Folder to download into. Roboflow creates it; this function does not.
    export_format : str, optional
        Roboflow export format. ``"yolov8"`` gives ``train/valid/test`` folders
        with ``images/`` and ``labels/`` and a ``data.yaml`` (same layout as the
        YOLO11/YOLO26 exports).
    api_key : str, optional
        Roboflow private API key. If omitted, read via
        :func:`porkqc.env.get_secret` from ``ROBOFLOW_API_KEY``.
    overwrite : bool, optional
        Re-download even if ``location`` already contains a dataset.
    retries : int, optional
        Attempts while the export is not ready (default 6).
    wait_seconds : float, optional
        Pause between those attempts (default 10 s).

    Returns
    -------
    pathlib.Path
        Folder containing the dataset and its ``data.yaml``.

    Raises
    ------
    RuntimeError
        If no API key is available, Roboflow returns an error, or the download
        finishes without producing ``data.yaml``.
    ImportError
        If the ``roboflow`` package is not installed.
    """
    location = Path(location)
    yaml_path = location / "data.yaml"
    if yaml_path.is_file() and not overwrite:
        return location

    if api_key is None:
        api_key = get_secret("ROBOFLOW_API_KEY")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError("Install the 'roboflow' package: pip install roboflow") from exc

    rf_version = Roboflow(api_key=api_key).workspace(workspace).project(project).version(version)
    force = overwrite or location.exists()  # roboflow skips existing folders unless overwrite=True
    for attempt in range(1, retries + 1):
        try:
            rf_version.download(export_format, location=str(location), overwrite=force)
            break
        except RuntimeError as exc:
            not_ready = "ready" in str(exc).lower()
            if not_ready and attempt < retries:
                time.sleep(wait_seconds)
                continue
            raise

    if not yaml_path.is_file():
        found = sorted(p.name for p in location.iterdir()) if location.exists() else []
        raise RuntimeError(f"Download finished but {yaml_path} is missing. Folder contains: {found[:10]}")
    return location


def _resolve_split_path(yaml_dir: Path, value: str | None, split_dir: str) -> Path:
    """Find the real images folder for one split, trying likely candidates."""
    candidates = []
    if value:
        raw = Path(value)
        candidates.append(raw if raw.is_absolute() else yaml_dir / raw)
        # Roboflow often writes '../train/images' although the folders sit next
        # to data.yaml; stripping the leading '../' fixes that case.
        parts = [p for p in raw.parts if p != ".."]
        if parts:
            candidates.append(yaml_dir.joinpath(*parts))
    candidates.append(yaml_dir / split_dir / "images")

    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find images folder for split '{split_dir}'. Tried: {tried}")


def fix_data_yaml(yaml_path: str | Path, write: bool = True) -> dict[str, Any]:
    """
    Rewrite the split paths in a Roboflow ``data.yaml`` as absolute paths.

    Roboflow exports often contain relative paths (``../train/images``) that do
    not resolve from where Ultralytics is run, causing "images not found"
    errors. This replaces ``train``, ``val`` and ``test`` with absolute paths to
    folders that actually exist.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to ``data.yaml``.
    write : bool, optional
        If ``True`` (default), save the corrected file in place and keep the
        original as ``data.yaml.orig`` (only the first time).

    Returns
    -------
    dict
        The corrected YAML contents.

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` or any split's images folder cannot be found.
    ValueError
        If the YAML has no ``names`` entry.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "names" not in data:
        raise ValueError(f"{yaml_path} has no 'names' entry.")

    yaml_dir = yaml_path.parent
    for key, split_dir in SPLIT_DIRS.items():
        data[key] = str(_resolve_split_path(yaml_dir, data.get(key), split_dir))

    if write:
        backup = yaml_path.with_name(yaml_path.name + ".orig")
        if not backup.exists():
            backup.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
    return data


def read_class_names(yaml_path: str | Path) -> list[str]:
    """
    Read the ordered class names from a ``data.yaml``.

    Handles both list form (``names: [a, b]``) and Ultralytics' dict form
    (``names: {0: a, 1: b}``).

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to ``data.yaml``.

    Returns
    -------
    list of str
        Class names in class-id order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If ``names`` is missing or malformed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fh:
        names = (yaml.safe_load(fh) or {}).get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        by_id = {int(k): str(v) for k, v in names.items()}
        keys = sorted(by_id)
        if keys != list(range(len(keys))):
            raise ValueError(f"names dict keys must be 0..N-1, got {keys}.")
        return [by_id[k] for k in keys]
    raise ValueError(f"{yaml_path}: 'names' must be a list or dict, got {type(names).__name__}.")
