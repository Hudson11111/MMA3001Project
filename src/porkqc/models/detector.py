"""
Thin, documented wrapper around Ultralytics YOLO for training and prediction.

Why a wrapper: every training run is described by a YAML experiment file
(``configs/experiments/``), runs resume automatically after a Colab disconnect,
and each run writes a ``run_meta.json`` recording settings, library versions,
hardware and wall-clock time — the information needed to reproduce or explain
a result later.

Ultralytics (AGPL-3.0) is imported lazily so the rest of the package and its
tests work without it.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Iterable

from ..config import load_yaml
from ..predictions import load_json, save_json

__all__ = ["load_experiment", "train_detector", "predict_images", "validate_detector"]

_REQUIRED_KEYS = ("name", "model", "train", "predict")


def load_experiment(path: str | Path) -> dict[str, Any]:
    """
    Load and check an experiment YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        e.g. ``configs/experiments/yolo11n_640.yaml``.

    Returns
    -------
    dict
        With keys ``name`` (run name), ``model`` (pretrained weights, e.g.
        ``"yolo11n.pt"``), ``train`` (keyword arguments for Ultralytics
        ``train``), ``predict`` (keyword arguments for ``predict``) and optional
        ``predict_variants`` (list of overrides, each with a ``tag``).

    Raises
    ------
    ValueError
        If a required key is missing or ``train``/``predict`` are not mappings.
    """
    exp = load_yaml(path)
    missing = [k for k in _REQUIRED_KEYS if k not in exp]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    for key in ("train", "predict"):
        if not isinstance(exp[key], dict):
            raise ValueError(f"{path}: '{key}' must be a mapping")
    exp.setdefault("predict_variants", [{"tag": "default"}])
    return exp


def _environment() -> dict[str, Any]:
    """Library versions and hardware, recorded with every run."""
    env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import ultralytics

        env.update(torch=torch.__version__, ultralytics=ultralytics.__version__,
                   cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def _is_resumable(checkpoint: Path) -> bool:
    """
    True if an Ultralytics checkpoint can be resumed (same test Ultralytics uses).

    A checkpoint is resumable when it still stores a non-negative ``epoch`` and
    the optimiser state; finished runs have both stripped.
    """
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)  # our own file
    return ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def train_detector(
    exp: dict[str, Any], data_yaml: str | Path, project_dir: str | Path, force: bool = False
) -> dict[str, Any]:
    """
    Train (or resume, or skip) one experiment.

    Behaviour:

    * ``run_meta.json`` exists in the run folder and ``force`` is False →
      training already finished; return the saved metadata.
    * ``weights/last.pt`` is a *resumable* checkpoint (still holds epoch and
      optimiser state) → resume the interrupted run.
    * ``weights/last.pt`` exists but is *not* resumable → training finished
      (Ultralytics strips the optimiser at the end, including after early
      stopping) but the session died before ``run_meta.json`` was written;
      write the metadata and skip. Without this check, Ultralytics would
      silently start a brand-new training run — found in a CPU smoke test.
    * otherwise → start from the pretrained ``exp["model"]`` weights.

    Parameters
    ----------
    exp : dict
        Experiment from :func:`load_experiment`.
    data_yaml : str or pathlib.Path
        Dataset ``data.yaml`` with absolute split paths (see
        :func:`porkqc.data.download.fix_data_yaml`).
    project_dir : str or pathlib.Path
        Parent folder for runs; on Colab use a Google Drive path so runs
        survive disconnects. The run folder is ``project_dir / exp["name"]``.
    force : bool, optional
        Retrain even if a finished run exists (the old folder is overwritten).

    Returns
    -------
    dict
        Run metadata: experiment, data path, run folder, best weights path,
        ``train_seconds`` (wall-clock time of the call that completed training;
        for a resumed run only the resumed part), environment, and flags
        ``resumed`` and ``skipped`` (``skipped`` is True when an already
        finished run was found and nothing was trained).
    """
    from ultralytics import YOLO

    run_dir = Path(project_dir) / exp["name"]
    meta_path = run_dir / "run_meta.json"
    last = run_dir / "weights" / "last.pt"

    if meta_path.exists() and not force:
        meta = load_json(meta_path)
        meta["skipped"] = True
        return meta

    def _meta(seconds: float | None, resumed: bool, note: str = "") -> dict[str, Any]:
        return {
            "experiment": exp,
            "data_yaml": str(data_yaml),
            "run_dir": str(run_dir),
            "best_weights": str(run_dir / "weights" / "best.pt"),
            "train_seconds": None if seconds is None else round(seconds, 1),
            "resumed": resumed,
            "skipped": False,
            "note": note,
            "environment": _environment(),
        }

    if last.exists() and not force and not _is_resumable(last):
        meta = _meta(None, False, "finished run found without run_meta.json; metadata reconstructed")
        save_json(meta, meta_path)
        meta["skipped"] = True
        return meta

    t0 = time.perf_counter()
    resumed = last.exists() and not force
    if resumed:
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(exp["model"]).train(
            data=str(data_yaml),
            project=str(project_dir),
            name=exp["name"],
            exist_ok=True,
            **exp["train"],
        )
    meta = _meta(time.perf_counter() - t0, resumed)
    save_json(meta, meta_path)
    return meta


def predict_images(
    weights: str | Path,
    image_paths: Iterable[str | Path],
    model_name: str,
    split: str,
    **predict_kwargs: Any,
) -> dict[str, Any]:
    """
    Run a trained detector on images and return a prediction dictionary.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained weights (``best.pt``).
    image_paths : iterable of path-like
        Images to process, in a fixed order.
    model_name : str
        Name recorded in the output (e.g. ``"yolo11n_640"``).
    split : str
        Split name recorded in the output.
    **predict_kwargs
        Passed to Ultralytics ``predict``: e.g. ``imgsz``, ``conf`` (use a low
        value such as 0.001 for precision–recall analysis), ``iou`` (NMS
        threshold), ``max_det``, ``quantize`` (16 for FP16), ``nms`` (False for
        YOLO26's NMS-free head), ``device``.

    Returns
    -------
    dict
        Prediction structure described in :mod:`porkqc.predictions`, including
        Ultralytics' per-image timing (pre-process / inference / post-process,
        milliseconds). Predictions are sorted by descending confidence.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = []
    for path in image_paths:
        # One image per call keeps the order explicit and timing per image.
        res = model.predict(str(path), verbose=False, **predict_kwargs)[0]
        h, w = res.orig_shape
        boxes = res.boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        order = conf.argsort()[::-1]
        pred = [
            [int(cls[i]), round(float(conf[i]), 5), *(round(float(v), 2) for v in xyxy[i])]
            for i in order
        ]
        images.append({
            "file": Path(path).name,
            "width": int(w),
            "height": int(h),
            "speed_ms": {k: round(float(v), 3) for k, v in res.speed.items()},
            "pred": pred,
        })
    return {
        "model": model_name,
        "weights": str(weights),
        "split": split,
        "settings": {k: v for k, v in predict_kwargs.items()},
        "environment": _environment(),
        "images": images,
    }


def validate_detector(weights: str | Path, data_yaml: str | Path, split: str = "val", **val_kwargs: Any) -> dict:
    """
    Ultralytics' own validation, used only as an independent cross-check.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained weights.
    data_yaml : str or pathlib.Path
        Dataset ``data.yaml``.
    split : str, optional
        ``"val"`` or ``"test"``. The test split is used once, at the end
        (DECISION_LOG D-008).
    **val_kwargs
        Passed to Ultralytics ``val`` (e.g. ``imgsz``, ``batch``, ``conf``).

    Returns
    -------
    dict
        ``map50``, ``map50_95``, mean ``precision`` and ``recall`` at
        Ultralytics' max-F1 operating point, per-class ``ap50_95`` and ``ap50``,
        and the per-image ``speed_ms`` Ultralytics measured.
    """
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, plots=False, verbose=False, **val_kwargs)
    box = m.box
    return {
        "split": split,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "classes_evaluated": [int(c) for c in box.ap_class_index],
        "ap50_per_class": [float(v) for v in box.ap50],
        "ap50_95_per_class": [float(v) for v in box.ap],
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
        "settings": val_kwargs,
    }
