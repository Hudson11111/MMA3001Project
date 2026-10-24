"""
Command-line inspection of tray images with the committed model.

Usage (from the repository root, after ``pip install -e ".[detector]"``)::

    python -m porkqc.cli path/to/frame.jpg [more.jpg ...] [--out results.json]

For each image it prints the tray decision (FAIL / REVIEW / PASS), the image
score and the detections above the review threshold, using the model in
``models/`` and the thresholds frozen in ``configs/decision.yaml`` (D-026).
This is the deployed form of the system: one image in, one decision out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["DEFAULT_WEIGHTS", "DEFAULT_DECISION", "inspect_images", "main"]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11n_640_blocked.pt"
DEFAULT_DECISION = ROOT / "configs" / "decision.yaml"


def inspect_images(
    paths, weights=DEFAULT_WEIGHTS, decision_cfg=DEFAULT_DECISION, device: str = "cpu"
) -> list[dict]:
    """
    Run the detector and the decision rule on image files.

    Parameters
    ----------
    paths : sequence of path-like
        Image files (JPEG/PNG, RGB; validated at 720 × 540).
    weights : path-like, optional
        Trained YOLO weights (default: the committed selected model).
    decision_cfg : path-like, optional
        YAML with ``t_review`` and ``t_fail`` (default: frozen thresholds).
    device : str, optional
        ``"cpu"`` (default) or ``"cuda"``/``"0"``.

    Returns
    -------
    list of dict
        Per image: ``file``, ``decision``, ``score`` (highest box confidence),
        ``detections`` (class name, confidence, ``[x1, y1, x2, y2]`` pixels) with
        confidence ≥ ``t_review``, and ``ms`` (pre/inference/post timing).

    Raises
    ------
    FileNotFoundError
        If an image, the weights or the decision config is missing.
    ValueError
        If a file is not a readable image.
    """
    from PIL import Image, UnidentifiedImageError

    from . import CLASS_NAMES
    from .config import load_yaml
    from .decision import decide
    from .models.detector import predict_images

    paths = [Path(p) for p in paths]
    for p in [*paths, Path(weights)]:
        if not p.is_file():
            raise FileNotFoundError(p)
    for p in paths:  # reject non-images before loading the model
        try:
            with Image.open(p) as im:
                im.verify()
        # Once Ultralytics is imported it patches Image.open to retry unknown files with
        # an optional HEIF plugin, so a non-image may surface as ImportError instead.
        except (UnidentifiedImageError, OSError, ImportError) as exc:
            raise ValueError(f"{p} is not a readable image") from exc
    cfg = load_yaml(decision_cfg)
    pred = predict_images(
        weights,
        paths,
        "yolo11n_640_blocked",
        "cli",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=device,
    )
    out = []
    for e in pred["images"]:
        score = max((b[1] for b in e["pred"]), default=0.0)
        out.append(
            {
                "file": e["file"],
                "decision": str(decide([score], cfg["t_review"], cfg["t_fail"])[0]),
                "score": round(score, 4),
                "detections": [
                    {"class": CLASS_NAMES[b[0]], "confidence": b[1], "box_xyxy": b[2:]}
                    for b in e["pred"]
                    if b[1] >= cfg["t_review"]
                ],
                "ms": e["speed_ms"],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for ``python -m porkqc.cli``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code: 0 on success, 2 for missing or unreadable files.
    """
    ap = argparse.ArgumentParser(description="Inspect pork-rasher tray images: FAIL / REVIEW / PASS.")
    ap.add_argument("images", nargs="+", help="image files")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights (.pt)")
    ap.add_argument("--decision", default=str(DEFAULT_DECISION), help="decision thresholds YAML")
    ap.add_argument("--device", default="cpu", help="cpu or cuda device id")
    ap.add_argument("--out", help="write full results as JSON to this file")
    args = ap.parse_args(argv)
    try:
        results = inspect_images(args.images, args.weights, args.decision, args.device)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for r in results:
        top = ", ".join(f"{d['class']} {d['confidence']:.2f}" for d in r["detections"][:3]) or "-"
        print(f"{r['decision']:6s}  score {r['score']:.3f}  {r['file']}  [{top}]")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
