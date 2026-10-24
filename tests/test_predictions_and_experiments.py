"""Tests for ground-truth export, prediction arrays and experiment configs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from porkqc import CLASS_NAMES
from porkqc.models.detector import load_experiment
from porkqc.predictions import export_ground_truth, gt_arrays, load_json, pred_arrays, save_json

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

REPO = Path(__file__).parents[1]


@pytest.fixture
def tiny_split(tmp_path):
    """Two 720x540 images: one with two boxes, one with none (a 'good' tray)."""
    for sub in ("images", "labels"):
        (tmp_path / "valid" / sub).mkdir(parents=True)
    for name, lines in [("a.jpg", "1 0.5 0.5 0.1 0.1\n4 0.25 0.25 0.5 0.5\n"), ("b.jpg", "")]:
        Image.new("RGB", (720, 540)).save(tmp_path / "valid" / "images" / name)
        (tmp_path / "valid" / "labels" / f"{Path(name).stem}.txt").write_text(lines)
    return tmp_path / "valid"


def test_export_ground_truth_pixels_and_null_image(tiny_split):
    gt = export_ground_truth(tiny_split, CLASS_NAMES)
    assert gt["split"] == "valid" and gt["class_names"] == list(CLASS_NAMES)
    a, b = gt["images"]
    assert (a["width"], a["height"]) == (720, 540)
    assert a["boxes"][0] == [1, 324.0, 243.0, 396.0, 297.0]
    assert a["boxes"][1] == [4, 0.0, 0.0, 360.0, 270.0]
    assert b["boxes"] == []  # null image kept, not dropped


def test_json_round_trip_and_arrays(tiny_split, tmp_path):
    gt = export_ground_truth(tiny_split, CLASS_NAMES)
    p = save_json(gt, tmp_path / "out" / "gt.json")
    back = load_json(p)
    cls, boxes = gt_arrays(back["images"][0])
    assert cls.tolist() == [1, 4] and boxes.shape == (2, 4)
    cls0, boxes0 = gt_arrays(back["images"][1])
    assert cls0.shape == (0,) and boxes0.shape == (0, 4)


def test_pred_arrays():
    entry = {"pred": [[2, 0.9, 1, 2, 3, 4], [0, 0.1, 5, 6, 7, 8]]}
    c, s, b = pred_arrays(entry)
    assert c.tolist() == [2, 0]
    np.testing.assert_allclose(s, [0.9, 0.1])
    assert b.shape == (2, 4)
    c, s, b = pred_arrays({"pred": []})
    assert c.shape == (0,) and b.shape == (0, 4)


@pytest.mark.parametrize("name", ["yolo11n_640", "yolo26n_640"])
def test_experiment_configs_load_and_are_comparable(name):
    exp = load_experiment(REPO / "configs" / "experiments" / f"{name}.yaml")
    assert exp["name"] == name
    assert exp["predict"]["conf"] <= 0.001  # low threshold needed for PR curves
    assert exp["predict_variants"][0]["tag"] == "default"


def test_both_experiments_share_training_settings():
    a = load_experiment(REPO / "configs" / "experiments" / "yolo11n_640.yaml")
    b = load_experiment(REPO / "configs" / "experiments" / "yolo26n_640.yaml")
    assert a["train"] == b["train"]  # fair comparison: only the model differs
    assert a["predict"] == b["predict"]


def test_load_experiment_missing_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "model": "y.pt", "train": {}}))
    with pytest.raises(ValueError, match="predict"):
        load_experiment(p)


def test_is_resumable_checkpoint_logic(tmp_path):
    """Finished (optimizer-stripped) checkpoints must not be 'resumed' (would retrain)."""
    torch = pytest.importorskip("torch")
    from porkqc.models.detector import _is_resumable

    torch.save({"epoch": 3, "optimizer": {"state": 1}}, tmp_path / "running.pt")
    torch.save({"epoch": -1, "optimizer": None}, tmp_path / "finished.pt")
    torch.save({"epoch": 5, "optimizer": None}, tmp_path / "stripped.pt")
    assert _is_resumable(tmp_path / "running.pt")
    assert not _is_resumable(tmp_path / "finished.pt")
    assert not _is_resumable(tmp_path / "stripped.pt")
