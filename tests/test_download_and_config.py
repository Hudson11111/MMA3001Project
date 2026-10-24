"""Tests for data.yaml repair, class-name checks and secret lookup (no network)."""

import pytest
import yaml

from porkqc import CLASS_NAMES
from porkqc.config import check_class_names, load_yaml
from porkqc.data.download import fix_data_yaml, read_class_names
from porkqc.env import get_secret


def _roboflow_yaml(root, **overrides):
    data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 5,
        "names": list(CLASS_NAMES),
    }
    data.update(overrides)
    p = root / "data.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


# ---------- fix_data_yaml ----------


def test_fix_data_yaml_makes_absolute_existing_paths(synthetic_dataset):
    p = _roboflow_yaml(synthetic_dataset)
    out = fix_data_yaml(p)
    for key, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        assert out[key] == str((synthetic_dataset / folder / "images").resolve())
    assert (synthetic_dataset / "data.yaml.orig").exists()
    assert yaml.safe_load(p.read_text())["train"] == out["train"]  # written back


def test_fix_data_yaml_dry_run_does_not_write(synthetic_dataset):
    p = _roboflow_yaml(synthetic_dataset)
    before = p.read_text()
    fix_data_yaml(p, write=False)
    assert p.read_text() == before


def test_fix_data_yaml_missing_split_folder(synthetic_dataset):
    import shutil

    shutil.rmtree(synthetic_dataset / "test")
    p = _roboflow_yaml(synthetic_dataset)
    with pytest.raises(FileNotFoundError, match="test"):
        fix_data_yaml(p)


def test_fix_data_yaml_requires_names(synthetic_dataset):
    p = synthetic_dataset / "data.yaml"
    p.write_text(yaml.safe_dump({"train": "train/images"}))
    with pytest.raises(ValueError, match="names"):
        fix_data_yaml(p)


# ---------- class names ----------


def test_read_class_names_list_and_dict(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump({"names": ["x", "y"]}))
    assert read_class_names(p) == ["x", "y"]
    p.write_text(yaml.safe_dump({"names": {1: "y", 0: "x"}}))
    assert read_class_names(p) == ["x", "y"]


def test_read_class_names_rejects_gappy_dict(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump({"names": {0: "x", 2: "z"}}))
    with pytest.raises(ValueError):
        read_class_names(p)


def test_check_class_names_order_matters():
    check_class_names(list(CLASS_NAMES), CLASS_NAMES)
    with pytest.raises(ValueError):
        check_class_names(list(reversed(CLASS_NAMES)), CLASS_NAMES)


def test_dataset_config_matches_package_constant():
    from pathlib import Path

    cfg = load_yaml(Path(__file__).parents[1] / "configs" / "dataset.yaml")
    check_class_names(cfg["class_names"], CLASS_NAMES)


def test_load_yaml_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_yaml(p)


# ---------- secrets ----------


def test_get_secret_from_environment(monkeypatch):
    monkeypatch.setenv("PORKQC_TEST_SECRET", "s3cret")
    assert get_secret("PORKQC_TEST_SECRET") == "s3cret"


def test_get_secret_missing_gives_helpful_error(monkeypatch):
    monkeypatch.delenv("PORKQC_DEFINITELY_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="key icon"):
        get_secret("PORKQC_DEFINITELY_MISSING")


def test_get_secret_empty_name():
    with pytest.raises(ValueError):
        get_secret("")
