"""Tests for porkqc.data.audit against a synthetic dataset with known answers."""

import zipfile

import pytest
import yaml

from porkqc import CLASS_NAMES
from porkqc.data.audit import (
    list_images,
    make_dev_sample,
    original_stem,
    summarise_dataset,
    summarise_split,
)

N_CLASSES = len(CLASS_NAMES)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("tray_012_jpg.rf.9f8e7d.jpg", "tray_012_jpg"),
        ("folder/x_png.rf.abc.jpg", "x_png"),
        ("plain.jpg", "plain"),
    ],
)
def test_original_stem(name, expected):
    assert original_stem(name) == expected


def test_list_images_filters_and_sorts(tmp_path):
    for n in ["b.jpg", "a.PNG", "notes.txt"]:
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.jpg"]


def test_list_images_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "nope")


def test_train_split_counts(synthetic_dataset):
    s = summarise_split(synthetic_dataset / "train", N_CLASSES)
    assert s.n_images == 5
    assert s.n_unique_originals == 4          # two copies of 'a_jpg'
    assert s.n_null_images == 1
    assert s.n_missing_labels == 1
    assert s.n_invalid_label_files == 0
    assert s.instances_per_class == [2, 0, 1, 0, 1]
    assert s.images_per_class == [2, 0, 1, 0, 1]
    assert s.n_instances == 4
    assert s.boxes_per_image_max == 2


def test_invalid_label_is_reported_not_raised(synthetic_dataset):
    s = summarise_split(synthetic_dataset / "valid", N_CLASSES)
    assert s.n_invalid_label_files == 1
    assert s.instances_per_class == [0, 0, 0, 1, 0]
    assert any("invalid label" in msg for msg in s.issues)


def test_summarise_dataset_and_to_dict(synthetic_dataset):
    out = summarise_dataset(synthetic_dataset, N_CLASSES)
    assert list(out) == ["train", "valid", "test"]
    d = out["test"].to_dict()
    assert d["split"] == "test" and d["instances_per_class"] == [0, 1, 0, 0, 0]


def test_dev_sample_contents_and_reproducibility(synthetic_dataset, tmp_path):
    sizes = {"train": 3, "valid": 1, "test": 1}
    z1 = make_dev_sample(synthetic_dataset, tmp_path / "s1.zip", CLASS_NAMES, sizes, seed=1)
    z2 = make_dev_sample(synthetic_dataset, tmp_path / "s2.zip", CLASS_NAMES, sizes, seed=1)
    with zipfile.ZipFile(z1) as a, zipfile.ZipFile(z2) as b:
        names = sorted(a.namelist())
        assert names == sorted(b.namelist())  # same seed -> same sample
        train_imgs = [n for n in names if n.startswith("dev_sample/train/images/")]
        assert len(train_imgs) == 3
        # every sampled image has its label file
        for img in train_imgs:
            stem = img.rsplit("/", 1)[1].rsplit(".", 1)[0]
            assert f"dev_sample/train/labels/{stem}.txt" in names
        meta = yaml.safe_load(a.read("dev_sample/data.yaml"))
        assert meta["names"] == list(CLASS_NAMES)
