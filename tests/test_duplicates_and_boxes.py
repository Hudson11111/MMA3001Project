"""Tests for near-duplicate hashing and box statistics."""

import numpy as np
import pytest

from porkqc.data.audit import box_table, image_classes
from porkqc.data.duplicates import (
    dhash,
    group_by_threshold,
    hamming_distance,
    hash_images,
    nearest_neighbours,
    paired_hamming,
    grey_thumbnail,
    thumbnail_correlation,
)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def scene(seed, shape=(540, 720)):
    """Smooth random 'photo' (blurred noise) so hashes are stable and distinct."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize(shape[::-1], Image.Resampling.BICUBIC).convert("RGB"))


def test_identical_images_distance_zero():
    a = scene(0)
    assert hamming_distance(dhash(a), dhash(a.copy()))[0, 0] == 0


def test_brightness_change_is_near_duplicate_but_new_scene_is_not():
    a = scene(1)
    brighter = np.clip(a.astype(int) * 1.1 + 10, 0, 255).astype(np.uint8)  # like exposure aug
    d_same = hamming_distance(dhash(a), dhash(brighter))[0, 0]
    d_other = hamming_distance(dhash(a), dhash(scene(2)))[0, 0]
    assert d_same <= 4
    assert d_other >= 15


def test_hamming_matches_naive_bit_count():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (5, 8), dtype=np.uint8)
    b = rng.integers(0, 256, (7, 8), dtype=np.uint8)
    naive = np.array([[np.unpackbits(x ^ y).sum() for y in b] for x in a])
    np.testing.assert_array_equal(hamming_distance(a, b), naive)


def test_paired_hamming_is_diagonal_of_pairwise():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (6, 8), dtype=np.uint8)
    b = rng.integers(0, 256, (6, 8), dtype=np.uint8)
    np.testing.assert_array_equal(paired_hamming(a, b), np.diag(hamming_distance(a, b)))
    with pytest.raises(ValueError):
        paired_hamming(a, b[:3])


def test_hamming_width_mismatch():
    with pytest.raises(ValueError):
        hamming_distance(np.zeros((1, 8), np.uint8), np.zeros((1, 4), np.uint8))


def test_nearest_neighbours_same_set_ignores_self():
    h = np.array([[0] * 8, [1] + [0] * 7, [255] * 8], dtype=np.uint8)
    d, i = nearest_neighbours(h, h, same_set=True, chunk=2)
    assert d.tolist() == [1, 1, 63] and i.tolist() == [1, 0, 1]


def test_group_by_threshold_chains_and_labels():
    h = np.array([[0] * 8, [1] + [0] * 7, [3] + [0] * 7, [255] * 8], dtype=np.uint8)
    # 0-1 differ by 1 bit, 1-2 by 1 bit (0-2 by 2): single linkage chains them.
    assert group_by_threshold(h, threshold=1) == [0, 0, 0, 1]
    # labels force a link between items 0 and 3
    assert group_by_threshold(h, threshold=0, labels=["x", "a", "b", "x"]) == [0, 1, 2, 0]


def test_hash_images_from_files(tmp_path):
    for k in range(3):
        Image.fromarray(scene(k)).save(tmp_path / f"{k}.jpg")
    h = hash_images(sorted(tmp_path.glob("*.jpg")))
    assert h.shape == (3, 8) and h.dtype == np.uint8


def test_box_table_and_image_classes(synthetic_dataset):
    rows = box_table(synthetic_dataset / "train", 5)
    assert len(rows) == 4  # instances in synthetic train split
    r = rows[0]
    assert r["original"] == "a_jpg" and r["area_frac"] == pytest.approx(0.04)
    ic = image_classes(synthetic_dataset / "train", 5)
    assert ic["b_jpg.rf.333.jpg"] == []          # null image
    assert ic["a_jpg.rf.111.jpg"] == [0, 2]
    assert "c_jpg.rf.444.jpg" not in ic          # missing label skipped


def test_thumbnail_correlation_separates_copies_from_other_frames():
    from PIL import ImageEnhance
    base = Image.fromarray(scene(10))
    aug = ImageEnhance.Brightness(ImageEnhance.Color(base).enhance(0.71)).enhance(1.1)  # Roboflow-range aug
    t0, t1, t2 = grey_thumbnail(base), grey_thumbnail(aug), grey_thumbnail(scene(11))
    assert abs(t0.mean()) < 1e-9 and t0.std() == pytest.approx(1.0)
    assert thumbnail_correlation(t0, t1) > 0.99
    assert thumbnail_correlation(t0, t2) < 0.9
    assert np.all(grey_thumbnail(np.full((10, 10, 3), 7, np.uint8)) == 0)   # constant image
    with pytest.raises(ValueError):
        thumbnail_correlation(t0, t0[:10])
