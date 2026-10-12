"""Tests for porkqc.data.labels — positive and negative cases."""

import numpy as np
import pytest

from porkqc.data.labels import (
    LabelFormatError,
    parse_label_line,
    read_label_file,
    yolo_to_xyxy,
)

N = 5


# ---------- positive cases ----------

def test_parse_valid_line():
    out = parse_label_line("3 0.5 0.4 0.2 0.1", N)
    np.testing.assert_allclose(out, [3, 0.5, 0.4, 0.2, 0.1])


def test_parse_accepts_integral_float_class_and_extra_whitespace():
    out = parse_label_line("  2.0\t0.5  0.5 0.2 0.2 \n", N)
    assert out[0] == 2


def test_box_touching_edge_within_tolerance():
    # right edge = 0.9 + 0.2/2 = 1.0000004 -> allowed by default tolerance
    parse_label_line("0 0.9000004 0.5 0.2 0.2", N)


def test_read_empty_file_means_no_objects(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    out = read_label_file(p, N)
    assert out.shape == (0, 5)


def test_read_file_ignores_blank_lines(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n\n4 0.1 0.1 0.1 0.1\n")
    out = read_label_file(p, N)
    assert out.shape == (2, 5)
    assert out[:, 0].tolist() == [0, 4]


def test_yolo_to_xyxy_known_values():
    # centre (360, 270), size 72 x 54 px on a 720 x 540 image
    boxes = np.array([[1, 0.5, 0.5, 0.1, 0.1]])
    out = yolo_to_xyxy(boxes, 720, 540)
    np.testing.assert_allclose(out, [[324, 243, 396, 297]])


def test_yolo_to_xyxy_accepts_four_columns():
    out = yolo_to_xyxy(np.array([[0.5, 0.5, 1.0, 1.0]]), 720, 540)
    np.testing.assert_allclose(out, [[0, 0, 720, 540]])


def test_yolo_to_xyxy_empty():
    assert yolo_to_xyxy(np.zeros((0, 5)), 720, 540).shape == (0, 4)


# ---------- negative cases ----------

@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.2",                     # too few fields
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2",  # polygon-style line
        "0 0.5 abc 0.2 0.2",                 # non-numeric
        "0 0.5 nan 0.2 0.2",                 # non-finite
        "5 0.5 0.5 0.2 0.2",                 # class id == n_classes
        "-1 0.5 0.5 0.2 0.2",                # negative class
        "1.5 0.5 0.5 0.2 0.2",               # non-integer class
        "0 0.5 0.5 0 0.2",                   # zero width
        "0 0.5 0.5 0.2 -0.1",                # negative height
        "0 0.95 0.5 0.2 0.2",                # right edge at 1.05
        "0 1.5 0.5 0.2 0.2",                 # centre outside image
    ],
)
def test_parse_rejects_invalid_lines(line):
    with pytest.raises(LabelFormatError):
        parse_label_line(line, N)


def test_parse_rejects_bad_n_classes():
    with pytest.raises(ValueError):
        parse_label_line("0 0.5 0.5 0.2 0.2", 0)


def test_read_file_error_names_file_and_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n")
    with pytest.raises(LabelFormatError, match=r"bad\.txt:2"):
        read_label_file(p, N)


def test_read_missing_file():
    with pytest.raises(FileNotFoundError):
        read_label_file("does/not/exist.txt", N)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 6)])
def test_yolo_to_xyxy_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        yolo_to_xyxy(np.zeros(shape), 720, 540)


def test_yolo_to_xyxy_rejects_bad_size():
    with pytest.raises(ValueError):
        yolo_to_xyxy(np.zeros((1, 4)), 0, 540)
