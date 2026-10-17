"""Tests for the committed model file and the command-line tool."""

import hashlib
from pathlib import Path

import pytest

from porkqc.cli import DEFAULT_WEIGHTS, main

ROOT = Path(__file__).parents[1]
SHA256 = "02e6a2275fa1292b5d07c7100ab2fe0c1e0dbae54db7156cd347c8bf1406a7ee"


def test_committed_weights_match_recorded_hash():
    """The model in the repo is exactly the one that produced the TEST results (models/README.md)."""
    assert DEFAULT_WEIGHTS.is_file()
    assert hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest() == SHA256


def test_cli_missing_image_returns_error_code(capsys):
    assert main(["does_not_exist.jpg"]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_rejects_a_file_that_is_not_an_image(tmp_path, capsys):
    pytest.importorskip("PIL")
    bad = tmp_path / "notes.jpg"
    bad.write_text("this is not an image")
    assert main([str(bad)]) == 2
    assert "not a readable image" in capsys.readouterr().err


def test_cli_end_to_end_on_a_synthetic_image(tmp_path):
    pytest.importorskip("ultralytics")
    import numpy as np
    from PIL import Image
    p = tmp_path / "frame.jpg"
    Image.fromarray(np.full((540, 720, 3), 30, np.uint8)).save(p)   # empty dark conveyor
    out = tmp_path / "r.json"
    assert main([str(p), "--out", str(out)]) == 0
    import json
    r = json.loads(out.read_text())[0]
    assert r["decision"] in {"PASS", "REVIEW", "FAIL"} and r["file"] == "frame.jpg"
