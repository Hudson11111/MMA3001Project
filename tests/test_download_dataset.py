"""
Tests for download_dataset using a fake 'roboflow' module that copies the real
package's behaviour (1.7.1): if the target folder exists and overwrite is
False, it returns immediately without downloading. No network access.
"""

import sys
import types

import pytest

from porkqc.data.download import download_dataset


class FakeVersion:
    def __init__(self, log, not_ready_times=0, writes_yaml=True):
        self.log, self.not_ready_left, self.writes_yaml = log, not_ready_times, writes_yaml

    def download(self, fmt, location, overwrite=False):
        from pathlib import Path

        self.log.append(
            {"fmt": fmt, "location": location, "overwrite": overwrite, "existed": Path(location).exists()}
        )
        if Path(location).exists() and not overwrite:  # real roboflow 1.7.1 behaviour
            return types.SimpleNamespace(location=location)
        if self.not_ready_left:
            self.not_ready_left -= 1
            raise RuntimeError("{'ready': False}")
        Path(location).mkdir(parents=True, exist_ok=True)
        if self.writes_yaml:
            (Path(location) / "data.yaml").write_text("names: [a]\n")
        return types.SimpleNamespace(location=location)


@pytest.fixture
def fake_roboflow(monkeypatch):
    def install(**kw):
        log = []
        version = FakeVersion(log, **kw)
        chain = types.SimpleNamespace(
            workspace=lambda w: types.SimpleNamespace(
                project=lambda p: types.SimpleNamespace(version=lambda v: version)
            )
        )
        module = types.ModuleType("roboflow")
        module.Roboflow = lambda api_key: chain
        monkeypatch.setitem(sys.modules, "roboflow", module)
        return log

    return install


def call(location, **kw):
    return download_dataset("ws", "proj", 4, location, api_key="k", wait_seconds=0, **kw)


def test_fresh_location_downloads_without_precreating_folder(fake_roboflow, tmp_path):
    log = fake_roboflow()
    out = call(tmp_path / "ds")
    assert (out / "data.yaml").is_file()
    assert log[0]["existed"] is False and log[0]["overwrite"] is False


def test_existing_empty_folder_forces_overwrite(fake_roboflow, tmp_path):
    """Regression for VERIFICATION_LOG V9 (the bug hit in Colab on 12 Oct)."""
    (tmp_path / "ds").mkdir()
    log = fake_roboflow()
    out = call(tmp_path / "ds")
    assert (out / "data.yaml").is_file()
    assert log[0]["overwrite"] is True


def test_existing_dataset_is_not_downloaded_again(fake_roboflow, tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "data.yaml").write_text("names: [a]\n")
    log = fake_roboflow()
    call(tmp_path / "ds")
    assert log == []


def test_retries_while_export_not_ready(fake_roboflow, tmp_path):
    log = fake_roboflow(not_ready_times=2)
    out = call(tmp_path / "ds", retries=3)
    assert (out / "data.yaml").is_file() and len(log) == 3


def test_gives_up_after_retries(fake_roboflow, tmp_path):
    fake_roboflow(not_ready_times=5)
    with pytest.raises(RuntimeError, match="ready"):
        call(tmp_path / "ds", retries=2)


def test_missing_yaml_after_download_is_an_error(fake_roboflow, tmp_path):
    fake_roboflow(writes_yaml=False)
    with pytest.raises(RuntimeError, match="data.yaml is missing"):
        call(tmp_path / "ds")
