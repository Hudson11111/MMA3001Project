"""Tests for porkqc.colab_utils using a local bare repository as the 'remote'."""

import shutil
import subprocess

import pytest

from porkqc.colab_utils import _normalise_remote, _push_url, push_results, redact

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def remote_and_clone(tmp_path):
    """A bare 'remote' with one commit on main, and a working clone of it."""
    remote = tmp_path / "remote.git"
    _git("init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    seed = tmp_path / "seed"
    _git("clone", "-q", str(remote), str(seed), cwd=tmp_path)
    (seed / "README.md").write_text("hi\n")
    _git("add", ".", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
         "commit", "-q", "-m", "init", cwd=seed)
    _git("push", "-q", "origin", "HEAD:main", cwd=seed)
    return remote, seed


def test_redact():
    assert redact("url https://x:abc123@github", "abc123") == "url https://x:***@github"
    assert redact("nothing", None) == "nothing"


def test_push_url_only_changes_https():
    assert _push_url("https://github.com/a/b.git", "T") == "https://x-access-token:T@github.com/a/b.git"
    assert _push_url("/local/path.git", "T") == "/local/path.git"


def test_push_results_creates_then_extends_branch(remote_and_clone):
    remote, work = remote_and_clone
    (work / "results").mkdir()
    (work / "results" / "m1.json").write_text("{}")
    push_results(work, ["results/m1.json"], "first", token=None, remote=str(remote))
    (work / "results" / "m2.json").write_text("{}")
    push_results(work, ["results/m2.json"], "second", token=None, remote=str(remote))

    files = _git("ls-tree", "-r", "--name-only", "colab-results", cwd=remote).split()
    assert {"README.md", "results/m1.json", "results/m2.json"} <= set(files)
    log = _git("log", "--format=%s", "colab-results", cwd=remote).split("\n")
    assert log[:2] == ["second", "first"]
    # main is untouched
    assert _git("ls-tree", "-r", "--name-only", "main", cwd=remote).split() == ["README.md"]


def test_push_results_refuses_main(remote_and_clone):
    remote, work = remote_and_clone
    with pytest.raises(ValueError):
        push_results(work, ["README.md"], "x", token=None, branch="main", remote=str(remote))


def test_push_results_missing_file(remote_and_clone):
    remote, work = remote_and_clone
    with pytest.raises(FileNotFoundError):
        push_results(work, ["nope.json"], "x", token=None, remote=str(remote))


def test_git_failure_message_is_scrubbed(remote_and_clone, tmp_path):
    _, work = remote_and_clone
    bad = "https://SECRET@invalid.invalid/none.git"
    _git("remote", "set-url", "origin", bad, cwd=work)  # pass the clone check, fail at git
    with pytest.raises(RuntimeError) as exc:
        push_results(work, ["README.md"], "x", token="SECRET", remote=bad)
    assert "SECRET" not in str(exc.value)


def test_refuses_to_push_from_a_folder_that_is_not_a_clone(remote_and_clone, tmp_path):
    """Regression test for VERIFICATION_LOG V8: a local dry run must never push."""
    remote, _ = remote_and_clone
    stray = tmp_path / "not_a_repo"
    (stray / "results").mkdir(parents=True)
    (stray / "results" / "fake.json").write_text("{}")
    with pytest.raises(ValueError, match="Refusing to push"):
        push_results(stray, ["results/fake.json"], "x", token=None, remote=str(remote))
    assert "colab-results" not in _git("branch", "--list", cwd=remote)


def test_normalise_remote():
    a = _normalise_remote("https://x-access-token:T@github.com/Hudson11111/MMA3001Project.git")
    assert a == _normalise_remote("https://github.com/hudson11111/mma3001project/")
