"""
Helpers for returning small result files from Colab to GitHub.

Colab's disk is temporary, and *File → Save a copy in GitHub* saves only the
notebook. :func:`push_results` commits chosen result files (metrics, predictions,
figures) to a dedicated ``colab-results`` branch — never ``main`` — so they can be
reviewed and merged like any other change (DECISION_LOG D-014, option b).

It needs a fine-grained GitHub token limited to this repository (Contents:
read and write) stored as the Colab secret ``GITHUB_TOKEN``. The token is never
printed: all git output is scrubbed before it is shown or raised.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

__all__ = ["redact", "push_results"]

DEFAULT_REMOTE = "https://github.com/Hudson11111/MMA3001Project.git"


def redact(text: str, secret: str | None) -> str:
    """
    Replace every occurrence of a secret in a string with ``***``.

    Parameters
    ----------
    text : str
        Text that may contain the secret (e.g. git error output).
    secret : str or None
        The secret; ``None`` or empty leaves the text unchanged.

    Returns
    -------
    str
        Scrubbed text.
    """
    return text.replace(secret, "***") if secret else text


def _git(args: list[str], cwd: Path, token: str | None) -> str:
    """Run git, returning stdout; on failure raise with the token scrubbed."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(redact(f"git {' '.join(args)} failed:\n{proc.stderr}", token))
    return proc.stdout


def _push_url(remote: str, token: str | None) -> str:
    """Insert the token into an https remote URL (local paths are returned unchanged)."""
    if token and remote.startswith("https://"):
        return remote.replace("https://", f"https://x-access-token:{token}@", 1)
    return remote


def push_results(
    repo_dir: str | Path,
    paths: Iterable[str | Path],
    message: str,
    token: str | None,
    branch: str = "colab-results",
    remote: str = DEFAULT_REMOTE,
    author: tuple[str, str] = ("Hudson11111", "Hudson11111@users.noreply.github.com"),
) -> str:
    """
    Commit files to a results branch on GitHub without touching ``main``.

    Works in a temporary clone, so the notebook's working copy is unaffected.
    If ``branch`` already exists on the remote, the new commit goes on top of it
    (results from several runs accumulate).

    Parameters
    ----------
    repo_dir : str or pathlib.Path
        Local repository the files live in; paths are kept relative to it.
    paths : iterable of path-like
        Files (or folders) relative to ``repo_dir`` to commit.
    message : str
        Commit message.
    token : str or None
        GitHub token for https remotes (``None`` for local test remotes).
    branch : str, optional
        Target branch (default ``"colab-results"``). Refuses ``"main"``.
    remote : str, optional
        Remote URL or local path.
    author : tuple of (str, str), optional
        Commit author name and e-mail (GitHub no-reply address by default).

    Returns
    -------
    str
        The new commit's hash.

    Raises
    ------
    ValueError
        If ``branch`` is ``"main"`` or no paths are given.
    FileNotFoundError
        If a path does not exist.
    RuntimeError
        If a git command fails (message scrubbed of the token).
    """
    if branch == "main":
        raise ValueError("push_results never writes to 'main'; use a results branch.")
    repo_dir = Path(repo_dir)
    rel_paths = [Path(p) for p in paths]
    if not rel_paths:
        raise ValueError("No paths given.")
    for p in rel_paths:
        if not (repo_dir / p).exists():
            raise FileNotFoundError(repo_dir / p)

    tmp = Path(tempfile.mkdtemp(prefix="results_push_"))
    try:
        url = _push_url(remote, token)
        try:  # existing results branch
            _git(["clone", "--quiet", "--depth", "1", "--branch", branch, url, str(tmp / "r")], tmp, token)
        except RuntimeError:  # first push: start from the default branch
            _git(["clone", "--quiet", "--depth", "1", url, str(tmp / "r")], tmp, token)
            _git(["checkout", "-q", "-b", branch], tmp / "r", token)
        work = tmp / "r"

        for p in rel_paths:
            src, dst = repo_dir / p, work / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        _git(["add", "--force", *[str(p) for p in rel_paths]], work, token)
        name, email = author
        _git(["-c", f"user.name={name}", "-c", f"user.email={email}", "-c", "commit.gpgsign=false",
              "commit", "-q", "--allow-empty", "-m", message], work, token)
        _git(["push", "-q", url, f"HEAD:refs/heads/{branch}"], work, token)
        return _git(["rev-parse", "HEAD"], work, token).strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
