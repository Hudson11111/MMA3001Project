"""
Environment helpers.

The same code runs in two places: Google Colab (GPU experiments) and a normal
Python environment (development, tests, CI). These helpers hide the
differences, most importantly how secrets such as the Roboflow API key are read.

Secrets are never stored in the repository or in configuration files.
"""

from __future__ import annotations

import os

__all__ = ["in_colab", "get_secret"]


def in_colab() -> bool:
    """
    Report whether the code is running inside Google Colab.

    Returns
    -------
    bool
        ``True`` if the ``google.colab`` module can be imported.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def get_secret(name: str) -> str:
    """
    Read a secret from the environment or from Colab Secrets.

    Lookup order:

    1. Environment variable ``name`` (used locally and in CI).
    2. Colab Secrets (the key icon in Colab's left sidebar), if running in Colab.

    Parameters
    ----------
    name : str
        Name of the secret, e.g. ``"ROBOFLOW_API_KEY"``.

    Returns
    -------
    str
        The secret value (never logged or printed by this package).

    Raises
    ------
    ValueError
        If ``name`` is empty.
    RuntimeError
        If the secret cannot be found, or Colab denies this notebook access to
        it. The message explains how to set it.
    """
    if not name:
        raise ValueError("Secret name must be a non-empty string.")

    value = os.environ.get(name)
    if value:
        return value

    if in_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except Exception as exc:  # SecretNotFoundError, NotebookAccessError, ...
            raise RuntimeError(
                f"Could not read Colab secret '{name}' ({type(exc).__name__}). "
                "In Colab, click the key icon in the left sidebar, add a secret "
                f"named '{name}', and switch on 'Notebook access'."
            ) from exc
        if value:
            return value

    raise RuntimeError(
        f"Secret '{name}' not found. In Colab, add it via the key icon in the left "
        f"sidebar (enable 'Notebook access'). Locally, run: export {name}=<value>"
    )
