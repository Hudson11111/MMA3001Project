#!/usr/bin/env bash
# Generate HTML API documentation from the package docstrings into docs/api/.
# Usage (from the repo root, after `pip install -e ".[dev]"`):  bash scripts/build_docs.sh
set -euo pipefail
rm -rf docs/api
python -m pdoc --docformat numpy --output-directory docs/api porkqc
echo "Docs written to docs/api/index.html"
