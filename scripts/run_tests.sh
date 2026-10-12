#!/usr/bin/env bash
# Run the test suite and write a self-contained HTML report to reports/.
# Usage (from the repo root):  bash scripts/run_tests.sh
set -euo pipefail
python -m pytest \
  --cov=porkqc --cov-report=term-missing \
  --html=reports/test_report.html --self-contained-html "$@"
