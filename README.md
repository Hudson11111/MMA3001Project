# Automated Inspection of Pork-Rasher Packaging

**MMA3001 Numerical Methods and Machine Learning — Individual Project (Monash University, 2026)**

> **Status: skeleton (v0.1).** Planning documents, package structure, data-access
> code and tests are in place. Experiments start once the dataset is downloaded in
> Colab. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

## The engineering problem (draft)

Pork rashers are packed into trays on a moving conveyor. Packaging and placement
defects — loose or twisted meat, unsealed film, wrinkles and other packaging
errors — must be caught before distribution, which today relies on manual
inspection. This project develops and evaluates a computer-vision method that:

1. **detects and localises** five defect types in an RGB image of a tray,
2. turns those detections into a **tray-level PASS / FAIL / REVIEW** decision, and
3. tests whether it is **accurate and fast enough** for real-time in-line use.

Full input/output contract: [`docs/IO_SPECIFICATION.md`](docs/IO_SPECIFICATION.md).

## Quick start (Google Colab)

| Notebook | Purpose | Open |
|---|---|---|
| `00_colab_setup_and_first_look` | Install, download dataset, verify splits, first class counts (CPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hudson11111/MMA3001Project/blob/main/notebooks/00_colab_setup_and_first_look.ipynb) |
| `01_data_audit` | Box statistics, augmentation accounting, near-duplicate leakage check (CPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hudson11111/MMA3001Project/blob/main/notebooks/01_data_audit.ipynb) |
| `02_train_detector` | Train YOLO11n / YOLO26n on the Roboflow or time-blocked split (resumable), training curves, export VALID predictions (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hudson11111/MMA3001Project/blob/main/notebooks/02_train_detector.ipynb) |
| `03_cross_evaluate` | Every trained model on every validation set: fair leakage measurement (GPU preferred) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hudson11111/MMA3001Project/blob/main/notebooks/03_cross_evaluate.ipynb) |

**One-time setup**

1. Create a free account at [roboflow.com](https://roboflow.com) and copy your
   **Private API Key** (Settings → API Keys).
2. In Colab, click the **key icon** in the left sidebar → *Add new secret* →
   name `ROBOFLOW_API_KEY`, paste the key, and switch **Notebook access** on.
   Never paste the key into a notebook cell or commit it.
3. *Runtime → Change runtime type → T4 GPU* (needed from training onwards).
4. *(Optional, saves manual uploads)* create a GitHub **fine-grained token**
   (GitHub → Settings → Developer settings → Fine-grained tokens) with access to
   **only this repository** and permission **Contents: Read and write**; add it as
   the Colab secret `GITHUB_TOKEN`. Notebooks then push result files to the
   `colab-results` branch (never `main`).

## Repository layout

```
├── configs/            YAML configuration (dataset, experiments)
├── docs/               plan, decision log, LO map, I/O spec, AI-use log, api/ (HTML docs)
├── notebooks/          Colab notebooks — run experiments, import the package
├── reports/            generated test report (pytest-html)
├── results/            small committed outputs: metrics (JSON/CSV) and figures
├── scripts/            run_tests.sh, build_docs.sh
├── src/porkqc/         the Python package (all reusable, tested code)
└── tests/              pytest unit tests (synthetic data, no GPU needed)
```

Large artefacts (dataset, trained weights, run folders) are **not** in Git; they
live in Colab / Google Drive.

## Local development

```bash
pip install -e ".[dev]"
bash scripts/run_tests.sh      # tests + reports/test_report.html
bash scripts/build_docs.sh     # HTML API docs into docs/api/
```

Tests also run automatically on every push (GitHub Actions, `.github/workflows/ci.yml`).

## Data

*Pork Rasher Error(Packaging)* computer-vision dataset, version 4, by the
Roboflow Universe workspace "Hello" (`hello-2aqe0`), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>

The dataset is downloaded at run time and not redistributed in this repository.

## AI use

AI assistance (Claude, Anthropic) is used throughout and disclosed in
[`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md). Commits made by the assistant are
authored as "Claude"; design decisions and their rationale are recorded in
[`docs/DECISION_LOG.md`](docs/DECISION_LOG.md).

## Licence

Code: [GNU AGPL-3.0](LICENSE) — chosen for compatibility with the Ultralytics
library (also AGPL-3.0) that this project imports (DECISION_LOG D-011).
Dataset: CC BY 4.0, licensed separately and not included in this repository.
