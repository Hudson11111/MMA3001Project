# Automated Inspection of Pork-Rasher Packaging

**MMA3001 Numerical Methods and Machine Learning — Individual Project, Monash University, 2026**
Author: Hudson · Licence: [AGPL-3.0](LICENSE) · [![tests](https://github.com/Hudson11111/MMA3001Project/actions/workflows/ci.yml/badge.svg)](https://github.com/Hudson11111/MMA3001Project/actions/workflows/ci.yml)

A computer-vision inspection system for a meat-packaging line: it finds packaging defects in
camera images of bacon trays, decides per tray whether to **FAIL**, **REVIEW** or **PASS** it,
and is evaluated for accuracy, robustness and real-time speed.

| | |
|---|---|
| **Report (draft)** | [`docs/REPORT_DRAFT.md`](docs/REPORT_DRAFT.md) |
| **Results, with numbers** | [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) |
| **Why each choice was made** | [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) |
| **How correctness was verified (bugs found and fixed)** | [`docs/VERIFICATION_LOG.md`](docs/VERIFICATION_LOG.md) |
| **Inputs and outputs** | [`docs/IO_SPECIFICATION.md`](docs/IO_SPECIFICATION.md) |
| **API documentation (HTML)** | [`docs/api/index.html`](docs/api/index.html) (pdoc; open locally or via the docs link below) |
| **Test report (HTML)** | [`reports/test_report.html`](reports/test_report.html) (pytest-html) |
| **AI use** | [`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md) |
| **Learning objectives evidenced** | [`docs/LEARNING_OBJECTIVES_MAP.md`](docs/LEARNING_OBJECTIVES_MAP.md) |

To browse the HTML files rendered: <https://htmlpreview.github.io/?https://github.com/Hudson11111/MMA3001Project/blob/main/docs/api/index.html>

## 1. Engineering problem

Bacon rashers are packed in clear trays, film-sealed and conveyed past an overhead camera.
Unsealed or lifted film, loose or twisted rashers, wrinkles and other packaging errors must be
removed before distribution; manual inspection of a fast line is tiring and inconsistent. This
is a **manufacturing-systems / automated quality-inspection** problem: it sits inside a
production line whose throughput (trays per minute), reject actuator timing and inspection
cost define the engineering requirements.

**Goal:** from one 720 × 540 RGB frame, (i) detect and localise five defect types, (ii) output
FAIL / REVIEW / PASS, (iii) meet **120 trays/min → ≤ 250 ms per image**, with a missed defect
costed at **10×** a false reject (assumptions and sensitivity: DECISION_LOG D-007).
Out of scope: meat-quality grading (no ground truth), seal-strength measurement, other lines.

## 2. Headline results

| Finding | Number | Source |
|---|---|---|
| Roboflow's random split leaks (neighbouring video frames in train and test) | optimism **+0.27 mAP50** (95% CI +0.06…+0.35), same model on identical images | R4 |
| **One-shot TEST** (time-blocked split, YOLO11n) | AP50 unsealed **0.91**, packaging-error **0.58**; headline mean **0.75** (CI 0.68–0.84) | R6 |
| Decision layer on TEST (thresholds frozen on VALID before TEST existed) | **1 miss in 156** defective images; 94% of automatic FAILs correct | R6, D-026 |
| Alternative: scikit-learn tree / SVM / MLP on hand-crafted features | image-level AP 0.81–0.96 vs detector 0.97–0.98 | R6 |
| Speed | **7 ms** per image (T4 GPU), 82 ms (CPU) — both far inside 250 ms; batch-1 latency is overhead-bound, not FLOP-bound | R6 |
| AP as a numerical integral | trapezoid/Simpson drop from O(h²)/O(h⁴) to ≈ O(h) on the step-function PR envelope; Romberg fails | R5 |
| Robustness | unaffected by ±50% exposure; blur σ 4 px, noise σ 25, JPEG q 10 cost 0.21–0.27 AP50 | R6 |

Main limitations: no fully visible good trays in the data (PASS cannot be validated), wrinkle has
no validation/test instances, labels miss some partly visible defects, single line and camera.

## 3. Repository layout

```
├── configs/            dataset + time-blocked split, experiment settings, frozen decision thresholds
├── models/             selected trained model (yolo11n_640_blocked.pt) + model card
├── docs/               report draft, results/decision/verification/AI logs, I/O spec, api/ (HTML docs)
├── notebooks/          00–04: Colab (data, training, GPU studies) · 05–06: CPU analysis, executed with outputs
├── reports/            test_report.html (pytest-html)
├── results/            every number in the report: JSON/CSV predictions, metrics, figures
├── scripts/            run_tests.sh, build_docs.sh
├── src/porkqc/         the Python package (all reusable code; NumPy-style docstrings)
│   ├── data/           download, label validation, audit, duplicates, time-blocked re-split
│   ├── metrics/        IoU, matching, AP integration rules, quadrature, evaluation + bootstrap
│   ├── models/         YOLO wrapper (train/resume/predict), scikit-learn baseline
│   ├── decision.py     FAIL / REVIEW / PASS rule and cost-based threshold search
│   ├── cli.py          command line: image(s) -> detections + FAIL / REVIEW / PASS
│   ├── perf.py         FLOPs, arithmetic intensity, machine balance, timing
│   └── robustness.py   controlled image perturbations
└── tests/              pytest suite (unit, regression and verification-against-library tests)
```

## 4. Workflow and how to reproduce

The dataset is **not** stored in the repository (size; CC BY 4.0 attribution below). It is
downloaded at run time from Roboflow with your own free API key. The **selected model** is committed
in `models/` (5.2 MB, model card + SHA-256, D-027); other runs' weights live on Google Drive
(`MyDrive/MMA3001Project/runs/`). All predictions are committed in `results/`, so every analysis can
be re-run on CPU without a GPU or the dataset.

**Try the model on an image** (CPU is fine):

```bash
pip install -e ".[detector]"
python -m porkqc.cli path/to/frame.jpg      # -> FAIL / REVIEW / PASS, score, top detections
```

| Step | Notebook | Runtime | Produces |
|---|---|---|---|
| 0 | `00_colab_setup_and_first_look` | Colab CPU | download check, class counts, development sample |
| 1 | `01_data_audit` | Colab CPU | leakage audit, scenes, unlabelled-image review sheet, blocked split |
| 2 | `02_train_detector` (run 3×: see table in notebook) | Colab T4 | trained weights on Drive, curves, VALID predictions |
| 3 | `03_cross_evaluate` | Colab T4 | every model on identical images (leakage measurement) |
| 4 | `04_test_performance_robustness` | Colab T4 | one-shot TEST, latency/FLOPs, resolution/FP16, robustness, baseline features |
| 5 | `05_ap_as_integral` | local CPU | numerical-integration study |
| 6 | `06_final_analysis` | local CPU | TEST metrics, decisions, baseline, roofline, robustness, failure modes |

Open any notebook in Colab from GitHub:
`https://colab.research.google.com/github/Hudson11111/MMA3001Project/blob/main/notebooks/<name>.ipynb`

**One-time Colab setup**
1. Free account at [roboflow.com](https://roboflow.com) → Settings → API Keys → copy the private key.
2. Colab → key icon → add secret `ROBOFLOW_API_KEY`, enable *Notebook access* (never paste keys into cells).
3. *Runtime → Change runtime type → T4 GPU* for notebooks 02–04.
4. Optional: secret `GITHUB_TOKEN` (fine-grained, this repo only, Contents read/write) lets notebooks
   push results to a `colab-results` branch; otherwise they download a zip.

Notebooks 00–04 run on Colab against the full dataset; their outputs are the files in `results/`
(each run also records library versions, GPU and git commit). Notebooks 05–06 are committed
**executed**, so their tables and figures are visible on GitHub.

### Local development (tests, docs, CPU analysis)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"          # numpy, pyyaml, pillow + pytest, pdoc, scipy, scikit-learn, pandas, matplotlib
bash scripts/run_tests.sh                 # pytest + coverage + reports/test_report.html
bash scripts/build_docs.sh                # HTML API docs into docs/api/
jupyter nbconvert --to notebook --execute --inplace notebooks/06_final_analysis.ipynb
```

Tests need no dataset or GPU (synthetic fixtures). Tests that compare against Ultralytics or need
PyTorch are skipped automatically unless `pip install -e ".[detector]"` is also installed; the
committed test report was generated with them installed. GitHub Actions runs the suite on every
push (Python 3.11 and 3.12).

### Environment and dependencies

| Where | Python | Key packages (pinned where results depend on them) |
|---|---|---|
| Colab (GPU work) | 3.13 | `ultralytics==8.4.175`, `roboflow==1.7.1` (`requirements-colab.txt`); torch 2.11 + CUDA (Colab default, recorded per run) |
| Local / CI | ≥ 3.10 | `pyproject.toml` extras: `dev`, `analysis`, `detector`, `data` |

## 5. Data

*Pork Rasher Error (Packaging)* computer-vision dataset, version 4, Roboflow Universe workspace
"Hello" (`hello-2aqe0`), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>.
3,349 / 120 / 80 images (1,411 distinct scenes) of 720 × 540 px with box labels for
loose-meat, packaging-error, twisted-meat, unsealed and wrinkle. All conclusions use a
**time-blocked re-split** (938 / 196 / 180 scenes; D-008) because the provided split leaks.

## 6. AI use

AI assistance (Claude, Anthropic) was used throughout and is disclosed in
[`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md). Every change arrived as a pull request that I
reviewed and merged; commits written by the assistant are authored "Claude". AI errors found
during the project are recorded in the verification log (e.g. V1, V8, V9, V10).

## 7. Licence and acknowledgements

Code © 2026 Hudson, licensed under the **GNU AGPL-3.0** ([LICENSE](LICENSE)), chosen for
compatibility with Ultralytics YOLO (AGPL-3.0), which this project imports. The dataset is
licensed separately (CC BY 4.0) and is not redistributed here.
