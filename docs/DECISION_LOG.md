# Decision Log

Every non-trivial technical choice is recorded here: what the options were, what
was chosen, why, and what evidence supports it. This is the source material for
the report's "justification of computational approach" section and for answering
"why did you do X instead of Y?" in the presentation.

**Status key:** ✅ Accepted · 🟡 Proposed (Claude's recommendation, awaiting your OK)
· ⏳ Pending your decision · 🔬 Pending data/evidence

Decisions marked ✅ on 12 Oct were either confirmed by you explicitly (D-001,
D-011, deadline/compute facts behind D-016) or proposed by Claude with no
objection raised; the latter are noted "(no objection)".

| ID | Decision | Status |
|---|---|---|
| D-001 | Problem framing and scope | ✅ confirmed 12 Oct |
| D-002 | Code architecture: package + thin Colab notebooks | ✅ (no objection) |
| D-003 | Docstring style: NumPy | ✅ (no objection) |
| D-004 | Data access via Roboflow API in Colab; data never committed | ✅ (no objection) |
| D-005 | Primary detector family and comparison | 🔬 |
| D-006 | Baselines / alternative solutions | ✅ (no objection) |
| D-007 | Engineering requirements: line speed and error-cost ratio | ✅ delegated to Claude 12 Oct |
| D-008 | Train/valid/test split strategy | 🔬 |
| D-009 | Testing strategy and CI | ✅ (no objection) |
| D-010 | Dependency pinning | ✅ (no objection) |
| D-011 | Repository licence: AGPL-3.0 | ✅ confirmed 12 Oct |
| D-012 | Commit authorship and AI attribution | ✅ (no objection) |
| D-013 | Branching strategy | ✅ (no objection) |
| D-014 | How Colab results return to GitHub | ✅ default (a) for now |
| D-015 | Export format from Roboflow | ✅ (no objection) |
| D-016 | Compute budget and scope trimming (free Colab, due 22 Oct) | ✅ |
| D-017 | Training settings: input size, epochs, early stopping | 🟡 |
| D-018 | Evaluate from exported predictions with our own metrics | 🟡 |
| D-019 | Prediction-to-ground-truth matching rule | 🟡 |

---

### D-001 — Problem framing and scope ✅ (confirmed 12 Oct; option (b))
- **Context:** Brief requires a narrow, achievable engineering problem with clear
  inputs/outputs and validation evidence.
- **Options:** (a) defect detection only; (b) detection + tray PASS/FAIL/REVIEW +
  real-time feasibility; (c) meat-quality (fat %) focus; (d) counting/orientation.
- **Recommendation:** (b), with (c) as a stretch.
- **Rationale:** (b) uses the labels that exist, gives an operator-level output,
  and creates a genuine reason to do performance analysis. (c) has no ground truth
  so validation would be weak — the rubric weights validation heavily. (d) has no
  count/orientation labels either.
- **Depends on:** WP1 finding defect-free images (otherwise PASS has no examples).

### D-002 — Code architecture: importable package + thin notebooks 🟡
- **Options:** (a) all code inside Colab notebooks; (b) Python package in `src/`
  with notebooks that import it.
- **Choice:** (b).
- **Rationale:** pytest cannot sensibly test notebook cells; pdoc generates HTML
  docs from modules, not notebooks; reusable functions avoid copy-paste drift
  between notebooks; `main` stays runnable. Notebooks remain the place to run
  experiments and show results.
- **Trade-off:** slightly more setup in Colab (clone + `sys.path`), handled by a
  standard first cell.
- **LO:** testing & validation (efficiently generate and manage test functions).

### D-003 — Docstring style: NumPy 🟡
- **Options:** Google, NumPy, Sphinx/reST (all accepted by rubric).
- **Choice:** NumPy.
- **Rationale:** same convention as NumPy/SciPy/scikit-learn used throughout the
  unit; renders well with `pdoc --docformat numpy`; good for array shapes/units.

### D-004 — Data access 🟡
- **Choice:** download Roboflow v4 inside Colab using the `roboflow` package and an
  API key stored in Colab Secrets (`ROBOFLOW_API_KEY`). Dataset and weights are
  git-ignored; large artefacts go to Google Drive.
- **Rationale:** reproducible (anyone with a free key can re-run); keeps the repo
  small; avoids redistributing the dataset (CC BY 4.0 still requires attribution,
  given in README); keeps secrets out of Git.
- **Note:** Claude's workspace cannot reach Roboflow, so Claude works from a small
  development sample exported by notebook 00.

### D-005 — Detector family and comparison 🔬
- **Proposal:** Ultralytics YOLO, comparing **YOLO11n** (mature, uses NMS
  post-processing) against **YOLO26n** (released Jan 2026, optional NMS-free
  end-to-end head, designed for low-latency deployment). Both are "nano" models
  sized for a free Colab GPU.
- **Why this pair:** similar parameter budgets but a genuinely different
  post-processing design, which directly matters for production-line latency.
- **Evidence still needed:** first training runs (accuracy, training time) before
  committing to a size sweep (n vs s) or input-resolution sweep.

### D-006 — Baselines and alternatives 🟡
- **B0** trivial always-PASS / always-FAIL — the floor any model must beat.
- **B1** scikit-learn image-level classifier: hand-crafted colour/texture features
  in a `Pipeline(StandardScaler, model)` comparing a decision tree, an RBF SVM and
  an MLP, hyperparameters chosen on VALID.
- **Rationale:** the brief requires at least one credible alternative; B1 directly
  exercises unit content (trees, SVM, MLP, scaling) and is cheap, interpretable and
  CPU-only — a realistic "do we even need deep learning?" check.

### D-007 — Engineering requirements ✅ (no line speed was given; you delegated "a reasonable pace" to Claude)
- **Evidence for a realistic range:** published equipment figures span roughly
  **24 trays/min** (an integrated bacon tray/case-packing line, Pearson Packaging)
  to **over 90 packs/min** (a high-speed meat tray sealer, Proseal GT2 as listed by
  Food Processing). References in the report.
- **Design requirement (R1, throughput):** **120 trays/min** — above the fastest
  cited sealer, so a system that meets it is not the line's bottleneck.
  ⇒ one tray every **500 ms**.
- **Latency budget (R2):** the vision computation may use at most **50%** of the
  tray interval, leaving the rest for image capture/transfer and the reject
  actuator ⇒ **≤ 250 ms per image**, end to end (pre-process + inference + post-process).
- **Error costs (R3):** a missed defective tray (e.g. unsealed pack reaching a
  customer: spoilage, complaint, possible recall) is assumed **10×** as costly as a
  false reject (manual re-check or re-pack). Used to choose decision thresholds.
- **Sensitivity:** results reported across 24–240 trays/min and cost ratios 1–50,
  so conclusions don't hinge on these assumed values.

### D-008 — Split strategy 🔬
- **Default:** keep Roboflow's predefined split (comparability with the brief).
- **Trigger for change:** if WP1 finds near-duplicate frames across splits, or the
  test set is too small to estimate per-class performance, propose a grouped
  re-split (groups = near-duplicate clusters / original image stems), keeping all
  augmented copies of an image in the same split.
- **Rule regardless:** TEST is used once, at the end. All selection happens on VALID.
- **LO:** data selection (random vs grouped split; leakage).

### D-009 — Testing strategy and CI 🟡
- pytest with **positive** (valid input → known output) and **negative**
  (invalid input → specific error) cases; synthetic fixtures so tests need no
  dataset or GPU; pytest-html report; GitHub Actions runs tests on every push.

### D-010 — Dependency pinning 🟡
- Pin `ultralytics` and `roboflow` exact versions in `requirements-colab.txt`
  (Ultralytics releases very frequently; default hyperparameters can change).
  Leave Colab's pre-installed torch/numpy as-is and record their versions in each
  run's metadata.

### D-011 — Repository licence ✅ AGPL-3.0 (confirmed 12 Oct; `LICENSE` added)
- **Constraint:** Ultralytics is licensed AGPL-3.0; our code imports it.
- **Options:** (a) AGPL-3.0 for the whole repo — simplest, clearly compatible;
  (b) MIT for our own code with a note that the Ultralytics dependency is AGPL —
  more permissive but the compatibility question is debated.
- **Recommendation:** (a). Dataset remains CC BY 4.0 (separate licence, not in repo).

### D-012 — Commit authorship and AI attribution 🟡
- Commits made by Claude are authored as "Claude" with a `Co-Authored-By` trailer;
  commits you make (e.g. from Colab) are under your name. This makes AI
  contribution visible in the history, which supports the AI-disclosure section.

### D-013 — Branching strategy 🟡
- `main` always runnable. One branch per work package (`wp1-data-audit`,
  `wp2-metrics`, …) merged via pull request with a short description. Gives the
  "evidence of branching and insightful commits" the rubric asks for.

### D-014 — Getting Colab results back to GitHub ✅ (a) for now
- (a) **File → Save a copy in GitHub** from Colab: saves the executed notebook
  (with outputs) — zero setup, but only the notebook is saved.
- (b) A notebook cell that commits `results/` using a fine-grained personal
  access token (this repo only, contents read/write) stored in Colab Secrets.
- **Recommendation:** start with (a); move to (b) once runs produce many files.

### D-015 — Roboflow export format 🟡
- Export as `yolov8` format: same directory layout and label format as the
  YOLO11/YOLO26 exports (one `.txt` per image; `class cx cy w h`, normalised to
  [0, 1]), and the longest-supported format string in the `roboflow` package.

### D-016 — Compute budget and scope trimming ✅
- **Facts (12 Oct):** free Colab only (one T4 GPU, limited daily GPU time, runtime
  disconnects); everything due **22 Oct** (10 days).
- **Consequences:**
  - Only **nano** models are trained (YOLO11n, YOLO26n), **one run each**, single
    seed. Run-to-run variance is stated as a limitation rather than measured.
  - The input-resolution trade-off is studied **at inference** (one model evaluated
    at several sizes) instead of retraining at each size.
  - Everything that does not need a GPU (metrics, baselines, AP study, decision
    thresholds, bootstrap) runs on CPU from saved predictions (D-018).
  - Training runs save to Google Drive every few epochs and can resume after a
    disconnect.
  - Stretch goals S1–S3 are dropped unless the core finishes early.

### D-017 — Training settings 🟡
- **Input size 640** (letterboxed 640 × 480 for these 720 × 540 images): the size
  the COCO-pretrained weights were trained at and the standard benchmark size.
  Native 736 would cost ~1.3× more pixels per image for an uncertain gain; the
  inference-resolution study (WP6) shows the accuracy/speed trade-off instead.
- **Transfer learning** from COCO-pretrained weights, not training from scratch —
  ~1,100 original training images are far too few to learn features from zero.
- **Epochs 50, early-stopping patience 15**, otherwise Ultralytics defaults
  (including its own online augmentation and mixed precision). The first run
  calibrates seconds/epoch; training curves are inspected for convergence and
  overfitting (validation loss rising while training loss falls).
- Identical settings and seed for both models so the comparison is fair.

### D-018 — Evaluate from exported predictions with our own metrics 🟡
- The GPU notebook exports raw predictions (low confidence threshold, as used for
  mAP) for the valid and test images to a small CSV, plus the ground-truth labels.
- All evaluation then runs on CPU from those files, using `porkqc.metrics`:
  reproducible, unit-tested, cheap to re-run with different settings, and
  independent of Ultralytics' internal code paths.
- **Verification:** our pipeline is checked against Ultralytics' own
  `compute_ap`/`ap_per_class` on identical inputs (should agree to rounding), and
  against `model.val()` end to end.

### D-019 — Matching rule 🟡
- **Context:** to count a prediction as a true positive it must be matched to an
  unmatched ground-truth box of the same class with IoU ≥ threshold. Libraries
  differ: COCO/VOC process predictions in **descending confidence**; Ultralytics
  (8.4.175, `match_predictions`) sorts candidate pairs by **IoU** then
  de-duplicates.
- **Choice:** confidence-ordered greedy matching (COCO/VOC convention) as the
  primary rule, because it is the documented standard and depends only on the
  ranking the operator would actually use; an `"ultralytics"` option reproduces
  their rule for verification. The size of any difference is reported.
