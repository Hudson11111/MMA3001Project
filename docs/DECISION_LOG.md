# Decision Log

Every non-trivial technical choice is recorded here: what the options were, what
was chosen, why, and what evidence supports it. This is the source material for
the report's "justification of computational approach" section and for answering
"why did you do X instead of Y?" in the presentation.

**Status key:** ✅ Accepted / implemented. (During the project, 🟡 marked proposals awaiting Hudson's
review, ⏳ decisions pending Hudson and 🔬 decisions pending evidence; all are now resolved — see git history.)

Decisions marked ✅ on 12 Oct were either confirmed by you explicitly (D-001,
D-011, deadline/compute facts behind D-016) or proposed by Claude with no
objection raised; the latter are noted "(no objection)".

| ID | Decision | Status |
|---|---|---|
| D-001 | Problem framing and scope | ✅ confirmed 12 Oct |
| D-002 | Code architecture: package + thin Colab notebooks | ✅ (proposed by Claude; no objection from Hudson) |
| D-003 | Docstring style: NumPy | ✅ (proposed by Claude; no objection from Hudson) |
| D-004 | Data access via Roboflow API in Colab; data never committed | ✅ (proposed by Claude; no objection from Hudson) |
| D-005 | Primary detector family and comparison | ✅ YOLO11n vs YOLO26n compared (R4); YOLO11n selected (D-025) |
| D-006 | Baselines / alternative solutions | ✅ (proposed by Claude; no objection from Hudson) |
| D-007 | Engineering requirements: line speed and error-cost ratio | ✅ delegated to Claude 12 Oct |
| D-008 | Train/valid/test split strategy: time-blocked split for all conclusions | ✅ evidence: +0.27 mAP50 leakage optimism (R4) |
| D-009 | Testing strategy and CI | ✅ (proposed by Claude; no objection from Hudson) |
| D-010 | Dependency pinning | ✅ (proposed by Claude; no objection from Hudson) |
| D-011 | Repository licence: AGPL-3.0 | ✅ confirmed 12 Oct |
| D-012 | Commit authorship and AI attribution | ✅ (proposed by Claude; no objection from Hudson) |
| D-013 | Branching strategy | ✅ (proposed by Claude; no objection from Hudson) |
| D-014 | How Colab results return to GitHub | ✅ default (a) for now |
| D-015 | Export format from Roboflow | ✅ (proposed by Claude; no objection from Hudson) |
| D-016 | Compute budget and scope trimming (free Colab, due 22 Oct) | ✅ |
| D-017 | Training settings: input size, epochs, early stopping | ✅ used for all three runs — proposed by Claude, accepted via PR review |
| D-018 | Evaluate from exported predictions with our own metrics | ✅ all reported metrics from exported predictions — proposed by Claude, accepted via PR review |
| D-019 | Prediction-to-ground-truth matching rule | ✅ confidence-ordered; Ultralytics rule reproduced for verification (V7) — proposed by Claude, accepted via PR review |
| D-020 | AP integration rule (exact step area as primary) | ✅ exact; rule spread 17× smaller than sampling CI (R5) — proposed by Claude, accepted via PR review |
| D-021 | Unit of data = scene (image + its augmented copies), found by content | ✅ controls confirmed on full data (R1) — proposed by Claude, accepted via PR review |
| D-022 | GPU run order: 11n/roboflow → 11n/blocked → 26n/blocked | ✅ all three runs completed — proposed by Claude, accepted via PR review |
| D-023 | Tray decision without validated PASS: FAIL/REVIEW + false alarms on empty frames | ✅ applied on TEST (R6) — proposed by Claude, accepted via PR review |
| D-024 | Headline classes: unsealed + packaging-error; others reported as not evaluable | ✅ applied in R4–R6 — proposed by Claude, accepted via PR review |
| D-025 | Model to evaluate once on blocked TEST: YOLO11n (blocked) | ✅ Hudson, 17 Oct |
| D-026 | Decision thresholds frozen on VALID before TEST | ✅ 17 Oct |

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

### D-002 — Code architecture: importable package + thin notebooks ✅
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

### D-003 — Docstring style: NumPy ✅
- **Options:** Google, NumPy, Sphinx/reST (all accepted by rubric).
- **Choice:** NumPy.
- **Rationale:** same convention as NumPy/SciPy/scikit-learn used throughout the
  unit; renders well with `pdoc --docformat numpy`; good for array shapes/units.

### D-004 — Data access ✅
- **Choice:** download Roboflow v4 inside Colab using the `roboflow` package and an
  API key stored in Colab Secrets (`ROBOFLOW_API_KEY`). Dataset and weights are
  git-ignored; large artefacts go to Google Drive.
- **Rationale:** reproducible (anyone with a free key can re-run); keeps the repo
  small; avoids redistributing the dataset (CC BY 4.0 still requires attribution,
  given in README); keeps secrets out of Git.
- **Note:** Claude's workspace cannot reach Roboflow, so Claude works from a small
  development sample exported by notebook 00.

### D-005 — Detector family and comparison ✅ (outcome: R4, D-025)
- **Proposal:** Ultralytics YOLO, comparing **YOLO11n** (mature, uses NMS
  post-processing) against **YOLO26n** (released Jan 2026, optional NMS-free
  end-to-end head, designed for low-latency deployment). Both are "nano" models
  sized for a free Colab GPU.
- **Why this pair:** similar parameter budgets but a genuinely different
  post-processing design, which directly matters for production-line latency.
- **Evidence still needed:** first training runs (accuracy, training time) before
  committing to a size sweep (n vs s) or input-resolution sweep.

### D-006 — Baselines and alternatives ✅
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

### D-008 — Split strategy ✅ (updated 12 Oct evening after the development sample; confirmed by R1 and R4)
- **New evidence (80-image development sample):** every file is a numbered video
  frame (`frame_with_red_<N>`), and Roboflow's split interleaves frames: e.g.
  frame 2247 is in VALID and frame 2248 (same tray, one frame later) is in TEST.
  In the sample, 40% of VALID and 40% of TEST frames have a TRAIN frame within 50
  frames. This is temporal leakage: test scores would partly measure recall of
  trays already seen.
- **Decision (proposed):** report results on **both** splits —
  1. Roboflow's split *as provided* (comparability with the brief's inputs);
  2. a **time-blocked split**: contiguous blocks of 300 frames assigned whole to
     train/valid/test (70/15/15 of scenes), plus a **purge gap** of 50 frames
     between different splits (`porkqc.data.resplit`, settings in
     `configs/dataset.yaml`). The difference between the two is the measured
     optimism caused by leakage. The model comparison uses the blocked split.
- **Why blocks + gap (and not image similarity):** perceptual hashing could not
  tell "same tray, next frame" from "different tray" on this dark, uniform
  background (VERIFICATION_LOG V10); frame numbers give the time order directly.
  Blocks also give larger VALID/TEST sets than 120/80 images.
- **To confirm in notebook 01:** how many frames one tray stays in view (sets the
  gap), and the purge cost (training scenes lost) — sensitivity table.
- **Original plan (kept):** TEST is used once, at the end. All selection happens on VALID.
- **Default:** keep Roboflow's predefined split (comparability with the brief).
- **Trigger for change:** if WP1 finds near-duplicate frames across splits, or the
  test set is too small to estimate per-class performance, propose a grouped
  re-split (groups = near-duplicate clusters / original image stems), keeping all
  augmented copies of an image in the same split.
- **Rule regardless:** TEST is used once, at the end. All selection happens on VALID.
- **Evidence method (notebook 01):** perceptual-hash (dHash) nearest-neighbour distances from every
  VALID/TEST image to TRAIN, with the threshold calibrated on a positive control (augmented
  copies of one photo) and a negative control (different photos), a threshold-sensitivity table,
  and a manual check of the closest pairs. Near-duplicate groups are saved for a grouped
  bootstrap or re-split.
- **LO:** data selection (random vs grouped split; leakage).

### D-009 — Testing strategy and CI ✅
- pytest with **positive** (valid input → known output) and **negative**
  (invalid input → specific error) cases; synthetic fixtures so tests need no
  dataset or GPU; pytest-html report; GitHub Actions runs tests on every push.

### D-010 — Dependency pinning ✅
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

### D-012 — Commit authorship and AI attribution ✅
- Commits made by Claude are authored as "Claude" with a `Co-Authored-By` trailer;
  commits you make (e.g. from Colab) are under your name. This makes AI
  contribution visible in the history, which supports the AI-disclosure section.

### D-013 — Branching strategy ✅
- `main` always runnable. One branch per work package (`wp1-data-audit`,
  `wp2-metrics`, …) merged via pull request with a short description. Gives the
  "evidence of branching and insightful commits" the rubric asks for.

### D-014 — Getting Colab results back to GitHub ✅ (a) for now
- (a) **File → Save a copy in GitHub** from Colab: saves the executed notebook
  (with outputs) — zero setup, but only the notebook is saved.
- (b) A notebook cell that commits `results/` using a fine-grained personal
  access token (this repo only, contents read/write) stored in Colab Secrets.
- **Recommendation:** start with (a); move to (b) once runs produce many files.

### D-015 — Roboflow export format ✅
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

### D-017 — Training settings ✅
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

### D-018 — Evaluate from exported predictions with our own metrics ✅
- The GPU notebook exports raw predictions (low confidence threshold, as used for
  mAP) for the valid and test images to a small CSV, plus the ground-truth labels.
- All evaluation then runs on CPU from those files, using `porkqc.metrics`:
  reproducible, unit-tested, cheap to re-run with different settings, and
  independent of Ultralytics' internal code paths.
- **Verification:** our pipeline is checked against Ultralytics' own
  `compute_ap`/`ap_per_class` on identical inputs (should agree to rounding), and
  against `model.val()` end to end.

### D-019 — Matching rule ✅
- **Context:** to count a prediction as a true positive it must be matched to an
  unmatched ground-truth box of the same class with IoU ≥ threshold. Libraries
  differ: COCO/VOC process predictions in **descending confidence**; Ultralytics
  (8.4.175, `match_predictions`) sorts candidate pairs by **IoU** then
  de-duplicates.
- **Choice:** confidence-ordered greedy matching (COCO/VOC convention) as the
  primary rule, because it is the documented standard and depends only on the
  ranking the operator would actually use; an `"ultralytics"` option reproduces
  their rule for verification. The size of any difference is reported.

### D-020 — AP integration rule ✅
- **Context:** AP is the integral of the interpolated-precision step function
  (see `porkqc.metrics.ap`). Libraries approximate it with different quadrature
  rules: COCO averages 101 samples; VOC2007 averages 11; Ultralytics applies the
  trapezoid rule to 101 samples of a linearly interpolated envelope.
- **Choice:** report the **exact** step-function area as our primary AP — it is
  the integral itself, with no discretisation error — and report the
  Ultralytics-rule value alongside for comparability with training logs.
- **Evidence so far:** unit tests show the rules differ on hand-computed curves
  (0.8333 exact vs 0.8350 COCO vs 0.8485 VOC for the same three predictions) and
  that Ultralytics' rule caps a perfect detector at 0.995 (VERIFICATION_LOG V6).
  The size of the difference on our real, coarse PR curves (few instances per
  class in VALID/TEST) is measured in WP8.

### D-021 — Unit of data: the scene ✅
- **Finding:** file names do not identify images uniquely — two *different* trays
  in the sample are both named `frame_with_red_992` (probably two videos numbered
  the same way). Grouping "augmented copies" by name would merge different images.
- **Choice:** a *scene* = one original image plus its augmented copies, found by
  correlating standardised 64 × 48 greyscale thumbnails (threshold 0.99).
- **Calibration:** manufactured positive control — real images colour-augmented
  within Roboflow's ranges correlate ≥ 0.9995; negative control — different frames
  ≤ 0.933 in the sample. Notebook 01 re-checks both controls on the full dataset.
- **Used for:** counting real originals, direct-leakage check, the blocked split
  (whole scenes move together; VALID/TEST keep one copy per scene).

### D-022 — GPU run order ✅
1. `yolo11n_640` on the Roboflow split (baseline "as provided");
2. `yolo11n_640` on the blocked split (same model → measures leakage optimism);
3. `yolo26n_640` on the blocked split (model comparison without leakage).
- Each is ~1 h on a free T4. If GPU time runs out, run 3 is dropped first and the
  comparison is reported as future work.



### D-023 — Tray-level decision without a validated PASS ✅
- **Evidence:** Hudson's independent review of the 26 unlabelled VALID/TEST images found
  23 E (edge/empty: tray not fully in view), 1 M (missed defect), 2 G (good tray); Claude's
  independent pass agreed that most are E (RESULTS_LOG R1).
- **Consequence:** "no label" ≠ "good tray", so PASS accuracy cannot be validated.
- **Choice:** the decision layer outputs **FAIL** (confident defect) / **REVIEW** (uncertain)
  / **no defect found**. We report (i) **miss rate** on images with labelled defects and
  (ii) **false-alarm rate** on E frames, and state that a deployed system needs a
  "tray fully in view" trigger or per-tray aggregation over consecutive frames.

### D-024 — Headline classes ✅
- wrinkle has no VALID/TEST instances; loose-meat and twisted-meat have 1–4 in blocked TEST,
  giving bootstrap CIs of roughly [0, 1] (RESULTS_LOG R2).
- **Choice:** headline metrics on **unsealed** and **packaging-error** (with all-class mAP
  reported alongside); rare classes are reported as "not reliably evaluable", not dropped
  from training.

### D-025 — Which model goes to the one-shot TEST evaluation ✅ YOLO11n (Hudson, 17 Oct)
- **Evidence (R4, blocked VALID, paired bootstrap):** YOLO26n vs YOLO11n — mAP50 +0.13 (CI −0.08…+0.27,
  not significant), driven by rare classes; headline classes: unsealed +0.02 (n.s.), packaging-error
  −0.08 (significant). Latency equal within noise.
- **Options:** (a) YOLO11n — better or equal on both headline classes, mature, simpler;
  (b) YOLO26n — higher all-class mAP and mAP50-95, NMS-free option.
- **Claude's recommendation:** (a) YOLO11n, because the decision layer is driven by the classes that
  can actually be evaluated (D-024), and "not significantly different overall" plus "significantly
  worse on the most common defect" favours the simpler model. Report YOLO26n alongside on VALID.
- **Rule:** whichever is chosen is evaluated on blocked TEST **once**; no further tuning afterwards.

### D-026 — Decision thresholds, frozen before TEST ✅
- **Method:** exhaustive grid search (0.00–1.00, step 0.01) of `t_review ≤ t_fail` on blocked VALID
  (196 images, 159 with labelled defects), YOLO11n-blocked predictions, minimising mean cost with
  miss = 10, false reject = 1, review = 0.25 (`porkqc.decision`). Grid search because the cost is a
  step function of the thresholds (no gradient).
- **Result:** `t_review = 0.01`, `t_fail = 0.16` → on VALID: 0 misses, 18 false rejects, 27 reviews,
  cost 0.126 per image. Trivial baselines: always-PASS 8.11, **always-FAIL 0.189**.
- **Honest reading:** VALID is 81% defective and its "clean" images are mostly edge frames (R1), so
  rejecting everything is already cheap; the model cuts the cost by a third, not by orders of
  magnitude. `t_review` sits at the bottom of the grid (any detection triggers review), and the
  thresholds are fitted to only 37 clean images — expect them to transfer imperfectly to TEST.
- **Sensitivity:** for miss costs 3–50 the optimum keeps 0 misses; the review cost decides the band
  width (review 0.1 → 98 reviews, 6 false rejects; review 1.0 → no review band, 31 false rejects).
- Frozen in `configs/decision.yaml`, committed before any TEST prediction exists (pre-registration).

### D-027 — Commit the selected model to the repository ✅
- **Options:** (a) keep weights only on Google Drive (originally planned, `.gitignore` excludes `*.pt`);
  (b) commit the selected model; (c) publish via a release / model hub.
- **Choice:** (b) for the one model that went to TEST (`models/yolo11n_640_blocked.pt`, 5.2 MB),
  found missing by the marker-style audit (V16). Other runs stay on Drive.
- **Rationale:** without the weights, nobody but Hudson could reproduce a single prediction; 5.2 MB is
  well within GitHub limits. A SHA-256 in `models/README.md` is checked by a test, so the file cannot
  silently change. A small command-line tool (`python -m porkqc.cli image.jpg`) runs the model plus the
  frozen decision rule — the deployed form of the system described in the I/O specification.
- **Provenance:** file supplied by Hudson from Drive (17 Oct); checked before use (V18).

### D-028 — Code style enforced in CI ✅
- **Choice:** ruff for linting (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, pep8-naming)
  and formatting, line length 110, checked by a separate CI job.
- **Rationale:** consistent style across 43 files and early detection of latent bugs (V21). Upper-case
  names such as `H`, `W`, `N` are allowed where they mirror the formulas in docstrings.
- **Provenance:** proposed by Claude during submission polish (24 Oct), reviewed via pull request.
