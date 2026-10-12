# Decision Log

Every non-trivial technical choice is recorded here: what the options were, what
was chosen, why, and what evidence supports it. This is the source material for
the report's "justification of computational approach" section and for answering
"why did you do X instead of Y?" in the presentation.

**Status key:** ✅ Accepted · 🟡 Proposed (Claude's recommendation, awaiting your OK)
· ⏳ Pending your decision · 🔬 Pending data/evidence

| ID | Decision | Status |
|---|---|---|
| D-001 | Problem framing and scope | ⏳ |
| D-002 | Code architecture: package + thin Colab notebooks | 🟡 |
| D-003 | Docstring style: NumPy | 🟡 |
| D-004 | Data access via Roboflow API in Colab; data never committed | 🟡 |
| D-005 | Primary detector family and comparison | 🔬 |
| D-006 | Baselines / alternative solutions | 🟡 |
| D-007 | Engineering requirements: line speed and error-cost ratio | ⏳ |
| D-008 | Train/valid/test split strategy | 🔬 |
| D-009 | Testing strategy and CI | 🟡 |
| D-010 | Dependency pinning | 🟡 |
| D-011 | Repository licence | ⏳ |
| D-012 | Commit authorship and AI attribution | 🟡 |
| D-013 | Branching strategy | 🟡 |
| D-014 | How Colab results return to GitHub | ⏳ |
| D-015 | Export format from Roboflow | 🟡 |

---

### D-001 — Problem framing and scope ⏳
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

### D-007 — Engineering requirements ⏳
- **Needed:** (1) line speed in trays/minute (sets the latency budget);
  (2) how much worse a *missed* defective tray is than a *false reject* (sets the
  decision thresholds).
- **If unknown:** assume a nominal value (e.g. 60 trays/min; miss cost 10× false
  reject), state it as an assumption, and run a sensitivity analysis across a
  range so conclusions don't hinge on one guess.

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

### D-011 — Repository licence ⏳
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

### D-014 — Getting Colab results back to GitHub ⏳
- (a) **File → Save a copy in GitHub** from Colab: saves the executed notebook
  (with outputs) — zero setup, but only the notebook is saved.
- (b) A notebook cell that commits `results/` using a fine-grained personal
  access token (this repo only, contents read/write) stored in Colab Secrets.
- **Recommendation:** start with (a); move to (b) once runs produce many files.

### D-015 — Roboflow export format 🟡
- Export as `yolov8` format: same directory layout and label format as the
  YOLO11/YOLO26 exports (one `.txt` per image; `class cx cy w h`, normalised to
  [0, 1]), and the longest-supported format string in the `roboflow` package.
