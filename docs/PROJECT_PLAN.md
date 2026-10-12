# Project Plan — Automated Inspection of Pork-Rasher Packaging

MMA3001 Numerical Methods and Machine Learning · Individual Project (25%)
Status: **v0.2 (12 Oct 2026)** — framing confirmed; deadline **Thu 22 Oct**; free Colab only.
Items marked ⏳ are waiting on data.

This plan is a living document. Every substantive change should also appear in
[`DECISION_LOG.md`](DECISION_LOG.md) so the reasoning is traceable for the report
and the presentation.

---

## 1. Problem statement ✅ (D-001, confirmed 12 Oct)

The brief asks for "a narrower and achievable problem". Proposed framing:

> **Given a 720 × 540 RGB image of a pork-rasher tray on a packaging line,
> detect and localise five packaging/placement defect types (loose-meat,
> packaging-error, twisted-meat, unsealed, wrinkle), convert the detections into a
> tray-level PASS / FAIL / REVIEW decision, and determine whether the method is
> accurate and fast enough for real-time in-line inspection.**

Why this framing:

- It uses the labels that actually exist in the dataset (bounding boxes for five
  defect classes). "Meat quality" (e.g. fat percentage) has **no ground truth** in
  the dataset, so it could not be validated properly — it is kept as a stretch goal.
- It combines three of the coordinator's suggested directions (detect/localise
  defects; pass/fail decision; real-time adequacy) into one coherent engineering
  question with a clear output a line operator would use.
- The "real-time adequacy" part creates a natural home for the unit's numerical
  performance content (FLOPs, arithmetic intensity, machine balance, profiling).
- The REVIEW band reflects real practice: uncertain trays go to a human, so the
  system *reduces* manual inspection rather than pretending to eliminate it.

Out of scope (stated as limitations): meat-quality grading, seal-strength
measurement, multi-camera fusion, deployment on real line hardware.

---

## 2. What we know about the data (and what we don't)

Facts from the Roboflow page (to be re-verified by the data audit):

| Item | Value |
|---|---|
| Source | Roboflow Universe, workspace `hello-2aqe0`, project `pork-rasher-error-packaging`, version 4 |
| Licence | CC BY 4.0 (attribution required) |
| Task / labels | Object detection, axis-aligned bounding boxes |
| Classes (5) | loose-meat, packaging-error, twisted-meat, unsealed, wrinkle |
| Project images | 1,955 originals |
| v4 images | 3,549 = 3,349 train (94%) + 120 valid (4%) + 80 test (2%) |
| Preprocessing | Auto-orient; resize "Fill (with centre crop)" to 720 × 540 |
| Augmentation (train only) | 3 outputs per training example; saturation ±29%; exposure ±10% |

Open questions the first notebook must answer:

1. **Class balance.** Per-class instance counts are not published. Rare classes
   will dominate the uncertainty of per-class AP.
2. **Are there defect-free ("good") trays?** If every image contains at least one
   defect, a tray-level PASS/FAIL decision cannot be validated against negatives —
   this would change D-001.
3. **Image accounting.** 3,349 / 3 ≈ 1,116.3 is not an integer, and
   ~1,116 + 120 + 80 ≈ 1,316 originals is well short of 1,955. Where did the other
   images go (unannotated? excluded? duplicated?).
4. **Leakage risk.** Images look like frames from conveyor video. Near-identical
   consecutive frames split randomly across train/valid/test would inflate test
   scores. We will check with perceptual hashing.
5. **Tiny evaluation sets.** 120 valid / 80 test images means wide confidence
   intervals. We will quantify this with bootstrapping and may propose a grouped
   re-split (⏳ D-008).

---

## 3. Approach at a glance

```
            ┌───────────────────────── Data (Roboflow v4, YOLO format) ─────────────────────────┐
            │ audit: counts, nulls, box stats, near-duplicate/leakage check, split decision       │
            └──────────────┬──────────────────────────────────────────────┬───────────────────────┘
                           │                                              │
             Baselines (alternatives)                         Primary detector(s)
   ┌────────────────────────────────────────┐        ┌──────────────────────────────────────┐
   │ B0 trivial: always-PASS / always-FAIL  │        │ YOLO11n  vs  YOLO26n (NMS-free head)  │
   │ B1 sklearn image-level classifier      │        │ (+ "s" size / input-resolution sweep) │
   │    (colour/texture features + Pipeline)│        └──────────────────┬───────────────────┘
   └───────────────────┬────────────────────┘                           │
                       └────────────────┬───────────────────────────────┘
                                        ▼
                   Tray-level decision: PASS / FAIL / REVIEW
                   (thresholds chosen on VALID only, cost-weighted)
                                        ▼
       Validation & credibility                         Optimisation & performance
   ┌──────────────────────────────────────┐   ┌──────────────────────────────────────────┐
   │ mAP50, mAP50-95, per-class AP,        │   │ theoretical GFLOPs, arithmetic intensity, │
   │ tray-level miss / false-reject rates  │   │ GPU machine balance (roofline),           │
   │ bootstrap CIs; own AP implementation  │   │ latency vs batch/resolution/FP16,         │
   │ verified against Ultralytics;         │   │ profiler breakdown (pre/infer/NMS/post),  │
   │ robustness to exposure/blur/noise;    │   │ accuracy–speed Pareto front vs line speed │
   │ failure gallery (reflections, overlap)│   └──────────────────────────────────────────┘
   └──────────────────────────────────────┘
          Numerical-methods study: AP is an integral, AP = ∫₀¹ p(r) dr
          → compare trapezoid / Simpson / 11- & 101-point / exact step rules,
            convergence (log–log) on a non-smooth integrand, why Romberg fails here
```

---

## 4. Work packages

Each package lists its deliverable, the rubric line it feeds, and the learning
objectives (LO) it evidences. Full LO mapping: [`LEARNING_OBJECTIVES_MAP.md`](LEARNING_OBJECTIVES_MAP.md).

### WP0 — Repository, environment and data access *(this session)*
- Package skeleton (`src/porkqc`), configs, CI, README, planning docs.
- Colab notebook 00: clone repo, install, read API key from Colab Secrets,
  download v4, verify split counts, first class/null counts, export a small
  development sample.
- Rubric: GitHub structure/README; inputs & outputs. LO: testing (positive/negative cases).

### WP1 — Data audit and split decision
- Class and box statistics (sizes, aspect ratios, positions, boxes per image).
- Null-image count → decides whether PASS/FAIL is validatable.
- Reconcile image accounting (originals vs augmented copies via Roboflow filename stems).
- Near-duplicate detection across splits (perceptual hash + Hamming distance).
- Decision: keep Roboflow split vs grouped re-split (⏳ D-008).
- Rubric: validation & engineering credibility. LO: data selection/leakage, grouped splits, distribution shift.

### WP2 — Metrics core (own implementation, tested)
- IoU, prediction-to-ground-truth matching, precision–recall curve.
- Average precision via several integration rules; verify against Ultralytics.
- Bootstrap confidence intervals.
- Rubric: validation; documentation & tests. LO: testing, numerical integration & error, uncertainty.

### WP3 — Detector training (Colab GPU)
- YOLO11n and YOLO26n from COCO-pretrained weights, same data/epochs/seed.
- Model selection on VALID only; test split stays locked until WP7.
- Record training curves; convergence criterion (early-stopping patience) explained.
- Rubric: computational solution; justification of approach. LO: overfitting (train vs validation), floating point (AMP/FP16).

### WP4 — Baselines and alternatives
- B0 trivial decisions (floor performance).
- B1 scikit-learn image-level classifier in a `Pipeline(StandardScaler, model)`,
  comparing decision tree / SVM (RBF) / MLP with validation-based hyperparameters.
- Rubric: alternative solutions. LO: trees, SVR→SVC, MLP, pipelines, scaling, permutation importance.

### WP5 — Tray-level decision and engineering metrics
- Map detections → PASS / FAIL / REVIEW with two confidence thresholds.
- Choose thresholds on VALID by minimising expected cost (⏳ D-007 cost ratio).
- Report miss rate (defective tray passed), false-reject rate, review rate.
- Rubric: engineering problem/outputs; validation. LO: metrics matched to consequences.

### WP6 — Performance and optimisation
- Theoretical: GPU peak FLOP/s and bandwidth → machine balance; model GFLOPs →
  arithmetic intensity; compute- vs memory-bound argument.
- Empirical: latency/throughput vs batch size, input resolution, FP32 vs FP16,
  CPU vs GPU; repeated timings with warm-up (median + IQR).
- Profiling: where time goes (pre-processing / inference / NMS / post-processing).
- Accuracy–speed Pareto front against the line-speed requirement (⏳ D-007).
- Rubric: optimisation & performance (baseline vs improved). LO: numerical performance, profiling, floating point.

### WP7 — Final evaluation, robustness and failure analysis
- One-shot evaluation of the selected configuration on the locked TEST split.
- Robustness/sensitivity: mAP vs exposure shift, blur, noise, JPEG quality.
- Failure gallery with defined criteria (reflections, occlusion, overlapping meat).
- Rubric: validation & credibility; honest limitations.

### WP8 — Numerical-methods study: AP as an integral
- AP = ∫₀¹ p(r) dr where p(r) is a non-smooth (step) function.
- Compare exact step-area, VOC 11-point, COCO 101-point, trapezoid, Simpson,
  Gauss–Legendre; N-point refinement study on a log–log plot (observed order);
  show why Simpson/Romberg smoothness assumptions fail on step integrands.
- Rubric: justification of computational approach. LO: Newton–Cotes, integration error, Richardson/Romberg limits, Gaussian quadrature, interpolation (precision envelope).

### WP9 — Documentation, tests report, report & presentation
- pdoc HTML API docs, pytest-html report, README results summary, LICENSE.
- 5–10 page report; AI reflection; 5-minute presentation + Q&A preparation notes.

### Stretch (only if core is solid)
- S1 meat-quality proxy (lean/fat pixel ratio per rasher via HSV segmentation) —
  validated only by synthetic images with known areas + defined manual criteria.
- S2 two-stage detector (torchvision Faster R-CNN) as a different paradigm.
- S3 ONNX / TensorRT export and INT8 quantisation.

---

## 5. Rubric → evidence matrix

| Rubric item | Where the evidence will live |
|---|---|
| Engineering problem, inputs & outputs | `docs/IO_SPECIFICATION.md`, README, report §1–2 |
| GitHub structure, commits, README | repo layout, feature branches + PRs, README quick-start & Colab badges |
| Inline comments, docstrings, HTML docs | NumPy-style docstrings in `src/porkqc`, `docs/api/` (pdoc) |
| Validation & engineering credibility | WP1, WP2, WP5, WP7 notebooks + `results/` |
| Justification of computational approach | WP3, WP4, WP6, WP8 + `DECISION_LOG.md` |
| AI use & critical reflection | `docs/AI_USAGE_LOG.md` → report AI section |

---

## 6. Timeline — everything due Thu 22 Oct (D-016)

GPU time on free Colab is the bottleneck, so GPU jobs are front-loaded and
everything else runs on CPU in parallel. "You" = run in Colab; "Claude" = code/analysis.

| Day | GPU (you, Colab) | CPU / code (Claude, and you reviewing) |
|---|---|---|
| Mon 12 | Notebook 00 (CPU): download + first look | Re-plan; licence; metrics core (WP2); training notebook |
| Mon 12–Tue 13 | **Run 1: YOLO11n, Roboflow split** (~1 h); notebook 01 audit (CPU) + your review of unlabelled images | Evaluate run 1 on VALID with own metrics; confirm blocked-split settings |
| Tue 13–Wed 14 | **Run 2: YOLO11n, blocked split**; **Run 3: YOLO26n, blocked split** (D-022) | Leakage optimism (run 1 vs 2); model comparison (run 2 vs 3); verify vs Ultralytics |
| Thu 15 | Short session: latency benchmarks (GPU vs CPU, batch, resolution, FP16) | Baselines B0/B1 (WP4); decision layer + thresholds (WP5) |
| Fri 16 | Robustness run (perturbed VALID images) | AP-as-integral study (WP8); roofline analysis (WP6) |
| Sat 17 | **One-shot TEST evaluation** of the selected model | Bootstrap CIs, failure gallery (WP7) |
| Sun 18 | — | README, HTML docs, test report; report outline + figures |
| Mon 19 | — | Report draft (Claude drafts technical sections; you edit) |
| Tue 20 | — | You: AI reflection + read-through; Claude: fixes, consistency checks |
| Wed 21 | — | Buffer; final repo check; zip clone; presentation notes |
| **Thu 22** | — | **Submit** report (Moodle), repo link + zipped clone |

Presentation (Week 14) prep notes are produced alongside the report.

**Trimmed for the deadline:** single seed per model; no model-size or training
resolution sweep (inference-resolution study instead); stretch goals dropped unless
the core is finished early.

---

## 7. How we work together (Claude ↔ Colab ↔ GitHub)

1. Claude writes/edits code, tests and notebooks and pushes them to GitHub
   (feature branch → pull request → merge to `main`; `main` must always run).
2. You open a notebook in Colab via its README badge and run it on a GPU runtime.
3. Notebooks save small artefacts (metrics JSON/CSV, figures) to `results/`;
   large artefacts (dataset, weights, run folders) go to Google Drive, never Git.
4. Results come back to GitHub ⏳ (D-014): either *File → Save a copy in GitHub*
   (zero setup) or a scripted push using a fine-grained token in Colab Secrets.
5. After each work package Claude writes a short "explain-it-back" note: the
   concept, the decision, the evidence and likely examiner questions — so you can
   defend the work without AI in the Q&A.

---

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No defect-free images | PASS/FAIL unvalidatable | Reframe tray decision as "defect present per class"; report as limitation |
| Near-duplicate leakage across splits | Inflated test scores | Hash-based audit; grouped re-split; report both |
| 80-image test set | Wide uncertainty | Bootstrap CIs; avoid over-claiming; consider re-split |
| Colab GPU limits / disconnects | Lost runs | Save runs to Drive; resume; small models (n) first |
| Ultralytics version drift | Irreproducible results | Pin versions in `requirements-colab.txt` |
| Ultralytics is AGPL-3.0 | Licence compatibility | Resolved: repo licensed AGPL-3.0 (D-011) |
| 10-day deadline + free Colab GPU limits | Unfinished experiments | GPU jobs first; nano models; single seed; CPU analysis from saved predictions (D-016, D-018) |
| Ambiguous class definitions (e.g. packaging-error vs unsealed) | Confused labels, low AP | Inspect examples; ask coordinator on edStem; per-class confusion analysis |
| Over-scoping | Shallow work everywhere | Core WPs first; stretch only after WP7 |
