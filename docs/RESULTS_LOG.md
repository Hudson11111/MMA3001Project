# Results Log

Running record of results as they come in (source files in `results/`). Every number
here is reproducible from the committed JSON/CSV files with `porkqc.metrics`.

## R1 — Data audit (notebook 01, full dataset, run 12–14 Oct)

| Item | Value |
|---|---|
| Images (train / valid / test) | 3,349 / 120 / 80 — matches Roboflow |
| Distinct **scenes** (one image + its augmented copies) | 1,211 / 120 / 80 = 1,411 |
| File names shared by *different* images | 128 (scene detection by thumbnail correlation was necessary, D-021) |
| Copy-detection controls | synthetic copies ≥ 0.9995; different images ≤ 0.910; threshold 0.99 separates them |
| Training copies per scene | 3 × 1,021 · 2 × 96 · 1 × 94 |
| Direct leakage (same scene in two splits) | none |
| Frames | 3 – 10,590, median spacing between dataset frames 4 |
| **Temporal leakage, Roboflow split** | median gap VALID→TRAIN **3 frames**, TEST→TRAIN **2.5**; 68% of VALID and TEST frames have a TRAIN frame within 5 frames; **100% within 50** |
| One tray stays in view for | roughly 5–15 consecutive frames (`01_frame_runs.png`) |
| **Blocked split** (block 300, gap 50) | 938 / 196 / 180 scenes, 97 purged; **0%** of VALID/TEST within 50 frames of TRAIN |
| Objects per class, all splits | packaging-error 1,769 · unsealed 1,185 · twisted-meat 79 · loose-meat 72 · wrinkle 18 |
| Not evaluable | **wrinkle**: 0 instances in VALID and TEST under both splits; loose-meat and twisted-meat have 1–4 instances in blocked TEST |
| Unlabelled scenes | 249 / 14 / 12 |

### Unlabelled-image review (26 VALID/TEST images, codes E/G/M/U)

| Reviewer | E (edge/empty) | G (good tray) | M (missed defect) | U |
|---|---|---|---|---|
| Hudson (independent; final, 14 Oct) | 25 | 0 *(2 initially G, re-coded E: not all tray edges visible)* | 1 *(frame 6496, to confirm)* | 0 |
| Claude (independent, low-resolution grid) | 19 | — | — | 7 (substantial tray visible: 132, 258, 595, 1195, 1433, 1892, 4088) |

**Conclusion:** both reviewers independently judged most unlabelled images as trays not fully in view. An unlabelled image almost always means "no tray fully in view", not "good
tray". There are too few confirmed good trays to validate a PASS decision (→ D-023).

## R2 — Run 1: YOLO11n, Roboflow split (as provided)

Trained 12 Oct on a Tesla T4: 41 epochs (early stopping, patience 15; best epoch 26),
33.9 min. Settings: `configs/experiments/yolo11n_640.yaml`.

| Metric (VALID, 120 images) | Value |
|---|---|
| mAP50 — Ultralytics `model.val()` | 0.661 |
| mAP50 — our pipeline on exported predictions, Ultralytics rules | 0.629 |
| mAP50 — our pipeline, exact AP (primary) | **0.630**, 95% bootstrap CI **0.44 – 0.84** |
| mAP50-95 — exact | **0.360**, CI 0.24 – 0.47 |
| mAP50 — COCO 101-point / VOC 11-point | 0.628 / 0.614 |
| Latency per image (T4, FP32, batch 1) | median **10.6 ms** (pre 1.5 + infer 8.0 + post 1.2), p95 13.9 ms |

| Class | VALID instances | AP50 (exact) | 95% CI |
|---|---|---|---|
| unsealed | 81 | 0.93 | 0.88 – 0.97 |
| packaging-error | 34 | 0.35 | 0.22 – 0.54 |
| loose-meat | 3 | 0.56 | 0.01 – 1.00 |
| twisted-meat | 3 | 0.67 | 0.01 – 1.00 |
| wrinkle | 0 | — | — |

**Observations**
1. **Unsealed is learned well; packaging-error is not.** Packaging-error boxes are tiny
   (median 0.37% of the image, 28% below COCO's "small" size), which is a known weakness
   at 640 px input.
2. **Loose-meat / twisted-meat APs are meaningless** with 3 instances each — the CIs span
   almost the whole [0, 1] range.
3. **The validation curve is very noisy** (epoch-to-epoch mAP50 jumps of ±0.1; the best
   epoch 26 at 0.66 sits between epochs at 0.48). Selecting the best epoch on such a small,
   noisy set favours a lucky epoch ("winner's curse"), so 0.66 is optimistic even before
   leakage is considered.
4. **Leakage:** all VALID images have near-twins in TRAIN (R1), so these scores are expected
   to be optimistic. Run 2 (same model, blocked split) measures by how much.
5. **Speed:** ~11 ms per image is ~20× inside the 250 ms budget (D-007).
6. **Integration rule matters slightly:** exact vs VOC 11-point differs by 0.016 mAP50.
7. **`model.val()` vs exported predictions differ by 0.03 mAP50** although our metric code
   reproduces Ultralytics' functions exactly (V7) — so the difference is in the inference
   path, not the metric (see VERIFICATION_LOG V11).

## R3 — Run 2: YOLO11n, time-blocked split

Trained 14 Oct on a T4: 43 epochs (early stopping, patience 15; best epoch 28, selected on val mAP50-95), 22.6 min
(fewer training images: 938 scenes vs 1,211). Settings identical to run 1.

| VALID set | Model | mAP50 (exact) | 95% CI, 50-frame-run bootstrap | mAP50-95 | AP50 unsealed | AP50 packaging-error |
|---|---|---|---|---|---|---|
| Roboflow VALID (120 img, 84 runs) | run 1 | 0.629 | 0.41 – 0.83 | 0.360 | 0.93 | 0.35 |
| Blocked VALID (196 img, **18 runs, 3 blocks**) | run 2 | 0.609 | 0.51 – 0.79 | 0.335 | 0.81 | 0.69 |

Ultralytics `val()`: run 2 mAP50 0.718 (vs our 0.609; same multi-label effect as V11, larger here).

**These two rows cannot be compared to measure leakage** — they are different images with a
different class mix (34 vs 117 packaging-error boxes), and blocked VALID comes from only three
300-frame blocks of video (frames 4217–8387), i.e. 18 independent runs of a tray. The fair
comparison scores both models on the *same* images: notebook 03 (cross-evaluation).
Observations so far:
- Validation curve is noisy again (mAP50 0.57–0.72 over the last 15 epochs).
- Validation box loss stays ~1.7 while training box loss falls 1.8 → 1.07: a generalisation gap
  that run 1 hid (its val box loss was ~1.35, closer to train) — consistent with leakage.
- Bootstrap by single images understates uncertainty: on blocked VALID the image bootstrap
  gives 0.54–0.71, the 50-frame-run bootstrap 0.51–0.79 (V12).

## R4 — Run 3 (YOLO26n, blocked split) and cross-evaluation (notebook 03)

Run 3 trained 14 Oct on a T4: 31 epochs (early stop; best epoch 16), 20.8 min.
Cross-evaluation: every model predicted on the same images (`results/xeval/`).
Differences use a **paired** bootstrap over 50-frame runs of blocked VALID
(`porkqc.metrics.evaluate.paired_bootstrap`, 1,000 resamples).

### Scores on the same images — blocked VALID (196 images, 18 runs)

| Model | Trained on | mAP50 | mAP50-95 | AP50 unsealed | AP50 packaging-error |
|---|---|---|---|---|---|
| YOLO11n, run 1 | Roboflow TRAIN (has seen these trays) | 0.878 | 0.609 | 0.906 | 0.835 |
| YOLO11n, run 2 | blocked TRAIN (clean) | 0.609 | 0.335 | 0.807 | 0.687 |
| YOLO26n, run 3 | blocked TRAIN (clean) | 0.735 | 0.434 | 0.829 | 0.607 |
| YOLO26n, run 3, NMS-free head | blocked TRAIN (clean) | 0.703 | 0.451 | — | — |

### Paired differences (A − B), 95% CI

| Question | Metric | Difference | 95% CI | P(diff ≤ 0) |
|---|---|---|---|---|
| **Leakage optimism** (run 1 − run 2) | mAP50 | **+0.269** | +0.062 … +0.351 | < 0.001 |
| | mAP50-95 | **+0.274** | +0.129 … +0.338 | < 0.001 |
| | AP50 unsealed | +0.099 | +0.033 … +0.183 | 0.001 |
| | AP50 packaging-error | +0.149 | +0.079 … +0.205 | < 0.001 |
| **Model** (YOLO26n − YOLO11n, both clean) | mAP50 | +0.126 | −0.077 … +0.268 | 0.32 |
| | mAP50-95 | +0.099 | −0.015 … +0.178 | 0.08 |
| | AP50 unsealed | +0.022 | −0.030 … +0.082 | 0.23 |
| | AP50 packaging-error | **−0.079** | −0.133 … −0.012 | 0.99 |
| **Head** (YOLO26n NMS-free − with NMS) | mAP50 | −0.032 | −0.040 … −0.000 | 0.98 |
| | mAP50-95 | +0.017 | −0.014 … +0.033 | 0.10 |

### Latency, T4, batch 1, FP32 (median per image, blocked VALID)

| Model | Pre | Inference | Post | Total (p95) |
|---|---|---|---|---|
| YOLO11n | 1.41 | 7.57 | 1.11 | 10.1 ms (14.5) |
| YOLO26n + NMS | 1.37 | 8.60 | 1.08 | 11.1 ms (15.5) |
| YOLO26n NMS-free | 1.38 | 8.90 | 0.39 | 10.7 ms (16.0) |

### Findings
1. **Leakage is large and statistically clear.** On identical images, the model that had seen
   neighbouring frames of the same trays scores **+0.27 mAP50 / +0.27 mAP50-95** higher than the
   identical model that had not. Roboflow's split therefore overstates performance by roughly
   that much; all further results use the blocked split.
2. **YOLO26n is not clearly better than YOLO11n.** Its higher overall mAP comes from the rare
   classes (3–4 validation instances each, D-024). On the headline classes it ties on unsealed
   and is significantly *worse* on packaging-error (−0.08).
3. **The NMS-free head saves ~0.7 ms of post-processing** (1.08 → 0.39 ms) but inference is slightly
   slower, so end-to-end latency is unchanged within noise; it costs ~0.03 mAP50.
4. **Every model is ~25× inside the 250 ms budget** (D-007); speed does not discriminate between
   them at this size — accuracy does.
5. Rows for clean models scored on *Roboflow* VALID (`cross_evaluation.csv`) are not interpretable:
   some Roboflow VALID images are inside blocked TRAIN.

## R5 — AP as a numerical integral (notebook 05, WP8)

Integrand: precision envelope of YOLO11n (blocked) on blocked VALID, IoU 0.5 (`05_pr_envelope.png`).

| | unsealed | packaging-error | mAP50 (4 classes with GT) |
|---|---|---|---|
| exact step area | 0.8070 | 0.6866 | 0.6091 |
| COCO 101-point | −0.0016 | −0.0019 | 0.6080 |
| VOC 11-point | **−0.0347** | −0.0232 | 0.5918 |
| Ultralytics (trapezoid, linear envelope) | +0.0014 | −0.0000 | 0.6084 |

**Convergence** (observed order p in |error| ∝ h^p, `05_quadrature_convergence.png`):

| Rule | smooth control eʳ | AP envelope |
|---|---|---|
| midpoint | 2.0 | 0.97 |
| trapezoid | 2.0 | 0.79 |
| Simpson | 4.0 | 0.99 |

- On the smooth control every rule shows its textbook order (verifies the implementations);
  on the step-function envelope **all rules drop to ≈ first order** and the error curve is
  erratic, because the panel containing each jump contributes O(h) regardless of the rule's degree.
- **Romberg / Richardson fails:** successive trapezoid error ratios on the envelope are
  3.2, 10.7, 3.6, 0.7, 4.0, 0.9, 1.5, 6.4, 0.5 — Romberg assumes a constant 4.
- Gauss–Legendre is exact to rounding on eʳ with 8 nodes, but on the envelope it is no better than
  the composite rules (nodes cannot adapt to jump locations).
- **Engineering conclusion:** the spread between integration rules (0.017 mAP50, dominated by VOC
  11-point) is **17× smaller** than the 95% bootstrap interval width from the small validation set
  (0.30). The exact rule is used as primary because it has no discretisation error at all, but the
  quadrature choice is not what limits confidence in the result — the data are.

## R6 — One-shot TEST, decision layer, baseline, performance, robustness (notebooks 04 + 06)

Notebook 04 ran 17 Oct on a Tesla T4 (code commit 3afd7bc). Decision thresholds were committed at
12:28:10 (+11:00, commit 0cbee97); TEST predictions were generated at 12:42:48 — pre-registration
holds. Analysis: `notebooks/06_final_analysis.ipynb`, tables in `results/final/`.

### TEST (180 images, 30 independent tray runs; never used for any choice)

| Metric | VALID (196 img) | **TEST** | TEST 95% CI |
|---|---|---|---|
| headline AP50 (unsealed + packaging-error) | 0.747 | **0.748** | 0.680 – 0.838 |
| AP50 unsealed (47 TEST boxes) | 0.807 | **0.912** | 0.830 – 0.991 |
| AP50 packaging-error (126) | 0.687 | **0.585** | 0.477 – 0.725 |
| mAP50, all classes with GT | 0.609 | 0.753 | 0.514 – 0.894 |
| mAP50-95 | 0.335 | 0.480 | 0.285 – 0.566 |

The headline metric is stable between VALID and TEST (0.747 vs 0.748); the all-class mAP moves
because loose-meat/twisted-meat have 1–2 TEST boxes (loose-meat AP = 1.0 from a single box).

### Decision layer with frozen thresholds (t_review 0.01, t_fail 0.16)

| | misses | false rejects | reviews | cost/image |
|---|---|---|---|---|
| TEST, model | **1 of 156** defective | 9 of 24 clean | 24 | 0.139 |
| TEST, always FAIL | 0 | 24 | 0 | 0.133 |
| TEST, always PASS | 156 | 0 | 0 | 8.667 |

- Miss rate 0.6%, auto-FAIL precision 94%. At TEST's 87% defect prevalence, rejecting everything is
  marginally cheaper — the data are not a realistic line.
- **Prevalence sensitivity** (`06_cost_vs_prevalence.png`): with TEST rates, the model beats
  always-FAIL for any prevalence below 86% (at 5%: 0.48 vs 0.95). At 1–5% prevalence, however,
  the 37.5% false-reject rate dominates and always-PASS is as cheap or cheaper (0.10 at 1%) —
  thresholds tuned on 81%-defective data are wrong for a real line and must be re-optimised for
  the line's prevalence; and the false-reject rate is measured on edge frames, not good trays.

### Alternative: scikit-learn image-level classifiers (TEST image-level AP, exact rule)

| Target | Decision tree | SVM (RBF) | MLP | **YOLO11n** | no-skill |
|---|---|---|---|---|---|
| any defect | 0.903 | 0.941 | 0.959 | **0.980** (0.957–0.994) | 0.867 |
| unsealed | 0.807 | 0.937 | 0.953 | **0.973** (0.930–0.999) | 0.261 |
| packaging-error | 0.841 | 0.924 | 0.914 | **0.971** (0.946–0.991) | 0.611 |

- The detector ranks images best for every target, most clearly for packaging-error (tiny
  defects); the SVM/MLP are respectable at image level but cannot localise.
- Cost: baseline ≈ 25 ms per image on one CPU core (features 23.6 ms + SVM 1.0 ms) and < 2 s to
  train; detector 82 ms on Colab CPU / 7 ms on a T4, 23 min GPU training.

### Speed (T4; forward pass, 100 timed runs after 20 warm-up)

| Setting | median ms | images/s | TFLOP/s achieved | share of peak |
|---|---|---|---|---|
| batch 1, 640, FP32 | 7.06 | 142 | 0.68 | 8.4% |
| batch 1, 640, FP16 | 8.52 | 117 | 0.56 | 0.9% |
| batch 32, 640, FP32 | 93.9 | 341 | 1.63 | 20% |
| batch 32, 640, FP16 | 48.8 | 655 | 3.14 | 4.8% |
| batch 1, CPU (Colab, 1 thread), 640 | 81.9 | 12 | — | — |

- Batch-1 latency is **flat (7.0–7.5 ms) from 320 to 800 px** although FLOPs grow 6× (1.3 → 7.6
  GFLOP): at batch 1 the GPU is limited by per-layer overhead (≈ 88 small convolutions → kernel
  launches), not by arithmetic or memory bandwidth. Achieved 0.17–1.1 TFLOP/s, far below both
  roofs; profiler: convolution kernels take 75% of GPU time, element-wise add/SiLU the rest.
- FP16 helps only once batches are large enough to fill the GPU (2× throughput at batch ≥ 8);
  at batch 1 it is slower (extra casts, no saturation). Memory 63 MB at batch 1.
- Intensity ≈ 37 FLOP/byte (FP32, upper bound) > T4 machine balance 25 → compute-bound in
  principle; with FP16 tensor cores the balance rises to 203 → memory-bound in principle. Neither
  roof is approached at batch 1, which is the line's operating point.
- **All settings, including CPU-only, meet the 250 ms budget** (end-to-end TEST p95 16.6 ms on T4).

### Resolution and precision (VALID, paired Δ headline AP50 vs 640 FP32)

| Input | Δ | 95% CI | packaging-error AP50 |
|---|---|---|---|
| 320 | −0.135 | −0.176 … −0.094 | 0.41 |
| 480 | −0.030 | −0.079 … +0.023 | 0.60 |
| **640 (trained)** | 0 | — | 0.69 |
| 800 | −0.035 | −0.071 … −0.002 | 0.62 |
| 960 | −0.074 | −0.118 … −0.025 | 0.58 |
| 640 FP16 | −0.002 | −0.006 … 0.000 | 0.68 |

Accuracy peaks at the training resolution; going smaller loses the tiny packaging-error boxes,
going larger mismatches the scale learned in training. Since latency is flat, there is no speed
reason to shrink the input. **Final design: 640 px, FP32, batch 1.**

### Robustness (VALID, paired Δ headline AP50 vs clean)

| Perturbation | mild | strong |
|---|---|---|
| exposure ×0.5 … ×1.5 | −0.002 … −0.013 (n.s.) | — |
| blur σ 1 / 2 / 4 px | −0.006 / −0.021 (n.s.) | **−0.271** |
| noise σ 10 / 25 | −0.037 | **−0.232** |
| JPEG q 30 / 10 | −0.033 | **−0.206** |

Robust to lighting changes of ±50%; sensitive to strong blur, noise and compression, almost
entirely through packaging-error (tiny, low-contrast defects: 0.69 → 0.28–0.31). Engineering
implication: control exposure time (motion blur) and avoid lossy compression on the camera link.

### Failure modes on TEST (notebook 06 §7, operating point confidence ≥ 0.16)

- **Small packaging-error boxes are the main failure:** recall 0.53 for boxes < 0.26% of the image
  (47 boxes) vs 0.74 for 0.26–1%; unsealed boxes > 5% of the image: recall 0.95.
- **Most false positives are localisation, not confusion:** of 94 false positives, 57 are the right
  class with IoU 0.1–0.5 ("mislocated"), 23 background, 13 on unlabelled (edge) frames, 1 wrong
  class. For a 20 × 20 px box, a 4-px offset already drops IoU below 0.6 — IoU is numerically very
  sensitive for tiny objects, so the 0.5 threshold penalises near-misses on packaging-error.
- Boxes touching the image edge (trays entering/leaving view) are found *more* often (0.79 vs 0.66),
  because they are mostly large unsealed boxes; the difficulty is object size, not truncation.
