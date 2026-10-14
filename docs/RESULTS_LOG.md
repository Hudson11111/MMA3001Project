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

Trained 14 Oct on a T4: 43 epochs (50 max; best by val fitness near the end), 22.6 min
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
