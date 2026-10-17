# Automated Inspection of Pork-Rasher Packaging — Project Report (DRAFT)

MMA3001 Numerical Methods and Machine Learning · Hudson · October 2026
Repository: https://github.com/Hudson11111/MMA3001Project

> Draft status: sections 1–3 and 7–8 drafted; 4–6 completed after notebook 04 (TEST,
> performance, robustness, baseline). ⟦…⟧ marks numbers still to be filled. Every number is
> traceable to `docs/RESULTS_LOG.md` (R-numbers) and decisions to `docs/DECISION_LOG.md` (D-numbers).

## 1. Engineering problem

**Context.** Bacon rashers are placed into clear plastic trays, film-sealed and conveyed past an
overhead camera. Defects — an unsealed or lifted film, loose or twisted rashers, wrinkled film and
other packaging errors — must be removed before distribution: an unsealed pack spoils and can
trigger a complaint or recall, while rejecting good packs wastes product and labour. Inspection
today relies on people watching a fast line, which is tiring and inconsistent.

**Problem addressed.** Given one 720 × 540 RGB frame from that camera, (i) detect and localise five
defect types, (ii) turn the detections into a tray-level action — FAIL (reject), REVIEW (send to an
inspector) or PASS — and (iii) determine whether the method is accurate and fast enough for in-line
use.

**Intended use.** Decision support that removes clearly defective packs automatically and routes
uncertain ones to a person, reducing — not eliminating — manual inspection.

**Requirements** (no line speed was specified; set from published equipment speeds, D-007):
throughput 120 trays/min (above a 90 packs/min high-speed tray sealer), hence a vision budget of
**≤ 250 ms per image**; a missed defect is taken to cost **10×** a false reject, with sensitivity
analysis over 1–50×.

**Limitations of scope.** Meat quality (e.g. fat ratio) is excluded because the dataset has no
ground truth for it. Seal integrity is judged visually only. One camera, one product line; no
deployment on line hardware.

## 2. Inputs, outputs and data

Full contract: `docs/IO_SPECIFICATION.md`; function-level I/O in the HTML API documentation.

- **Input:** RGB image, `(H, W, 3)` uint8, nominally 720 × 540 px; pixel coordinates.
- **Outputs:** boxes `[class, confidence, x1, y1, x2, y2]` in original-image pixels; a
  FAIL/REVIEW/PASS decision per image; per-stage timing in ms.
- **Invalid input** is rejected with specific exceptions (malformed labels with file and line,
  wrong array shapes/dtypes, mismatched class lists or image sizes) rather than silently accepted.

**Data.** Roboflow Universe "Pork Rasher Error (Packaging)", version 4 (CC BY 4.0): 3,349 / 120 / 80
train/valid/test images at 720 × 540 with box labels for loose-meat, packaging-error,
twisted-meat, unsealed and wrinkle. An audit (notebook 01, R1) found:

1. **Every image is a numbered frame of conveyor video**, and Roboflow assigned frames to splits at
   random. The median gap from a VALID/TEST frame to the nearest TRAIN frame is **3 frames**; one
   tray stays in view for roughly 5–15 frames. VALID and TEST are therefore full of near-twins of
   training images (temporal leakage).
2. **File names are not unique**: 128 names are shared by different images (several videos numbered
   alike), so augmented copies were identified by image content (correlation of standardised
   thumbnails, calibrated with a manufactured positive control: copies ≥ 0.9995, different images
   ≤ 0.910; D-021). Behind the 3,349 training files are 1,211 distinct scenes.
3. **Class imbalance:** packaging-error 1,769 boxes, unsealed 1,185, twisted-meat 79, loose-meat 72,
   wrinkle 18 (none in VALID or TEST). Packaging-error boxes are tiny (median 0.37% of the image).
4. **"No label" does not mean "good tray".** An independent manual review of all 26 unlabelled
   VALID/TEST frames with defined codes (E edge/empty, G good, M missed defect, U unsure) gave
   25 E and 1 M (visible seal lifting, frame 6496); my own and Claude's independent reviews agreed
   that most show a tray only partly in view (R1).

**Split used for all conclusions (D-008):** a time-blocked re-split — contiguous blocks of 300
frames assigned whole to train/valid/test (70/15/15% of scenes) with a 50-frame purge gap, giving
938 / 196 / 180 scenes and no VALID/TEST frame within 50 frames of TRAIN.

## 3. Computational approach

**Detector.** YOLO11n (Ultralytics 8.4.175, 2.6 M parameters, 6.4 GFLOPs at 640 px), fine-tuned
from COCO-pretrained weights — transfer learning, because ~1,000 training scenes are far too few to
learn visual features from scratch. A one-stage detector was chosen over a two-stage one because
the latency budget and free GPU time favour a small, fast model, and over image classification
because operators need to see *where* the defect is.

**Settings (D-017):** input 640 (letterboxed 640 × 480), up to 50 epochs with early stopping
(patience 15) on validation mAP50-95, batch 16, seed 0, Ultralytics defaults otherwise (mixed
precision, online augmentation). Identical across all runs so that only the model or split differs.
Training curves were inspected for convergence and overfitting; on the blocked split training box
loss fell 1.80 → 1.07 while validation box loss stayed ≈ 1.7, a generalisation gap that the leaky
Roboflow split had hidden (R3).

**Decision layer (D-023, D-026).** Image score = highest box confidence; FAIL if ≥ t_fail, REVIEW
if ≥ t_review, else PASS. Thresholds minimise expected cost (miss 10, false reject 1, review 0.25)
by exhaustive grid search on VALID — the cost is a step function of the thresholds, so there is no
gradient to follow. Result: t_review = 0.01, t_fail = 0.16, **frozen and committed before any TEST
prediction existed** (the commit is stamped into the TEST file).

**Evaluation pipeline (own, tested).** Predictions are exported once and scored on CPU by
`porkqc.metrics`: IoU, confidence-ordered matching, precision–recall curves, average precision as
the exact area under the precision envelope, and bootstrap confidence intervals that resample
**50-frame runs** (neighbouring frames are not independent; V12). It reproduces Ultralytics'
`match_predictions`, `compute_ap` and `ap_per_class` exactly on random inputs (V7).

**Engineering workflow.** Python package `porkqc` (about 180 pytest cases, CI on every push), thin Colab
notebooks for GPU work, YAML configs, resumable training on Google Drive, results committed as
JSON/CSV, decisions and bugs logged as they happened.

## 4. Alternative solutions ⟦after notebook 04⟧

- YOLO26n (Jan 2026, optional NMS-free head) vs YOLO11n on identical images (paired bootstrap, R4):
  mAP50 +0.13 (95% CI −0.08…+0.27, not significant), driven by classes with 3–4 validation boxes;
  packaging-error −0.08 (significant); unsealed +0.02 (n.s.). NMS-free head: −0.7 ms post-processing,
  −0.03 mAP50. **Selected: YOLO11n** (my decision, D-025).
- Trivial baselines (always FAIL / always PASS) for the decision layer.
- scikit-learn image-level baseline (decision tree, RBF-SVM, MLP on 324 hand-crafted features) ⟦…⟧.

## 5. Validation ⟦TEST numbers after notebook 04⟧

**What was validated and how.**
1. *Code correctness (verification):* unit tests with hand-computed values (IoU, AP rules, FLOP
   formula, quadrature orders), regression tests for every bug found, agreement with trusted
   implementations (Ultralytics metrics exactly; FLOP counter within 1%, V13). `docs/VERIFICATION_LOG.md` lists 14 findings.
2. *Data validity:* frame-number leakage analysis; content-based copy detection with controls;
   manual review of unlabelled frames with defined codes.
3. *Model performance:* held-out TEST evaluated once with pre-registered thresholds; VALID used for
   all choices (epochs, model, thresholds), which is why the VALID numbers are optimistic.
4. *Leakage quantified:* the same architecture trained with and without access to neighbouring
   frames, scored on identical images: **+0.27 mAP50 (95% CI +0.06…+0.35)** optimism from the random
   split (R4).
5. *Robustness / sensitivity:* exposure, blur, noise and compression ⟦…⟧; cost-ratio sensitivity of
   the thresholds (R-decision).

**What validation cannot establish:** performance on other lines, cameras or lighting; seal
strength; rare classes (wrinkle cannot be evaluated at all; loose-/twisted-meat CIs span ≈ [0, 1]);
true PASS accuracy, because the data contain almost no fully visible good trays.

## 6. Optimisation and performance ⟦after notebook 04⟧

⟦latency vs batch/resolution/FP16/CPU; profiler; FLOPs 4.8 GFLOP at 640×480; arithmetic intensity
≈ 37 FLOP/byte (FP32) vs T4 machine balance 25 FLOP/byte (FP32) / 203 (FP16) → mostly compute-bound
in FP32 but memory-bound in FP16; accuracy vs input size; final choice⟧

**AP as a numerical integral (R5).** The precision envelope is a step function, so AP can be computed
exactly. On our curves VOC 11-point under-estimates by up to 0.035 AP; COCO and Ultralytics are within
0.002. Composite rules that are O(h²)/O(h⁴) on a smooth control (verified) fall to ≈ O(h) on the
envelope, and Romberg's error ratios are erratic (0.5–10.7 instead of 4). The spread between rules
(0.017 mAP50) is 17× smaller than the bootstrap CI width (0.30): the data, not the quadrature, limit
the result.

## 7. Limitations and lessons learned ⟦finalise⟧

- The provided split leaks; without the audit, the headline score would have been ~0.27 mAP50 too high.
- Labels are incomplete for partly visible trays, which penalises correct detections as false alarms.
- A real line needs a "tray fully in view" trigger or per-tray aggregation over consecutive frames.
- A visually convincing check can fail: perceptual hashing looked perfect on synthetic data and was
  useless on the real conveyor background (V10). Controls on real data are essential.

## 8. AI use and reflection ⟦Hudson to rewrite in own words⟧

*Draft scaffold from `docs/AI_USAGE_LOG.md`; must be rewritten by Hudson.*
- Tool: Claude (Anthropic), with GitHub access; used for planning, code, tests, notebooks, analysis
  and drafting documentation. Approximate contribution: most code and first drafts; decisions,
  data review and execution were mine.
- Checking: every change arrived as a pull request that I reviewed and merged; CI tests; I ran every
  notebook on the real data, which exposed bugs the AI's synthetic tests missed (V9).
- Decisions that remained mine: problem framing, licence, the manual review of unlabelled frames
  (25 E / 1 M), the model sent to the one-shot test.
- AI errors: ⟦e.g. `.gitignore` excluded a code folder (V1); a local dry run pushed fake results to my
  repository (V8); the planned hash-based leakage check did not work on real images (V10); a download
  function that silently did nothing (V9).⟧
- Effect on learning: ⟦…⟧

## References
- Roboflow Universe, *Pork Rasher Error (Packaging)*, v4, CC BY 4.0.
- Ultralytics YOLO11 / YOLO26 documentation, https://docs.ultralytics.com.
- Everingham et al., *The PASCAL VOC Challenge*, IJCV 2010 (11-point AP); Lin et al., *Microsoft COCO*, ECCV 2014 (101-point AP).
- Pearson Packaging Systems, bacon tray and bag case-packing line (24 trays/min); Proseal GT2 tray sealer (> 90 packs/min), Food Processing.
- NVIDIA Tesla T4 datasheet (8.1 TFLOPS FP32, 65 TFLOPS FP16, 320 GB/s).
