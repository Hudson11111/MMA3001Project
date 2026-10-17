# Input / Output Specification

Top-level contract of the inspection system. Function-level inputs and outputs are
documented in NumPy-style docstrings and rendered in the HTML API docs (`docs/api/index.html`).

## 1. Inputs

### 1.1 Inspection image (primary input)
| Property | Specification |
|---|---|
| Content | One RGB frame from the overhead camera of the pork-rasher packaging conveyor, as in the Roboflow dataset |
| Format | JPEG/PNG file (`str`/`Path`) or `numpy.ndarray` |
| Array shape / dtype | `(H, W, 3)`, `uint8`, 0–255, RGB |
| Nominal size | 720 × 540 px (W × H). The detector letterboxes to 640 × 480 (multiple of 32). Other sizes run, but accuracy was validated only at 720 × 540 |
| Units | Pixels; origin top-left, x right, y down |
| Validated domain | Lighting, camera pose, tray type and products of the dataset videos (frames 3–10,590). Measured robustness (R6): unaffected by exposure ×0.5–×1.5; small losses (≤ 0.04 AP50) for blur σ ≤ 2 px, noise σ ≤ 10, JPEG quality ≥ 30; large losses (0.21–0.27) at blur σ 4, noise σ 25, JPEG q 10. Inputs outside this envelope, or from other cameras/lines, are unsupported |
| Important property | A frame may show a tray only partly (entering/leaving the view). 25 of 26 unlabelled VALID/TEST frames were of this kind (RESULTS_LOG R1) |

### 1.2 Training / evaluation data
| Property | Specification |
|---|---|
| Source | Roboflow Universe `hello-2aqe0/pork-rasher-error-packaging`, version 4, `yolov8` export (CC BY 4.0) |
| Layout | `train/`, `valid/`, `test/`, each `images/` + `labels/`; `data.yaml` |
| Label format | One `.txt` per image, one line per box: `class_id cx cy w h`, normalised to [0, 1]; empty file = no labelled box |
| Classes | 0 loose-meat · 1 packaging-error · 2 twisted-meat · 3 unsealed · 4 wrinkle (checked against `data.yaml` at every run) |
| Split used for conclusions | Time-blocked re-split of all 1,411 scenes (blocks of 300 frames, 50-frame purge gap): 938 / 196 / 180 scenes (D-008) |

### 1.3 Configuration
YAML in `configs/`: dataset and split (`dataset.yaml`), training/prediction settings
(`experiments/*.yaml`), decision thresholds and costs (`decision.yaml`). Secrets (Roboflow key,
optional GitHub token) are read from the environment or Colab Secrets, never from files.

## 2. Outputs

| Output | Form | Meaning / intended use |
|---|---|---|
| Detections | per image: list of `[class_id, confidence, x1, y1, x2, y2]`, pixels in the original image, sorted by confidence (`porkqc.models.detector.predict_images`) | Where each suspected defect is and how confident the model is — for the operator display and for metrics |
| Tray decision | `FAIL` / `REVIEW` / `PASS` per image (`porkqc.decision.decide`) with thresholds `t_review = 0.01`, `t_fail = 0.16` | FAIL → reject automatically; REVIEW → human inspector; PASS → "no defect found". Thresholds minimise expected cost with miss : false reject : review = 10 : 1 : 0.25 on VALID |
| Timing | ms per image for pre-processing / inference / post-processing | Check against the 250 ms per-tray budget (120 trays/min, D-007) |
| Command-line report | one line per image (`decision  score  file  [top detections]`), optional full JSON (`--out`); `python -m porkqc.cli` with the committed model `models/yolo11n_640_blocked.pt` | The deployed form: image in, decision out |
| Evaluation files | ground-truth and prediction JSON (`porkqc.predictions`) | Reproducible CPU-side evaluation with `porkqc.metrics` |

## 3. Invalid, missing or unsupported input

| Situation | Behaviour (function) |
|---|---|
| Label/config/data file missing | `FileNotFoundError` naming the path (`read_label_file`, `load_yaml`, `fix_data_yaml`) |
| Label line with wrong field count, non-numeric/non-finite value, class id out of range, non-positive size, box outside the image | `LabelFormatError` with file and line number (`parse_label_line`); the audit counts and reports such files instead of stopping |
| Class names in `data.yaml` differ from the expected list or order | `ValueError` before any training/evaluation (`check_class_names`) |
| Image array not `(H, W, 3)` | `ValueError`; non-`uint8` → `TypeError` (`perturb`, `image_features`) |
| Box arrays with wrong shape, NaN/inf, or x2 < x1 | `ValueError` (`box_iou`, `box_area`) |
| Predictions for images not in the ground truth, or image size mismatch between files | `ValueError` (`match_dataset`) — prevents silently comparing different coordinate frames |
| Thresholds out of order / outside [0, 1] | `ValueError` (`decide`) |
| Missing API key | `RuntimeError` explaining how to add the Colab secret (`get_secret`) |
| Roboflow download produced no `data.yaml` / export not ready | retry, then `RuntimeError` (`download_dataset`) |
| Command line (`python -m porkqc.cli`): image or weights file missing, or a file that is not a readable image | Error message naming the file, exit code 2, model not loaded (`porkqc.cli`) |
| Image outside the validated domain (different camera, lighting, product) | Processed, but results are unvalidated — documented limitation |
