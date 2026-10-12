# Input / Output Specification (DRAFT)

Top-level contract of the inspection system. Function-level inputs and outputs are
documented in docstrings and rendered in the HTML API docs (`docs/api/`).
Items marked ⏳ depend on open decisions (see `DECISION_LOG.md`).

## 1. Inputs

### 1.1 Inspection image (primary input)
| Property | Specification |
|---|---|
| Content | One RGB photograph of a pork-rasher tray on the packaging conveyor, top-down view, as in the Roboflow dataset |
| Format | JPEG or PNG file, or an in-memory `numpy.ndarray` |
| Array shape / dtype | `(H, W, 3)`, `uint8`, values 0–255, channel order documented per function (OpenCV loads BGR; converted at the boundary) |
| Nominal size | 720 × 540 px (W × H), the dataset's preprocessing size. Other sizes are accepted and resized to the model input size (multiple of 32), but accuracy is only validated at the nominal size |
| Units | Pixel coordinates; origin top-left, x right, y down |
| Domain (validated) | Lighting, camera pose, tray type and product as represented in the dataset. Anything else is **extrapolation** and is flagged as unsupported |

### 1.2 Training/evaluation data
| Property | Specification |
|---|---|
| Source | Roboflow Universe `hello-2aqe0/pork-rasher-error-packaging`, version 4, exported in YOLO (`yolov8`) format |
| Layout | `train/`, `valid/`, `test/`, each with `images/` and `labels/`; `data.yaml` with class names |
| Label format | One `.txt` per image; one line per object: `class_id cx cy w h`; `class_id` ∈ {0..4}; coordinates normalised to [0, 1] by image width/height; empty file = no defects ⏳ (to be confirmed by audit) |
| Classes | 0 loose-meat · 1 packaging-error · 2 twisted-meat · 3 unsealed · 4 wrinkle (order to be confirmed from `data.yaml`) |

### 1.3 Configuration
YAML files in `configs/` (dataset location, model, input size, thresholds, seeds).
Secrets (Roboflow API key) are **never** in config files — read from the
environment or Colab Secrets.

## 2. Outputs

| Output | Form | Meaning / intended use |
|---|---|---|
| Detections | list of `(class_name, confidence, x1, y1, x2, y2)` per image; pixel coordinates in the *original* image | Where each defect is, for an operator display and for metric computation |
| Tray decision | `PASS` / `FAIL` / `REVIEW` + list of reasons | PASS → continue; FAIL → reject; REVIEW → route to manual inspection. Thresholds chosen on validation data ⏳ (D-007) |
| Severity proxy ⏳ | defect box area as a fraction of tray area | Rough "extent" indicator — not validated against a ground-truth severity |
| Run report | CSV/JSON per image + annotated image | Audit trail for quality records |
| Timing | milliseconds per stage (pre-process / inference / post-process) | Checks against the line-speed budget ⏳ (D-007) |

## 3. Invalid, missing or unsupported input handling (planned behaviour)

| Situation | Behaviour |
|---|---|
| File missing / unreadable / not an image | Raise `FileNotFoundError` / `ValueError` with the path; batch runs log and skip |
| Wrong shape (e.g. grayscale `(H, W)`, RGBA `(H, W, 4)`) | Raise `ValueError` describing expected `(H, W, 3)` (no silent conversion) |
| Wrong dtype (float image) | Raise `TypeError` |
| Label line with wrong number of fields | Raise `LabelFormatError` naming file and line |
| Class id outside 0..4 | Raise `LabelFormatError` |
| Coordinates outside [0, 1] or non-positive width/height | Raise `LabelFormatError` (small tolerance for rounding) |
| Image size very different from 720 × 540 | Process, but flag result as outside validated domain |
| Missing API key | Raise `RuntimeError` explaining how to set `ROBOFLOW_API_KEY` |
