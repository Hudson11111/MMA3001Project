# Trained model

| | |
|---|---|
| File | `yolo11n_640_blocked.pt` (5.2 MB, Ultralytics checkpoint, FP16 weights) |
| SHA-256 | `02e6a2275fa1292b5d07c7100ab2fe0c1e0dbae54db7156cd347c8bf1406a7ee` (checked by `tests/test_cli_and_model.py`) |
| Architecture | YOLO11n (Ultralytics 8.4.175), 2.6 M parameters, 6.4 GFLOPs at 640 × 640 |
| Trained | 14 Oct 2026, Colab Tesla T4, 43 epochs (early stop; best epoch 28), seed 0, `configs/experiments/yolo11n_640.yaml` |
| Training data | time-blocked TRAIN split of Roboflow *Pork Rasher Error (Packaging)* v4 (938 scenes, each with its Roboflow augmented copies; `results/audit/blocked_split_assignment.csv`) |
| Selected for TEST | D-025; one-shot TEST results in `docs/RESULTS_LOG.md` R6 |
| Classes | 0 loose-meat · 1 packaging-error · 2 twisted-meat · 3 unsealed · 4 wrinkle |
| Decision thresholds | `configs/decision.yaml` (t_review 0.01, t_fail 0.16) |

**Provenance check (VERIFICATION_LOG V18):** this file was downloaded from the training run on
Google Drive; re-running it on CPU reproduces the committed GPU predictions for the
development-sample images (identical classes and confidences, boxes within 0.01 px).

Run it:

```bash
pip install -e ".[detector]"
python -m porkqc.cli path/to/frame.jpg            # prints FAIL / REVIEW / PASS + top detections
```

Licence: weights derived from Ultralytics COCO-pretrained YOLO11n (AGPL-3.0) fine-tuned on
CC BY 4.0 data; distributed under this repository's AGPL-3.0 licence.
