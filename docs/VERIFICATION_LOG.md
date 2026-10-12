# Verification and Debugging Log

Evidence for "how did you verify that your method was implemented correctly?".
The unit distinguishes three activities, and this log keeps them apart:

- **Testing** — automated checks that code does what its specification says
  (pytest, positive and negative cases, CI on every push).
- **Debugging** — finding the cause of a failure once a test or run exposes it.
- **Validation** — checking the *method* against reality (held-out data, ground
  truth, benchmarks). Recorded in the results notebooks, not here.

| # | Date | How it was found | Problem | Root cause | Fix | Lesson |
|---|---|---|---|---|---|---|
| V1 | 12 Oct | **CI** on first push (fresh clone) | `ModuleNotFoundError: porkqc.data` on GitHub, although all 61 tests passed locally | `.gitignore` pattern `data/` (meant for the dataset folder) also matched `src/porkqc/data/`, so that code was never committed | Anchor patterns to the repo root (`/data/`) | Local tests only check the files on *my* disk; CI on a fresh clone checks what was actually committed |
| V2 | 12 Oct | Reading the **pinned** Ultralytics 8.4.175 source | Common tutorials use `half=True` for FP16 | In this version `half` is deprecated in favour of `quantize=16`; `nms=False` selects YOLO26's NMS-free head | Use the pinned version's argument names | Pin versions and read *that* version's code/docs (D-010) |
| V3 | 12 Oct | **CPU smoke test** of the training wrapper (synthetic images, 1 epoch) | Deleting `run_meta.json` and calling `train_detector` again started a *new* training run instead of skipping | Ultralytics strips optimiser state from `last.pt` when training finishes; `resume=True` on such a file logs a warning and silently starts fresh | `_is_resumable()` applies the same test Ultralytics uses; finished runs are detected and skipped | On free Colab a disconnect at the wrong moment would have cost an hour of GPU time |
| V4 | 12 Oct | **Simulated disconnect** (process killed during epoch 2) | — (check that resume actually works) | — | Resumed "from epoch 2 to 30" and completed | Resume path verified before relying on it in Colab |
| V5 | 12 Oct | Reading Ultralytics `compute_ap` / `match_predictions` | Our metrics must be comparable to the library's | AP: 101-point **trapezoid** integration of a linearly interpolated precision envelope; matching sorts candidate pairs by **IoU**, not confidence | Implement both conventions; verify equality on identical inputs (WP2) | "mAP" is not one number — it depends on the quadrature rule and matching rule |
| V6 | 12 Oct | **Unit test** of our transcription of Ultralytics' AP rule (a perfect detector should score 1) | Test failed: Ultralytics' rule gives **0.995** | Not a bug in our code — a property of Ultralytics' quadrature: the sentinel makes the interpolated envelope 0 at recall 1, so the last trapezoid panel averages 1 and 0. Explains the familiar "mAP50 0.995" ceiling in YOLO logs | Test corrected to document the behaviour; confirmed against Ultralytics' own `compute_ap` | A wrong *expectation* in a test is also a finding — it showed the metric itself carries a quadrature artefact |
| V7 | 12 Oct | **Verification tests** against Ultralytics' functions on random scenes (20 seeds each) | — | — | Our `"ultralytics"` matching rule and AP rule reproduce `match_predictions`, `compute_ap` and `ap_per_class` exactly (to 1e-9) | Our evaluation pipeline can reproduce the library's numbers, so differences from the default `"confidence"`/`"exact"` choices are attributable to those choices |

