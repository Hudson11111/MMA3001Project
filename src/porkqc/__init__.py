"""
porkqc: automated inspection of pork-rasher packaging images.

MMA3001 Numerical Methods and Machine Learning, individual project
(Monash University, 2026). Repository:
https://github.com/Hudson11111/MMA3001Project

The package holds all reusable, tested code. The notebooks in ``notebooks/``
import it, so every function behind a reported result has a docstring and,
where practical, a unit test.

Pipeline
--------
image -> :mod:`porkqc.models.detector` (YOLO11n boxes)
-> :mod:`porkqc.decision` (FAIL / REVIEW / PASS)
-> :mod:`porkqc.metrics` (evaluation against labels, with bootstrap intervals).
:mod:`porkqc.cli` runs the first two steps on image files.

Modules
-------
porkqc.data
    Dataset download, strict label parsing, audit, copy detection and the
    time-blocked re-split that removes temporal leakage.
porkqc.models
    YOLO training/prediction wrapper and the scikit-learn baseline.
porkqc.metrics
    IoU, matching, average precision under several integration rules,
    quadrature, dataset-level evaluation and bootstrap.
porkqc.decision
    Cost-based FAIL / REVIEW / PASS rule and threshold search.
porkqc.perf
    FLOP counting, arithmetic intensity, machine balance and timing.
porkqc.robustness
    Controlled image perturbations.
porkqc.predictions
    Ground-truth and prediction JSON files used for CPU-side evaluation.
porkqc.cli
    Command line: ``python -m porkqc.cli frame.jpg``.
porkqc.config, porkqc.env, porkqc.colab_utils, porkqc.plotting
    Configuration, secrets, Colab helpers and figure style.
"""

__version__ = "1.0.0"

#: Defect classes in the Roboflow v4 export, in class-id order.
#: Verified against the downloaded ``data.yaml`` by
#: :func:`porkqc.config.check_class_names`.
CLASS_NAMES: tuple[str, ...] = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
)
