"""
porkqc — automated inspection of pork-rasher packaging images.

MMA3001 Numerical Methods and Machine Learning, individual project.

The package holds all reusable, tested code. Colab notebooks in ``notebooks/``
import it to run experiments, so that every function used to produce a reported
result has a docstring and (where practical) a unit test.

Subpackages
-----------
porkqc.env
    Environment helpers (reading secrets locally or in Google Colab).
porkqc.config
    Loading and checking YAML configuration files.
porkqc.data
    Dataset download, YOLO label parsing/validation and dataset audit.
porkqc.metrics
    Detection metrics (IoU now; matching, precision-recall and average
    precision to follow in work package WP2).

Planned (see ``docs/PROJECT_PLAN.md``): detector wrappers, scikit-learn
baselines, the tray-level PASS/FAIL/REVIEW decision, performance benchmarking
and robustness tests.
"""

__version__ = "0.1.0"

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
