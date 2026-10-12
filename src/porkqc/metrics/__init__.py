"""
Detection metrics.

Implemented
-----------
porkqc.metrics.iou
    Box area and pairwise IoU.

Planned (WP2)
-------------
Prediction-to-ground-truth matching, precision-recall curves, average precision
computed with several numerical-integration rules (and verified against
Ultralytics), and bootstrap confidence intervals.
"""

from .iou import box_area, box_iou

__all__ = ["box_area", "box_iou"]
