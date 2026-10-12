"""
Detection metrics, implemented explicitly and unit-tested (WP2).

porkqc.metrics.iou
    Box area and pairwise IoU.
porkqc.metrics.matching
    True/false positives per image (confidence-ordered or Ultralytics rule).
porkqc.metrics.ap
    Precision–recall curves and average precision under several integration
    rules (exact step area, COCO 101-point, VOC 11-point, Ultralytics).
porkqc.metrics.quadrature
    Generic quadrature rules (midpoint, trapezoid, Simpson, Gauss–Legendre,
    Romberg) and observed order of convergence, for the AP-as-integral study.
porkqc.metrics.evaluate
    Dataset-level matching, mAP summaries, operating-point counts and
    bootstrap confidence intervals.
"""
