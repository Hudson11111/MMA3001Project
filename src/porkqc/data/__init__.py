"""
Dataset access, label handling and audit.

Modules
-------
porkqc.data.download
    Roboflow download and ``data.yaml`` path repair.
porkqc.data.labels
    Strict YOLO label parsing/validation and coordinate conversion.
porkqc.data.audit
    Per-split counts (classes, null images, missing/invalid labels, originals)
    and a reproducible development sample.
porkqc.data.duplicates
    Image hashing and thumbnail correlation for copy and scene detection.
porkqc.data.resplit
    Time-blocked re-split with a purge gap (removes temporal leakage).
"""
