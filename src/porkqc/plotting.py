"""
Consistent figure style for report figures.

Categorical colours are taken in a fixed order from a palette validated for
colour-vision deficiency (adjacent pairs separable under protan/deutan/tritan
simulation); series keep their colour across figures, and every figure with
two or more series has a legend, so identity never depends on colour alone.
"""

from __future__ import annotations

__all__ = ["SERIES", "INK", "MUTED", "GRID", "apply_style"]

#: Fixed categorical order (slots 1–8).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply_style() -> None:
    """Set matplotlib defaults: thin lines, recessive grid, fixed colour cycle, 2 px lines."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=SERIES),
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9,
    })
