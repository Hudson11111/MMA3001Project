r"""
Numerical integration rules on an interval, for the AP-as-integral study (report Section 5.4).

All rules approximate :math:`I = \int_a^b f(x)\,dx` for a vectorised callable
``f``. They are implemented here (rather than imported) so that each one can be
tested for its textbook properties — polynomial exactness and order of
convergence — before being applied to the precision–recall step function, where
those properties are expected to break down.

=================  =======================  ====================  ==================
Rule               Nodes (n panels)         Exact for degree ≤    Error, smooth f
=================  =======================  ====================  ==================
midpoint           n                        1                     O(h²)
trapezoid          n + 1                    1                     O(h²)
Simpson 1/3        n + 1 (n even)           3                     O(h⁴)
Gauss–Legendre     n                        2n − 1                very fast
Romberg            2^L + 1                  —                     O(h^{2L+2})
=================  =======================  ====================  ==================

For a function with a jump discontinuity, every composite rule's error is
dominated by the panel containing the jump, giving only O(h) convergence — the
higher-order rules lose their advantage, and Richardson extrapolation (Romberg)
rests on an error expansion in even powers of h that no longer exists.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "midpoint_rule",
    "trapezoid_rule",
    "simpson_rule",
    "gauss_legendre_rule",
    "romberg_table",
    "observed_order",
]

Integrand = Callable[[np.ndarray], np.ndarray]


def _check(a: float, b: float, n: int) -> None:
    """Validate an interval ``[a, b]`` and a panel/node count; raise ValueError if invalid."""
    if not b > a:
        raise ValueError(f"Need b > a, got a={a}, b={b}.")
    if n < 1:
        raise ValueError(f"Need at least one panel/node, got n={n}.")


def midpoint_rule(f: Integrand, a: float, b: float, n: int) -> float:
    """
    Composite midpoint rule with ``n`` equal panels.

    Parameters
    ----------
    f : callable
        Vectorised integrand ``f(x) -> array``.
    a, b : float
        Interval, ``b > a``.
    n : int
        Number of panels (≥ 1).

    Returns
    -------
    float
        Approximation of the integral.
    """
    _check(a, b, n)
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return float(h * np.sum(f(x)))


def trapezoid_rule(f: Integrand, a: float, b: float, n: int) -> float:
    """
    Composite trapezoid rule with ``n`` equal panels (``n + 1`` nodes).

    Parameters
    ----------
    f : callable
        Vectorised integrand.
    a, b : float
        Interval, ``b > a``.
    n : int
        Number of panels (≥ 1).

    Returns
    -------
    float
        Approximation of the integral.
    """
    _check(a, b, n)
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def simpson_rule(f: Integrand, a: float, b: float, n: int) -> float:
    """
    Composite Simpson 1/3 rule with ``n`` equal panels (``n`` even).

    .. math:: S = \\frac{h}{3}\\left[f_0 + 4\\sum_{odd} f_i + 2\\sum_{even} f_i + f_n\\right]

    Parameters
    ----------
    f : callable
        Vectorised integrand.
    a, b : float
        Interval, ``b > a``.
    n : int
        Number of panels; must be even and ≥ 2.

    Returns
    -------
    float
        Approximation of the integral.

    Raises
    ------
    ValueError
        If ``n`` is odd.
    """
    _check(a, b, n)
    if n % 2:
        raise ValueError(f"Simpson's rule needs an even number of panels, got n={n}.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1]))


def gauss_legendre_rule(f: Integrand, a: float, b: float, n: int) -> float:
    """
    ``n``-point Gauss–Legendre quadrature mapped from [-1, 1] to [a, b].

    Parameters
    ----------
    f : callable
        Vectorised integrand.
    a, b : float
        Interval, ``b > a``.
    n : int
        Number of nodes (≥ 1). Exact for polynomials of degree ≤ ``2n - 1``.

    Returns
    -------
    float
        Approximation of the integral.
    """
    _check(a, b, n)
    t, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * t + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.sum(w * f(x)))


def romberg_table(f: Integrand, a: float, b: float, levels: int) -> np.ndarray:
    r"""
    Romberg table from trapezoid estimates and repeated Richardson extrapolation.

    ``R[i, 0]`` is the trapezoid rule with :math:`2^i` panels and

    .. math:: R_{i,j} = R_{i,j-1} + \frac{R_{i,j-1} - R_{i-1,j-1}}{4^j - 1}.

    Parameters
    ----------
    f : callable
        Vectorised integrand.
    a, b : float
        Interval, ``b > a``.
    levels : int
        Number of rows (≥ 1); the finest trapezoid uses ``2**(levels-1)`` panels.

    Returns
    -------
    numpy.ndarray
        Shape ``(levels, levels)``, lower triangular (NaN above the diagonal).
        ``R[-1, -1]`` is the most extrapolated estimate.
    """
    _check(a, b, levels)
    R = np.full((levels, levels), np.nan)
    for i in range(levels):
        R[i, 0] = trapezoid_rule(f, a, b, 2**i)
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    return R


def observed_order(h: np.ndarray, errors: np.ndarray) -> float:
    """
    Observed order of convergence: slope of log(error) against log(h).

    Fits :math:`\\log|E| = p \\log h + c` by least squares. For an O(h^p)
    method on a suitable integrand the slope is close to ``p``.

    Parameters
    ----------
    h : numpy.ndarray
        Step sizes (> 0).
    errors : numpy.ndarray
        Absolute errors at those step sizes; zeros are ignored.

    Returns
    -------
    float
        Estimated order ``p``.

    Raises
    ------
    ValueError
        If fewer than two non-zero errors remain.
    """
    h = np.asarray(h, dtype=np.float64)
    e = np.abs(np.asarray(errors, dtype=np.float64))
    keep = e > 0
    if keep.sum() < 2:
        raise ValueError("Need at least two non-zero errors to fit an order.")
    slope, _ = np.polyfit(np.log(h[keep]), np.log(e[keep]), 1)
    return float(slope)
