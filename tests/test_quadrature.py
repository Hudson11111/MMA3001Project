"""
Tests for the quadrature rules: polynomial exactness and convergence order on a
smooth integrand, then the expected loss of order on a step function.
"""

import numpy as np
import pytest

from porkqc.metrics.quadrature import (
    gauss_legendre_rule,
    midpoint_rule,
    observed_order,
    romberg_table,
    simpson_rule,
    trapezoid_rule,
)


def poly(deg):
    """x**deg on [0, 2]; exact integral 2**(deg+1)/(deg+1)."""
    return (lambda x: x**deg), 2.0 ** (deg + 1) / (deg + 1)


# ---------- polynomial exactness (degree of precision) ----------


@pytest.mark.parametrize("deg", [0, 1])
def test_trapezoid_and_midpoint_exact_for_linear(deg):
    f, exact = poly(deg)
    assert trapezoid_rule(f, 0, 2, 3) == pytest.approx(exact)
    assert midpoint_rule(f, 0, 2, 3) == pytest.approx(exact)


def test_trapezoid_not_exact_for_quadratic():
    f, exact = poly(2)
    assert abs(trapezoid_rule(f, 0, 2, 4) - exact) > 1e-3


@pytest.mark.parametrize("deg", [0, 1, 2, 3])
def test_simpson_exact_up_to_cubic(deg):
    f, exact = poly(deg)
    assert simpson_rule(f, 0, 2, 2) == pytest.approx(exact)


def test_simpson_not_exact_for_quartic():
    f, exact = poly(4)
    assert abs(simpson_rule(f, 0, 2, 2) - exact) > 1e-3


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_gauss_legendre_exact_to_degree_2n_minus_1(n):
    f, exact = poly(2 * n - 1)
    assert gauss_legendre_rule(f, 0, 2, n) == pytest.approx(exact)
    f, exact = poly(2 * n)
    assert abs(gauss_legendre_rule(f, 0, 2, n) - exact) > 1e-6


# ---------- convergence on a smooth integrand ----------


def _orders(rule, f, exact, ns):
    errs = [rule(f, 0.0, 1.0, n) - exact for n in ns]
    return observed_order(1.0 / np.array(ns), errs)


def test_observed_orders_smooth():
    f, exact = np.exp, np.e - 1
    ns = [4, 8, 16, 32, 64]
    assert _orders(trapezoid_rule, f, exact, ns) == pytest.approx(2.0, abs=0.05)
    assert _orders(midpoint_rule, f, exact, ns) == pytest.approx(2.0, abs=0.05)
    assert _orders(simpson_rule, f, exact, ns) == pytest.approx(4.0, abs=0.1)


def test_romberg_beats_trapezoid_on_smooth():
    R = romberg_table(np.exp, 0.0, 1.0, 5)
    exact = np.e - 1
    assert abs(R[-1, -1] - exact) < 1e-12
    assert abs(R[-1, 0] - exact) > 1e-5
    # column 1 of Romberg is Simpson's rule
    assert R[2, 1] == pytest.approx(simpson_rule(np.exp, 0.0, 1.0, 4))


# ---------- step function: high order is lost ----------


def step(x):
    """1 on [0, 0.3), 0.6 on [0.3, 0.7), 0 beyond — integral 0.3 + 0.24 = 0.54."""
    return np.where(x < 0.3, 1.0, np.where(x < 0.7, 0.6, 0.0))


def test_step_function_first_order_only():
    exact = 0.54
    ns = np.array([10, 20, 40, 80, 160, 320, 640]) + 1  # +1 avoids nodes landing exactly on jumps
    ns_even = ns + 1
    p_trap = _orders(trapezoid_rule, step, exact, ns)
    p_simp = _orders(simpson_rule, step, exact, ns_even)
    assert 0.6 < p_trap < 1.4
    assert 0.6 < p_simp < 1.4  # Simpson is no better than first order here


# ---------- argument checks ----------


def test_bad_arguments():
    with pytest.raises(ValueError):
        simpson_rule(np.exp, 0, 1, 3)
    with pytest.raises(ValueError):
        trapezoid_rule(np.exp, 1, 0, 4)
    with pytest.raises(ValueError):
        midpoint_rule(np.exp, 0, 1, 0)
    with pytest.raises(ValueError):
        observed_order([0.1, 0.05], [0.0, 0.0])


def test_simpson_matches_scipy():
    scipy_integrate = pytest.importorskip("scipy.integrate")
    x = np.linspace(0, 1, 9)
    assert simpson_rule(np.exp, 0, 1, 8) == pytest.approx(scipy_integrate.simpson(np.exp(x), x=x))
