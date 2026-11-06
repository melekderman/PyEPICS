"""Tests for the order-extended helpers in pyepics.physics.elastic."""

from __future__ import annotations

import numpy as np
import pytest

from pyepics.physics.elastic import (
    _moments_resolved,
    legendre_moments_screened_rutherford,
    tabulated_legendre_moments,
)


def _reference_sr(eta, order, n=400):
    """Independent: Gauss-Legendre in mu on a resolved split of [-1, 1]."""
    # Substitution t = 1 - mu = eta * (exp(u) - 1) is what the code uses, so
    # use mpmath-free brute force instead: composite GL on geometric panels.
    edges = np.concatenate([[0.0], eta * np.geomspace(1e-3, 2.0 / eta + 1.0, 60)])
    edges = np.unique(np.clip(edges, 0.0, 2.0))
    edges[-1] = 2.0
    x, w = np.polynomial.legendre.leggauss(32)
    total = np.zeros(order + 1)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        t = 0.5 * (b - a) * (x + 1.0) + a
        pdf = eta * (eta + 2.0) / (2.0 * (eta + t) ** 2)
        total += (
            0.5 * (b - a) * (w * pdf) @ np.polynomial.legendre.legvander(1.0 - t, order)
        )
    return total


class TestScreenedRutherfordOrder:
    @pytest.mark.parametrize("eta", [1e-10, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 100.0])
    def test_default_unchanged_shape(self, eta):
        assert legendre_moments_screened_rutherford(eta).shape == (4,)

    @pytest.mark.parametrize("eta", [1e-12, 1e-8, 1e-6, 1e-3, 0.1, 1.0, 50.0])
    def test_resolved_matches_analytic_for_order_three(self, eta):
        analytic = legendre_moments_screened_rutherford(eta)
        resolved = _moments_resolved(eta, 3)
        np.testing.assert_allclose(resolved, analytic, rtol=1e-11, atol=2e-14)

    @pytest.mark.parametrize("eta", [1e-8, 1e-6, 1e-4, 1e-2, 0.5, 5.0])
    def test_high_order_against_independent_quadrature(self, eta):
        g = legendre_moments_screened_rutherford(eta, order=15)
        assert g.shape == (16,) and g[0] == 1.0
        np.testing.assert_allclose(g, _reference_sr(eta, 15), rtol=1e-8, atol=1e-10)

    def test_moments_decrease_and_stay_bounded(self):
        g = legendre_moments_screened_rutherford(1e-5, order=15)
        assert np.all(np.abs(g) <= 1.0)
        assert np.all(np.diff(g) < 0.0)

    def test_negative_order(self):
        with pytest.raises(ValueError, match="non-negative"):
            legendre_moments_screened_rutherford(0.1, order=-1)


class TestTabulatedMoments:
    def test_linear_pdf_closed_form(self):
        mu = np.array([-1.0, 1.0])
        g = tabulated_legendre_moments(mu, 1.0 + mu, 3)  # (1 + mu) / 2
        np.testing.assert_allclose(g, [1.0, 1.0 / 3.0, 0.0, 0.0], atol=1e-14)

    def test_coarse_table_is_exact_for_piecewise_linear(self):
        # three-point table; compare to a very fine trapezoid of the same function
        mu = np.array([-1.0, 0.2, 1.0])
        pdf = np.array([0.1, 0.9, 0.4])
        fine = np.linspace(-1.0, 1.0, 400001)
        p = np.interp(fine, mu, pdf)
        norm = np.trapezoid(p, fine)
        ref = [
            np.trapezoid(p * np.polynomial.legendre.legval(fine, [0] * n + [1]), fine)
            / norm
            for n in range(16)
        ]
        np.testing.assert_allclose(
            tabulated_legendre_moments(mu, pdf, 15), ref, atol=1e-9
        )

    def test_isotropic_has_zero_higher_moments(self):
        g = tabulated_legendre_moments([-1.0, 0.0, 1.0], [0.5, 0.5, 0.5], 10)
        np.testing.assert_allclose(g, [1.0] + [0.0] * 10, atol=1e-14)

    @pytest.mark.parametrize(
        "mu, pdf, match",
        [
            ([0.0], [1.0], "two"),
            ([1.0, 0.0], [1.0, 1.0], "non-decreasing"),
            ([0.0, 1.0], [0.0, 0.0], "non-positive"),
        ],
    )
    def test_bad_tables(self, mu, pdf, match):
        with pytest.raises(ValueError, match=match):
            tabulated_legendre_moments(mu, pdf, 3)
