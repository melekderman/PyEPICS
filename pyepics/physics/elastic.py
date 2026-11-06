#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Screened Rutherford elastic scattering

Legendre moments of the screened Rutherford angular distribution that
describes small-angle elastic scattering beyond the EEDL large-angle
tables.  The screening parameter ``eta`` comes from
:func:`pyepics.utils.parsing.small_angle_eta`.  Also holds the log-log
densification of the large-angle cosine tables used to interpolate moments
in energy.
"""

from __future__ import annotations

from functools import cache

import numpy as np

# Largest energy ratio between adjacent moment tables (April / FRENSIE).
ANCHOR_MAX_ENERGY_RATIO = 1.1
# Gauss-Legendre order for the resolved screened Rutherford moments (eta > 0.1).
SR_QUADRATURE_ORDER = 200

_NO_DEFLECTION = np.array([1.0, 0.0, 0.0, 0.0])

# np.trapezoid needs NumPy >= 2.0; np.trapz is the same rule in older releases.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _legendre(mu):
    """P_1, P_2 and P_3 at *mu*."""
    return (mu, 0.5 * (3.0 * mu * mu - 1.0), 0.5 * (5.0 * mu * mu * mu - 3.0 * mu))


@cache
def _sr_quadrature():
    return np.polynomial.legendre.leggauss(SR_QUADRATURE_ORDER)


def legendre_moments_screened_rutherford(eta, order=3):
    """Legendre moments g_0..g_order of the screened Rutherford PDF
    eta (eta + 2) / (2 (eta + 1 - mu)^2) on [-1, 1].

    For ``order = 3`` (the default), integrating powers of t = 1 - mu gives
    m_k = <t^k> analytically.  For eta <= 0.1 these expressions avoid large
    cancelling terms and resolve arbitrarily narrow peaks.  At larger eta the
    PDF is smooth; quadrature avoids cancellation in the analytical,
    near-isotropic limit.

    Higher orders use :func:`_moments_resolved` for every eta, which agrees
    with the analytical result for ``order = 3`` to about 1e-12.

    Returns
    -------
    numpy.ndarray
        Shape ``(order + 1,)``; ``g[0]`` is exactly 1.
    """
    if order != 3:
        return _moments_resolved(eta, order)
    if eta <= 0.1:
        # L = log(1 + 2/eta), without overflowing 2/eta for tiny eta.
        L = np.log(eta + 2.0) - np.log(eta)
        m1 = eta * ((1.0 + 0.5 * eta) * L - 1.0)
        m2 = 2.0 * eta * (1.0 - m1)
        m3 = 2.0 * eta - 1.5 * eta * m2
        return np.array(
            [
                1.0,
                1.0 - m1,
                1.0 - 3.0 * m1 + 1.5 * m2,
                1.0 - 6.0 * m1 + 7.5 * m2 - 2.5 * m3,
            ],
            dtype="f8",
        )
    nodes, weights = _sr_quadrature()
    pdf = (1.0 / (eta + 1.0 - nodes) ** 2) / (2.0 / (eta * (eta + 2.0)))
    g = _NO_DEFLECTION.copy()
    g[1:] = [np.dot(weights, p * pdf) for p in _legendre(nodes)]
    return g


def insert_log_log_tables(energy, offset, value, pdf):
    """Insert log-log interpolated cosine tables so that no two adjacent
    energies differ by more than :data:`ANCHOR_MAX_ENERGY_RATIO`.

    The evaluated tables are returned unchanged.
    """
    ends = list(offset[1:]) + [len(value)]
    tables = [
        (energy[i], value[offset[i] : ends[i]], pdf[offset[i] : ends[i]])
        for i in range(len(energy))
    ]
    out = []
    for (e0, mu0, p0), (e1, mu1, p1) in zip(tables[:-1], tables[1:], strict=True):
        out.append((e0, mu0, p0))
        ratio = e1 / e0
        if ratio <= ANCHOR_MAX_ENERGY_RATIO:
            continue
        n = max(1, int(np.ceil(np.log(ratio) / np.log(ANCHOR_MAX_ENERGY_RATIO))) - 1)
        mu = np.union1d(mu0, mu1)
        lo, hi = np.interp(mu, mu0, p0), np.interp(mu, mu1, p1)
        if np.any(lo <= 0.0) or np.any(hi <= 0.0):
            raise ValueError(
                f"Cosine tables at {e0} and {e1} eV have non-positive PDF values; "
                "log-log interpolation is undefined."
            )
        lo, hi = np.log(lo), np.log(hi)
        for e in np.exp(np.linspace(np.log(e0), np.log(e1), n + 2)[1:-1]):
            t = (np.log(e) - np.log(e0)) / (np.log(e1) - np.log(e0))
            p = np.exp(lo + t * (hi - lo))
            out.append((e, mu, p / _trapezoid(p, mu)))
    out.append(tables[-1])

    new_offset = np.cumsum([0] + [len(t[1]) for t in out[:-1]])
    return (
        np.array([t[0] for t in out], dtype="f8"),
        new_offset.astype(np.asarray(offset).dtype),
        np.concatenate([t[1] for t in out]),
        np.concatenate([t[2] for t in out]),
    )


def _moments_resolved(eta, order):
    """Moments g_0..g_order by quadrature in u = log(1 + (1 - mu) / eta)

    Then p(mu) |dmu| = (eta + 2) / 2 exp(-u) du on [0, log(1 + 2 / eta)],
    which has no narrow peak for any eta, so a fixed Gauss-Legendre rule
    resolves it and the Legendre polynomials are evaluated directly.
    """
    if order < 0:
        raise ValueError("Legendre order must be non-negative.")
    nodes, weights = _sr_quadrature()
    length = np.log1p(2.0 / eta)
    u = 0.5 * length * (nodes + 1.0)
    mu = np.clip(1.0 - eta * np.expm1(u), -1.0, 1.0)
    w = weights * 0.5 * length * (1.0 + 0.5 * eta) * np.exp(-u)
    g = w @ np.polynomial.legendre.legvander(mu, order)
    g[0] = 1.0
    return g


def tabulated_legendre_moments(mu, pdf, order):
    """Legendre moments g_0..g_order of a lin-lin tabulated cosine PDF

    The PDF is normalised by the trapezoid rule, which is exact for a
    piecewise-linear function.  Each interval is integrated with a
    Gauss-Legendre rule exact for the product of the linear PDF and
    ``P_order``, so the moments carry no quadrature error however coarse the
    table.

    Parameters
    ----------
    mu : array_like
        Cosines, shape ``(N,)``, ``N >= 2``, non-decreasing.
    pdf : array_like
        PDF values at *mu*, shape ``(N,)``, with a positive integral.
    order : int
        Highest Legendre order, ``>= 0``.

    Returns
    -------
    numpy.ndarray
        Shape ``(order + 1,)``; ``g[0]`` is exactly 1.

    Raises
    ------
    ValueError
        If the table is too short, not sorted, or has a non-positive area.
    """
    mu = np.asarray(mu, dtype="f8")
    pdf = np.asarray(pdf, dtype="f8")
    if mu.ndim != 1 or mu.shape != pdf.shape or mu.size < 2:
        raise ValueError("Cosine table needs at least two matching mu and pdf values.")
    if np.any(np.diff(mu) < 0.0):
        raise ValueError("Cosine table needs non-decreasing mu values.")
    if order < 0:
        raise ValueError("Legendre order must be non-negative.")
    norm = _trapezoid(pdf, mu)
    if norm <= 0.0:
        raise ValueError("Cosine table has a non-positive integral.")
    nodes, weights = np.polynomial.legendre.leggauss(order // 2 + 2)
    half = 0.5 * np.diff(mu)[:, None]
    mid = 0.5 * (mu[1:] + mu[:-1])[:, None]
    x = mid + half * nodes[None, :]
    frac = 0.5 * (nodes[None, :] + 1.0)
    p = pdf[:-1, None] + (pdf[1:] - pdf[:-1])[:, None] * frac
    g = np.einsum(
        "ij,ijl->l",
        half * weights[None, :] * p,
        np.polynomial.legendre.legvander(x, order),
    )
    g = g / norm
    g[0] = 1.0
    return g
