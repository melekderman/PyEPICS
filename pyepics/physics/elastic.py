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
:func:`pyepics.utils.parsing.small_angle_eta`.
"""

from __future__ import annotations

from functools import cache

import numpy as np

# Gauss-Legendre order for the resolved screened Rutherford moments (eta > 0.1).
SR_QUADRATURE_ORDER = 200

_NO_DEFLECTION = np.array([1.0, 0.0, 0.0, 0.0])


def _legendre(mu):
    """P_1, P_2 and P_3 at *mu*."""
    return (mu, 0.5 * (3.0 * mu * mu - 1.0), 0.5 * (5.0 * mu * mu * mu - 3.0 * mu))


@cache
def _sr_quadrature():
    return np.polynomial.legendre.leggauss(SR_QUADRATURE_ORDER)


def legendre_moments_screened_rutherford(eta):
    """Legendre moments g_0..g_3 of the screened Rutherford PDF
    eta (eta + 2) / (2 (eta + 1 - mu)^2) on [-1, 1].

    Integrating powers of t = 1 - mu gives m_k = <t^k> analytically.
    For eta <= 0.1 these expressions avoid large cancelling terms and
    resolve arbitrarily narrow peaks.  At larger eta the PDF is smooth;
    quadrature avoids cancellation in the analytical, near-isotropic limit.
    """
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
