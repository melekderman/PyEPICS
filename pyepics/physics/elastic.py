#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Angular distributions of elastic electron scattering

EEDL tabulates elastic scattering only for deflections with scattering cosine
below ``1 - 1e-6``.  The remaining small-angle scattering is described by the
screened Rutherford distribution, whose screening parameter ``eta`` comes from
:func:`pyepics.utils.parsing.small_angle_eta`.

This module provides the Legendre moments of both pieces: the analytic moments
of the screened Rutherford distribution, and the moments of the tabulated
large-angle cosine distributions together with the log-log densification of
those tables in energy.
"""

from __future__ import annotations

from functools import cache

import numpy as np

# Largest energy ratio between adjacent moment tables (April / FRENSIE).
ANCHOR_MAX_ENERGY_RATIO = 1.1
# Gauss-Legendre order for the screened Rutherford moments by quadrature.
SR_QUADRATURE_ORDER = 200

# np.trapezoid needs NumPy >= 2.0; np.trapz is the same rule in older releases.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@cache
def _sr_quadrature():
    return np.polynomial.legendre.leggauss(SR_QUADRATURE_ORDER)


def legendre_moments_screened_rutherford(eta, order=3):
    """Legendre moments of the screened Rutherford scattering distribution

    The distribution of the scattering cosine ``mu`` is
    ``eta (eta + 2) / (2 (eta + 1 - mu)^2)`` on ``[-1, 1]``.  Its Legendre
    moments ``g_0..g_order`` measure how forward-peaked the scattering is:
    ``g_l`` is the average of ``P_l(mu)`` over a collision.

    For ``order = 3`` and ``eta <= 0.1`` the moments are evaluated in closed
    form, which resolves arbitrarily narrow forward peaks.  For larger
    ``eta`` the distribution is smooth and is integrated by quadrature, which
    avoids the cancellation of the closed form in the near-isotropic limit.
    Other orders always use quadrature (:func:`_moments_resolved`), which
    agrees with the closed form to about 1e-12.

    Parameters
    ----------
    eta : float
        Screening parameter of the distribution.
    order : int, optional
        Highest Legendre order.

    Returns
    -------
    numpy.ndarray
        Moments, shape ``(order + 1,)``; ``g[0]`` is exactly 1.
    """
    if order != 3:
        return _moments_resolved(eta, order)
    if eta <= 0.1:
        # Averages <t^k> of the deflection t = 1 - mu, from which the Legendre
        # moments follow; log_term = log(1 + 2/eta) without overflow for tiny eta.
        log_term = np.log(eta + 2.0) - np.log(eta)
        t_mean = eta * ((1.0 + 0.5 * eta) * log_term - 1.0)
        t2_mean = 2.0 * eta * (1.0 - t_mean)
        t3_mean = 2.0 * eta - 1.5 * eta * t2_mean
        return np.array(
            [
                1.0,
                1.0 - t_mean,
                1.0 - 3.0 * t_mean + 1.5 * t2_mean,
                1.0 - 6.0 * t_mean + 7.5 * t2_mean - 2.5 * t3_mean,
            ]
        )
    mu, weights = _sr_quadrature()
    pdf = eta * (eta + 2.0) / (2.0 * (eta + 1.0 - mu) ** 2)
    moments = (weights * pdf) @ np.polynomial.legendre.legvander(mu, order)
    moments[0] = 1.0
    return moments


def insert_log_log_tables(energy, offset, value, pdf):
    """Densify the tabulated cosine distributions in incident energy

    Where two adjacent tabulated energies differ by more than a factor
    :data:`ANCHOR_MAX_ENERGY_RATIO`, distributions at intermediate energies
    are inserted by log-log interpolation in energy and probability density,
    each renormalised to unit area.  The evaluated distributions are returned
    unchanged.

    Parameters
    ----------
    energy : numpy.ndarray
        Incident energies, eV, shape ``(E,)``, ascending.
    offset : numpy.ndarray
        Start of each energy's cosine table within *value* and *pdf*.
    value, pdf : numpy.ndarray
        Scattering cosines and probability densities of all tables, joined.

    Returns
    -------
    energy, offset, value, pdf : numpy.ndarray
        The same quantities for the densified set of tables.

    Raises
    ------
    ValueError
        If a pair of tables to interpolate between has a non-positive density.
    """
    table_end = list(offset[1:]) + [len(value)]
    tables = [
        (energy[i], value[offset[i] : table_end[i]], pdf[offset[i] : table_end[i]])
        for i in range(len(energy))
    ]
    densified = []
    for (energy_lo, mu_lo, pdf_lo), (energy_hi, mu_hi, pdf_hi) in zip(
        tables[:-1], tables[1:], strict=True
    ):
        densified.append((energy_lo, mu_lo, pdf_lo))
        ratio = energy_hi / energy_lo
        if ratio <= ANCHOR_MAX_ENERGY_RATIO:
            continue
        n_inserted = max(
            1, int(np.ceil(np.log(ratio) / np.log(ANCHOR_MAX_ENERGY_RATIO))) - 1
        )
        mu = np.union1d(mu_lo, mu_hi)
        pdf_at_lo = np.interp(mu, mu_lo, pdf_lo)
        pdf_at_hi = np.interp(mu, mu_hi, pdf_hi)
        if np.any(pdf_at_lo <= 0.0) or np.any(pdf_at_hi <= 0.0):
            raise ValueError(
                f"Cosine tables at {energy_lo} and {energy_hi} eV have non-positive "
                "PDF values; log-log interpolation is undefined."
            )
        log_pdf_lo, log_pdf_hi = np.log(pdf_at_lo), np.log(pdf_at_hi)
        inserted = np.exp(
            np.linspace(np.log(energy_lo), np.log(energy_hi), n_inserted + 2)[1:-1]
        )
        for new_energy in inserted:
            fraction = (np.log(new_energy) - np.log(energy_lo)) / (
                np.log(energy_hi) - np.log(energy_lo)
            )
            new_pdf = np.exp(log_pdf_lo + fraction * (log_pdf_hi - log_pdf_lo))
            densified.append((new_energy, mu, new_pdf / _trapezoid(new_pdf, mu)))
    densified.append(tables[-1])

    new_offset = np.cumsum([0] + [len(table[1]) for table in densified[:-1]])
    return (
        np.array([table[0] for table in densified], dtype="f8"),
        new_offset.astype(np.asarray(offset).dtype),
        np.concatenate([table[1] for table in densified]),
        np.concatenate([table[2] for table in densified]),
    )


def _moments_resolved(eta, order):
    """Screened Rutherford Legendre moments by quadrature, for any ``eta``

    Changing variables to ``u = log(1 + (1 - mu) / eta)`` turns the
    distribution into ``(eta + 2) / 2 exp(-u) du`` on ``[0, log(1 + 2 / eta)]``.
    This has no narrow forward peak for any ``eta``, so a fixed Gauss-Legendre
    rule resolves it.
    """
    if order < 0:
        raise ValueError("Legendre order must be non-negative.")
    nodes, weights = _sr_quadrature()
    u_max = np.log1p(2.0 / eta)
    u = 0.5 * u_max * (nodes + 1.0)
    mu = np.clip(1.0 - eta * np.expm1(u), -1.0, 1.0)
    u_weights = weights * 0.5 * u_max * (1.0 + 0.5 * eta) * np.exp(-u)
    moments = u_weights @ np.polynomial.legendre.legvander(mu, order)
    moments[0] = 1.0
    return moments


def tabulated_legendre_moments(mu, pdf, order):
    """Legendre moments of a tabulated scattering-cosine distribution

    The distribution is linear between table points and normalised to unit
    area.  Each interval is integrated with a Gauss-Legendre rule that is
    exact for the product of the linear density and ``P_order``, so the
    moments carry no quadrature error however coarse the table.

    Parameters
    ----------
    mu : array_like
        Cosines, shape ``(N,)``, ``N >= 2``, non-decreasing.
    pdf : array_like
        Probability density at *mu*, shape ``(N,)``, with a positive integral.
    order : int
        Highest Legendre order, ``>= 0``.

    Returns
    -------
    numpy.ndarray
        Moments ``g_0..g_order``, shape ``(order + 1,)``; ``g[0]`` is exactly 1.

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
    half_width = 0.5 * np.diff(mu)[:, None]
    midpoint = 0.5 * (mu[1:] + mu[:-1])[:, None]
    node_mu = midpoint + half_width * nodes[None, :]
    position = 0.5 * (nodes[None, :] + 1.0)  # fraction of the way across the interval
    node_pdf = pdf[:-1, None] + (pdf[1:] - pdf[:-1])[:, None] * position
    moments = np.einsum(
        "ij,ijl->l",
        half_width * weights[None, :] * node_pdf,
        np.polynomial.legendre.legvander(node_mu, order),
    )
    moments = moments / norm
    moments[0] = 1.0
    return moments
