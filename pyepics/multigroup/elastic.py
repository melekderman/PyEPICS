#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Pointwise elastic Legendre cross sections from EEDL data

The elastic cross section is split, as in
:mod:`pyepics.converters.mcdc_gfp`, into

* the large-angle part ``xs_lge`` (EEDL MT525 cosine tables, mu below
  ``1 - 1e-6``), whose Legendre moments come from the tables, and
* the small-angle remainder ``max(xs_el - xs_lge, 0)``, a screened Rutherford
  distribution with the Moliere parameter of
  :func:`pyepics.utils.parsing.small_angle_eta`.

The moments of the large-angle tables are interpolated linearly in energy
after the tables are densified by log-log interpolation, as in the converter.
This replaces the Mott model of Bienvenue et al. (J. Comput. Phys. 524 (2025)
113740, Eqs. 46-59), whose coefficient tables are not part of EEDL, but keeps
their high-order cleanup (Eq. 60).
"""

from __future__ import annotations

import numpy as np

from pyepics.exceptions import ValidationError
from pyepics.models.records import DistributionRecord, EEDLDataset
from pyepics.physics.elastic import (
    insert_log_log_tables,
    legendre_moments_screened_rutherford,
    tabulated_legendre_moments,
)
from pyepics.utils.parsing import small_angle_eta

# Eq. 60 zeroes moments that exceed sigma_0 in magnitude; this much slack absorbs
# round-off in nearly forward moments, which equal 1 to many digits.
_EQ60_SLACK = 1e-9


def large_angle_moment_anchors(
    table: DistributionRecord, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Energies and Legendre moments ``g_0..g_order`` of the densified tables

    Parameters
    ----------
    table : DistributionRecord
        EEDL large-angle cosine PDFs (``ang_lge``): incident energy, mu, pdf.
    order : int
        Highest Legendre order.

    Returns
    -------
    energy : numpy.ndarray
        Anchor energies, eV, shape ``(A,)``.
    moments : numpy.ndarray
        Shape ``(A, order + 1)``.
    """
    inc = np.asarray(table.inc_energy, dtype="f8")
    first = np.flatnonzero(np.r_[True, np.diff(inc) != 0.0])
    energy, offset, mu, pdf = insert_log_log_tables(
        inc[first], first, table.value, table.probability
    )
    ends = list(offset[1:]) + [len(mu)]
    moments = np.array(
        [
            tabulated_legendre_moments(mu[a:b], pdf[a:b], order)
            for a, b in zip(offset, ends, strict=True)
        ]
    )
    return energy, moments


def elastic_legendre_cross_sections(
    dataset: EEDLDataset, energy: np.ndarray, order: int
) -> np.ndarray:
    """Elastic Legendre cross sections ``Sigma_{s,l}(E)``, ``l = 0..order``

    ``Sigma_{s,l} = sigma_el g_l`` with ``g_l`` the cross-section-weighted mix
    of the large-angle and screened Rutherford moments.  A moment whose
    magnitude exceeds ``sigma_el`` (numerical garbage at high order) is set to
    zero together with all higher ones (Eq. 60).

    Parameters
    ----------
    dataset : EEDLDataset
        Must hold ``xs_el`` and, if ``xs_lge`` is positive anywhere asked
        for, ``xs_lge`` and the ``ang_lge`` tables.
    energy : numpy.ndarray
        Energies, eV, shape ``(N,)``.
    order : int
        Highest Legendre order ``L``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, order + 1)``, barns.  Zero where ``xs_el`` is zero.

    Raises
    ------
    ValidationError
        If ``xs_el`` is missing, or large-angle scattering is present
        without its cosine tables.
    """
    owner = "elastic_legendre_cross_sections"
    xs = dataset.cross_sections
    if "xs_el" not in xs:
        raise ValidationError(f"{owner}: dataset has no 'xs_el'.")
    energy = np.asarray(energy, dtype="f8")

    def on_energy(label):
        if label not in xs:
            return np.zeros_like(energy)
        rec = xs[label]
        return np.interp(energy, rec.energy, rec.cross_section, left=0.0, right=0.0)

    sigma_el = on_energy("xs_el")
    sigma_la = np.minimum(on_energy("xs_lge"), sigma_el)
    sigma_sa = sigma_el - sigma_la

    g_la = np.zeros((energy.size, order + 1))
    g_la[:, 0] = 1.0
    if np.any(sigma_la > 0.0):
        table = dataset.distributions.get("ang_lge")
        if table is None:
            raise ValidationError(
                f"{owner}: 'xs_lge' needs the 'ang_lge' cosine tables."
            )
        anchor_e, anchor_g = large_angle_moment_anchors(table, order)
        # linear in E between anchors, constant beyond them
        for l_order in range(order + 1):
            g_la[:, l_order] = np.interp(energy, anchor_e, anchor_g[:, l_order])

    g_sa = np.zeros_like(g_la)
    g_sa[:, 0] = 1.0
    use_sa = sigma_sa > 0.0
    if np.any(use_sa):
        eta = small_angle_eta(dataset.Z, energy[use_sa])
        g_sa[use_sa] = [legendre_moments_screened_rutherford(e, order) for e in eta]

    with np.errstate(invalid="ignore", divide="ignore"):
        g = (sigma_la[:, None] * g_la + sigma_sa[:, None] * g_sa) / sigma_el[:, None]
    g[sigma_el <= 0.0] = 0.0
    g[sigma_el > 0.0, 0] = 1.0
    sigma = sigma_el[:, None] * g

    # Eq. 60: once a moment exceeds sigma_0, drop it and every higher one.
    too_big = np.abs(sigma[:, 1:]) > np.abs(sigma[:, :1]) * (1.0 + _EQ60_SLACK)
    dropped = np.logical_or.accumulate(too_big, axis=1)
    sigma[:, 1:][dropped] = 0.0
    return sigma
