#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Elastic scattering Legendre cross sections from EEDL data

The elastic cross section is split, as in :mod:`pyepics.converters.mcdc_gfp`,
into

* the large-angle part ``xs_lge`` (EEDL cosine tables, scattering cosine below
  ``1 - 1e-6``), whose angular distribution is tabulated, and
* the small-angle remainder ``max(xs_el - xs_lge, 0)``, described by a
  screened Rutherford distribution with the Moliere screening parameter of
  :func:`pyepics.utils.parsing.small_angle_eta`.

The Legendre moments of the large-angle tables are interpolated linearly in
energy after the tables are densified by log-log interpolation, as in the
converter.  This replaces the Mott-model elastic cross sections of Bienvenue
et al. (J. Comput. Phys. 524 (2025) 113740), whose coefficient tables are not
part of EEDL, but keeps their cleanup of unstable high-order moments.
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

# Relative slack before a Legendre moment counts as exceeding the zeroth one;
# absorbs round-off in nearly forward moments, which equal 1 to many digits.
_MOMENT_SLACK = 1e-9


def large_angle_moment_anchors(
    table: DistributionRecord, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Legendre moments of the large-angle scattering at a set of anchor energies

    The tabulated cosine distributions are first densified in energy by
    log-log interpolation; the Legendre moments of each resulting
    distribution are then evaluated exactly.

    Parameters
    ----------
    table : DistributionRecord
        EEDL large-angle cosine distributions (``ang_lge``): incident energy,
        scattering cosine and probability density.
    order : int
        Highest Legendre order.

    Returns
    -------
    energy : numpy.ndarray
        Anchor energies, eV, shape ``(A,)``.
    moments : numpy.ndarray
        Legendre moments at each anchor energy, shape ``(A, order + 1)``.
    """
    row_energy = np.asarray(table.inc_energy, dtype="f8")
    table_start = np.flatnonzero(np.r_[True, np.diff(row_energy) != 0.0])
    energy, table_start, mu, pdf = insert_log_log_tables(
        row_energy[table_start], table_start, table.value, table.probability
    )
    table_end = list(table_start[1:]) + [len(mu)]
    moments = np.array(
        [
            tabulated_legendre_moments(mu[start:end], pdf[start:end], order)
            for start, end in zip(table_start, table_end, strict=True)
        ]
    )
    return energy, moments


def elastic_legendre_cross_sections(
    dataset: EEDLDataset, energy: np.ndarray, order: int
) -> np.ndarray:
    """Elastic scattering Legendre cross sections ``Sigma_{s,l}(E)``

    ``Sigma_{s,l} = sigma_el g_l``, where ``g_l`` is the mean of ``P_l(mu)``
    over an elastic collision: the cross-section-weighted mix of the
    large-angle (tabulated) and small-angle (screened Rutherford) moments.
    A moment whose magnitude exceeds ``sigma_el`` is unphysical, a sign of
    numerical error at high order, and is set to zero together with all
    higher moments.

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
    cross_sections = dataset.cross_sections
    if "xs_el" not in cross_sections:
        raise ValidationError(f"{owner}: dataset has no 'xs_el'.")
    energy = np.asarray(energy, dtype="f8")

    def cross_section_at_energy(label):
        if label not in cross_sections:
            return np.zeros_like(energy)
        record = cross_sections[label]
        return np.interp(
            energy, record.energy, record.cross_section, left=0.0, right=0.0
        )

    sigma_el = cross_section_at_energy("xs_el")
    sigma_la = np.minimum(cross_section_at_energy("xs_lge"), sigma_el)
    sigma_sa = sigma_el - sigma_la

    moments_la = np.zeros((energy.size, order + 1))
    moments_la[:, 0] = 1.0
    if np.any(sigma_la > 0.0):
        table = dataset.distributions.get("ang_lge")
        if table is None:
            raise ValidationError(
                f"{owner}: 'xs_lge' needs the 'ang_lge' cosine tables."
            )
        anchor_energy, anchor_moments = large_angle_moment_anchors(table, order)
        # moments are linear in E between anchors and constant beyond them
        for legendre_order in range(order + 1):
            moments_la[:, legendre_order] = np.interp(
                energy, anchor_energy, anchor_moments[:, legendre_order]
            )

    moments_sa = np.zeros_like(moments_la)
    moments_sa[:, 0] = 1.0
    has_small_angle = sigma_sa > 0.0
    if np.any(has_small_angle):
        eta = small_angle_eta(dataset.Z, energy[has_small_angle])
        moments_sa[has_small_angle] = [
            legendre_moments_screened_rutherford(eta_i, order) for eta_i in eta
        ]

    with np.errstate(invalid="ignore", divide="ignore"):
        moments = (
            sigma_la[:, None] * moments_la + sigma_sa[:, None] * moments_sa
        ) / sigma_el[:, None]
    moments[sigma_el <= 0.0] = 0.0
    moments[sigma_el > 0.0, 0] = 1.0
    sigma = sigma_el[:, None] * moments

    too_big = np.abs(sigma[:, 1:]) > np.abs(sigma[:, :1]) * (1.0 + _MOMENT_SLACK)
    dropped = np.logical_or.accumulate(too_big, axis=1)
    sigma[:, 1:][dropped] = 0.0
    return sigma
