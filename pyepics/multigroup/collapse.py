#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Collapse pointwise xs (or BFP coefficients) into multigroup values

A group value is the weighted average over the group,

.. math:: f_g = \\frac{\\int_g f(E)\\, w(E)\\, dE}{\\int_g w(E)\\, dE},

which for ``w = 1`` is Eqs. (4), (8) and (9) of Bienvenue et al. (J. Comput.
Phys. 524 (2025) 113740).  Bounds, weighting and data are provided by the
caller; this module contains only integration numerics.

The integral over each group is a sum of Gauss-Legendre rules, one per
segment between the break energies the caller supplies (normally the energy
grid of the pointwise table), so a piecewise-linear table is integrated
exactly for a uniform weight and to quadrature accuracy otherwise.
:func:`group_quadrature` builds the rule once, so every quantity of a
generation run shares it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyepics.exceptions import ValidationError
from pyepics.models.records import CrossSectionRecord
from pyepics.multigroup.models import GroupStructure, GroupXS
from pyepics.multigroup.weighting import WeightingSpectrum, resolve, uniform


@dataclass(frozen=True)
class GroupQuadrature:
    """Quadrature nodes for the integral over every group of a structure

    Parameters
    ----------
    energy : numpy.ndarray
        Node energies, eV, shape ``(Nq,)``.
    weight : numpy.ndarray
        Quadrature weights (segment-length scaled), eV, shape ``(Nq,)``;
        they sum to the width of each group within that group.
    group : numpy.ndarray
        Index of the group each node belongs to, shape ``(Nq,)``; groups are
        stored in order, so it is non-decreasing.
    n_groups : int
        Number of groups ``G``.
    lowest : float
        Lowest group bound, eV.  A spectrum that is singular at zero cannot
        weight a structure whose ``lowest`` is 0.
    """

    energy: np.ndarray
    weight: np.ndarray
    group: np.ndarray
    n_groups: int
    lowest: float


def group_quadrature(
    structure: GroupStructure,
    breaks: np.ndarray | None = None,
    n_points: int = 4,
    max_ratio: float = 1.5,
) -> GroupQuadrature:
    """Gauss-Legendre rule over each group, split at *breaks*

    Segments wider than a factor *max_ratio* in energy are split further into
    geometrically spaced pieces, so the accuracy for smooth weights such as
    ``1/E`` does not depend on how coarse *breaks* is.

    Parameters
    ----------
    structure : GroupStructure
        Group structure.
    breaks : numpy.ndarray, optional
        Energies (eV) at which to split the groups into segments, typically
        the energy grid of the pointwise data.  Values outside a group are
        ignored.
    n_points : int, optional
        Gauss-Legendre points per segment.
    max_ratio : float, optional
        Largest ratio between the ends of a segment, ``> 1``.
    """
    if n_points < 1:
        raise ValidationError("group_quadrature: n_points must be at least 1.")
    if max_ratio <= 1.0:
        raise ValidationError("group_quadrature: max_ratio must exceed 1.")
    breaks = (
        np.empty(0) if breaks is None else np.unique(np.asarray(breaks, dtype="f8"))
    )
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    energies, wts, groups = [], [], []
    bounds = structure.energy
    for g in range(structure.n_groups):
        lo, hi = bounds[g + 1], bounds[g]
        inside = breaks[(breaks > lo) & (breaks < hi)]
        edges = _split_wide(np.concatenate([[lo], inside, [hi]]), max_ratio)
        half = 0.5 * np.diff(edges)[:, None]
        mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
        energies.append((mid + half * nodes[None, :]).ravel())
        wts.append((half * weights[None, :]).ravel())
        groups.append(np.full(energies[-1].size, g))
    return GroupQuadrature(
        np.concatenate(energies),
        np.concatenate(wts),
        np.concatenate(groups),
        structure.n_groups,
        float(structure.energy[-1]),
    )


def _split_wide(edges: np.ndarray, max_ratio: float) -> np.ndarray:
    """Subdivide segments whose end energies differ by more than *max_ratio*"""
    out = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if a > 0.0 and b / a > max_ratio:
            n = int(np.ceil(np.log(b / a) / np.log(max_ratio)))
            out.extend(np.geomspace(a, b, n + 1)[1:])
        else:
            out.append(b)
    return np.array(out)


def collapse_nodes(
    values: np.ndarray, quadrature: GroupQuadrature, weight: WeightingSpectrum | str
) -> np.ndarray:
    """Weighted average of *values* (evaluated at the quadrature nodes) per group

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(Nq, ...)``; the average is taken over the first axis.
    quadrature : GroupQuadrature
        Rule the values were evaluated on.
    weight : WeightingSpectrum | str
        Weighting spectrum (see :func:`pyepics.multigroup.weighting.resolve`).

    Returns
    -------
    numpy.ndarray
        Shape ``(G, ...)``.

    Raises
    ------
    ValidationError
        If the weight integrates to zero over a group, or the spectrum is
        singular at zero (``1/E``) and the lowest group bound is 0.
    """
    spectrum = resolve(weight)
    if spectrum.requires_positive_energy and quadrature.lowest <= 0.0:
        raise ValidationError(
            f"collapse: weighting spectrum '{spectrum.name}' is singular at "
            f"E = 0 but the lowest group bound is {quadrature.lowest:g} eV."
        )
    values = np.asarray(values, dtype="f8")
    if values.shape[0] != quadrature.energy.size:
        raise ValidationError(
            f"collapse: values have {values.shape[0]} nodes, quadrature has "
            f"{quadrature.energy.size}."
        )
    w = quadrature.weight * spectrum(quadrature.energy)
    norm = np.bincount(quadrature.group, weights=w, minlength=quadrature.n_groups)
    if np.any(norm <= 0.0):
        raise ValidationError(
            f"collapse: weighting spectrum '{spectrum.name}' integrates to zero "
            f"over group {int(np.argmax(norm <= 0.0))}."
        )
    flat = values.reshape(values.shape[0], -1)
    total = np.zeros((quadrature.n_groups, flat.shape[1]))
    np.add.at(total, quadrature.group, w[:, None] * flat)
    return (total / norm[:, None]).reshape((quadrature.n_groups,) + values.shape[1:])


# collapse( EEDL CE cross-section, structure : GroupStructure, weighting) -> GroupXS
def collapse(
    pointwise_xs: CrossSectionRecord,
    structure: GroupStructure,
    weight: WeightingSpectrum | str = uniform,
) -> GroupXS:
    """Collapse a pointwise cross-section table onto *structure*

    The table is interpolated linearly in energy (ENDF ``INT = 2``) and taken
    as zero outside its energy range.

    Parameters
    ----------
    pointwise_xs : CrossSectionRecord
        Pointwise table, barns.
    structure : GroupStructure
        Target group structure.
    weight : WeightingSpectrum | str, optional
        Weighting spectrum; defaults to uniform.

    Returns
    -------
    GroupXS
        Group cross section with the label of *pointwise_xs*, barns.
    """
    quadrature = group_quadrature(structure, pointwise_xs.energy)
    at_nodes = np.interp(
        quadrature.energy,
        pointwise_xs.energy,
        pointwise_xs.cross_section,
        left=0.0,
        right=0.0,
    )
    return GroupXS(pointwise_xs.label, collapse_nodes(at_nodes, quadrature, weight))
