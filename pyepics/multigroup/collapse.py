#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Collapse of pointwise cross sections and BFP coefficients onto energy groups

A multigroup value is the average of the pointwise quantity over the group,
weighted by an approximate particle flux spectrum ``w(E)``,

.. math:: f_g = \\frac{\\int_g f(E)\\, w(E)\\, dE}{\\int_g w(E)\\, dE}.

With constant flux (``w = 1``) this is the group-averaged cross section,
stopping power and straggling of Bienvenue et al. (J. Comput. Phys. 524
(2025) 113740).  The group bounds, weighting spectrum and pointwise data are
provided by the caller.

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
    """Quadrature nodes and weights for integrating over every energy group

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
    """Gauss-Legendre integration rule for every group of a structure

    Each group is split into segments at *breaks*, and a Gauss-Legendre rule
    is placed on each segment.  Segments wider than a factor *max_ratio* in
    energy are split further into geometrically spaced pieces, so the
    accuracy for smooth weights such as ``1/E`` does not depend on how
    coarse *breaks* is.

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

    Returns
    -------
    GroupQuadrature
        Nodes and weights of all groups.

    Raises
    ------
    ValidationError
        If *n_points* is below 1 or *max_ratio* does not exceed 1.
    """
    if n_points < 1:
        raise ValidationError("group_quadrature: n_points must be at least 1.")
    if max_ratio <= 1.0:
        raise ValidationError("group_quadrature: max_ratio must exceed 1.")
    breaks = (
        np.empty(0) if breaks is None else np.unique(np.asarray(breaks, dtype="f8"))
    )
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(n_points)
    node_energies, node_weights, node_groups = [], [], []
    bounds = structure.energy
    for group in range(structure.n_groups):
        group_lower, group_upper = bounds[group + 1], bounds[group]
        breaks_in_group = breaks[(breaks > group_lower) & (breaks < group_upper)]
        segment_edges = _split_wide(
            np.concatenate([[group_lower], breaks_in_group, [group_upper]]), max_ratio
        )
        half_width = 0.5 * np.diff(segment_edges)[:, None]
        midpoint = 0.5 * (segment_edges[1:] + segment_edges[:-1])[:, None]
        node_energies.append((midpoint + half_width * gauss_nodes[None, :]).ravel())
        node_weights.append((half_width * gauss_weights[None, :]).ravel())
        node_groups.append(np.full(node_energies[-1].size, group))
    return GroupQuadrature(
        np.concatenate(node_energies),
        np.concatenate(node_weights),
        np.concatenate(node_groups),
        structure.n_groups,
        float(structure.energy[-1]),
    )


def _split_wide(edges: np.ndarray, max_ratio: float) -> np.ndarray:
    """Split segments whose end energies differ by more than *max_ratio*"""
    split_edges = [edges[0]]
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if lower > 0.0 and upper / lower > max_ratio:
            n_pieces = int(np.ceil(np.log(upper / lower) / np.log(max_ratio)))
            split_edges.extend(np.geomspace(lower, upper, n_pieces + 1)[1:])
        else:
            split_edges.append(upper)
    return np.array(split_edges)


def collapse_nodes(
    values: np.ndarray, quadrature: GroupQuadrature, weight: WeightingSpectrum | str
) -> np.ndarray:
    """Flux-weighted group average of a quantity evaluated at the quadrature nodes

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
    flux_weight = quadrature.weight * spectrum(quadrature.energy)
    group_flux = np.bincount(
        quadrature.group, weights=flux_weight, minlength=quadrature.n_groups
    )
    if np.any(group_flux <= 0.0):
        raise ValidationError(
            f"collapse: weighting spectrum '{spectrum.name}' integrates to zero "
            f"over group {int(np.argmax(group_flux <= 0.0))}."
        )
    flat_values = values.reshape(values.shape[0], -1)
    weighted_sum = np.zeros((quadrature.n_groups, flat_values.shape[1]))
    np.add.at(weighted_sum, quadrature.group, flux_weight[:, None] * flat_values)
    return (weighted_sum / group_flux[:, None]).reshape(
        (quadrature.n_groups,) + values.shape[1:]
    )


def collapse(
    pointwise_xs: CrossSectionRecord,
    structure: GroupStructure,
    weight: WeightingSpectrum | str = uniform,
) -> GroupXS:
    """Group-averaged cross section from a pointwise cross-section table

    The cross section is interpolated linearly in energy (ENDF ``INT = 2``)
    and taken as zero outside the table's energy range, then averaged over
    each group with the weighting spectrum.

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
