#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Boltzmann-Fokker-Planck (BFP) electron data from EEDL

Electron collisions are split by a cutoff energy into *soft* collisions, which
lose little energy and are treated as continuous slowing down (stopping power
and energy straggling), and *catastrophic* collisions, which are kept as
discrete group-to-group scattering (see :func:`cutoff_energy`).

The pointwise soft stopping power and straggling are computed from the EEDL
energy-loss spectra, the catastrophic scattering from the same spectra and the
collision kinematics, and :func:`generate` collapses everything onto an energy
group structure, following Bienvenue et al. (J. Comput. Phys. 524 (2025)
113740).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import numpy as np

from pyepics.exceptions import ValidationError
from pyepics.models.records import DistributionRecord, EEDLDataset
from pyepics.multigroup.collapse import collapse_nodes, group_quadrature
from pyepics.multigroup.elastic import elastic_legendre_cross_sections
from pyepics.multigroup.models import (
    MOMENTUM_TRANSFER_LABEL,
    BFPCoefficients,
    GroupStructure,
    GroupXS,
    ScatteringMatrix,
)
from pyepics.multigroup.weighting import WeightingSpectrum


class CutoffModel(StrEnum):
    """Convention that separates soft from catastrophic collisions

    See :func:`cutoff_energy`.
    """

    RADIANT = "RADIANT"
    CEPXS = "CEPXS"


def _group_of(structure: GroupStructure, energy: np.ndarray) -> np.ndarray:
    """Group index ``g`` with ``e[g + 1] < E <= e[g]``, for an array of energies

    Vectorised version of :meth:`GroupStructure.group_of`; energies outside the
    structure get an index outside ``0..G-1``.
    """
    bounds = structure.energy
    return bounds.size - 1 - np.searchsorted(bounds[::-1], energy, side="left")


def cutoff_energy(
    energy: np.ndarray,
    structure: GroupStructure,
    cutoff: CutoffModel | str = CutoffModel.RADIANT,
) -> np.ndarray:
    """Cutoff energy ``Ec(E)`` separating soft from catastrophic collisions

    A collision of an electron of energy ``E`` is *soft* if the electron
    leaves it with at least ``Ec(E)``, i.e. its energy loss is at most
    ``E - Ec(E)``; otherwise it is *catastrophic*.  With the group bounds
    ``e[0] > e[1] > ... > e[G]`` and ``e[G + 1] = 0``, for ``E`` in group ``g``
    (the interval ``(e[g + 1], e[g]]``):

    ``RADIANT``
        One group below the current energy, linear in ``E`` through
        ``(e[g], e[g + 1])`` and ``(e[g + 1], e[g + 2])``
        (Bienvenue et al., J. Comput. Phys. 524 (2025) 113740, cutoff energy
        definition).  Continuous across group bounds.
    ``CEPXS``
        The lower bound of the adjacent group, ``e[g + 2]``: soft collisions
        are those scattering into non-adjacent energy groups. Piecewise
        constant, so discontinuous at group bounds.

    Outside the structure: below ``e[G]`` the cutoff is 0 (every collision is
    soft; this is also the limit of both models as ``E -> e[G]``), and above
    ``e[0]`` the line of the top group is continued.

    Parameters
    ----------
    energy : numpy.ndarray
        Electron energies, eV.
    structure : GroupStructure
        Group structure the cutoff is defined from.
    cutoff : CutoffModel | str, optional
        Cutoff convention.

    Returns
    -------
    numpy.ndarray
        ``Ec(E)`` in eV, same shape as *energy*, with ``0 <= Ec <= E``.
    """
    cutoff = _coerce_cutoff(cutoff)
    energy = np.asarray(energy, dtype="f8")
    bounds = structure.energy
    bounds_to_zero = np.append(bounds, 0.0)  # e[G + 1] = 0
    group = np.clip(_group_of(structure, energy), 0, structure.n_groups - 1)

    if cutoff == CutoffModel.CEPXS:
        energy_cutoff = bounds_to_zero[group + 2]
    else:
        e_g = bounds_to_zero[group]
        e_g1 = bounds_to_zero[group + 1]
        e_g2 = bounds_to_zero[group + 2]
        energy_cutoff = e_g1 + (e_g1 - e_g2) / (e_g - e_g1) * (energy - e_g)
    energy_cutoff = np.where(energy < bounds[-1], 0.0, energy_cutoff)
    return np.clip(energy_cutoff, 0.0, energy)


def _interp_table(energy, table_energy, values) -> np.ndarray:
    """Tabulated quantity at *energy*: linear in energy, zero outside the table"""
    return np.interp(energy, table_energy, values, left=0.0, right=0.0)


def _ionization_subshells(dataset: EEDLDataset, owner: str):
    """Yield ``(knock_on_spectra, binding_energy, cross_section_record)`` per subshell

    The binding energy is the threshold (first tabulated energy) of the
    subshell's electroionization cross section.
    """
    xs = dataset.cross_sections
    for name, spectra in dataset.distributions.items():
        if not name.startswith("spec_"):
            continue
        xs_name = "xs_" + name[len("spec_") :]
        if xs_name not in xs:
            raise ValidationError(
                f"{owner}: '{name}' has no cross section '{xs_name}'."
            )
        yield spectra, float(xs[xs_name].energy[0]), xs[xs_name]


def _bracket(incident_energies: np.ndarray, energy: np.ndarray):
    """Tabulated incident energies on either side of each *energy*, and the mixing weight

    Returns ``(lower, upper, upper_weight)``: indices of the neighbouring
    incident energies and ``ln(E / E_lower) / ln(E_upper / E_lower)`` clipped to
    ``[0, 1]``.  Outside the tabulated range the nearest spectrum is used.
    """
    n_incident = incident_energies.size
    lower = np.clip(
        np.searchsorted(incident_energies, energy, side="right") - 1, 0, n_incident - 1
    )
    upper = np.minimum(lower + 1, n_incident - 1)
    log_span = np.log(incident_energies[upper] / incident_energies[lower])
    positive_energy = np.maximum(energy, np.finfo("f8").tiny)
    upper_weight = np.divide(
        np.log(positive_energy / incident_energies[lower]),
        log_span,
        out=np.zeros_like(energy),
        where=log_span > 0.0,
    )
    return lower, upper, np.clip(upper_weight, 0.0, 1.0)


def _spectra_by_incident_energy(record: DistributionRecord):
    """Incident energies and the ``(energy_loss, probability)`` spectrum at each

    Each spectrum is sorted by energy loss.
    """
    incident_energies = np.unique(record.inc_energy)
    spectra = []
    for incident_energy in incident_energies:
        rows = record.inc_energy == incident_energy
        by_loss = np.argsort(record.value[rows], kind="stable")
        spectra.append((record.value[rows][by_loss], record.probability[rows][by_loss]))
    return incident_energies, spectra


def _partial_moments(
    loss: np.ndarray, probability: np.ndarray, loss_limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrals of ``loss^n * probability`` for ``n = 0, 1, 2`` up to *loss_limit*

    The spectrum is linear between table points and zero outside
    ``[loss[0], loss[-1]]``; *loss_limit* is an array of upper limits
    (``inf`` allowed).  The integrals are the probability of a loss up to the
    limit, and the first and second moments of that part of the spectrum.
    """
    loss_lo, loss_hi = loss[:-1], loss[1:]
    probability_lo = probability[:-1]
    width = loss_hi - loss_lo
    slope = np.divide(
        probability[1:] - probability_lo,
        width,
        out=np.zeros_like(width),
        where=width > 0.0,
    )
    # length of each interval below the limit, shape (limits, intervals)
    length = np.clip(np.minimum(loss_hi, loss_limit[:, None]) - loss_lo, 0.0, None)
    zeroth = probability_lo * length + slope * length**2 / 2.0
    first = (
        loss_lo * probability_lo * length
        + (loss_lo * slope + probability_lo) * length**2 / 2.0
        + slope * length**3 / 3.0
    )
    second = (
        loss_lo**2 * probability_lo * length
        + (loss_lo**2 * slope + 2.0 * loss_lo * probability_lo) * length**2 / 2.0
        + (2.0 * loss_lo * slope + probability_lo) * length**3 / 3.0
        + slope * length**4 / 4.0
    )
    return zeroth.sum(axis=1), first.sum(axis=1), second.sum(axis=1)


def _mixture_moments(
    record: DistributionRecord, energy: np.ndarray, loss_limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partial moments of the energy-loss spectrum at each incident *energy*

    The spectrum at ``E`` is the mix ``(1 - t) p_a + t p_b`` of the tabulated
    spectra at the incident energies ``E_a <= E <= E_b`` around it, with
    ``t = ln(E / E_a) / ln(E_b / E_a)``; outside the tabulated range the
    nearest spectrum is used.  Returns the probability, mean and second moment
    of the energy loss, each integrated up to *loss_limit*, for every entry of
    *energy*.
    """
    incident_energies, spectra = _spectra_by_incident_energy(record)
    moments = np.empty((3, incident_energies.size, energy.size))
    for i, (loss, probability) in enumerate(spectra):
        moments[:, i] = _partial_moments(loss, probability, loss_limit)
    lower, upper, upper_weight = _bracket(incident_energies, energy)
    column = np.arange(energy.size)
    mixed = (1.0 - upper_weight) * moments[:, lower, column]
    mixed += upper_weight * moments[:, upper, column]
    return mixed[0], mixed[1], mixed[2]


def _soft_loss_moment(
    dataset: EEDLDataset,
    structure: GroupStructure,
    cutoff: CutoffModel,
    order: int,
    energy: np.ndarray,
) -> np.ndarray:
    """Moment of the energy loss over the soft collisions, summed over channels

    ``sum_x integral (energy loss)^order dsigma_x`` over collisions whose loss
    is at most ``E - Ec(E)``, for ``order`` 1 (stopping power) or 2
    (straggling).
    """
    owner = "soft energy-loss moment"
    cross_sections = dataset.cross_sections
    average_losses = dataset.average_energy_losses
    largest_soft_loss = energy - cutoff_energy(energy, structure, cutoff)
    total = np.zeros_like(energy)

    if "xs_exc" in cross_sections and "loss_exc" in average_losses:
        xs = cross_sections["xs_exc"]
        sigma = _interp_table(energy, xs.energy, xs.cross_section)
        loss_table = average_losses["loss_exc"]
        loss = _interp_table(energy, loss_table.energy, loss_table.avg_loss)
        total += np.where(loss <= largest_soft_loss, sigma * loss**order, 0.0)

    if "xs_brem" in cross_sections:
        photon_spectra = dataset.bremsstrahlung_spectra
        if photon_spectra is None or "loss_brem_spec" not in average_losses:
            raise ValidationError(
                f"{owner}: 'xs_brem' needs bremsstrahlung spectra and 'loss_brem_spec'."
            )
        xs = cross_sections["xs_brem"]
        sigma = _interp_table(energy, xs.energy, xs.cross_section)
        _, first_soft, second_soft = _mixture_moments(
            photon_spectra, energy, largest_soft_loss
        )
        if order == 1:
            _, first_all, _ = _mixture_moments(photon_spectra, energy, energy)
            loss_table = average_losses["loss_brem_spec"]
            loss = _interp_table(energy, loss_table.energy, loss_table.avg_loss)
            soft_fraction = np.divide(
                first_soft,
                first_all,
                out=np.zeros_like(first_soft),
                where=first_all > 0.0,
            )
            total += sigma * loss * soft_fraction
        else:
            total += sigma * second_soft

    for knock_on_spectra, binding, xs in _ionization_subshells(dataset, owner):
        sigma = _interp_table(energy, xs.energy, xs.cross_section)
        # the knock-on is the slower electron, so it takes at most half the energy
        knock_on_limit = np.maximum(energy - binding, 0.0) / 2.0
        probability, first, second = _mixture_moments(
            knock_on_spectra,
            energy,
            np.minimum(largest_soft_loss - binding, knock_on_limit),
        )
        if order == 1:
            total += sigma * (binding * probability + first)
        else:
            total += sigma * (binding**2 * probability + 2.0 * binding * first + second)
    return total


def pointwise_stopping_power(
    dataset: EEDLDataset,
    structure: GroupStructure,
    cutoff: CutoffModel | str = CutoffModel.RADIANT,
    energy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restricted (soft) electron stopping power at each energy, from EEDL data

    The soft stopping power is the mean energy lost per unit path length in
    collisions that leave the electron with at least ``Ec(E)``, summed over
    the inelastic channels (see :func:`cutoff_energy`).  It equals the total
    stopping power minus the catastrophic part, with the totals taken from
    EEDL (Bienvenue et al., J. Comput. Phys. 524 (2025) 113740; Beling 2024,
    first energy-loss moment).  Elastic scattering loses no energy and
    contributes nothing.

    Channels
    --------
    excitation
        Every collision loses the tabulated average ``loss_exc``, so it is
        soft or catastrophic as a whole.
    bremsstrahlung
        The loss is the photon energy.  The soft fraction of the tabulated
        average loss ``loss_brem_spec`` is taken from the photon spectrum,
        because the spectra's means sit about 1.6 % below that table; the
        unrestricted limit therefore reproduces the table.
    electroionization (each subshell)
        The primary loses the binding energy ``U`` plus the knock-on energy
        ``W`` from ``spec_<subshell>``.  ``U`` is the threshold of
        ``xs_<subshell>`` (its first tabulated energy).  Collisions are soft
        when ``U + W <= E - Ec(E)``; the paper starts the catastrophic range
        at ``W = E - Ec(E)``, which leaves ``U`` out.

    Between the tabulated incident energies the two neighbouring spectra are
    mixed with weights linear in ``ln E``, and within a spectrum the
    probability is linear in the energy loss.  The file states weights linear
    in ``E``, but EEDL tabulates only a handful of incident energies per
    subshell, and that reading puts the unrestricted stopping power of iron
    about 5 % below the Bethe formula at 1-3 MeV, against about 1 % with
    ``ln E`` weights.

    Mixing in the higher-energy spectrum would give knock-ons more than the
    electron has, so the knock-on energy is capped at ``(E - U) / 2`` (the
    knock-on is the slower electron) and the photon energy at ``E``.  With the
    cap, the unrestricted total (infinite cutoff) is a further 2-5 % lower at
    0.1-1 MeV; the soft value is unchanged wherever ``E - Ec`` is below
    ``(E - U) / 2``.

    Parameters
    ----------
    dataset : EEDLDataset
        Parsed EEDL data.
    structure : GroupStructure
        Group structure defining the cutoff.
    cutoff : CutoffModel | str, optional
        Cutoff convention.
    energy : numpy.ndarray, optional
        Energies to evaluate at, eV.  Defaults to the ``xs_tot`` grid.

    Returns
    -------
    energy : numpy.ndarray
        The energies, eV, shape ``(N,)``.
    stopping_power : numpy.ndarray
        Soft stopping power at those energies, eV*barns, shape ``(N,)``.

    Raises
    ------
    ValidationError
        If *dataset* has no ``xs_tot`` table (and no *energy* is given), has
        ``xs_brem`` without bremsstrahlung spectra or ``loss_brem_spec``, has
        a subshell spectrum without its cross section, or *cutoff* is
        unknown.
    """
    grid = _evaluation_grid(dataset, energy, "pointwise_stopping_power")
    cutoff = _coerce_cutoff(cutoff)
    return grid.copy(), _soft_loss_moment(dataset, structure, cutoff, 1, grid)


def pointwise_straggling(
    dataset: EEDLDataset,
    structure: GroupStructure,
    cutoff: CutoffModel | str = CutoffModel.RADIANT,
    energy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft energy straggling at each energy: the mean squared energy loss

    ``Omega(E) = sum_x integral (E - E')^2 dsigma_x`` over soft collisions, in
    barns*eV^2, with the same channels, cutoff and interpolation as
    :func:`pointwise_stopping_power`.  The paper has no energy straggling, so
    this follows the stopping-power definition with the squared energy loss
    and no factor 1/2.  Bremsstrahlung uses the photon spectrum directly,
    since only the mean loss is tabulated separately.

    Parameters and return values are those of
    :func:`pointwise_stopping_power`; the straggling is in barns*eV^2.
    """
    grid = _evaluation_grid(dataset, energy, "pointwise_straggling")
    cutoff = _coerce_cutoff(cutoff)
    return grid.copy(), _soft_loss_moment(dataset, structure, cutoff, 2, grid)


def _evaluation_grid(dataset: EEDLDataset, energy, owner: str) -> np.ndarray:
    """Requested energies, or by default the total cross-section grid"""
    if energy is not None:
        return np.asarray(energy, dtype="f8")
    if "xs_tot" not in dataset.cross_sections:
        raise ValidationError(f"{owner}: dataset has no 'xs_tot' to define the grid.")
    return np.asarray(dataset.cross_sections["xs_tot"].energy, dtype="f8")


def _coerce_cutoff(cutoff: CutoffModel | str) -> CutoffModel:
    """Cutoff convention from its name"""
    try:
        return CutoffModel(cutoff)
    except ValueError:
        valid = ", ".join(repr(str(m)) for m in CutoffModel)
        raise ValidationError(
            f"cutoff must be one of {valid} (got {cutoff!r})."
        ) from None


# ---------------------------------------------------------------------------
# Catastrophic collisions: totals and group-to-group matrices at each energy
# ---------------------------------------------------------------------------

_MEC2 = 510998.95  # electron rest energy, eV
_BIN_NODES = np.polynomial.legendre.leggauss(6)


def _scattering_cosine(energy: float, energy_out: np.ndarray) -> np.ndarray:
    """Cosine of the deflection of an electron that leaves with *energy_out*

    From the kinematics of an ionization collision of an electron of energy
    ``E``, ``mu = sqrt(E' (E + 2) / (E (E' + 2)))`` with energies in units of
    the electron rest energy; valid for the scattered primary and the knock-on.
    """
    energy_mec2 = energy / _MEC2
    energy_out_mec2 = np.maximum(energy_out, 0.0) / _MEC2
    return np.sqrt(
        energy_out_mec2 * (energy_mec2 + 2.0) / (energy_mec2 * (energy_out_mec2 + 2.0))
    )


def _primary_cosine(energy: float, binding: float, knock_on_energy: np.ndarray):
    """Cosine of the deflection of the primary that ionizes and frees a knock-on"""
    return _scattering_cosine(energy, energy - binding - knock_on_energy)


def _bin_integrals(
    transfer, density, bin_edges, order, cosine, limit=np.inf
) -> np.ndarray:
    """Legendre moments of the deflection, per energy bin, of a spectrum

    The spectrum ``density(transfer)`` of the energy given to the secondary
    particle (knock-on electron or photon) is linear between table points and
    zero outside its table and above *limit*.  Integrates
    ``density P_l(mu)`` over each bin between the ascending *bin_edges* in
    that energy, for ``l = 0..order``, where *cosine* gives the deflection
    cosine ``mu`` of the electron.  Returns shape ``(len(bin_edges) - 1, order + 1)``.
    """
    moments = np.zeros((len(bin_edges) - 1, order + 1))
    breakpoints = np.concatenate([transfer, [limit]])
    breakpoints = breakpoints[
        (breakpoints > bin_edges[0]) & (breakpoints < bin_edges[-1])
    ]
    segment_edges = np.unique(np.concatenate([bin_edges, breakpoints]))
    segment_lo, segment_hi = segment_edges[:-1], segment_edges[1:]
    in_table = (segment_hi > transfer[0]) & (segment_lo < min(transfer[-1], limit))
    if not np.any(in_table):
        return moments
    segment_lo, segment_hi = segment_lo[in_table], segment_hi[in_table]
    nodes, weights = _BIN_NODES
    half_width = 0.5 * (segment_hi - segment_lo)[:, None]
    midpoint = 0.5 * (segment_lo + segment_hi)
    node_energy = midpoint[:, None] + half_width * nodes[None, :]
    node_density = np.interp(node_energy, transfer, density, left=0.0, right=0.0)
    mu = np.clip(cosine(node_energy), -1.0, 1.0)
    legendre = np.polynomial.legendre.legvander(mu.ravel(), order).reshape(
        node_energy.shape + (order + 1,)
    )
    weighted_density = half_width * weights[None, :] * node_density
    segment_moments = (weighted_density[..., None] * legendre).sum(axis=1)
    bin_of_segment = np.searchsorted(bin_edges, midpoint, side="right") - 1
    np.add.at(moments, bin_of_segment, segment_moments)
    return moments


def _catastrophic_ionization(dataset, structure, cutoff, energy, order):
    """Catastrophic electroionization cross sections at each energy

    Returns the total cross section, shape ``(Nq,)``, and the group-to-group
    Legendre scattering matrices, shape ``(Nq, L+1, G)``.

    The scattered primary (energy ``E' = E - U - W``) is catastrophic when
    ``E' < Ec(E)``; every knock-on with ``W`` in the target group is produced,
    whatever the primary does.  The deflections follow from the kinematics of
    the collision (:func:`_scattering_cosine`).
    """
    owner = "catastrophic ionization"
    n_groups = structure.n_groups
    bounds = structure.energy
    energy_cutoff = cutoff_energy(energy, structure, cutoff)
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, n_groups))
    for knock_on_spectra, binding, xs in _ionization_subshells(dataset, owner):
        sigma = _interp_table(energy, xs.energy, xs.cross_section)
        knock_on_limit = np.maximum(energy - binding, 0.0) / 2.0
        probability_all, _, _ = _mixture_moments(
            knock_on_spectra, energy, knock_on_limit
        )
        probability_soft, _, _ = _mixture_moments(
            knock_on_spectra,
            energy,
            np.minimum(energy - binding - energy_cutoff, knock_on_limit),
        )
        total += sigma * (probability_all - probability_soft)

        incident_energies, spectra = _spectra_by_incident_energy(knock_on_spectra)
        lower, upper, upper_weight = _bracket(incident_energies, energy)
        for node in np.flatnonzero((sigma > 0.0) & (energy > binding)):
            incident_energy = energy[node]
            energy_after_binding = incident_energy - binding
            catastrophic_onset = energy_after_binding - energy_cutoff[node]

            primary_cosine = partial(_primary_cosine, incident_energy, binding)
            knock_on_cosine = partial(_scattering_cosine, incident_energy)

            # primary: bins in knock-on energy aligned with the group bounds of the primary
            primary_bin_edges = np.unique(
                np.concatenate(
                    [energy_after_binding - bounds, [catastrophic_onset]]
                ).clip(0.0, None)
            )
            bin_center = 0.5 * (primary_bin_edges[1:] + primary_bin_edges[:-1])
            primary_energy_out = energy_after_binding - bin_center
            catastrophic_bin = (
                (bin_center >= catastrophic_onset)
                & (primary_energy_out > bounds[-1])
                & (primary_energy_out <= bounds[0])
            )
            primary_group = _group_of(structure, primary_energy_out[catastrophic_bin])
            for mix_weight, table in (
                (1.0 - upper_weight[node], lower[node]),
                (upper_weight[node], upper[node]),
            ):
                if mix_weight == 0.0:
                    continue
                knock_on_energy, density = spectra[table]
                primary = _bin_integrals(
                    knock_on_energy,
                    density,
                    primary_bin_edges,
                    order,
                    primary_cosine,
                    knock_on_limit[node],
                )
                # knock-on: bins in knock-on energy are the groups, ascending in energy
                knock_on = _bin_integrals(
                    knock_on_energy,
                    density,
                    bounds[::-1],
                    order,
                    knock_on_cosine,
                    knock_on_limit[node],
                )
                weighted = sigma[node] * mix_weight
                np.add.at(
                    matrix[node].T, primary_group, weighted * primary[catastrophic_bin]
                )
                matrix[node] += weighted * knock_on[::-1].T
    return total, matrix


def _catastrophic_bremsstrahlung(dataset, structure, cutoff, energy, order):
    """Catastrophic bremsstrahlung cross sections at each energy

    Returns the total cross section, shape ``(Nq,)``, and the group-to-group
    Legendre scattering matrices of the electron, shape ``(Nq, L+1, G)``.

    The electron keeps its direction (``mu = 1``), so all Legendre moments of
    the matrix are equal.  A collision is catastrophic when the electron
    leaves below ``Ec(E)``.
    """
    n_groups = structure.n_groups
    bounds = structure.energy
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, n_groups))
    if "xs_brem" not in dataset.cross_sections:
        return total, matrix
    photon_spectra = dataset.bremsstrahlung_spectra
    if photon_spectra is None:
        raise ValidationError(
            "catastrophic bremsstrahlung: 'xs_brem' needs bremsstrahlung spectra."
        )
    xs = dataset.cross_sections["xs_brem"]
    sigma = _interp_table(energy, xs.energy, xs.cross_section)
    energy_cutoff = cutoff_energy(energy, structure, cutoff)
    probability_all, _, _ = _mixture_moments(photon_spectra, energy, energy)
    probability_soft, _, _ = _mixture_moments(
        photon_spectra, energy, energy - energy_cutoff
    )
    total = sigma * (probability_all - probability_soft)

    # The photon takes E - E_electron; target group g holds the electron energies
    # in (e[g+1], min(e[g], Ec)], which fixes the photon energy range per group.
    energy_per_group = np.repeat(energy, n_groups)
    electron_energy_max = np.minimum(bounds[None, :-1], energy_cutoff[:, None])
    photon_energy_min = (energy[:, None] - electron_energy_max).ravel()
    photon_energy_max = (energy[:, None] - bounds[None, 1:]).ravel()
    probability_below_min, _, _ = _mixture_moments(
        photon_spectra, energy_per_group, np.clip(photon_energy_min, 0.0, None)
    )
    probability_below_max, _, _ = _mixture_moments(
        photon_spectra, energy_per_group, np.clip(photon_energy_max, 0.0, None)
    )
    group_probability = np.clip(
        (probability_below_max - probability_below_min).reshape(energy.size, n_groups),
        0.0,
        None,
    )
    group_probability[electron_energy_max <= bounds[None, 1:]] = 0.0
    matrix[:] = (sigma[:, None] * group_probability)[:, None, :]
    return total, matrix


def _catastrophic_excitation(dataset, structure, cutoff, energy, order):
    """Catastrophic excitation cross sections at each energy

    Returns the total cross section, shape ``(Nq,)``, and the group-to-group
    Legendre scattering matrices, shape ``(Nq, L+1, G)``.

    Excitation is an undeflected down-scatter by the average loss
    ``loss_exc``, which the RADIANT paper does not treat.  A collision is
    catastrophic when its loss exceeds ``E - Ec(E)``, which for EEDL's
    tens-of-eV losses only happens in the lowest groups.  The sharp
    soft/catastrophic switch is sampled at the quadrature nodes, so this
    contribution is only as accurate as that sampling.
    """
    cross_sections = dataset.cross_sections
    average_losses = dataset.average_energy_losses
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, structure.n_groups))
    if "xs_exc" not in cross_sections or "loss_exc" not in average_losses:
        return total, matrix
    xs = cross_sections["xs_exc"]
    sigma = _interp_table(energy, xs.energy, xs.cross_section)
    loss_table = average_losses["loss_exc"]
    loss = _interp_table(energy, loss_table.energy, loss_table.avg_loss)
    catastrophic = loss > energy - cutoff_energy(energy, structure, cutoff)
    total = np.where(catastrophic, sigma, 0.0)
    energy_out = energy - loss
    lands_in_structure = (
        catastrophic
        & (energy_out > structure.energy[-1])
        & (energy_out <= structure.energy[0])
    )
    nodes = np.flatnonzero(lands_in_structure)
    matrix[nodes, :, _group_of(structure, energy_out[nodes])] = sigma[nodes, None]
    return total, matrix


# ---------------------------------------------------------------------------
# Transport correction and assembly
# ---------------------------------------------------------------------------


def landesman_morel_correction(
    total: np.ndarray, elastic_diagonal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Landesman-Morel transport correction with momentum transfer

    Removes the forward-peaked part of elastic scattering from the Boltzmann
    operator and hands it to the Fokker-Planck operator as a soft momentum
    transfer.  For each group, with ``S[l] = Sigma_{s,l,g->g}`` the
    within-group elastic moments and ``L`` the last non-zero one::

        T_g       = (S[L-1] - S[L]) / (2 L)
        total'    = total - S[L] - T_g L (L + 1)
        S'[l]     = S[l] - S[L] - T_g (L (L + 1) - l (l + 1))

    Groups with ``L < 1`` (isotropic) are left alone with ``T_g = 0``.  A
    negative ``T_g`` (non-monotone moments) is set to 0, which leaves the plain
    extended transport correction.

    Parameters
    ----------
    total : numpy.ndarray
        Group total cross section, shape ``(G,)``.
    elastic_diagonal : numpy.ndarray
        Within-group elastic Legendre cross sections, shape ``(G, L+1)``.

    Returns
    -------
    total_corrected : numpy.ndarray
        Shape ``(G,)``.
    diagonal_corrected : numpy.ndarray
        Shape ``(G, L+1)``; entries above each group's ``L`` stay zero.
    momentum_transfer : numpy.ndarray
        ``T_g``, shape ``(G,)``.
    """
    n_groups, n_moments = elastic_diagonal.shape
    total_corrected = total.copy()
    diagonal_corrected = elastic_diagonal.copy()
    momentum_transfer = np.zeros(n_groups)
    orders = np.arange(n_moments)
    for group in range(n_groups):
        moments = elastic_diagonal[group]
        nonzero = np.flatnonzero(moments)
        if nonzero.size == 0 or nonzero[-1] < 1:
            continue
        last = int(nonzero[-1])
        transfer = max((moments[last - 1] - moments[last]) / (2.0 * last), 0.0)
        total_corrected[group] = (
            total[group] - moments[last] - transfer * last * (last + 1)
        )
        corrected = (
            moments
            - moments[last]
            - transfer * (last * (last + 1) - orders * (orders + 1))
        )
        diagonal_corrected[group, : last + 1] = corrected[: last + 1]
        momentum_transfer[group] = transfer
    return total_corrected, diagonal_corrected, momentum_transfer


@dataclass
class BFPData:
    """Multigroup BFP data produced by :func:`generate`, ready for ``ParticleMG``

    Parameters
    ----------
    coefficients : BFPCoefficients
        Stopping power and straggling per group and at the group bounds.
    cross_sections : dict[str, GroupXS]
        ``"xs_tot"`` (transport-corrected total, barns) and
        ``"momentum_transfer"`` (``T_g``, barns).
    scattering : dict[str, ScatteringMatrix]
        Catastrophic electron-to-electron matrices by interaction:
        ``"elastic"`` (transport-corrected), ``"ionization"``,
        ``"bremsstrahlung"`` and, if non-zero, ``"excitation"``.
    """

    coefficients: BFPCoefficients
    cross_sections: dict[str, GroupXS]
    scattering: dict[str, ScatteringMatrix]


def generate(
    dataset: EEDLDataset,
    structure: GroupStructure,
    weight: WeightingSpectrum | str,
    cutoff: CutoffModel | str = CutoffModel.RADIANT,
    legendre_order: int = 15,
) -> BFPData:
    """Multigroup Boltzmann-Fokker-Planck electron data from EEDL

    Averages over each group of *structure*, weighted by the flux spectrum
    *weight* (constant flux gives the group averages of Bienvenue et al.,
    J. Comput. Phys. 524 (2025) 113740), and builds:

    * the soft stopping power and energy straggling, per group and at the
      group bounds (:func:`pointwise_stopping_power`,
      :func:`pointwise_straggling`);
    * the catastrophic scattering matrices of elastic scattering,
      electroionization (scattered primary and knock-on), bremsstrahlung
      (deflected electron) and excitation;
    * the total cross section and the momentum transfer after the
      Landesman-Morel transport correction (:func:`landesman_morel_correction`).

    The momentum transfer is the elastic one only: bremsstrahlung leaves the
    electron undeflected (``T = 0``) and the paper neglects it for ionization.

    Parameters
    ----------
    dataset : EEDLDataset
        Parsed EEDL data.
    structure : GroupStructure
        Electron group structure.
    weight : WeightingSpectrum | str
        Weighting spectrum or the name of a pre-defined one.
    cutoff : CutoffModel | str, optional
        Soft/catastrophic convention.
    legendre_order : int, optional
        Highest Legendre order ``L`` of the matrices (RADIANT uses 15).

    Returns
    -------
    BFPData
        Soft coefficients, corrected total and momentum-transfer cross
        sections, and the catastrophic scattering matrices.

    Raises
    ------
    ValidationError
        If *legendre_order* is below 1, *cutoff* is unknown, or the dataset
        lacks the tables the channels need (see
        :func:`pointwise_stopping_power`).
    """
    cutoff = _coerce_cutoff(cutoff)
    if legendre_order < 1:
        raise ValidationError("generate: legendre_order must be at least 1.")
    pointwise_energies = _evaluation_grid(dataset, None, "generate")
    quadrature = group_quadrature(structure, pointwise_energies)
    node_energy = quadrature.energy

    def group_average(values):
        return collapse_nodes(values, quadrature, weight)

    # Soft stopping power and straggling: averaged over each group and at the group bounds
    bounds = structure.energy
    coefficients = BFPCoefficients(
        stopping_power=group_average(
            _soft_loss_moment(dataset, structure, cutoff, 1, node_energy)
        ),
        boundary_stopping_power=_soft_loss_moment(
            dataset, structure, cutoff, 1, bounds
        ),
        straggling=group_average(
            _soft_loss_moment(dataset, structure, cutoff, 2, node_energy)
        ),
        boundary_straggling=_soft_loss_moment(dataset, structure, cutoff, 2, bounds),
    )

    # Elastic scattering stays within each group; its total is its zeroth moment.
    elastic = elastic_legendre_cross_sections(dataset, node_energy, legendre_order)
    elastic_diagonal = group_average(elastic)

    total = elastic[:, 0].copy()
    catastrophic_matrices = {}
    for label, build_catastrophic in (
        ("ionization", _catastrophic_ionization),
        ("bremsstrahlung", _catastrophic_bremsstrahlung),
        ("excitation", _catastrophic_excitation),
    ):
        catastrophic_total, catastrophic_matrix = build_catastrophic(
            dataset, structure, cutoff, node_energy, legendre_order
        )
        total += catastrophic_total
        if np.any(catastrophic_matrix):
            # group-averaged over the source energy: (G_from, L+1, G_to) -> (L+1, G_from, G_to)
            catastrophic_matrices[label] = group_average(catastrophic_matrix).transpose(
                1, 0, 2
            )

    total_corrected, diagonal_corrected, momentum_transfer = landesman_morel_correction(
        group_average(total), elastic_diagonal
    )
    n_groups = structure.n_groups
    elastic_matrix = np.zeros((legendre_order + 1, n_groups, n_groups))
    elastic_matrix[:, np.arange(n_groups), np.arange(n_groups)] = diagonal_corrected.T

    scattering = {"elastic": ScatteringMatrix("elastic", elastic_matrix)}
    for label, matrix in catastrophic_matrices.items():
        scattering[label] = ScatteringMatrix(label, matrix)
    return BFPData(
        coefficients=coefficients,
        cross_sections={
            "xs_tot": GroupXS("xs_tot", total_corrected),
            MOMENTUM_TRANSFER_LABEL: GroupXS(
                MOMENTUM_TRANSFER_LABEL, momentum_transfer
            ),
        },
        scattering=scattering,
    )
