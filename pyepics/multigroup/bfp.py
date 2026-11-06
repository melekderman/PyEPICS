#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Calculation of Bolzmann Fokker-Planck (BFP) coefficients from EEDL data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
    RADIANT = "RADIANT"
    CEPXS = "CEPXS"


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
        (Bienvenue et al., J. Comput. Phys. 524 (2025) 113740, Eq. 10).
        Continuous across group bounds.
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
    e = np.asarray(energy, dtype="f8")
    bounds = structure.energy
    n_groups = structure.n_groups
    padded = np.append(bounds, 0.0)  # e[G + 1] = 0

    # Group index g with e[g + 1] < E <= e[g]; same rule as GroupStructure.group_of
    g = bounds.size - 1 - np.searchsorted(bounds[::-1], e, side="left")
    g = np.clip(g, 0, n_groups - 1)

    if cutoff == CutoffModel.CEPXS:
        ec = padded[g + 2]
    else:
        slope = (padded[g + 1] - padded[g + 2]) / (padded[g] - padded[g + 1])
        ec = padded[g + 1] + slope * (e - padded[g])
    ec = np.where(e < bounds[-1], 0.0, ec)
    return np.clip(ec, 0.0, e)


_EXCITATION = "excitation"


def _bracket(incident: np.ndarray, energy: np.ndarray):
    """Neighbouring tabulated incident energies and the mixing weight

    Returns ``(lo, hi, t)`` with ``t = ln(E / E_lo) / ln(E_hi / E_lo)`` clipped
    to ``[0, 1]``: outside the tabulated range the nearest spectrum is used.
    """
    n = incident.size
    lo = np.clip(np.searchsorted(incident, energy, side="right") - 1, 0, n - 1)
    hi = np.minimum(lo + 1, n - 1)
    span = np.log(incident[hi] / incident[lo])
    safe = np.maximum(energy, np.finfo("f8").tiny)
    t = np.divide(
        np.log(safe / incident[lo]), span, out=np.zeros_like(energy), where=span > 0.0
    )
    return lo, hi, np.clip(t, 0.0, 1.0)


def _spectra(record: DistributionRecord):
    """Incident energies and the ``(w, p)`` table at each, sorted by ``w``"""
    incident = np.unique(record.inc_energy)
    tables = []
    for e_in in incident:
        mask = record.inc_energy == e_in
        order = np.argsort(record.value[mask], kind="stable")
        tables.append((record.value[mask][order], record.probability[mask][order]))
    return incident, tables


def _partial_moments(
    w: np.ndarray, p: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``integral w^n p dw`` for ``n = 0, 1, 2`` over ``w <= upper``

    ``p`` is piecewise linear in ``w`` and zero outside ``[w[0], w[-1]]``;
    *upper* is an array of limits (``inf`` allowed).
    """
    w0, w1 = w[:-1][None, :], w[1:][None, :]
    dw = (w1 - w0)[0]
    slope = np.divide(p[1:] - p[:-1], dw, out=np.zeros_like(dw), where=dw > 0.0)[
        None, :
    ]
    # distance integrated into each interval, for every limit: shape (n, intervals)
    d = np.clip(np.minimum(w1, upper[:, None]) - w0, 0.0, None)
    p0 = p[:-1][None, :]
    zeroth = p0 * d + slope * d**2 / 2.0
    first = w0 * p0 * d + (w0 * slope + p0) * d**2 / 2.0 + slope * d**3 / 3.0
    second = (
        w0**2 * p0 * d
        + (w0**2 * slope + 2.0 * w0 * p0) * d**2 / 2.0
        + (2.0 * w0 * slope + p0) * d**3 / 3.0
        + slope * d**4 / 4.0
    )
    return zeroth.sum(axis=1), first.sum(axis=1), second.sum(axis=1)


def _mixture_moments(
    record: DistributionRecord, energy: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partial moments of the spectrum at each incident *energy*

    The spectrum at ``E`` is the mix ``(1 - t) p_a + t p_b`` of the tabulated
    spectra at the incident energies ``E_a <= E <= E_b`` around it, with
    ``t = ln(E / E_a) / ln(E_b / E_a)``; outside the tabulated range the
    nearest spectrum is used.  Returns ``(integral p dw, integral w p dw,
    integral w^2 p dw)`` over ``w <= upper`` for each entry of *energy*.
    """
    incident, tables = _spectra(record)
    moments = np.empty((3, incident.size, energy.size))
    for i, (w, p) in enumerate(tables):
        moments[:, i] = _partial_moments(w, p, upper)
    lo, hi, t = _bracket(incident, energy)
    cols = np.arange(energy.size)
    mixed = (1.0 - t) * moments[:, lo, cols] + t * moments[:, hi, cols]
    return mixed[0], mixed[1], mixed[2]


def _soft_loss_moment(
    dataset: EEDLDataset,
    structure: GroupStructure,
    cutoff: CutoffModel,
    order: int,
    energy: np.ndarray,
) -> np.ndarray:
    """``sum_x integral over soft collisions of (energy loss)^order dsigma``"""
    owner = "soft energy-loss moment"
    xs = dataset.cross_sections
    losses = dataset.average_energy_losses
    loss_cut = energy - cutoff_energy(energy, structure, cutoff)  # largest soft loss

    def on_grid(record_energy, values):  # lin-lin (INT = 2), zero outside the table
        return np.interp(energy, record_energy, values, left=0.0, right=0.0)

    total = np.zeros_like(energy)

    # Excitation
    if "xs_exc" in xs and "loss_exc" in losses:
        sigma = on_grid(xs["xs_exc"].energy, xs["xs_exc"].cross_section)
        loss = on_grid(losses["loss_exc"].energy, losses["loss_exc"].avg_loss)
        total += np.where(loss <= loss_cut, sigma * loss**order, 0.0)

    # Bremsstrahlung
    if "xs_brem" in xs:
        spectra = dataset.bremsstrahlung_spectra
        if spectra is None or "loss_brem_spec" not in losses:
            raise ValidationError(
                f"{owner}: 'xs_brem' needs bremsstrahlung spectra and 'loss_brem_spec'."
            )
        sigma = on_grid(xs["xs_brem"].energy, xs["xs_brem"].cross_section)
        _, first_soft, second_soft = _mixture_moments(spectra, energy, loss_cut)
        if order == 1:
            _, first_all, _ = _mixture_moments(spectra, energy, energy)
            mean_loss = on_grid(
                losses["loss_brem_spec"].energy, losses["loss_brem_spec"].avg_loss
            )
            fraction = np.divide(
                first_soft,
                first_all,
                out=np.zeros_like(first_soft),
                where=first_all > 0.0,
            )
            total += sigma * mean_loss * fraction
        else:
            total += sigma * second_soft

    # Electroionization, one subshell at a time
    for name, spectrum in dataset.distributions.items():
        if not name.startswith("spec_"):
            continue
        xs_name = "xs_" + name[len("spec_") :]
        if xs_name not in xs:
            raise ValidationError(
                f"{owner}: '{name}' has no cross section '{xs_name}'."
            )
        binding = float(xs[xs_name].energy[0])
        sigma = on_grid(xs[xs_name].energy, xs[xs_name].cross_section)
        limit = (
            np.maximum(energy - binding, 0.0) / 2.0
        )  # knock-on is the slower electron
        prob, first, second = _mixture_moments(
            spectrum, energy, np.minimum(loss_cut - binding, limit)
        )
        if order == 1:
            total += sigma * (binding * prob + first)
        else:
            total += sigma * (binding**2 * prob + 2.0 * binding * first + second)
    return total


def pointwise_stopping_power(
    dataset: EEDLDataset,
    structure: GroupStructure,
    cutoff: CutoffModel | str = CutoffModel.RADIANT,
    energy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise soft (restricted) electron stopping power from EEDL data

    This is the first moment of the energy loss over soft collisions,
    ``S(E) = sum_x  integral_{loss <= E - Ec(E)} (E - E') dsigma_x``, summed
    over the inelastic channels ``x`` (Beling 2024, Eq. 3.12, with
    ``F_{1,0}`` the first energy-loss moment and the zeroth angular moment;
    the angular integral is trivial because EEDL gives the inelastic cross
    sections already integrated over angle).  Equivalent to Eqs. 45 and 81
    of Bienvenue et al. (J. Comput. Phys. 524 (2025) 113740), the total
    stopping power minus the catastrophic part, with the totals taken from
    EEDL.  See :func:`cutoff_energy` for ``Ec(E)``.

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
        The primary loses ``U + W``, with ``W`` the knock-on energy from
        ``spec_<subshell>`` and ``U`` the binding energy, taken as the
        threshold of ``xs_<subshell>`` (its first tabulated energy).  The
        soft collisions are those with ``U + W <= E - Ec(E)``.  RADIANT's
        Eq. 45 starts the catastrophic range at ``W = E - Ec(E)``, which
        leaves ``U`` out.
    elastic
        No energy loss, so it contributes nothing (EEDL's angular cutoff
        ``1 - 1e-6`` matters for the momentum transfer, not here).

    Between the tabulated incident energies the two neighbouring spectra are
    mixed with weights linear in ``ln E``, and within a spectrum ``p`` is
    linear in ``E'`` (``LEP = 2``).  Mixing in the higher-energy spectrum
    would give knock-ons more than the electron has, so the knock-on energy
    is capped at ``(E - U) / 2`` (the EEDL convention: the knock-on is the
    slower electron) and the photon energy at ``E``.  The file states ``INT = 2`` (weights
    linear in ``E``), but EEDL tabulates only a handful of incident energies
    per subshell and that reading puts the unrestricted stopping power of
    iron about 5 % below the Bethe formula at 1-3 MeV, against about 1 % with
    ``ln E`` weights.  With the cap, the unrestricted total (infinite cutoff)
    is a further 2-5 % lower at 0.1-1 MeV; the soft value is unchanged
    wherever the cutoff ``E - Ec`` is below ``(E - U) / 2``.

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
    """Pointwise soft energy straggling: the second energy-loss moment

    ``Omega(E) = sum_x integral_{loss <= E - Ec(E)} (E - E')^2 dsigma_x``,
    in barns*eV^2, over the same channels, cutoff and interpolation as
    :func:`pointwise_stopping_power` (``F_{2,0}`` in the notation of its
    reference).  The paper has no energy straggling, so this follows the
    stopping-power definition with ``(E - E')^2`` for ``E - E'``, with no
    factor 1/2.  Bremsstrahlung uses the photon spectrum directly, since only
    the first moment is tabulated separately.
    """
    grid = _evaluation_grid(dataset, energy, "pointwise_straggling")
    cutoff = _coerce_cutoff(cutoff)
    return grid.copy(), _soft_loss_moment(dataset, structure, cutoff, 2, grid)


def _evaluation_grid(dataset: EEDLDataset, energy, owner: str) -> np.ndarray:
    if energy is not None:
        return np.asarray(energy, dtype="f8")
    if "xs_tot" not in dataset.cross_sections:
        raise ValidationError(f"{owner}: dataset has no 'xs_tot' to define the grid.")
    return np.asarray(dataset.cross_sections["xs_tot"].energy, dtype="f8")


def _coerce_cutoff(cutoff: CutoffModel | str) -> CutoffModel:
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
    """Cosine of the deflection of an electron of *energy_out* (Eq. 25)

    ``mu = sqrt(E' (E + 2) / (E (E' + 2)))`` with energies in units of the
    electron rest energy; used for the scattered primary and the knock-on.
    """
    e = energy / _MEC2
    out = np.maximum(energy_out, 0.0) / _MEC2
    return np.sqrt(out * (e + 2.0) / (e * (out + 2.0)))


def _bin_integrals(w, p, edges, order, cosine, limit=np.inf) -> np.ndarray:
    """``integral p(w) P_l(mu(w)) dw`` over each bin between ascending *edges*

    ``p`` is piecewise linear in ``w`` (zero outside its table and above
    *limit*) and ``cosine`` maps ``w`` to ``mu``.  Returns shape
    ``(len(edges) - 1, order + 1)``.
    """
    out = np.zeros((len(edges) - 1, order + 1))
    inside = np.concatenate([w, [limit]])
    inside = inside[(inside > edges[0]) & (inside < edges[-1])]
    points = np.unique(np.concatenate([edges, inside]))
    a, b = points[:-1], points[1:]
    keep = (b > w[0]) & (a < min(w[-1], limit))
    if not np.any(keep):
        return out
    a, b = a[keep], b[keep]
    nodes, weights = _BIN_NODES
    half = 0.5 * (b - a)[:, None]
    x = 0.5 * (a + b)[:, None] + half * nodes[None, :]
    density = np.interp(x, w, p, left=0.0, right=0.0)
    mu = np.clip(cosine(x), -1.0, 1.0)
    legendre = np.polynomial.legendre.legvander(mu.ravel(), order).reshape(
        x.shape + (order + 1,)
    )
    segment = ((half * weights[None, :] * density)[..., None] * legendre).sum(axis=1)
    np.add.at(out, np.searchsorted(edges, 0.5 * (a + b), side="right") - 1, segment)
    return out


def _group_of(structure: GroupStructure, energy: np.ndarray) -> np.ndarray:
    """Group index ``g`` with ``e[g + 1] < E <= e[g]`` (the ``GroupStructure.group_of`` rule)"""
    bounds = structure.energy
    return bounds.size - 1 - np.searchsorted(bounds[::-1], energy, side="left")


def _catastrophic_ionization(dataset, structure, cutoff, energy, order):
    """Catastrophic electroionization: totals ``(Nq,)`` and matrices ``(Nq, L+1, G)``

    The scattered primary (energy ``E' = E - U - W``) is catastrophic when
    ``E' < Ec(E)`` (Eq. 29); every knock-on with ``W`` in the target group is
    produced, whatever the primary does (Eq. 32).  Cosines come from the
    kinematics of Eq. 25.
    """
    n_groups = structure.n_groups
    bounds = structure.energy
    ec = cutoff_energy(energy, structure, cutoff)
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, n_groups))
    xs = dataset.cross_sections
    for name, record in dataset.distributions.items():
        if not name.startswith("spec_"):
            continue
        xs_name = "xs_" + name[len("spec_") :]
        if xs_name not in xs:
            raise ValidationError(
                f"catastrophic ionization: '{name}' has no cross section '{xs_name}'."
            )
        binding = float(xs[xs_name].energy[0])
        sigma = np.interp(
            energy, xs[xs_name].energy, xs[xs_name].cross_section, left=0.0, right=0.0
        )
        limit = (
            np.maximum(energy - binding, 0.0) / 2.0
        )  # knock-on is the slower electron
        all_prob, _, _ = _mixture_moments(record, energy, limit)
        soft_prob, _, _ = _mixture_moments(
            record, energy, np.minimum(energy - binding - ec, limit)
        )
        total += sigma * (all_prob - soft_prob)

        incident, tables = _spectra(record)
        lo, hi, t = _bracket(incident, energy)
        for q in np.flatnonzero((sigma > 0.0) & (energy > binding)):
            e_in = energy[q]
            available = e_in - binding
            # primary: bins in W aligned with the target groups' bounds in E'
            cut = available - ec[q]
            edges_primary = np.unique(
                np.concatenate([available - bounds, [cut]]).clip(0.0, None)
            )
            mid = 0.5 * (edges_primary[1:] + edges_primary[:-1])
            e_primary = available - mid
            usable = (mid >= cut) & (e_primary > bounds[-1]) & (e_primary <= bounds[0])
            target = _group_of(structure, e_primary[usable])
            for weight, index in ((1.0 - t[q], lo[q]), (t[q], hi[q])):
                if weight == 0.0:
                    continue
                w, p = tables[index]
                primary = _bin_integrals(
                    w,
                    p,
                    edges_primary,
                    order,
                    lambda x, e_in=e_in, available=available: _scattering_cosine(
                        e_in, available - x
                    ),
                    limit[q],
                )
                # knock-on: bins in W are the groups themselves, ascending in energy
                knock = _bin_integrals(
                    w,
                    p,
                    bounds[::-1],
                    order,
                    lambda x, e_in=e_in: _scattering_cosine(e_in, x),
                    limit[q],
                )
                np.add.at(matrix[q].T, target, sigma[q] * weight * primary[usable])
                matrix[q] += sigma[q] * weight * knock[::-1].T
    return total, matrix


def _catastrophic_bremsstrahlung(dataset, structure, cutoff, energy, order):
    """Catastrophic bremsstrahlung: totals and matrices for the deflected electron

    The electron keeps its direction (``mu = 1``, Eq. 70), so all Legendre
    moments of the group-to-group matrix are equal.  A collision is
    catastrophic when the electron leaves below ``Ec(E)`` (Eq. 78).
    """
    n_groups = structure.n_groups
    bounds = structure.energy
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, n_groups))
    xs = dataset.cross_sections
    if "xs_brem" not in xs:
        return total, matrix
    spectra = dataset.bremsstrahlung_spectra
    if spectra is None:
        raise ValidationError(
            "catastrophic bremsstrahlung: 'xs_brem' needs bremsstrahlung spectra."
        )
    sigma = np.interp(
        energy, xs["xs_brem"].energy, xs["xs_brem"].cross_section, left=0.0, right=0.0
    )
    ec = cutoff_energy(energy, structure, cutoff)
    all_prob, _, _ = _mixture_moments(spectra, energy, energy)
    soft_prob, _, _ = _mixture_moments(spectra, energy, energy - ec)
    total = sigma * (all_prob - soft_prob)

    # photon energy k = E - E_e; target group g holds E_e in (e[g+1], min(e[g], Ec)]
    rep = np.repeat(energy, n_groups)
    top = np.minimum(bounds[None, :-1], ec[:, None])
    k_low = (energy[:, None] - top).ravel()
    k_high = (energy[:, None] - bounds[None, 1:]).ravel()
    cdf_low, _, _ = _mixture_moments(spectra, rep, np.clip(k_low, 0.0, None))
    cdf_high, _, _ = _mixture_moments(spectra, rep, np.clip(k_high, 0.0, None))
    fraction = np.clip((cdf_high - cdf_low).reshape(energy.size, n_groups), 0.0, None)
    fraction[top <= bounds[None, 1:]] = 0.0
    matrix[:] = (sigma[:, None] * fraction)[:, None, :]
    return total, matrix


def _catastrophic_excitation(dataset, structure, cutoff, energy, order):
    """Catastrophic excitation: an undeflected down-scatter by ``loss_exc``

    Not in the RADIANT paper.  A collision is catastrophic when its loss
    exceeds ``E - Ec(E)``, which for EEDL's tens-of-eV losses only happens
    for the lowest groups.  The indicator is evaluated at the quadrature
    nodes, so this contribution is only as accurate as that sampling.
    """
    n_groups = structure.n_groups
    total = np.zeros(energy.size)
    matrix = np.zeros((energy.size, order + 1, n_groups))
    xs = dataset.cross_sections
    losses = dataset.average_energy_losses
    if "xs_exc" not in xs or "loss_exc" not in losses:
        return total, matrix
    sigma = np.interp(
        energy, xs["xs_exc"].energy, xs["xs_exc"].cross_section, left=0.0, right=0.0
    )
    loss = np.interp(
        energy,
        losses["loss_exc"].energy,
        losses["loss_exc"].avg_loss,
        left=0.0,
        right=0.0,
    )
    catastrophic = loss > energy - cutoff_energy(energy, structure, cutoff)
    total = np.where(catastrophic, sigma, 0.0)
    e_out = energy - loss
    inside = (
        catastrophic & (e_out > structure.energy[-1]) & (e_out <= structure.energy[0])
    )
    rows = np.flatnonzero(inside)
    matrix[rows, :, _group_of(structure, e_out[rows])] = sigma[rows, None]
    return total, matrix


# ---------------------------------------------------------------------------
# Transport correction and assembly
# ---------------------------------------------------------------------------


def landesman_morel_correction(
    total: np.ndarray, elastic_diagonal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extended transport correction with momentum transfer (Eqs. 62-66)

    For each group, with ``S[l] = Sigma_{s,l,g->g}`` the within-group elastic
    moments and ``L`` the last non-zero one::

        T_g       = (S[L-1] - S[L]) / (2 L)                          (63)
        total'    = total - S[L] - T_g L (L + 1)                     (65)
        S'[l]     = S[l] - S[L] - T_g (L (L + 1) - l (l + 1))        (66)

    ``T_g`` is the soft momentum transfer handed to the Fokker-Planck
    operator.  Groups with ``L < 1`` (isotropic) are left alone with
    ``T_g = 0``.  A negative ``T_g`` (non-monotone moments) is set to 0,
    which leaves the plain extended transport correction.

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
    n_groups, n_orders = elastic_diagonal.shape
    total_out = total.copy()
    diagonal = elastic_diagonal.copy()
    momentum = np.zeros(n_groups)
    ell = np.arange(n_orders)
    for g in range(n_groups):
        nonzero = np.flatnonzero(elastic_diagonal[g])
        if nonzero.size == 0 or nonzero[-1] < 1:
            continue
        last = int(nonzero[-1])
        moments = elastic_diagonal[g]
        t_g = max((moments[last - 1] - moments[last]) / (2.0 * last), 0.0)
        total_out[g] = total[g] - moments[last] - t_g * last * (last + 1)
        corrected = (
            moments - moments[last] - t_g * (last * (last + 1) - ell * (ell + 1))
        )
        diagonal[g, : last + 1] = corrected[: last + 1]
        momentum[g] = t_g
    return total_out, diagonal, momentum


@dataclass
class BFPData:
    """Multigroup output of :func:`generate`, ready for ``ParticleMG``

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
    """Multigroup Boltzmann-Fokker-Planck data for electrons from EEDL

    Builds, on *structure* and with the weighting *weight* (Eqs. 4-9 of
    Bienvenue et al., J. Comput. Phys. 524 (2025) 113740, with ``w = 1``
    their constant-flux average):

    * the soft stopping power and energy straggling, per group and at the
      group bounds (:func:`pointwise_stopping_power`,
      :func:`pointwise_straggling`);
    * the catastrophic scattering matrices of elastic scattering,
      electroionization (scattered primary and knock-on), bremsstrahlung
      (deflected electron) and excitation;
    * the total cross section and the momentum transfer after the elastic
      Landesman-Morel correction (:func:`landesman_morel_correction`).

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
    """
    cutoff = _coerce_cutoff(cutoff)
    if legendre_order < 1:
        raise ValidationError("generate: legendre_order must be at least 1.")
    grid = _evaluation_grid(dataset, None, "generate")
    quadrature = group_quadrature(structure, grid)
    nodes = quadrature.energy

    def average(values):
        return collapse_nodes(values, quadrature, weight)

    # Soft stopping power and straggling: group averages (Eq. 8) and bound values (Eq. 7)
    bounds = structure.energy
    coefficients = BFPCoefficients(
        stopping_power=average(_soft_loss_moment(dataset, structure, cutoff, 1, nodes)),
        boundary_stopping_power=_soft_loss_moment(
            dataset, structure, cutoff, 1, bounds
        ),
        straggling=average(_soft_loss_moment(dataset, structure, cutoff, 2, nodes)),
        boundary_straggling=_soft_loss_moment(dataset, structure, cutoff, 2, bounds),
    )

    # Elastic scattering stays within each group; Sigma_t of elastic is Sigma_{s,0} (Eq. 61)
    elastic = elastic_legendre_cross_sections(dataset, nodes, legendre_order)
    elastic_diagonal = average(elastic)

    total = elastic[:, 0].copy()
    matrices = {}
    for label, build in (
        ("ionization", _catastrophic_ionization),
        ("bremsstrahlung", _catastrophic_bremsstrahlung),
        (_EXCITATION, _catastrophic_excitation),
    ):
        cat_total, cat_matrix = build(dataset, structure, cutoff, nodes, legendre_order)
        total += cat_total
        if np.any(cat_matrix):
            # (G', L+1, G) from the averaged source group -> (L+1, G', G)
            matrices[label] = average(cat_matrix).transpose(1, 0, 2)

    total_corrected, diagonal, momentum = landesman_morel_correction(
        average(total), elastic_diagonal
    )
    n_groups = structure.n_groups
    elastic_matrix = np.zeros((legendre_order + 1, n_groups, n_groups))
    elastic_matrix[:, np.arange(n_groups), np.arange(n_groups)] = diagonal.T

    scattering = {"elastic": ScatteringMatrix("elastic", elastic_matrix)}
    scattering.update({k: ScatteringMatrix(k, v) for k, v in matrices.items()})
    return BFPData(
        coefficients=coefficients,
        cross_sections={
            "xs_tot": GroupXS("xs_tot", total_corrected),
            MOMENTUM_TRANSFER_LABEL: GroupXS(MOMENTUM_TRANSFER_LABEL, momentum),
        },
        scattering=scattering,
    )
