#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Typed dataclass models for EPICS parsed datasets

Every model is a frozen-safe ``dataclass`` carrying scalar metadata and
NumPy arrays.  Models are the sole output of the reader layer and the
sole input accepted by the converter layer, enforcing strict separation
of concerns.

Hierarchy
---------
::

    CrossSectionRecord   — energy / xs pair with optional interpolation info
    DistributionRecord   — flat (inc_energy, value, probability) triple
    FormFactorRecord     — x / y pair (momentum transfer / form factor)
    SubshellTransition   — single radiative or Auger transition
    SubshellRelaxation   — one subshell's relaxation data
    EEDLDataset          — full electron (EEDL) parsed output
    EPDLDataset          — full photon  (EPDL) parsed output
    EADLDataset          — full atomic  (EADL) parsed output

Units
-----
* Energies are in **eV** unless explicitly noted otherwise.
* Cross sections are in **barns** (ENDF convention).
* Momentum-transfer values are in **1/Å** (inverse ångström).
* Probabilities are dimensionless fractions summing to ≈ 1 per subshell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Atomic-level building blocks
# ---------------------------------------------------------------------------

@dataclass
class CrossSectionRecord:
    """A single cross-section table (energy vs. σ)

    Parameters
    ----------
    label : str
        Short mnemonic name (e.g. ``"xs_tot"``, ``"xs_K"``).
    energy : numpy.ndarray
        Incident-energy grid, shape ``(N,)``, units eV.
        Must be monotonically non-decreasing.
    cross_section : numpy.ndarray
        Cross-section values, shape ``(N,)``, units barns.
        Must be non-negative.
    breakpoints : numpy.ndarray | None
        ENDF TAB1 interpolation-region breakpoints, or ``None``.
    interpolation : numpy.ndarray | None
        ENDF TAB1 interpolation law codes, or ``None``.
    """

    label: str
    energy: np.ndarray
    cross_section: np.ndarray
    breakpoints: np.ndarray | None = None
    interpolation: np.ndarray | None = None


@dataclass
class DistributionRecord:
    """A flat ENDF distribution (incident energy → outgoing value / PDF)

    The three arrays are aligned element-wise: for each index *k*,
    ``inc_energy[k]`` is the incident energy, ``value[k]`` is the
    outgoing quantity (cosine μ or secondary energy E'), and
    ``probability[k]`` is the probability density.

    Parameters
    ----------
    label : str
        Short mnemonic name (e.g. ``"ang_lge"``, ``"spec_K"``).
    inc_energy : numpy.ndarray
        Incident energies, shape ``(M,)``, units eV.
    value : numpy.ndarray
        Outgoing quantity, shape ``(M,)``.
    probability : numpy.ndarray
        Probability-density values, shape ``(M,)``.
    """

    label: str
    inc_energy: np.ndarray
    value: np.ndarray
    probability: np.ndarray


@dataclass
class AverageEnergyLoss:
    """Average energy loss vs. incident energy

    Parameters
    ----------
    label : str
        Short mnemonic name (e.g. ``"loss_exc"``).
    energy : numpy.ndarray
        Incident-energy grid, shape ``(N,)``, units eV.
    avg_loss : numpy.ndarray
        Average energy loss per collision, shape ``(N,)``, units eV.
    """

    label: str
    energy: np.ndarray
    avg_loss: np.ndarray


@dataclass
class FormFactorRecord:
    """A form-factor or scattering-function table

    Parameters
    ----------
    label : str
        Short mnemonic (e.g. ``"ff_coherent"``, ``"sf_incoherent"``).
    x : numpy.ndarray
        Independent variable — momentum transfer (1/Å) or energy (eV).
    y : numpy.ndarray
        Form factor or scattering function values, same shape as *x*.
    breakpoints : numpy.ndarray | None
        ENDF TAB1 interpolation-region breakpoints.
    interpolation : numpy.ndarray | None
        ENDF TAB1 interpolation law codes.
    """

    label: str
    x: np.ndarray
    y: np.ndarray
    breakpoints: np.ndarray | None = None
    interpolation: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Atomic relaxation building blocks
# ---------------------------------------------------------------------------

@dataclass
class SubshellTransition:
    """A single atomic-relaxation transition

    Parameters
    ----------
    origin_designator : int
        EADL subshell designator of the originating electron.
    origin_label : str
        Human-readable label (e.g. ``"L1"``).
    secondary_designator : int
        Designator for the subshell that fills the vacancy.
        Zero (0) indicates a radiative (X-ray) transition.
    secondary_label : str
        ``"radiative"`` when *secondary_designator* is 0, else the
        subshell label (e.g. ``"M2"``).
    energy_eV : float
        Transition energy (eV).
    probability : float
        Fractional probability of this transition occurring.
    is_radiative : bool
        ``True`` for X-ray emission, ``False`` for Auger / Coster-Kronig.
    """

    origin_designator: int
    origin_label: str
    secondary_designator: int
    secondary_label: str
    energy_eV: float
    probability: float
    is_radiative: bool


@dataclass
class SubshellRelaxation:
    """Relaxation data for a single atomic subshell

    Parameters
    ----------
    designator : int
        EADL numeric subshell designator (1 = K, 2 = L1, …).
    name : str
        Standard label (e.g. ``"K"``, ``"L1"``).
    binding_energy_eV : float
        Binding energy of the subshell (eV).
    n_electrons : float
        Number of electrons in the neutral atom.
    transitions : list[SubshellTransition]
        All radiative and non-radiative transitions from this subshell.
    """

    designator: int
    name: str
    binding_energy_eV: float
    n_electrons: float
    transitions: list[SubshellTransition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level dataset models  (one per EPICS library)
# ---------------------------------------------------------------------------

@dataclass
class EEDLDataset:
    """Complete parsed output of an EEDL (Evaluated Electron Data Library) file

    Instances are returned by :meth:`EEDLReader.read` and consumed by
    :func:`convert_dataset_to_hdf5`.

    Parameters
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element symbol (e.g. ``"Fe"``).
    atomic_weight_ratio : float
        AWR from the ENDF material header.
    ZA : float
        ZA identifier (Z × 1000 + A).
    cross_sections : dict[str, CrossSectionRecord]
        Keyed by abbreviation (e.g. ``"xs_tot"``, ``"xs_K"``).
    distributions : dict[str, DistributionRecord]
        Keyed by abbreviation (e.g. ``"ang_lge"``, ``"spec_K"``).
    average_energy_losses : dict[str, AverageEnergyLoss]
        Keyed by abbreviation (e.g. ``"loss_exc"``, ``"loss_brem_spec"``).
    bremsstrahlung_spectra : DistributionRecord | None
        Bremsstrahlung photon energy spectrum, if present.
    """

    Z: int
    symbol: str
    atomic_weight_ratio: float
    ZA: float
    cross_sections: dict[str, CrossSectionRecord] = field(default_factory=dict)
    distributions: dict[str, DistributionRecord] = field(default_factory=dict)
    average_energy_losses: dict[str, AverageEnergyLoss] = field(default_factory=dict)
    bremsstrahlung_spectra: DistributionRecord | None = None


@dataclass
class EPDLDataset:
    """Complete parsed output of an EPDL (Evaluated Photon Data Library) file

    Instances are returned by :meth:`EPDLReader.read` and consumed by
    :func:`convert_dataset_to_hdf5`.

    Parameters
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element symbol.
    atomic_weight_ratio : float
        AWR from the ENDF material header.
    ZA : float
        ZA identifier.
    cross_sections : dict[str, CrossSectionRecord]
        Photon cross sections keyed by abbreviation.
    form_factors : dict[str, FormFactorRecord]
        Form factors / scattering functions keyed by abbreviation.
    """

    Z: int
    symbol: str
    atomic_weight_ratio: float
    ZA: float
    cross_sections: dict[str, CrossSectionRecord] = field(default_factory=dict)
    form_factors: dict[str, FormFactorRecord] = field(default_factory=dict)


@dataclass
class EADLDataset:
    """Complete parsed output of an EADL (Evaluated Atomic Data Library) file

    Instances are returned by :meth:`EADLReader.read` and consumed by
    :func:`convert_dataset_to_hdf5`.

    Parameters
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element symbol.
    atomic_weight_ratio : float
        AWR from the ENDF material header.
    ZA : float
        ZA identifier.
    n_subshells : int
        Number of subshells for which relaxation data is given.
    subshells : dict[str, SubshellRelaxation]
        Relaxation data keyed by subshell label (e.g. ``"K"``).
    """

    Z: int
    symbol: str
    atomic_weight_ratio: float
    ZA: float
    n_subshells: int = 0
    subshells: dict[str, SubshellRelaxation] = field(default_factory=dict)
