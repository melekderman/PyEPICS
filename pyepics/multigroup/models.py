#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Typed dataclass models for collapsed EPICS datasets

Every model is a frozen-safe ``dataclass`` carrying scalar metadata and
NumPy arrays.  Models are the sole output of the physics/collapse layers and
the sole input of the converter layer, enforcing strict separation of concerns.

Hierarchy
---------
::
    GroupStructure     : Descending energy group bounds in eV
    GroupXS            : Cross sections for a single reaction, all groups
    ScatteringMatrix   : Scattering matrix, group-to-group, within species
    BFPCoefficients    : Stopping power, energy straggling, and momentum transfer cross section, all groups
    CutoffSpec         : CEPXS or RADIANT convention. String, for provenance
    ParticleMG         : Contains all data for a single particle
    MGElement          : Contains electron and/or photon ParticleMG data
    MGLibrary          : A library of multiple MGElements

Units
-----
* Energies are in **eV** unless explicitly noted otherwise.
* Cross sections are in **barns** (ENDF convention).
* Stopping power is in **eV*barns**, energy straggling coefficient is in **barns*eV^2**
* Momentum-transfer values are in **1/Å** (inverse ångström).

Group ordering
--------------
Groups are numbered from the highest energy down: group ``g`` spans
``[energy[g + 1], energy[g]]``, so group 0 is the highest-energy group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyepics.exceptions import ValidationError
from pyepics.utils.validation import validate_non_negative, validate_strictly_descending


@dataclass(frozen=True, eq=False)
class GroupStructure:
    """An energy group structure (boundaries only)

    Immutable: the boundary array is copied on construction and set
    read-only, so a structure shared between several models cannot be
    changed under them.  Equality is by identity (``eq=False``) because
    comparing the array field with ``==`` is ambiguous.

    Parameters
    ----------
    label : str
        Label for group structure
    energy : numpy.ndarray
        Energy group boundaries, shape ``(G+1,)``, units eV.
        Must be strictly descending, finite, and non-negative (a lowest
        boundary of exactly 0 is allowed).  Must contain at least two
        values.  Any array-like is converted to ``float64``.

    Raises
    ------
    ValidationError
        If *energy* is not 1-D, has fewer than two values, contains
        non-finite or negative values, or is not strictly descending.
    """

    label: str
    energy: np.ndarray

    def __post_init__(self) -> None:
        try:
            energy = np.array(self.energy, dtype="f8")  # copy
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Group structure '{self.label}': energy bounds must be "
                f"numeric ({exc})."
            ) from exc
        if energy.ndim != 1:
            raise ValidationError(
                f"Group structure '{self.label}': energy grid must be a 1D "
                f"array (got {energy.ndim}D)."
            )
        if energy.size < 2:
            raise ValidationError(
                f"Group structure '{self.label}': energy grid must contain "
                f"at least two values (got {energy.size})."
            )
        if not np.all(np.isfinite(energy)):
            raise ValidationError(
                f"Group structure '{self.label}': energy grid contains "
                f"non-finite values."
            )
        validate_non_negative(energy, f"{self.label} energy")
        validate_strictly_descending(energy, f"{self.label} energy")
        energy.setflags(write=False)
        object.__setattr__(self, "energy", energy)

    @property
    def n_groups(self) -> int:
        """Number of groups, ``G``"""
        return int(self.energy.size - 1)

    @property
    def upper(self) -> np.ndarray:
        """Upper boundary of each group, shape ``(G,)``"""
        return self.energy[:-1]

    @property
    def lower(self) -> np.ndarray:
        """Lower boundary of each group, shape ``(G,)``"""
        return self.energy[1:]

    @property
    def width(self) -> np.ndarray:
        """Group widths ``upper - lower``, shape ``(G,)``, units eV"""
        return self.upper - self.lower

    def group_of(self, energy: float) -> int:
        """Index of the group containing *energy*

        A value exactly on an interior boundary belongs to the group
        below it (the upper bound is exclusive for all but group 0).

        Raises
        ------
        ValidationError
            If *energy* lies outside ``[energy[-1], energy[0]]``.
        """
        if not (self.energy[-1] <= energy <= self.energy[0]):
            raise ValidationError(
                f"Energy {energy:.6e} eV is outside group structure "
                f"'{self.label}' [{self.energy[-1]:.6e}, {self.energy[0]:.6e}]."
            )
        # np.searchsorted needs ascending input; reverse the view.
        k = int(np.searchsorted(self.energy[::-1], energy, side="left"))
        g = self.energy.size - 1 - k
        return min(max(g, 0), self.n_groups - 1)
