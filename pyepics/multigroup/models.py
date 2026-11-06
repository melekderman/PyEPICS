#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Typed dataclass models for collapsed EPICS datasets

Every model is a ``dataclass`` carrying scalar metadata and NumPy arrays,
validated on construction.  The models hold the output of the group collapse
and are the only input the writers need.

Hierarchy
---------
::
    GroupStructure     : Descending energy group bounds in eV
    GroupXS            : Cross sections for a single reaction, all groups
    ScatteringMatrix   : Scattering matrix, group-to-group, within species
    BFPCoefficients    : Stopping power and (optional) energy straggling, per group and at group boundaries
    Particle           : Enum, ``electron`` or ``photon``
    TransportModel     : Enum, ``bfp``, ``bcsd``, ``fp`` or ``boltzmann``
    TransportCorrection: Enum, ``none``, ``extended`` or ``landesman_morel``
    ParticleMG         : Contains all data for a single particle; the transport model
                         decides which fields are required.  Also records the cutoff
                         convention (a plain string, e.g. ``"CEPXS"``) for provenance
    MGElement          : One element (Z, symbol, AWR) with electron and/or photon ParticleMG data
    MGLibrary          : A labelled library of MGElements, keyed by Z, sharing one
                         group structure / model / correction / cutoff per particle

Units
-----
* Energies are in **eV** unless explicitly noted otherwise.
* Cross sections are in **barns** (ENDF convention).
* Stopping power is in **eV*barns**, energy straggling coefficient is in **barns*eV^2**

Group ordering
--------------
Groups are numbered from the highest energy down: group ``g`` spans
``[energy[g + 1], energy[g]]``, so group 0 is the highest-energy group.

Scattering matrix axes
----------------------
``ScatteringMatrix.matrix[l, g_from, g_to]`` is the Legendre-order-``l``
cross section for a particle starting in group ``g_from`` and ending in
group ``g_to``.  Because group 0 is the highest energy, down-scatter has
``g_to >= g_from`` (upper triangle including the diagonal).  Writers for
codes that use the transposed convention must transpose on output; the
model never does.

Transport models
----------------
``ParticleMG.transport_model`` says what the data is for, and so which fields
must (or must not) be present:

=============  ===============  ================  =========  ================
field          bfp              bcsd              fp         boltzmann
=============  ===============  ================  =========  ================
coefficients   required         required          required   forbidden
straggling     optional         forbidden         required   forbidden
momentum xs    required         forbidden         required   forbidden
scattering     required         required          forbidden  required
cutoff         required         required          forbidden  forbidden
correction     landesman_morel  none or extended  none       none or extended
=============  ===============  ================  =========  ================

* ``bfp``: Boltzmann-Fokker-Planck.  Soft collisions are the continuous
  terms (stopping power, optional straggling, momentum transfer); collisions
  above the cutoff are catastrophic and held in the scattering matrices.
* ``bcsd``: Boltzmann-continuous slowing down.  As ``bfp`` but soft
  collisions are reduced to the stopping power alone: no straggling and no
  momentum transfer.  Not the classical CSDA, which has no scattering.
* ``fp``: pure Fokker-Planck.  Every collision is soft, so there is no cutoff
  and no scattering matrix.
* ``boltzmann``: no continuous terms; all scattering is in the matrices.

*momentum xs* is the ``GroupXS`` stored under
:data:`MOMENTUM_TRANSFER_LABEL` in ``cross_sections``.  For ``bfp`` and
``bcsd`` the scattering matrices are the catastrophic ones (above the
cutoff).  Photons only support ``boltzmann`` and no transport correction.
*correction* is :attr:`ParticleMG.transport_correction`; see
:class:`TransportModel` for the physics behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from pyepics.exceptions import ValidationError
from pyepics.utils.constants import PERIODIC_TABLE
from pyepics.utils.validation import (
    validate_atomic_number,
    validate_non_negative,
    validate_strictly_descending,
)


def _float_array(owner: str, name: str, values: np.ndarray, ndim: int) -> np.ndarray:
    """Copy *values* to a finite ``float64`` array of the given dimension"""
    try:
        array = np.array(values, dtype="f8")  # copy
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{owner}: {name} must be numeric ({exc}).") from exc
    if array.ndim != ndim:
        raise ValidationError(
            f"{owner}: {name} must be a {ndim}D array (got {array.ndim}D)."
        )
    if array.size == 0:
        raise ValidationError(f"{owner}: {name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{owner}: {name} contains non-finite values.")
    return array


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
        owner = f"Group structure '{self.label}'"
        energy = _float_array(owner, "energy grid", self.energy, 1)
        if energy.size < 2:
            raise ValidationError(
                f"{owner}: energy grid must contain at least two values "
                f"(got {energy.size})."
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
        n_bounds_below = int(np.searchsorted(self.energy[::-1], energy, side="left"))
        group = self.energy.size - 1 - n_bounds_below
        return min(max(group, 0), self.n_groups - 1)


@dataclass
class GroupXS:
    """Group-averaged cross section for a single reaction

    Parameters
    ----------
    label : str
        Short mnemonic name (e.g. ``"xs_tot"``).
    cross_section : numpy.ndarray
        Cross-section values, shape ``(G,)``, units barns, indexed by
        group (group 0 is the highest energy).  Must be finite and
        non-negative.  Any array-like is converted to ``float64``.

    Raises
    ------
    ValidationError
        If *cross_section* is not a non-empty 1-D finite array, or
        contains negative values.
    """

    label: str
    cross_section: np.ndarray

    def __post_init__(self) -> None:
        owner = f"GroupXS '{self.label}'"
        self.cross_section = _float_array(owner, "cross_section", self.cross_section, 1)
        validate_non_negative(self.cross_section, f"{self.label} cross_section")

    @property
    def n_groups(self) -> int:
        """Number of groups, ``G``"""
        return int(self.cross_section.size)


@dataclass
class ScatteringMatrix:
    """Group-to-group scattering matrix within one particle species

    Parameters
    ----------
    label : str
        Short mnemonic name (e.g. ``"elastic"``, ``"catastrophic_elastic"``).
    matrix : numpy.ndarray
        Cross sections, shape ``(L+1, G, G)``, units barns, indexed
        ``matrix[l, g_from, g_to]`` (see "Scattering matrix axes" in the
        module docstring).  Must be finite.  The ``l = 0`` moment must be
        non-negative; higher moments may be negative.  Any array-like is
        converted to ``float64``.

    Raises
    ------
    ValidationError
        If *matrix* is not a non-empty 3-D finite array with equal last
        two axes, or its ``l = 0`` moment contains negative values.
    """

    label: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        owner = f"ScatteringMatrix '{self.label}'"
        self.matrix = _float_array(owner, "matrix", self.matrix, 3)
        if self.matrix.shape[1] != self.matrix.shape[2]:
            raise ValidationError(
                f"{owner}: group axes must be square, got shape "
                f"{self.matrix.shape} (expected (L+1, G, G))."
            )
        validate_non_negative(self.matrix[0].ravel(), f"{self.label} matrix[l=0]")

    @property
    def n_groups(self) -> int:
        """Number of groups, ``G``"""
        return int(self.matrix.shape[1])

    @property
    def order(self) -> int:
        """Highest Legendre order, ``L``"""
        return int(self.matrix.shape[0] - 1)

    def is_downscatter_only(self, rtol: float = 0.0) -> bool:
        """Whether no scattering goes to a higher-energy group

        True when every entry below the diagonal (``g_to < g_from``) is
        zero, for all Legendre orders.  With *rtol* > 0, entries up to
        ``rtol`` times the largest absolute entry are treated as zero.
        """
        below = np.tril(np.ones(self.matrix.shape[1:], dtype=bool), k=-1)
        scale = float(np.max(np.abs(self.matrix)))
        return bool(np.all(np.abs(self.matrix[:, below]) <= rtol * scale))


@dataclass
class BFPCoefficients:
    """Boltzmann-Fokker-Planck soft-collision coefficients

    The arrays are only meaningful together: all come from the same
    soft/catastrophic split.  The cutoff that defines the split is recorded
    on :class:`ParticleMG`, not here.  The momentum-transfer cross section
    is *not* part of this model.

    Each coefficient is stored twice: averaged over each group (``(G,)``)
    and evaluated at the group boundaries (``(G+1,)``, same indexing as
    :attr:`GroupStructure.energy`, so ``boundary_*[i]`` is the value at
    ``energy[i]``).

    Parameters
    ----------
    stopping_power : numpy.ndarray
        Restricted stopping power per group, shape ``(G,)``, units
        eV*barns.  Must be finite and non-negative.
    boundary_stopping_power : numpy.ndarray
        Restricted stopping power at the group boundaries, shape
        ``(G+1,)``, units eV*barns.  Must be finite and non-negative.
    straggling : numpy.ndarray | None
        Energy-straggling coefficient per group, shape ``(G,)``, units
        barns*eV^2.  Must be finite and non-negative.  ``None`` if
        straggling is not included.
    boundary_straggling : numpy.ndarray | None
        Energy-straggling coefficient at the group boundaries, shape
        ``(G+1,)``, units barns*eV^2.  Must be finite and non-negative.
        Required if *straggling* is given, and forbidden if it is not.

    Raises
    ------
    ValidationError
        If an array is not a non-empty 1-D finite array, contains negative
        values, or has the wrong length (``G`` for group values, ``G+1``
        for boundary values, with ``G`` taken from *stopping_power*); or if
        exactly one of *straggling* and *boundary_straggling* is given.
    """

    stopping_power: np.ndarray
    boundary_stopping_power: np.ndarray
    straggling: np.ndarray | None = None
    boundary_straggling: np.ndarray | None = None

    def __post_init__(self) -> None:
        owner = "BFPCoefficients"
        self.stopping_power = _float_array(
            owner, "stopping_power", self.stopping_power, 1
        )
        self.boundary_stopping_power = _float_array(
            owner, "boundary_stopping_power", self.boundary_stopping_power, 1
        )
        validate_non_negative(self.stopping_power, "stopping_power")
        validate_non_negative(self.boundary_stopping_power, "boundary_stopping_power")
        n_groups = self.stopping_power.size
        if self.boundary_stopping_power.size != n_groups + 1:
            raise ValidationError(
                f"{owner}: stopping_power has {n_groups} groups, so "
                f"boundary_stopping_power must have {n_groups + 1} values "
                f"(got {self.boundary_stopping_power.size})."
            )

        if (self.straggling is None) != (self.boundary_straggling is None):
            given, missing = (
                ("straggling", "boundary_straggling")
                if self.boundary_straggling is None
                else ("boundary_straggling", "straggling")
            )
            raise ValidationError(
                f"{owner}: {given} was given without {missing}; provide "
                f"both or neither."
            )
        if self.straggling is not None:
            self.straggling = _float_array(owner, "straggling", self.straggling, 1)
            self.boundary_straggling = _float_array(
                owner, "boundary_straggling", self.boundary_straggling, 1
            )
            validate_non_negative(self.straggling, "straggling")
            validate_non_negative(self.boundary_straggling, "boundary_straggling")
            if self.straggling.size != n_groups:
                raise ValidationError(
                    f"{owner}: stopping_power has {n_groups} groups but straggling "
                    f"has {self.straggling.size}."
                )
            if self.boundary_straggling.size != n_groups + 1:
                raise ValidationError(
                    f"{owner}: stopping_power has {n_groups} groups, so "
                    f"boundary_straggling must have {n_groups + 1} values "
                    f"(got {self.boundary_straggling.size})."
                )

    @property
    def n_groups(self) -> int:
        """Number of groups, ``G``"""
        return int(self.stopping_power.size)

    @property
    def has_straggling(self) -> bool:
        """Whether straggling (group and boundary values) is included"""
        return self.straggling is not None


class Particle(StrEnum):
    """Particle species of a :class:`ParticleMG`"""

    ELECTRON = "electron"
    PHOTON = "photon"


class TransportModel(StrEnum):
    """Transport formulation a :class:`ParticleMG` is generated for

    See "Transport models" in the module docstring for what each requires.

    Transport correction
    --------------------
    The Boltzmann scattering operator must be transport-corrected, with
    different formulas depending on which physics model is used. The ``bfp``
    correction removes soft scattering, and all models correct the total
    and scattering cross sections using the highest Legendre moment.

    Reference: Bienvenue, Naceur, Hebert, Carrier, J. Comput. Phys. 524
    (2025) 113740, the extended transport correction with momentum transfer
    (Landesman-Morel).  All quantities are the *within-group* (``g -> g``)
    elastic moments of each group ``g``.  With ``L`` the last non-zero
    Legendre moment::

        T_g          = (S[L-1] - S[L]) / (2 L)
        sigma_t'     = sigma_t - S[L] - T_g L (L + 1)
        S'[l]        = S[l]    - S[L] - T_g (L (L + 1) - l (l + 1))

    where ``S[l]`` is ``Sigma_{s,l,g->g}``.  ``S'[0]`` equals ``sigma_t'``
    and ``S'[L]`` is zero.

    * ``bfp`` -- Landesman-Morel (all terms).  ``T_g`` is the elastic
      momentum transfer passed to the Fokker-Planck operator, so it comes
      from the matrix moments, not from integrating the DCS, and depends on
      ``L`` (and on the paper's zeroing of unstable high-order moments).
    * ``bcsd`` -- no Fokker-Planck operator, so nothing takes up the soft
      part.  The ``T_g`` terms must **not** be applied: they would delete the
      angular deflection of soft elastic scattering from the problem.  The
      ``T_g = 0`` limit, ``sigma_t' = sigma_t - S[L]`` and
      ``S'[l] = S[l] - S[L]``, is the extended transport correction (Morel
      1979; Drumm et al.) and stays inside the Boltzmann operator.
    * ``fp`` -- no scattering matrices, so no correction.
    * ``boltzmann`` -- as ``bcsd``: ``none`` or ``extended``.

    .. note::
       The paper does not treat ``bcsd``.  Using the ``T_g = 0`` limit
       there is derived from the corrected total and scattering cross
       sections above, and rests on the higher moments being close to
       ``S[L]`` (forward-peaked scattering).  Benchmark it
       against ``bfp`` before relying on it.
    """

    BFP = "bfp"
    BCSD = "bcsd"
    FP = "fp"
    BOLTZMANN = "boltzmann"


class TransportCorrection(StrEnum):
    """Transport correction applied to the elastic scattering data

    A corrected total cross section is *not* the physical total, so the
    correction is recorded on :class:`ParticleMG` and must travel with the
    data.  Which values a model accepts, and the equations, are described
    under "Transport correction" in :class:`TransportModel`.
    """

    NONE = "none"
    EXTENDED = "extended"
    LANDESMAN_MOREL = "landesman_morel"


_ALLOWED_CORRECTIONS: dict[TransportModel, frozenset[TransportCorrection]] = {
    TransportModel.BFP: frozenset({TransportCorrection.LANDESMAN_MOREL}),
    TransportModel.BCSD: frozenset(
        {TransportCorrection.NONE, TransportCorrection.EXTENDED}
    ),
    TransportModel.FP: frozenset({TransportCorrection.NONE}),
    TransportModel.BOLTZMANN: frozenset(
        {TransportCorrection.NONE, TransportCorrection.EXTENDED}
    ),
}


#: Label of the restricted momentum-transfer cross section in
#: :attr:`ParticleMG.cross_sections`.
MOMENTUM_TRANSFER_LABEL = "momentum_transfer"

_REQUIRED, _OPTIONAL, _FORBIDDEN = "required", "optional", "forbidden"

# What each transport model needs (see "Transport models" in the module docstring).
_MODEL_RULES: dict[TransportModel, dict[str, str]] = {
    TransportModel.BFP: dict(
        coefficients=_REQUIRED,
        straggling=_OPTIONAL,
        momentum_transfer=_REQUIRED,
        scattering=_REQUIRED,
        cutoff=_REQUIRED,
    ),
    TransportModel.BCSD: dict(
        coefficients=_REQUIRED,
        straggling=_FORBIDDEN,
        momentum_transfer=_FORBIDDEN,
        scattering=_REQUIRED,
        cutoff=_REQUIRED,
    ),
    TransportModel.FP: dict(
        coefficients=_REQUIRED,
        straggling=_REQUIRED,
        momentum_transfer=_REQUIRED,
        scattering=_FORBIDDEN,
        cutoff=_FORBIDDEN,
    ),
    TransportModel.BOLTZMANN: dict(
        coefficients=_FORBIDDEN,
        straggling=_FORBIDDEN,
        momentum_transfer=_FORBIDDEN,
        scattering=_REQUIRED,
        cutoff=_FORBIDDEN,
    ),
}


def _coerce_enum(owner: str, enum_cls: type[StrEnum], value, name: str):
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in enum_cls)
        raise ValidationError(
            f"{owner}: {name} must be one of {valid} (got {value!r})."
        ) from None


@dataclass
class ParticleMG:
    """All multigroup data for one particle species

    Parameters
    ----------
    particle : Particle | str
        ``"electron"`` or ``"photon"``.
    transport_model : TransportModel | str
        ``"bfp"``, ``"bcsd"``, ``"fp"`` or ``"boltzmann"``.  Decides which
        of the fields below are required, optional or forbidden (see
        "Transport models" in the module docstring).  Photons must use
        ``"boltzmann"``.
    group_structure : GroupStructure
        The energy groups every array below is defined on.
    cross_sections : dict[str, GroupXS]
        Group cross sections keyed by reaction label.  Each key must equal
        its value's ``label``.  The momentum-transfer cross section, where
        the model needs it, is stored under :data:`MOMENTUM_TRANSFER_LABEL`.
    scattering : dict[str, ScatteringMatrix]
        Scattering matrices keyed by label (key must equal ``label``).  For
        ``bfp`` and ``bcsd`` these are the catastrophic matrices.
    coefficients : BFPCoefficients | None
        Stopping power and straggling.
    cutoff : str | None
        Convention that defined the soft/catastrophic split (e.g.
        ``"CEPXS"`` or ``"RADIANT"``), kept for provenance.  Non-empty if
        given.
    transport_correction : TransportCorrection | str
        Correction already applied to the elastic data in *scattering* and
        *cross_sections* (default ``"none"``).  ``bfp`` requires
        ``"landesman_morel"``; ``bcsd`` and ``boltzmann`` allow ``"none"``
        or ``"extended"``; ``fp`` and photons allow only ``"none"``.  See
        :class:`TransportModel`.

    Raises
    ------
    ValidationError
        If *particle* or *transport_model* is not a valid value; if a
        photon is not ``boltzmann``; if a dict key differs from its
        value's label or a value has the wrong type; if any array's group
        count differs from ``group_structure.n_groups``; if
        *transport_correction* is not allowed for the model; or if a field
        the transport model requires is missing or one it forbids is
        present.
    """

    particle: Particle
    transport_model: TransportModel
    group_structure: GroupStructure
    cross_sections: dict[str, GroupXS] = field(default_factory=dict)
    scattering: dict[str, ScatteringMatrix] = field(default_factory=dict)
    coefficients: BFPCoefficients | None = None
    cutoff: str | None = None
    transport_correction: TransportCorrection = TransportCorrection.NONE

    def __post_init__(self) -> None:
        owner = "ParticleMG"
        self.particle = _coerce_enum(owner, Particle, self.particle, "particle")
        self.transport_model = _coerce_enum(
            owner, TransportModel, self.transport_model, "transport_model"
        )
        if not isinstance(self.group_structure, GroupStructure):
            raise ValidationError(f"{owner}: group_structure must be a GroupStructure.")
        n_groups = self.group_structure.n_groups

        if self.particle is Particle.PHOTON and (
            self.transport_model is not TransportModel.BOLTZMANN
        ):
            raise ValidationError(
                f"{owner}: photons only support transport_model 'boltzmann' "
                f"(got '{self.transport_model}')."
            )

        self.transport_correction = _coerce_enum(
            owner,
            TransportCorrection,
            self.transport_correction,
            "transport_correction",
        )
        allowed = _ALLOWED_CORRECTIONS[self.transport_model]
        if self.particle is Particle.PHOTON:
            allowed = allowed & {TransportCorrection.NONE}
        if self.transport_correction not in allowed:
            valid = ", ".join(repr(str(c)) for c in sorted(allowed))
            raise ValidationError(
                f"{owner}: transport_model '{self.transport_model}' "
                f"({self.particle}) allows transport_correction {valid}; "
                f"got '{self.transport_correction}'."
            )

        self.cross_sections = self._check_leaves(
            "cross_sections", self.cross_sections, GroupXS, n_groups
        )
        self.scattering = self._check_leaves(
            "scattering", self.scattering, ScatteringMatrix, n_groups
        )

        if self.coefficients is not None:
            if not isinstance(self.coefficients, BFPCoefficients):
                raise ValidationError(f"{owner}: coefficients must be BFPCoefficients.")
            if self.coefficients.n_groups != n_groups:
                raise ValidationError(
                    f"{owner}: coefficients have {self.coefficients.n_groups} "
                    f"groups but group_structure has {n_groups}."
                )
        if self.cutoff is not None and (
            not isinstance(self.cutoff, str) or not self.cutoff.strip()
        ):
            raise ValidationError(f"{owner}: cutoff must be a non-empty string.")

        self._check_model_fields()

    @staticmethod
    def _check_leaves(name: str, leaves, leaf_type, n_groups: int) -> dict:
        owner = f"ParticleMG.{name}"
        out = dict(leaves)
        for key, leaf in out.items():
            if not isinstance(leaf, leaf_type):
                raise ValidationError(
                    f"{owner}['{key}'] must be a {leaf_type.__name__}."
                )
            if leaf.label != key:
                raise ValidationError(
                    f"{owner}: key '{key}' does not match label '{leaf.label}'."
                )
            if leaf.n_groups != n_groups:
                raise ValidationError(
                    f"{owner}['{key}'] has {leaf.n_groups} groups but "
                    f"group_structure has {n_groups}."
                )
        return out

    def _check_model_fields(self) -> None:
        rules = _MODEL_RULES[self.transport_model]
        coefficients = self.coefficients
        present = dict(
            coefficients=coefficients is not None,
            straggling=coefficients is not None and coefficients.has_straggling,
            momentum_transfer=MOMENTUM_TRANSFER_LABEL in self.cross_sections,
            scattering=bool(self.scattering),
            cutoff=self.cutoff is not None,
        )
        names = dict(
            coefficients="coefficients",
            straggling="straggling in coefficients",
            momentum_transfer=f"the '{MOMENTUM_TRANSFER_LABEL}' cross section",
            scattering="scattering matrices",
            cutoff="cutoff",
        )
        for key, rule in rules.items():
            if key == "straggling" and not present["coefficients"]:
                continue  # already reported as missing/forbidden
            if rule == _REQUIRED and not present[key]:
                raise ValidationError(
                    f"ParticleMG: transport_model '{self.transport_model}' "
                    f"requires {names[key]}."
                )
            if rule == _FORBIDDEN and present[key]:
                raise ValidationError(
                    f"ParticleMG: transport_model '{self.transport_model}' "
                    f"does not allow {names[key]}."
                )

    @property
    def n_groups(self) -> int:
        """Number of groups, ``G``"""
        return self.group_structure.n_groups


@dataclass
class MGElement:
    """Multigroup data for one element

    Mirrors the identification fields of the parsed datasets
    (:class:`~pyepics.models.records.EEDLDataset` and friends) and the
    ``electron`` / ``photon`` slots of
    :class:`~pyepics.client.ElementProperties`.  There is no ``ZA``: the
    EPICS libraries are natural elements, so it is just ``Z * 1000``.

    Parameters
    ----------
    Z : int
        Atomic number, 1 to 118.
    symbol : str
        Element symbol (e.g. ``"Fe"``); must match *Z*.
    atomic_weight_ratio : float
        AWR from the ENDF material header.  Needed to turn the per-atom
        cross sections into macroscopic ones.  Must be finite and positive.
    electron : ParticleMG | None
        Electron data; its ``particle`` must be ``electron``.
    photon : ParticleMG | None
        Photon data; its ``particle`` must be ``photon``.

    Raises
    ------
    ValidationError
        If *Z* is not an integer in range, *symbol* does not match *Z*,
        *atomic_weight_ratio* is not finite and positive, a slot holds the
        wrong type or the wrong particle, or both slots are empty.
    """

    Z: int
    symbol: str
    atomic_weight_ratio: float
    electron: ParticleMG | None = None
    photon: ParticleMG | None = None

    def __post_init__(self) -> None:
        owner = "MGElement"
        if isinstance(self.Z, bool) or not isinstance(self.Z, (int, np.integer)):
            raise ValidationError(f"{owner}: Z must be an integer (got {self.Z!r}).")
        self.Z = int(self.Z)
        validate_atomic_number(self.Z)
        expected = PERIODIC_TABLE[self.Z]["symbol"]
        if self.symbol != expected:
            raise ValidationError(
                f"{owner}: symbol {self.symbol!r} does not match Z={self.Z} "
                f"({expected!r})."
            )
        try:
            ratio = float(self.atomic_weight_ratio)
        except (TypeError, ValueError):
            raise ValidationError(
                f"{owner}: atomic_weight_ratio must be a number "
                f"(got {self.atomic_weight_ratio!r})."
            ) from None
        if not (np.isfinite(ratio) and ratio > 0.0):
            raise ValidationError(
                f"{owner}: atomic_weight_ratio must be finite and positive "
                f"(got {ratio})."
            )
        self.atomic_weight_ratio = ratio

        for slot, particle in (
            ("electron", Particle.ELECTRON),
            ("photon", Particle.PHOTON),
        ):
            particle_data = getattr(self, slot)
            if particle_data is None:
                continue
            if not isinstance(particle_data, ParticleMG):
                raise ValidationError(f"{owner}: {slot} must be a ParticleMG.")
            if particle_data.particle is not particle:
                raise ValidationError(
                    f"{owner}: {slot} holds {particle_data.particle} data."
                )
        if self.electron is None and self.photon is None:
            raise ValidationError(
                f"{owner} {self.symbol}: needs electron and/or photon data."
            )

    @property
    def particles(self) -> dict[Particle, ParticleMG]:
        """The ``ParticleMG`` of each particle present, electron first"""
        out: dict[Particle, ParticleMG] = {}
        if self.electron is not None:
            out[Particle.ELECTRON] = self.electron
        if self.photon is not None:
            out[Particle.PHOTON] = self.photon
        return out


@dataclass
class MGLibrary:
    """A multigroup library: elements that can be mixed into one problem

    Elements of one library are meant to be used together, so for each
    particle every element that has it must agree on the group structure
    (by energy values; labels are ignored), ``transport_model``,
    ``transport_correction`` and ``cutoff``.  Electron and photon groups
    may differ from each other.

    Parameters
    ----------
    label : str
        Name of the library.  Must be non-empty.
    elements : dict[int, MGElement]
        Elements keyed by atomic number; each key must equal its value's
        ``Z``.  May be empty.

    Raises
    ------
    ValidationError
        If *label* is empty, a value is not an ``MGElement``, a key differs
        from its element's ``Z``, or two elements disagree on any of the
        quantities above for the same particle.
    """

    label: str
    elements: dict[int, MGElement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        owner = "MGLibrary"
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValidationError(f"{owner}: label must be a non-empty string.")
        owner = f"MGLibrary '{self.label}'"
        self.elements = dict(self.elements)
        for key, element in self.elements.items():
            if not isinstance(element, MGElement):
                raise ValidationError(
                    f"{owner}: elements[{key!r}] must be an MGElement."
                )
            if key != element.Z:
                raise ValidationError(
                    f"{owner}: key {key!r} does not match Z={element.Z} "
                    f"({element.symbol})."
                )
        self._check_consistent(owner)

    def _check_consistent(self, owner: str) -> None:
        for particle in Particle:
            reference = None
            for atomic_number in sorted(self.elements):
                element = self.elements[atomic_number]
                data = element.particles.get(particle)
                if data is None:
                    continue
                if reference is None:
                    reference_element, reference = element, data
                    continue
                agreement = {
                    "group_structure": np.array_equal(
                        data.group_structure.energy, reference.group_structure.energy
                    ),
                    "transport_model": data.transport_model
                    == reference.transport_model,
                    "transport_correction": (
                        data.transport_correction == reference.transport_correction
                    ),
                    "cutoff": data.cutoff == reference.cutoff,
                }
                for name, agrees in agreement.items():
                    if not agrees:
                        raise ValidationError(
                            f"{owner}: {particle} {name} of {element.symbol} "
                            f"(Z={element.Z}) differs from {reference_element.symbol} "
                            f"(Z={reference_element.Z})."
                        )

    @property
    def particles(self) -> list[Particle]:
        """Particles present in at least one element, electron first"""
        return [
            particle
            for particle in Particle
            if any(particle in element.particles for element in self.elements.values())
        ]

    def group_structure(self, particle: Particle | str) -> GroupStructure:
        """The group structure shared by every element for *particle*

        Raises
        ------
        ValidationError
            If *particle* is invalid or no element has data for it.
        """
        particle = _coerce_enum("MGLibrary", Particle, particle, "particle")
        for atomic_number in sorted(self.elements):
            particle_data = self.elements[atomic_number].particles.get(particle)
            if particle_data is not None:
                return particle_data.group_structure
        raise ValidationError(
            f"MGLibrary '{self.label}': no element has {particle} data."
        )

    def __getitem__(self, atomic_number: int) -> MGElement:
        return self.elements[atomic_number]

    def __contains__(self, atomic_number: object) -> bool:
        return atomic_number in self.elements

    def __len__(self) -> int:
        return len(self.elements)
