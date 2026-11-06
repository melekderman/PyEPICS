#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Weighting spectra for group collapse

A group cross section is the average of the pointwise one over the group,
weighted by the particle flux spectrum within the group,

.. math:: \\Sigma_g = \\frac{\\int_g \\Sigma(E)\\, w(E)\\, dE}{\\int_g w(E)\\, dE},

so a weighting spectrum is just a function ``w(E)`` of energy in eV that
approximates the flux.  Only its shape matters: scaling ``w`` by a constant
changes nothing.

Pre-defined spectra
-------------------
``uniform``
    ``w(E) = 1``.  Constant flux in each group, as assumed in the group
    averages of Bienvenue et al. (J. Comput. Phys. 524 (2025) 113740).
``inverse_energy``
    ``w(E) = 1/E``.  The flux of a particle slowing down in a medium.
    Singular at zero, so it cannot weight a group whose lower bound is 0.

A 1/S spectrum (inverse stopping power) is planned.

User-supplied spectra
---------------------
* :class:`WeightingSpectrum` wraps any function of energy, or
* :meth:`WeightingSpectrum.from_table` builds one from a tabulated flux.

This module only defines and validates spectra; the group averaging is done
in :mod:`pyepics.multigroup.collapse`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyepics.exceptions import ValidationError

__all__ = [
    "WeightingSpectrum",
    "uniform",
    "inverse_energy",
    "PREDEFINED",
    "resolve",
]


@dataclass(frozen=True)
class WeightingSpectrum:
    """A named flux weighting function ``w(E)``

    Calling the instance evaluates the function and checks the result.

    Parameters
    ----------
    name : str
        Name recorded for provenance.  Must be non-empty.
    func : callable
        Maps an array of energies (eV) to an array of weights of the same
        shape.  Must be vectorised.  Results must be finite and
        non-negative.
    requires_positive_energy : bool, optional
        True if ``func`` is singular at ``E = 0``.  Evaluating at
        ``E = 0`` then raises, and the collapse refuses groups whose lower
        bound is 0.

    Raises
    ------
    ValidationError
        On construction if *name* is empty or *func* is not callable; on
        evaluation if the energies or the weights are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> flux = WeightingSpectrum("one_over_sqrt_e", lambda e: e**-0.5)
    >>> flux(np.array([1.0, 4.0]))
    array([1. , 0.5])
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    requires_positive_energy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("WeightingSpectrum: name must be a non-empty string.")
        if not callable(self.func):
            raise ValidationError(
                f"WeightingSpectrum '{self.name}': func must be callable."
            )

    def __call__(self, energy) -> np.ndarray:
        """Flux weight ``w(E)`` at *energy* (eV), a scalar or array"""
        owner = f"WeightingSpectrum '{self.name}'"
        energy = np.asarray(energy, dtype="f8")
        if not np.all(np.isfinite(energy)):
            raise ValidationError(f"{owner}: energies must be finite.")
        if np.any(energy < 0.0) or (
            self.requires_positive_energy and np.any(energy == 0.0)
        ):
            bound = "positive" if self.requires_positive_energy else "non-negative"
            raise ValidationError(f"{owner}: energies must be {bound}.")
        weights = np.asarray(self.func(energy), dtype="f8")
        if weights.shape != energy.shape:
            raise ValidationError(
                f"{owner}: func returned shape {weights.shape} for energies of "
                f"shape {energy.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValidationError(f"{owner}: weights must be finite and non-negative.")
        return weights

    @classmethod
    def from_table(
        cls,
        name: str,
        energy: np.ndarray,
        weight: np.ndarray,
        interpolation: str = "loglog",
    ) -> WeightingSpectrum:
        """Weighting spectrum interpolated from a tabulated flux, e.g. a previous run

        Parameters
        ----------
        name : str
            Name recorded for provenance.
        energy : numpy.ndarray
            Energies in eV, shape ``(N,)``, ``N >= 2``, strictly ascending
            and positive.
        weight : numpy.ndarray
            Flux values at *energy*, shape ``(N,)``.  Finite and
            non-negative, and strictly positive for ``"loglog"``.
        interpolation : {"loglog", "linear"}, optional
            Interpolation between table points.  Log-log suits a flux that
            spans decades on a log-spaced grid.

        Notes
        -----
        The table is not extrapolated: evaluating outside
        ``[energy[0], energy[-1]]`` raises :class:`ValidationError`, so the
        table must cover every group it is used for.

        Raises
        ------
        ValidationError
            If the table is malformed or *interpolation* is unknown.
        """
        owner = f"WeightingSpectrum '{name}'"
        if interpolation not in ("loglog", "linear"):
            raise ValidationError(
                f"{owner}: interpolation must be 'loglog' or 'linear' "
                f"(got {interpolation!r})."
            )
        try:
            table_energy = np.array(energy, dtype="f8")
            table_weight = np.array(weight, dtype="f8")
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{owner}: table must be numeric ({exc}).") from exc
        if (
            table_energy.ndim != 1
            or table_energy.shape != table_weight.shape
            or table_energy.size < 2
        ):
            raise ValidationError(
                f"{owner}: energy and weight must be 1-D arrays of equal "
                f"length >= 2 (got shapes {table_energy.shape} and "
                f"{table_weight.shape})."
            )
        if not (
            np.all(np.isfinite(table_energy)) and np.all(np.isfinite(table_weight))
        ):
            raise ValidationError(f"{owner}: table contains non-finite values.")
        if table_energy[0] <= 0.0 or np.any(np.diff(table_energy) <= 0.0):
            raise ValidationError(
                f"{owner}: energies must be positive and strictly ascending."
            )
        if np.any(table_weight < 0.0) or (
            interpolation == "loglog" and np.any(table_weight <= 0.0)
        ):
            raise ValidationError(
                f"{owner}: weights must be "
                f"{'positive' if interpolation == 'loglog' else 'non-negative'} "
                f"for {interpolation} interpolation."
            )
        table_energy.setflags(write=False)
        table_weight.setflags(write=False)
        lowest, highest = float(table_energy[0]), float(table_energy[-1])

        def tabulated(energy: np.ndarray) -> np.ndarray:
            if np.any(energy < lowest) or np.any(energy > highest):
                raise ValidationError(
                    f"{owner}: energies outside the table range "
                    f"[{lowest:.6e}, {highest:.6e}] eV."
                )
            if interpolation == "linear":
                return np.interp(energy, table_energy, table_weight)
            return np.exp(
                np.interp(np.log(energy), np.log(table_energy), np.log(table_weight))
            )

        return cls(name, tabulated, requires_positive_energy=interpolation == "loglog")


uniform = WeightingSpectrum("uniform", np.ones_like)
inverse_energy = WeightingSpectrum(
    "inverse_energy", np.reciprocal, requires_positive_energy=True
)

#: Pre-defined spectra by name, for APIs and the command line.
PREDEFINED: dict[str, WeightingSpectrum] = {
    spectrum.name: spectrum for spectrum in (uniform, inverse_energy)
}


def resolve(weight: WeightingSpectrum | str | Callable) -> WeightingSpectrum:
    """Weighting spectrum for a spectrum object, a pre-defined name or a function

    Parameters
    ----------
    weight : WeightingSpectrum | str | callable
        A spectrum (returned as is), the name of a pre-defined one, or a
        plain function of energy, which is wrapped using its ``__name__``.

    Returns
    -------
    WeightingSpectrum

    Raises
    ------
    ValidationError
        If a name is unknown or *weight* is none of the above.
    """
    if isinstance(weight, WeightingSpectrum):
        return weight
    if isinstance(weight, str):
        try:
            return PREDEFINED[weight]
        except KeyError:
            valid = ", ".join(repr(n) for n in PREDEFINED)
            raise ValidationError(
                f"Unknown weighting spectrum {weight!r}; pre-defined spectra "
                f"are {valid}."
            ) from None
    if callable(weight):
        return WeightingSpectrum(getattr(weight, "__name__", "custom"), weight)
    raise ValidationError(
        f"weight must be a WeightingSpectrum, a name or a callable "
        f"(got {type(weight).__name__})."
    )
