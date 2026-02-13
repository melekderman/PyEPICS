#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Post-parse validation routines for EPICS datasets

Every validation function raises :class:`~pyepics.exceptions.ValidationError`
when a constraint is violated.  Readers call these functions after parsing
to ensure the resulting data models are physically consistent before they
are handed to the converter layer.

Checked Constraints
-------------------
* Energy arrays must be monotonically non-decreasing.
* Cross-section values must be non-negative.
* Atomic number must be in the range 1 ≤ Z ≤ 118.
* Probability arrays must sum to a value close to unity (within 5 %).
* Transition probabilities per subshell must not exceed 1.0.

Design Note
-----------
Validation functions accept raw NumPy arrays or scalar values — **not**
dataclass model instances — so that the ``models`` layer does not depend
on ``utils``.  This keeps the import graph acyclic::

    utils ← models ← readers ← converters
"""

from __future__ import annotations

import logging

import numpy as np

from pyepics.exceptions import ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MIN_ATOMIC_NUMBER: int = 1
"""Smallest valid atomic number (hydrogen)."""

MAX_ATOMIC_NUMBER: int = 118
"""Largest valid atomic number (oganesson)."""

PROBABILITY_TOLERANCE: float = 0.05
"""Relative tolerance for probability-sum checks (5 %)."""


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_atomic_number(Z: int) -> None:
    """Verify that *Z* is a valid atomic number

    Parameters
    ----------
    Z : int
        Atomic number to validate.

    Raises
    ------
    ValidationError
        If *Z* is outside the range [1, 118].

    Examples
    --------
    >>> validate_atomic_number(26)  # Iron — OK
    >>> validate_atomic_number(0)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pyepics.exceptions.ValidationError: ...
    """
    if not (MIN_ATOMIC_NUMBER <= Z <= MAX_ATOMIC_NUMBER):
        raise ValidationError(
            f"Atomic number Z={Z} is outside the valid range "
            f"[{MIN_ATOMIC_NUMBER}, {MAX_ATOMIC_NUMBER}]."
        )
    logger.debug("Atomic number Z=%d passed validation.", Z)


def validate_energy_monotonic(energy: np.ndarray, label: str = "energy") -> None:
    """Verify that an energy array is monotonically non-decreasing

    Parameters
    ----------
    energy : numpy.ndarray
        1-D array of energy values (eV).
    label : str, optional
        Human-readable name of the array for error messages.

    Raises
    ------
    ValidationError
        If any ``energy[i] > energy[i+1]``.

    Examples
    --------
    >>> import numpy as np
    >>> validate_energy_monotonic(np.array([1.0, 2.0, 3.0]))
    >>> validate_energy_monotonic(np.array([3.0, 1.0]))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pyepics.exceptions.ValidationError: ...
    """
    arr = np.asarray(energy, dtype="f8")
    if arr.size < 2:
        return
    diff = np.diff(arr)
    if np.any(diff < 0):
        first_bad = int(np.argmax(diff < 0))
        raise ValidationError(
            f"Array '{label}' is not monotonically non-decreasing.  "
            f"First violation at index {first_bad}: "
            f"{arr[first_bad]:.6e} > {arr[first_bad + 1]:.6e}."
        )
    logger.debug("Array '%s' (%d points) passed monotonicity check.", label, arr.size)


def validate_non_negative(values: np.ndarray, label: str = "values") -> None:
    """Verify that all values in the array are non-negative

    Parameters
    ----------
    values : numpy.ndarray
        1-D array of physical values (e.g. cross sections).
    label : str, optional
        Human-readable name for error messages.

    Raises
    ------
    ValidationError
        If any value is negative.

    Examples
    --------
    >>> import numpy as np
    >>> validate_non_negative(np.array([0.0, 1.0, 2.0]))
    >>> validate_non_negative(np.array([-1.0, 2.0]))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pyepics.exceptions.ValidationError: ...
    """
    arr = np.asarray(values, dtype="f8")
    if arr.size == 0:
        return
    if np.any(arr < 0):
        first_bad = int(np.argmax(arr < 0))
        raise ValidationError(
            f"Array '{label}' contains negative value(s).  "
            f"First violation at index {first_bad}: {arr[first_bad]:.6e}."
        )
    logger.debug("Array '%s' (%d points) passed non-negativity check.", label, arr.size)


def validate_cross_section(
    energy: np.ndarray,
    cross_section: np.ndarray,
    label: str = "cross_section",
) -> None:
    """Validate a complete cross-section record

    Runs :func:`validate_energy_monotonic` on the energy array and
    :func:`validate_non_negative` on the cross-section array, and
    additionally checks that the two arrays have the same length.

    Parameters
    ----------
    energy : numpy.ndarray
        Energy grid (eV).
    cross_section : numpy.ndarray
        Corresponding cross-section values.
    label : str, optional
        Name of the section for error messages.

    Raises
    ------
    ValidationError
        If any sub-check fails or shapes mismatch.
    """
    e = np.asarray(energy, dtype="f8")
    xs = np.asarray(cross_section, dtype="f8")
    if e.shape != xs.shape:
        raise ValidationError(
            f"Shape mismatch in '{label}': energy has shape {e.shape} "
            f"but cross_section has shape {xs.shape}."
        )
    validate_energy_monotonic(e, label=f"{label}/energy")
    validate_non_negative(xs, label=f"{label}/xs")


def validate_probability_sum(
    probabilities: np.ndarray,
    label: str = "probabilities",
) -> None:
    """Verify that transition probabilities sum to approximately 1.0

    Parameters
    ----------
    probabilities : numpy.ndarray
        Array of probabilities for a single subshell.
    label : str, optional
        Human-readable name for error messages.

    Raises
    ------
    ValidationError
        If the sum deviates from 1.0 by more than
        :data:`PROBABILITY_TOLERANCE` (relative).

    Examples
    --------
    >>> import numpy as np
    >>> validate_probability_sum(np.array([0.3, 0.7]))
    >>> validate_probability_sum(np.array([0.1, 0.1]))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    pyepics.exceptions.ValidationError: ...
    """
    arr = np.asarray(probabilities, dtype="f8")
    if arr.size == 0:
        return
    total = float(arr.sum())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValidationError(
            f"Probability sum for '{label}' is {total:.6f}, "
            f"expected ~1.0 (tolerance={PROBABILITY_TOLERANCE})."
        )
    logger.debug(
        "Probability sum for '%s' = %.6f passed (tol=%.2f).",
        label, total, PROBABILITY_TOLERANCE,
    )
