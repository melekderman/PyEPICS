#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Backward-compatibility shim for the legacy ``pyeedl`` API

This module re-exports every public symbol from the new ``pyepics``
package under the old ``pyeedl``-style names so that existing scripts
continue to work with zero modifications::

    # Old usage (still works)
    from pyeedl_compat import extract_sections, PERIODIC_TABLE

    # New recommended usage
    from pyepics.readers.eedl import EEDLReader


.. deprecated:: 0.1.0
   Import from ``pyepics`` instead.  This shim will be removed in a
   future release.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "pyeedl_compat is a backward-compatibility shim.  "
    "Please migrate to 'import pyepics' for future releases.",
    DeprecationWarning,
    stacklevel=2,
)

# ── Version ─────────────────────────────────────────────────────────────
__version__ = "1.0.0"

# ── Data constants & mappings ───────────────────────────────────────────
# ── Converter ──────────────────────────────────────────────────────────
from pyepics.converters.hdf5 import convert_dataset_to_hdf5  # noqa: E402, F401
from pyepics.readers.eadl import EADLReader  # noqa: E402, F401

# ── Readers (thin wrappers returning old-style dicts are NOT provided;
#    users of the legacy functions should call readers directly) ─────────
from pyepics.readers.eedl import EEDLReader  # noqa: E402, F401
from pyepics.readers.epdl import EPDLReader  # noqa: E402, F401
from pyepics.utils.constants import (  # noqa: E402, F401
    # Atomic (EADL)
    ATOMIC_MF_MT,
    ATOMIC_SECTIONS_ABBREVS,
    BARN_TO_CM2,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    # Electron (EEDL) — canonical names
    ELECTRON_MF_MT,
    ELECTRON_SECTIONS_ABBREVS,
    ELECTRON_SUBSHELL_LABELS,
    # Physical constants
    FINE_STRUCTURE,
    MF23,
    MF26,
    MF27,
    MF28,
    # Electron (EEDL) — backward-compatible aliases
    MF_MT,
    PERIODIC_TABLE,
    # Photon (EPDL)
    PHOTON_MF_MT,
    PHOTON_SECTIONS_ABBREVS,
    PLANCK_CONSTANT,
    SECTIONS_ABBREVS,
    SPEED_OF_LIGHT,
    SUBSHELL_DESIGNATORS,
    SUBSHELL_DESIGNATORS_INV,
    SUBSHELL_LABELS,
)

# ── Utility / math functions ───────────────────────────────────────────
from pyepics.utils.parsing import (  # noqa: E402, F401
    build_pdf,
    float_endf,
    int_endf,
    linear_interpolation,
    small_angle_eta,
    small_angle_scattering_cosine,
)

__all__ = [
    # Version
    "__version__",
    # Readers (new API)
    "EEDLReader",
    "EPDLReader",
    "EADLReader",
    # Converter (new API)
    "convert_dataset_to_hdf5",
    # Data mappings - Electron (canonical)
    "ELECTRON_MF_MT",
    "ELECTRON_SECTIONS_ABBREVS",
    "ELECTRON_SUBSHELL_LABELS",
    # Data mappings - Electron (backward-compatible aliases)
    "MF_MT",
    "SECTIONS_ABBREVS",
    "SUBSHELL_LABELS",
    "MF23",
    "MF26",
    # Data mappings - Photon
    "PHOTON_MF_MT",
    "PHOTON_SECTIONS_ABBREVS",
    "MF27",
    # Data mappings - Atomic
    "ATOMIC_MF_MT",
    "ATOMIC_SECTIONS_ABBREVS",
    "MF28",
    "SUBSHELL_DESIGNATORS",
    "SUBSHELL_DESIGNATORS_INV",
    # Common
    "PERIODIC_TABLE",
    # Constants
    "FINE_STRUCTURE",
    "ELECTRON_MASS",
    "BARN_TO_CM2",
    "PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "ELECTRON_CHARGE",
    # Utility functions
    "float_endf",
    "int_endf",
    "linear_interpolation",
    "build_pdf",
    "small_angle_eta",
    "small_angle_scattering_cosine",
]
