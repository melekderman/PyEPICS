#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Tests for data-dictionary / mapping-table completeness

Ensures that every (MF, MT) pair found in the shipped ENDF data files
is present in the corresponding PyEPICS mapping dictionary.  This
prevents silent data loss when new evaluations are added.

References
----------
ENDF-6 Formats Manual (BNL-90365-2009-Rev.2), §0.2 and Appendix B,
describe the MF/MT numbering scheme used throughout EEDL, EPDL, and
EADL evaluated libraries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyepics.utils.constants import (
    ATOMIC_MF_MT,
    ATOMIC_SECTIONS_ABBREVS,
    ELECTRON_MF_MT,
    ELECTRON_SECTIONS_ABBREVS,
    ELECTRON_SUBSHELL_LABELS,
    PERIODIC_TABLE,
    PHOTON_MF_MT,
    PHOTON_SECTIONS_ABBREVS,
    SUBSHELL_DESIGNATORS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PYEPICS_ROOT = Path(__file__).resolve().parent.parent
ENDF_DIR = PYEPICS_ROOT / "data" / "endf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_endf_mf_mt(lib_name: str) -> set[tuple[int, int]]:
    """Return the set of (MF, MT) pairs present in ENDF files for *lib_name*.

    Scans *all* ``.endf`` files under ``data/endf/<lib_name>/``.
    Requires the ``endf`` package (``pip install endf``).
    """
    endf = pytest.importorskip("endf", reason="endf package required")
    lib_dir = ENDF_DIR / lib_name
    if not lib_dir.exists():
        pytest.skip(f"ENDF directory {lib_dir} not found")
    pairs: set[tuple[int, int]] = set()
    for fpath in sorted(lib_dir.glob("*.endf")):
        tape = endf.Material(fpath)
        # section_data keys are already (MF, MT) integer tuples
        for mf_mt in tape.section_data:
            pairs.add(mf_mt)
    if not pairs:
        pytest.skip(f"No .endf files found in {lib_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Parametrised completeness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lib_name, desc_dict, abbrev_dict",
    [
        ("eedl", ELECTRON_MF_MT, ELECTRON_SECTIONS_ABBREVS),
        ("epdl", PHOTON_MF_MT, PHOTON_SECTIONS_ABBREVS),
        ("eadl", ATOMIC_MF_MT, ATOMIC_SECTIONS_ABBREVS),
    ],
    ids=["EEDL", "EPDL", "EADL"],
)
class TestMappingCompleteness:
    """All (MF, MT) pairs in ENDF files must be present in mappings."""

    def test_description_dict_covers_endf(self, lib_name, desc_dict, abbrev_dict):
        """Every ENDF (MF, MT) pair must have a human-readable description."""
        endf_pairs = _collect_endf_mf_mt(lib_name)
        missing = sorted(endf_pairs - set(desc_dict.keys()))
        assert not missing, (
            f"{lib_name.upper()} description dict is missing (MF, MT) pairs: "
            f"{missing}.  Add them to the corresponding *_MF_MT dict in "
            f"pyepics/utils/constants.py."
        )

    def test_abbreviation_dict_covers_endf(self, lib_name, desc_dict, abbrev_dict):
        """Every ENDF (MF, MT) pair must have a short abbreviation."""
        endf_pairs = _collect_endf_mf_mt(lib_name)
        missing = sorted(endf_pairs - set(abbrev_dict.keys()))
        assert not missing, (
            f"{lib_name.upper()} abbreviation dict is missing (MF, MT) pairs: "
            f"{missing}.  Add them to the corresponding *_SECTIONS_ABBREVS dict "
            f"in pyepics/utils/constants.py."
        )

    def test_desc_and_abbrev_keys_match(self, lib_name, desc_dict, abbrev_dict):
        """Description and abbreviation dicts must have identical key sets."""
        desc_keys = set(desc_dict.keys())
        abbr_keys = set(abbrev_dict.keys())
        only_desc = sorted(desc_keys - abbr_keys)
        only_abbr = sorted(abbr_keys - desc_keys)
        assert not only_desc, (
            f"{lib_name.upper()}: keys in description dict but not abbreviation: "
            f"{only_desc}"
        )
        assert not only_abbr, (
            f"{lib_name.upper()}: keys in abbreviation dict but not description: "
            f"{only_abbr}"
        )


# ---------------------------------------------------------------------------
# Internal-consistency tests  (always run, no ENDF files required)
# ---------------------------------------------------------------------------

class TestInternalConsistency:
    """Checks that mapping dicts are internally well-formed."""

    def test_periodic_table_has_all_elements(self):
        assert len(PERIODIC_TABLE) >= 118

    def test_subshell_labels_non_empty(self):
        assert len(ELECTRON_SUBSHELL_LABELS) >= 1

    def test_subshell_designators_non_empty(self):
        assert len(SUBSHELL_DESIGNATORS) >= 1

    def test_sections_abbrevs_non_empty(self):
        assert len(ELECTRON_SECTIONS_ABBREVS) >= 1

    def test_no_duplicate_abbreviations_eedl(self):
        """No two EEDL sections share the same abbreviation."""
        vals = list(ELECTRON_SECTIONS_ABBREVS.values())
        assert len(vals) == len(set(vals)), "Duplicate abbreviations in ELECTRON_SECTIONS_ABBREVS"

    def test_no_duplicate_abbreviations_epdl(self):
        """No two EPDL sections share the same abbreviation."""
        vals = list(PHOTON_SECTIONS_ABBREVS.values())
        assert len(vals) == len(set(vals)), "Duplicate abbreviations in PHOTON_SECTIONS_ABBREVS"

    def test_no_duplicate_abbreviations_eadl(self):
        """No two EADL sections share the same abbreviation."""
        vals = list(ATOMIC_SECTIONS_ABBREVS.values())
        assert len(vals) == len(set(vals)), "Duplicate abbreviations in ATOMIC_SECTIONS_ABBREVS"
