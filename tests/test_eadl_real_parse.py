#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Regression tests for :class:`pyepics.readers.eadl.EADLReader`

Exercises the reader against a committed ENDF fixture (LLNL EPICS 2025
``EADL.ZA026000``) and asserts on the resulting
:class:`~pyepics.models.records.EADLDataset` contents.

Assertions intentionally inspect only the public dataset surface (no
``endf`` library internals) so the suite acts as a safety net for the
later native-parser migration (Phase 2).

Coverage
--------
* MF=28 / MT=533 atomic-relaxation data populates ``subshells`` with
  the expected K / L1 / L2 / L3 / M-series entries.
* Each ``SubshellRelaxation`` carries the correct binding energy and
  electron count.
* Per-shell transitions are correctly reconstructed from the parallel
  ``SUBJ`` / ``SUBK`` / ``ETR`` / ``FTR`` arrays — the ``SUBK == 0``
  rows become radiative, the rest non-radiative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyepics.models.records import EADLDataset
from pyepics.readers.eadl import EADLReader

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "endf" / "EADL.ZA026000.endf"


@pytest.fixture(scope="module")
def eadl_fe_dataset() -> EADLDataset:
    """Parsed EADL dataset for Iron (Z=26) from the committed fixture"""
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
    reader = EADLReader()
    return reader.read(FIXTURE_PATH)


class TestEADLRealParseMetadata:
    """Top-level metadata extracted from EADL Fe (Z=26)"""

    def test_atomic_number(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.Z == 26

    def test_symbol(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.symbol == "Fe"

    def test_atomic_weight_ratio(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.atomic_weight_ratio == pytest.approx(
            55.3672319,
            rel=1e-6,
        )

    def test_za_identifier(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.ZA == pytest.approx(26000.0)

    def test_subshell_count(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.n_subshells == 10


class TestEADLRealParseSubshells:
    """MF=28 / MT=533 subshell relaxation data

    Pins down the Bug 2 fix: ``EADLReader`` must read the ``endf``
    key ``'shells'`` (not ``'subshells'``) and reconstruct individual
    transitions from the parallel ``SUBJ`` / ``SUBK`` / ``ETR`` / ``FTR``
    arrays.
    """

    def test_subshells_populated(self, eadl_fe_dataset: EADLDataset) -> None:
        assert eadl_fe_dataset.subshells, "subshells dict must not be empty"

    def test_all_expected_shells_present(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        expected = {"K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1"}
        assert expected.issubset(
            eadl_fe_dataset.subshells.keys()
        ), f"missing shells: {expected - set(eadl_fe_dataset.subshells)}"

    def test_k_shell_binding_energy(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        assert eadl_fe_dataset.subshells["K"].binding_energy_eV == pytest.approx(
            7117.0,
        )

    def test_k_shell_electrons(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        assert eadl_fe_dataset.subshells["K"].n_electrons == pytest.approx(2.0)

    def test_k_shell_designator(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        assert eadl_fe_dataset.subshells["K"].designator == 1

    def test_l3_shell_electrons(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        assert eadl_fe_dataset.subshells["L3"].n_electrons == pytest.approx(4.0)


class TestEADLRealParseTransitions:
    """Per-shell transition reconstruction from parallel arrays"""

    def test_k_shell_transition_count(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        k = eadl_fe_dataset.subshells["K"]
        assert len(k.transitions) == 48

    def test_k_shell_radiative_count(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        k = eadl_fe_dataset.subshells["K"]
        radiative = [t for t in k.transitions if t.is_radiative]
        assert len(radiative) == 6

    def test_k_shell_first_transition_is_radiative(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        first = eadl_fe_dataset.subshells["K"].transitions[0]
        assert first.is_radiative is True
        assert first.secondary_designator == 0
        assert first.secondary_label == "radiative"

    def test_k_shell_first_transition_values(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        """First K-shell transition: SUBJ=3 (L2), ETR=6349.85, FTR=0.101391"""
        first = eadl_fe_dataset.subshells["K"].transitions[0]
        assert first.origin_designator == 3
        assert first.origin_label == "L2"
        assert first.energy_eV == pytest.approx(6349.85)
        assert first.probability == pytest.approx(0.101391)

    def test_k_shell_transition_probabilities_sum(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        k = eadl_fe_dataset.subshells["K"]
        total = sum(t.probability for t in k.transitions)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_outer_shell_no_transitions(
        self,
        eadl_fe_dataset: EADLDataset,
    ) -> None:
        """Outermost shells (NTR=0 in the EADL data) carry no transitions"""
        for name in ("M5", "N1"):
            shell = eadl_fe_dataset.subshells.get(name)
            if shell is not None:
                assert shell.transitions == []
