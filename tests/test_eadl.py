#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Tests for EADL reader

Covers atomic relaxation extraction, subshell transition parsing,
and validation of binding energies and transition probabilities.
"""

from __future__ import annotations

import pytest

from pyepics.exceptions import FileFormatError
from pyepics.models.records import EADLDataset, SubshellRelaxation


class TestEADLDataset:
    """Tests using the sample_eadl_dataset fixture"""

    def test_atomic_number(self, sample_eadl_dataset: EADLDataset) -> None:
        assert sample_eadl_dataset.Z == 26

    def test_symbol(self, sample_eadl_dataset: EADLDataset) -> None:
        assert sample_eadl_dataset.symbol == "Fe"

    def test_subshell_count(self, sample_eadl_dataset: EADLDataset) -> None:
        assert sample_eadl_dataset.n_subshells == 2

    def test_k_shell_exists(self, sample_eadl_dataset: EADLDataset) -> None:
        assert "K" in sample_eadl_dataset.subshells

    def test_k_binding_energy(self, sample_eadl_dataset: EADLDataset) -> None:
        assert sample_eadl_dataset.subshells["K"].binding_energy_eV == pytest.approx(7112.0)

    def test_k_transitions(self, sample_eadl_dataset: EADLDataset) -> None:
        k_shell = sample_eadl_dataset.subshells["K"]
        assert len(k_shell.transitions) == 2
        rad = [t for t in k_shell.transitions if t.is_radiative]
        aug = [t for t in k_shell.transitions if not t.is_radiative]
        assert len(rad) == 1
        assert len(aug) == 1

    def test_radiative_transition_energy(self, sample_eadl_dataset: EADLDataset) -> None:
        k_shell = sample_eadl_dataset.subshells["K"]
        rad = [t for t in k_shell.transitions if t.is_radiative][0]
        assert rad.energy_eV == pytest.approx(6391.0)

    def test_transition_probabilities_sum(self, sample_eadl_dataset: EADLDataset) -> None:
        k_shell = sample_eadl_dataset.subshells["K"]
        total = sum(t.probability for t in k_shell.transitions)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_l1_shell_electrons(self, sample_eadl_dataset: EADLDataset) -> None:
        assert sample_eadl_dataset.subshells["L1"].n_electrons == 2.0


class TestEADLReader:
    """Tests for EADL reader error paths"""

    def test_file_not_found(self, tmp_path) -> None:
        from pyepics.readers.eadl import EADLReader

        reader = EADLReader()
        with pytest.raises(FileFormatError):
            reader.read(tmp_path / "nonexistent.endf")

    def test_bad_filename(self, tmp_path) -> None:
        from pyepics.readers.eadl import EADLReader

        bad_file = tmp_path / "no_za_pattern.txt"
        bad_file.write_text("dummy")
        reader = EADLReader()
        with pytest.raises(FileFormatError):
            reader.read(bad_file)
