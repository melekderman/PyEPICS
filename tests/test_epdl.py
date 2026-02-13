#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Tests for EPDL reader

Covers photon cross-section extraction, form factor parsing,
and validation.
"""

from __future__ import annotations

import pytest

from pyepics.exceptions import FileFormatError
from pyepics.models.records import EPDLDataset


class TestEPDLDataset:
    """Tests using the sample_epdl_dataset fixture"""

    def test_atomic_number(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert sample_epdl_dataset.Z == 1

    def test_symbol(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert sample_epdl_dataset.symbol == "H"

    def test_total_xs_exists(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert "xs_tot" in sample_epdl_dataset.cross_sections

    def test_coherent_xs_exists(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert "xs_coherent" in sample_epdl_dataset.cross_sections

    def test_incoherent_xs_exists(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert "xs_incoherent" in sample_epdl_dataset.cross_sections

    def test_form_factor_coherent(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert "ff_coherent" in sample_epdl_dataset.form_factors
        ff = sample_epdl_dataset.form_factors["ff_coherent"]
        assert ff.x.shape == (4,)
        assert ff.y.shape == (4,)

    def test_scattering_function(self, sample_epdl_dataset: EPDLDataset) -> None:
        assert "sf_incoherent" in sample_epdl_dataset.form_factors

    def test_cross_section_values_positive(self, sample_epdl_dataset: EPDLDataset) -> None:
        import numpy as np
        for key, rec in sample_epdl_dataset.cross_sections.items():
            assert np.all(rec.cross_section >= 0), f"{key} has negative values"


class TestEPDLReader:
    """Tests for EPDL reader error paths"""

    def test_file_not_found(self, tmp_path) -> None:
        from pyepics.readers.epdl import EPDLReader

        reader = EPDLReader()
        with pytest.raises(FileFormatError):
            reader.read(tmp_path / "nonexistent.endf")

    def test_bad_filename(self, tmp_path) -> None:
        from pyepics.readers.epdl import EPDLReader

        bad_file = tmp_path / "wrong_name.dat"
        bad_file.write_text("dummy")
        reader = EPDLReader()
        with pytest.raises(FileFormatError):
            reader.read(bad_file)
