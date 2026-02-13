#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Tests for the two-step pipeline: raw HDF5 + MCDC HDF5 writers

Uses the same synthetic fixtures from conftest.py to verify that
both output formats are correct and contain the expected data.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import h5py
except ImportError:
    pytest.skip("h5py not installed", allow_module_level=True)

from pyepics.converters.raw_hdf5 import (
    write_raw_eedl,
    write_raw_epdl,
    write_raw_eadl,
)
from pyepics.converters.mcdc_hdf5 import (
    write_mcdc_eedl,
    write_mcdc_epdl,
    write_mcdc_eadl,
)
from pyepics.models.records import EADLDataset, EEDLDataset, EPDLDataset


# -----------------------------------------------------------------------
# Raw EEDL
# -----------------------------------------------------------------------

class TestRawEEDL:
    """Test raw EEDL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        assert out.exists()

    def test_metadata(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert int(h5f["metadata/Z"][()]) == 1
            assert h5f["metadata/Sym"].asstr()[()] == "H"

    def test_total_xs_group(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "total_xs/cross_section/energy" in h5f
            assert "total_xs/cross_section/cross_section" in h5f
            assert h5f["total_xs/cross_section/energy"].attrs["units"] == "eV"

    def test_preserves_original_grid(self, tmp_path, sample_eedl_dataset):
        """Raw format must keep the original energy grid, not a common one."""
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            orig_e = sample_eedl_dataset.cross_sections["xs_tot"].energy
            saved_e = h5f["total_xs/cross_section/energy"][:]
            np.testing.assert_array_equal(saved_e, orig_e)

    def test_elastic_scatter_groups(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "elastic_scatter/cross_section/total" in h5f
            assert "elastic_scatter/cross_section/large_angle" in h5f

    def test_ionization_subshell(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "ionization/cross_section/total" in h5f

    def test_excitation_energy_loss(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "raw_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "excitation/distributions/loss_inc_energy" in h5f
            assert "excitation/distributions/avg_loss" in h5f


# -----------------------------------------------------------------------
# MCDC EEDL
# -----------------------------------------------------------------------

class TestMcdcEEDL:
    """Test MCDC EEDL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "mcdc_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eedl(h5f, sample_eedl_dataset)
        assert out.exists()

    def test_top_level_metadata(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "mcdc_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert int(h5f["atomic_number"][()]) == 1
            assert h5f["element_name"].asstr()[()] == "H"

    def test_common_energy_grid(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "mcdc_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "electron_reactions/xs_energy_grid" in h5f

    def test_mcdc_groups_exist(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "mcdc_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            er = "electron_reactions"
            assert f"{er}/total/xs" in h5f
            assert f"{er}/elastic_scattering/xs" in h5f
            assert f"{er}/elastic_scattering/large_angle/xs" in h5f
            assert f"{er}/elastic_scattering/small_angle/xs" in h5f
            assert f"{er}/bremsstrahlung/xs" in h5f
            assert f"{er}/excitation/xs" in h5f
            assert f"{er}/ionization/xs" in h5f

    def test_large_angle_scattering_cosine(self, tmp_path, sample_eedl_dataset):
        out = tmp_path / "mcdc_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eedl(h5f, sample_eedl_dataset)
        with h5py.File(str(out), "r") as h5f:
            sc = "electron_reactions/elastic_scattering/large_angle/scattering_cosine"
            assert f"{sc}/energy_grid" in h5f
            assert f"{sc}/energy_offset" in h5f
            assert f"{sc}/value" in h5f
            assert f"{sc}/PDF" in h5f


# -----------------------------------------------------------------------
# Raw EPDL
# -----------------------------------------------------------------------

class TestRawEPDL:
    """Test raw EPDL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_epdl_dataset):
        out = tmp_path / "raw_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_epdl(h5f, sample_epdl_dataset)
        assert out.exists()

    def test_photoelectric_groups(self, tmp_path, sample_epdl_dataset):
        out = tmp_path / "raw_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_epdl(h5f, sample_epdl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "coherent_scattering/cross_section" in h5f
            assert "incoherent_scattering/cross_section" in h5f


# -----------------------------------------------------------------------
# MCDC EPDL
# -----------------------------------------------------------------------

class TestMcdcEPDL:
    """Test MCDC EPDL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_epdl_dataset):
        out = tmp_path / "mcdc_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_epdl(h5f, sample_epdl_dataset)
        assert out.exists()

    def test_photon_reactions_group(self, tmp_path, sample_epdl_dataset):
        out = tmp_path / "mcdc_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_epdl(h5f, sample_epdl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "photon_reactions/xs_energy_grid" in h5f
            assert "photon_reactions/total/xs" in h5f
            assert "photon_reactions/coherent_scattering/xs" in h5f


# -----------------------------------------------------------------------
# Raw EADL
# -----------------------------------------------------------------------

class TestRawEADL:
    """Test raw EADL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_eadl_dataset):
        out = tmp_path / "raw_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eadl(h5f, sample_eadl_dataset)
        assert out.exists()

    def test_preserves_all_transitions(self, tmp_path, sample_eadl_dataset):
        """Raw format keeps all transitions together (not split by type)."""
        out = tmp_path / "raw_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_raw_eadl(h5f, sample_eadl_dataset)
        with h5py.File(str(out), "r") as h5f:
            assert "atomic_relaxation/subshells/K/transitions" in h5f
            # Raw has is_radiative flag, not separate groups
            assert "atomic_relaxation/subshells/K/transitions/is_radiative" in h5f


# -----------------------------------------------------------------------
# MCDC EADL
# -----------------------------------------------------------------------

class TestMcdcEADL:
    """Test MCDC EADL HDF5 output."""

    def test_creates_file(self, tmp_path, sample_eadl_dataset):
        out = tmp_path / "mcdc_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eadl(h5f, sample_eadl_dataset)
        assert out.exists()

    def test_splits_radiative_nonradiative(self, tmp_path, sample_eadl_dataset):
        """MCDC format must split transitions into radiative/non_radiative."""
        out = tmp_path / "mcdc_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eadl(h5f, sample_eadl_dataset)
        with h5py.File(str(out), "r") as h5f:
            k_grp = "atomic_relaxation/subshells/K"
            # Should have either radiative or non_radiative (or both)
            has_rad = f"{k_grp}/radiative" in h5f
            has_nonrad = f"{k_grp}/non_radiative" in h5f
            assert has_rad or has_nonrad

    def test_fluorescence_yield(self, tmp_path, sample_eadl_dataset):
        out = tmp_path / "mcdc_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            write_mcdc_eadl(h5f, sample_eadl_dataset)
        with h5py.File(str(out), "r") as h5f:
            k_grp = "atomic_relaxation/subshells/K"
            if f"{k_grp}/radiative" in h5f:
                assert f"{k_grp}/radiative/fluorescence_yield" in h5f
