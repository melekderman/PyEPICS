#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Tests for HDF5 converter

Covers file creation, group layout verification, dataset shapes,
attribute values, and error conditions for all three dataset types.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import h5py
except ImportError:
    pytest.skip("h5py not installed", allow_module_level=True)

from pyepics.converters.hdf5 import convert_dataset_to_hdf5
from pyepics.exceptions import ConversionError
from pyepics.models.records import EADLDataset, EEDLDataset, EPDLDataset


# -----------------------------------------------------------------------
# Direct writer tests (using dataset fixtures)
# -----------------------------------------------------------------------

class TestWriteEEDL:
    """Test EEDL HDF5 output from a synthetic dataset"""

    def test_creates_file(self, tmp_path, sample_eedl_dataset: EEDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eedl, _write_metadata

        out = tmp_path / "test_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eedl_dataset)
            _write_eedl(h5f, sample_eedl_dataset)

        assert out.exists()

    def test_metadata_group(self, tmp_path, sample_eedl_dataset: EEDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eedl, _write_metadata

        out = tmp_path / "test_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eedl_dataset)
            _write_eedl(h5f, sample_eedl_dataset)

        with h5py.File(str(out), "r") as h5f:
            assert "metadata" in h5f
            assert int(h5f["metadata/Z"][()]) == 1
            assert h5f["metadata/symbol"].asstr()[()] == "H"

    def test_eedl_groups_exist(self, tmp_path, sample_eedl_dataset: EEDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eedl, _write_metadata

        out = tmp_path / "test_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eedl_dataset)
            _write_eedl(h5f, sample_eedl_dataset)

        with h5py.File(str(out), "r") as h5f:
            assert "EEDL/Z_001" in h5f
            assert "EEDL/Z_001/total" in h5f
            assert "EEDL/Z_001/elastic_scattering" in h5f
            assert "EEDL/Z_001/bremsstrahlung" in h5f
            assert "EEDL/Z_001/excitation" in h5f
            assert "EEDL/Z_001/ionization" in h5f

    def test_energy_grid_shape(self, tmp_path, sample_eedl_dataset: EEDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eedl, _write_metadata

        out = tmp_path / "test_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eedl_dataset)
            _write_eedl(h5f, sample_eedl_dataset)

        with h5py.File(str(out), "r") as h5f:
            eg = h5f["EEDL/Z_001/xs_energy_grid"][:]
            assert eg.shape == (4,)

    def test_units_attribute(self, tmp_path, sample_eedl_dataset: EEDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eedl, _write_metadata

        out = tmp_path / "test_eedl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eedl_dataset)
            _write_eedl(h5f, sample_eedl_dataset)

        with h5py.File(str(out), "r") as h5f:
            ds = h5f["EEDL/Z_001/xs_energy_grid"]
            assert ds.attrs["units"] == "eV"
            ds_xs = h5f["EEDL/Z_001/total/xs"]
            assert ds_xs.attrs["units"] == "barns"


class TestWriteEPDL:
    """Test EPDL HDF5 output from a synthetic dataset"""

    def test_creates_file(self, tmp_path, sample_epdl_dataset: EPDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_epdl, _write_metadata

        out = tmp_path / "test_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_epdl_dataset)
            _write_epdl(h5f, sample_epdl_dataset)

        assert out.exists()

    def test_epdl_groups(self, tmp_path, sample_epdl_dataset: EPDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_epdl, _write_metadata

        out = tmp_path / "test_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_epdl_dataset)
            _write_epdl(h5f, sample_epdl_dataset)

        with h5py.File(str(out), "r") as h5f:
            assert "EPDL/Z_001" in h5f
            assert "EPDL/Z_001/total" in h5f
            assert "EPDL/Z_001/coherent_scattering" in h5f
            assert "EPDL/Z_001/incoherent_scattering" in h5f
            assert "EPDL/Z_001/photoelectric" in h5f
            assert "EPDL/Z_001/pair_production" in h5f

    def test_form_factor_data(self, tmp_path, sample_epdl_dataset: EPDLDataset) -> None:
        from pyepics.converters.hdf5 import _write_epdl, _write_metadata

        out = tmp_path / "test_epdl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_epdl_dataset)
            _write_epdl(h5f, sample_epdl_dataset)

        with h5py.File(str(out), "r") as h5f:
            ff = h5f["EPDL/Z_001/coherent_scattering/form_factor"]
            assert "momentum_transfer" in ff
            assert ff["momentum_transfer"].shape == (4,)


class TestWriteEADL:
    """Test EADL HDF5 output from a synthetic dataset"""

    def test_creates_file(self, tmp_path, sample_eadl_dataset: EADLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eadl, _write_metadata

        out = tmp_path / "test_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eadl_dataset)
            _write_eadl(h5f, sample_eadl_dataset)

        assert out.exists()

    def test_subshell_structure(self, tmp_path, sample_eadl_dataset: EADLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eadl, _write_metadata

        out = tmp_path / "test_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eadl_dataset)
            _write_eadl(h5f, sample_eadl_dataset)

        with h5py.File(str(out), "r") as h5f:
            assert "EADL/Z_026/subshells/K" in h5f
            assert "EADL/Z_026/subshells/L1" in h5f

    def test_binding_energy_value(self, tmp_path, sample_eadl_dataset: EADLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eadl, _write_metadata

        out = tmp_path / "test_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eadl_dataset)
            _write_eadl(h5f, sample_eadl_dataset)

        with h5py.File(str(out), "r") as h5f:
            be = float(h5f["EADL/Z_026/subshells/K/binding_energy_eV"][()])
            assert be == pytest.approx(7112.0)

    def test_radiative_transitions(self, tmp_path, sample_eadl_dataset: EADLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eadl, _write_metadata

        out = tmp_path / "test_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eadl_dataset)
            _write_eadl(h5f, sample_eadl_dataset)

        with h5py.File(str(out), "r") as h5f:
            assert "EADL/Z_026/subshells/K/radiative" in h5f
            assert "EADL/Z_026/subshells/K/non_radiative" in h5f
            fy = float(h5f["EADL/Z_026/subshells/K/radiative/fluorescence_yield"][()])
            assert fy == pytest.approx(0.342)

    def test_summary_arrays(self, tmp_path, sample_eadl_dataset: EADLDataset) -> None:
        from pyepics.converters.hdf5 import _write_eadl, _write_metadata

        out = tmp_path / "test_eadl.h5"
        with h5py.File(str(out), "w") as h5f:
            _write_metadata(h5f, sample_eadl_dataset)
            _write_eadl(h5f, sample_eadl_dataset)

        with h5py.File(str(out), "r") as h5f:
            be_arr = h5f["EADL/Z_026/binding_energies_eV"][:]
            assert be_arr.shape == (2,)
            assert be_arr[0] == pytest.approx(7112.0)


# -----------------------------------------------------------------------
# Error conditions
# -----------------------------------------------------------------------

class TestConvertErrors:
    """Tests for error handling in convert_dataset_to_hdf5"""

    def test_invalid_dataset_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset_type"):
            convert_dataset_to_hdf5("INVALID", "src.endf", "out.h5")

    def test_overwrite_false_existing_file(self, tmp_path) -> None:
        out = tmp_path / "existing.h5"
        out.touch()
        with pytest.raises(ConversionError, match="already exists"):
            convert_dataset_to_hdf5("EEDL", "src.endf", out, overwrite=False)
