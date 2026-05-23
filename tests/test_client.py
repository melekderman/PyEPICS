#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Tests for the high-level client API (EPICSClient, ElementProperties)
and the optional plotting module.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pyepics.client import (
    ElementProperties,
    EPICSClient,
    _resolve_element,
)
from pyepics.exceptions import ValidationError
from pyepics.models.records import (
    AverageEnergyLoss,
    CrossSectionRecord,
    DistributionRecord,
    EADLDataset,
    EEDLDataset,
    EPDLDataset,
    FormFactorRecord,
    SubshellRelaxation,
    SubshellTransition,
)

# ---------------------------------------------------------------------------
# Helpers — re-use conftest-style synthetic datasets
# ---------------------------------------------------------------------------


def _make_eedl(z: int = 26, symbol: str = "Fe") -> EEDLDataset:
    energy = np.array([10.0, 100.0, 1000.0, 10000.0], dtype="f8")
    xs_vals = np.array([1.0e-20, 5.0e-21, 1.0e-21, 2.0e-22], dtype="f8")
    return EEDLDataset(
        Z=z,
        symbol=symbol,
        atomic_weight_ratio=55.845,
        ZA=z * 1000.0,
        cross_sections={
            "xs_tot": CrossSectionRecord("xs_tot", energy, xs_vals),
            "xs_el": CrossSectionRecord("xs_el", energy, xs_vals * 0.5),
        },
        distributions={
            "ang_lge": DistributionRecord(
                "ang_lge",
                np.array([100.0, 100.0], dtype="f8"),
                np.array([-1.0, 0.0], dtype="f8"),
                np.array([0.3, 0.7], dtype="f8"),
            ),
        },
        average_energy_losses={
            "loss_exc": AverageEnergyLoss(
                "loss_exc",
                energy,
                np.array([5.0, 10.0, 20.0, 50.0], dtype="f8"),
            ),
        },
    )


def _make_epdl(z: int = 26, symbol: str = "Fe") -> EPDLDataset:
    energy = np.array([100.0, 1000.0, 10000.0, 100000.0], dtype="f8")
    xs_vals = np.array([5e-22, 3e-22, 1e-22, 5e-23], dtype="f8")
    return EPDLDataset(
        Z=z,
        symbol=symbol,
        atomic_weight_ratio=55.845,
        ZA=z * 1000.0,
        cross_sections={
            "xs_tot": CrossSectionRecord("xs_tot", energy, xs_vals),
        },
        form_factors={
            "ff_coherent": FormFactorRecord(
                "ff_coherent",
                np.array([0.0, 1.0, 2.0], dtype="f8"),
                np.array([1.0, 0.8, 0.5], dtype="f8"),
            ),
        },
    )


def _make_eadl(z: int = 26, symbol: str = "Fe") -> EADLDataset:
    return EADLDataset(
        Z=z,
        symbol=symbol,
        atomic_weight_ratio=55.845,
        ZA=z * 1000.0,
        n_subshells=2,
        subshells={
            "K": SubshellRelaxation(
                designator=1,
                name="K",
                binding_energy_eV=7112.0,
                n_electrons=2.0,
                transitions=[
                    SubshellTransition(
                        origin_designator=2,
                        origin_label="L1",
                        secondary_designator=0,
                        secondary_label="radiative",
                        energy_eV=6391.0,
                        probability=0.342,
                        is_radiative=True,
                    ),
                ],
            ),
            "L1": SubshellRelaxation(
                designator=2,
                name="L1",
                binding_energy_eV=844.6,
                n_electrons=2.0,
                transitions=[],
            ),
        },
    )


# ===================================================================
# Tests: _resolve_element
# ===================================================================


class TestResolveElement:
    """Tests for the element-identifier resolver."""

    def test_resolve_by_z(self):
        z, sym = _resolve_element(26)
        assert z == 26
        assert sym == "Fe"

    def test_resolve_by_symbol(self):
        z, sym = _resolve_element("Fe")
        assert z == 26
        assert sym == "Fe"

    def test_resolve_by_symbol_case_insensitive(self):
        z, sym = _resolve_element("fe")
        assert z == 26
        z2, sym2 = _resolve_element("FE")
        assert z2 == 26

    def test_resolve_by_name(self):
        z, sym = _resolve_element("Iron")
        assert z == 26
        assert sym == "Fe"

    def test_resolve_by_name_case_insensitive(self):
        z, sym = _resolve_element("iron")
        assert z == 26

    def test_resolve_hydrogen(self):
        z, sym = _resolve_element(1)
        assert z == 1 and sym == "H"

    def test_resolve_oganesson(self):
        z, sym = _resolve_element(118)
        assert z == 118 and sym == "Og"

    def test_invalid_z_zero(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            _resolve_element(0)

    def test_invalid_z_negative(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            _resolve_element(-1)

    def test_invalid_z_over_118(self):
        with pytest.raises(ValidationError, match="outside the valid range"):
            _resolve_element(119)

    def test_invalid_string(self):
        with pytest.raises(ValidationError, match="Unknown element"):
            _resolve_element("Unobtanium")

    def test_invalid_type(self):
        with pytest.raises(ValidationError, match="int or str"):
            _resolve_element(3.14)  # type: ignore

    def test_numpy_integer(self):
        z, sym = _resolve_element(np.int64(26))
        assert z == 26 and sym == "Fe"


# ===================================================================
# Tests: ElementProperties
# ===================================================================


class TestElementProperties:
    """Tests for the ElementProperties container."""

    def test_basic_attributes(self):
        ep = ElementProperties(
            26,
            "Fe",
            "Iron",
            electron=_make_eedl(),
            photon=_make_epdl(),
            atomic=_make_eadl(),
        )
        assert ep.Z == 26
        assert ep.symbol == "Fe"
        assert ep.name == "Iron"

    def test_binding_energies(self):
        ep = ElementProperties(26, "Fe", "Iron", atomic=_make_eadl())
        be = ep.binding_energies
        assert be["K"] == 7112.0
        assert be["L1"] == 844.6

    def test_binding_energies_no_atomic(self):
        ep = ElementProperties(26, "Fe", "Iron")
        assert ep.binding_energies == {}

    def test_cross_section_labels(self):
        ep = ElementProperties(26, "Fe", "Iron", electron=_make_eedl())
        assert "xs_tot" in ep.electron_cross_section_labels
        assert "xs_el" in ep.electron_cross_section_labels

    def test_photon_cross_section_labels(self):
        ep = ElementProperties(26, "Fe", "Iron", photon=_make_epdl())
        assert "xs_tot" in ep.photon_cross_section_labels

    def test_subshells(self):
        ep = ElementProperties(26, "Fe", "Iron", atomic=_make_eadl())
        assert ep.subshells == ["K", "L1"]
        assert ep.n_subshells == 2

    def test_no_libraries(self):
        ep = ElementProperties(1, "H", "Hydrogen")
        assert ep.electron is None
        assert ep.photon is None
        assert ep.atomic is None
        assert ep.subshells == []
        assert ep.n_subshells == 0
        assert ep.electron_cross_section_labels == []
        assert ep.photon_cross_section_labels == []

    def test_to_dict(self):
        ep = ElementProperties(
            26,
            "Fe",
            "Iron",
            electron=_make_eedl(),
            atomic=_make_eadl(),
        )
        d = ep.to_dict()
        assert d["Z"] == 26
        assert d["symbol"] == "Fe"
        assert d["name"] == "Iron"
        assert "K" in d["binding_energies"]
        assert "xs_tot" in d["electron_cross_sections"]
        assert "atomic_weight_ratio" in d

    def test_repr(self):
        ep = ElementProperties(26, "Fe", "Iron", electron=_make_eedl())
        r = repr(ep)
        assert "Fe" in r
        assert "EEDL" in r
        assert "Z=26" in r

    def test_getitem(self):
        ep = ElementProperties(26, "Fe", "Iron", electron=_make_eedl())
        assert ep["Z"] == 26
        assert ep["symbol"] == "Fe"

    def test_contains(self):
        ep = ElementProperties(26, "Fe", "Iron", electron=_make_eedl())
        assert "Z" in ep
        assert "nonexistent" not in ep


# ===================================================================
# Tests: EPICSClient
# ===================================================================


class TestEPICSClient:
    """Tests for EPICSClient with mocked file I/O."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        """Client with pre-populated cache (no real files needed)."""
        client = EPICSClient(tmp_path)
        # Pre-fill cache
        client._cache[26] = (
            _make_eedl(26, "Fe"),
            _make_epdl(26, "Fe"),
            _make_eadl(26, "Fe"),
        )
        client._cache[29] = (
            _make_eedl(29, "Cu"),
            _make_epdl(29, "Cu"),
            _make_eadl(29, "Cu"),
        )
        client._cache[79] = (
            _make_eedl(79, "Au"),
            _make_epdl(79, "Au"),
            _make_eadl(79, "Au"),
        )
        return client

    def test_get_element_by_symbol(self, mock_client):
        ep = mock_client.get_element("Fe")
        assert ep.Z == 26
        assert ep.symbol == "Fe"
        assert ep.name == "Iron"

    def test_get_element_by_z(self, mock_client):
        ep = mock_client.get_element(26)
        assert ep.symbol == "Fe"

    def test_get_element_by_name(self, mock_client):
        ep = mock_client.get_element("Iron")
        assert ep.Z == 26

    def test_get_properties(self, mock_client):
        d = mock_client.get_properties("Fe")
        assert isinstance(d, dict)
        assert d["Z"] == 26
        assert d["symbol"] == "Fe"

    def test_compare(self, mock_client):
        rows = mock_client.compare(["Fe", "Cu", "Au"])
        assert len(rows) == 3
        symbols = [r["symbol"] for r in rows]
        assert symbols == ["Fe", "Cu", "Au"]

    def test_compare_with_properties_filter(self, mock_client):
        rows = mock_client.compare(["Fe", "Cu"], properties=["Z", "symbol"])
        assert all(set(r.keys()) == {"Z", "symbol"} for r in rows)

    def test_get_cross_section(self, mock_client):
        energy, xs = mock_client.get_cross_section("Fe", "xs_tot")
        assert isinstance(energy, np.ndarray)
        assert isinstance(xs, np.ndarray)
        assert len(energy) == len(xs)

    def test_get_cross_section_missing_label(self, mock_client):
        with pytest.raises(KeyError, match="xs_nonexistent"):
            mock_client.get_cross_section("Fe", "xs_nonexistent")

    def test_get_cross_section_bad_library(self, mock_client):
        with pytest.raises(ValidationError, match="not"):
            mock_client.get_cross_section("Fe", "xs_tot", library="EADL")

    def test_clear_cache(self, mock_client):
        assert len(mock_client._cache) > 0
        mock_client.clear_cache()
        assert len(mock_client._cache) == 0

    def test_invalid_element(self, mock_client):
        with pytest.raises(ValidationError):
            mock_client.get_element("Unobtanium")

    def test_available_elements_no_dir(self, tmp_path):
        client = EPICSClient(tmp_path)
        assert client.available_elements == []

    def test_available_elements_with_files(self, tmp_path):
        eedl_dir = tmp_path / "eedl"
        eedl_dir.mkdir()
        (eedl_dir / "EEDL.ZA001000.endf").write_text("dummy")
        (eedl_dir / "EEDL.ZA026000.endf").write_text("dummy")
        client = EPICSClient(tmp_path)
        available = client.available_elements
        assert 1 in available
        assert 26 in available


# ===================================================================
# Tests: compare_df and binding_energy_table (pandas-dependent)
# ===================================================================


class TestPandasIntegration:
    """Tests that require pandas."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        client = EPICSClient(tmp_path)
        for z, sym in [(26, "Fe"), (29, "Cu")]:
            client._cache[z] = (
                _make_eedl(z, sym),
                _make_epdl(z, sym),
                _make_eadl(z, sym),
            )
        return client

    def test_compare_df(self, mock_client):
        pd = pytest.importorskip("pandas")
        df = mock_client.compare_df(["Fe", "Cu"])
        assert len(df) == 2
        assert list(df["symbol"]) == ["Fe", "Cu"]

    def test_compare_df_with_properties(self, mock_client):
        pd = pytest.importorskip("pandas")
        df = mock_client.compare_df(["Fe", "Cu"], properties=["Z", "symbol"])
        assert set(df.columns) == {"Z", "symbol"}

    def test_binding_energy_table(self, mock_client):
        pd = pytest.importorskip("pandas")
        df = mock_client.binding_energy_table(["Fe", "Cu"])
        assert "K" in df.columns
        assert "L1" in df.columns
        assert df.loc["Fe", "K"] == 7112.0


# ===================================================================
# Tests: plotting module
# ===================================================================


class TestPlotting:
    """Tests for pyepics.plotting functions (with matplotlib mocked)."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        client = EPICSClient(tmp_path)
        for z, sym in [(26, "Fe"), (29, "Cu")]:
            client._cache[z] = (
                _make_eedl(z, sym),
                _make_epdl(z, sym),
                _make_eadl(z, sym),
            )
        return client

    def test_plot_cross_sections_import_error(self, mock_client):
        """Verify helpful error when matplotlib missing."""
        from pyepics import plotting

        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                plotting.plot_cross_sections(mock_client, "Fe", show=False)

    def test_plot_cross_sections(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import plot_cross_sections

        ax = plot_cross_sections(mock_client, "Fe", show=False)
        assert ax is not None

    def test_compare_cross_sections(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import compare_cross_sections

        ax = compare_cross_sections(mock_client, ["Fe", "Cu"], "xs_tot", show=False)
        assert ax is not None

    def test_plot_binding_energies(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import plot_binding_energies

        ax = plot_binding_energies(mock_client, ["Fe", "Cu"], show=False)
        assert ax is not None

    def test_plot_binding_energies_single_subshell(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import plot_binding_energies

        ax = plot_binding_energies(mock_client, ["Fe", "Cu"], subshell="K", show=False)
        assert ax is not None

    def test_plot_shell_binding_energies(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import plot_shell_binding_energies

        ax = plot_shell_binding_energies(mock_client, "Fe", show=False)
        assert ax is not None

    def test_plot_shell_no_eadl(self, mock_client):
        pytest.importorskip("matplotlib")
        from pyepics.plotting import plot_shell_binding_energies

        # Override cache with no EADL
        mock_client._cache[1] = (_make_eedl(1, "H"), None, None)
        with pytest.raises(ValueError, match="No EADL"):
            plot_shell_binding_energies(mock_client, "H", show=False)
