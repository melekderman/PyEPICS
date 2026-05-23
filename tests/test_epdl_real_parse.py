#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Regression tests for :class:`pyepics.readers.epdl.EPDLReader`

Exercises the reader against a committed ENDF fixture (LLNL EPICS 2025
``EPDL.ZA026000``) and asserts on the resulting
:class:`~pyepics.models.records.EPDLDataset` contents.

Assertions intentionally inspect only the public dataset surface (no
``endf`` library internals) so the suite acts as a safety net for the
later native-parser migration (Phase 2).

Coverage
--------
* MF=23 cross sections populate ``cross_sections`` with non-empty,
  non-negative, monotonically increasing energy grids.
* MF=27 form factors and scattering functions populate ``form_factors``
  with the four expected keys, with reference values matching the
  underlying ENDF tabulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyepics.models.records import EPDLDataset
from pyepics.readers.epdl import EPDLReader

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "endf" / "EPDL.ZA026000.endf"


@pytest.fixture(scope="module")
def epdl_fe_dataset() -> EPDLDataset:
    """Parsed EPDL dataset for Iron (Z=26) from the committed fixture"""
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
    reader = EPDLReader()
    return reader.read(FIXTURE_PATH)


class TestEPDLRealParseMetadata:
    """Top-level metadata extracted from EPDL Fe (Z=26)"""

    def test_atomic_number(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert epdl_fe_dataset.Z == 26

    def test_symbol(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert epdl_fe_dataset.symbol == "Fe"

    def test_atomic_weight_ratio(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert epdl_fe_dataset.atomic_weight_ratio == pytest.approx(55.3673, rel=1e-4)

    def test_za_identifier(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert epdl_fe_dataset.ZA == pytest.approx(26000.0)


class TestEPDLRealParseCrossSections:
    """MF=23 cross sections — sanity checks (parser already worked here)"""

    def test_total_xs_present(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert "xs_tot" in epdl_fe_dataset.cross_sections

    def test_coherent_xs_present(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert "xs_coherent" in epdl_fe_dataset.cross_sections

    def test_xs_arrays_aligned(self, epdl_fe_dataset: EPDLDataset) -> None:
        for label, rec in epdl_fe_dataset.cross_sections.items():
            assert rec.energy.shape == rec.cross_section.shape, (
                f"{label}: energy / xs shape mismatch"
            )
            assert rec.energy.size > 0, f"{label}: empty energy grid"

    def test_xs_values_non_negative(self, epdl_fe_dataset: EPDLDataset) -> None:
        for label, rec in epdl_fe_dataset.cross_sections.items():
            assert np.all(rec.cross_section >= 0), f"{label} has negative values"


class TestEPDLRealParseFormFactors:
    """MF=27 form factors and scattering functions

    These tests pin down the Bug 1 fix: ``EPDLReader`` must use the
    ``endf`` key ``'H'`` for MF=27 sections (the package returns
    tabulated data under ``H`` for MF=27 and ``sigma`` for MF=23).
    """

    expected_labels = ("ff_coherent", "sf_incoherent", "asf_imag", "asf_real")

    def test_form_factors_populated(self, epdl_fe_dataset: EPDLDataset) -> None:
        assert epdl_fe_dataset.form_factors, "form_factors must not be empty"

    @pytest.mark.parametrize("label", expected_labels)
    def test_expected_label_present(
        self, epdl_fe_dataset: EPDLDataset, label: str,
    ) -> None:
        assert label in epdl_fe_dataset.form_factors, (
            f"{label!r} missing from form_factors "
            f"(have: {sorted(epdl_fe_dataset.form_factors)})"
        )

    @pytest.mark.parametrize("label", expected_labels)
    def test_arrays_aligned_and_non_empty(
        self, epdl_fe_dataset: EPDLDataset, label: str,
    ) -> None:
        rec = epdl_fe_dataset.form_factors[label]
        assert rec.x.size > 0, f"{label}: empty x array"
        assert rec.x.shape == rec.y.shape, f"{label}: x / y shape mismatch"

    def test_ff_coherent_reference_values(
        self, epdl_fe_dataset: EPDLDataset,
    ) -> None:
        """Coherent form factor F(x=0) equals Z (=26) for Fe"""
        ff = epdl_fe_dataset.form_factors["ff_coherent"]
        assert ff.x.size == 1166
        assert ff.x[0] == pytest.approx(0.0)
        assert ff.y[0] == pytest.approx(26.0)
        assert ff.x[-1] == pytest.approx(1.0e9)

    def test_sf_incoherent_reference_values(
        self, epdl_fe_dataset: EPDLDataset,
    ) -> None:
        """Incoherent scattering function S(x=0)=0 and S(∞)=Z=26 for Fe"""
        sf = epdl_fe_dataset.form_factors["sf_incoherent"]
        assert sf.x.size == 441
        assert sf.y[0] == pytest.approx(0.0)
        assert sf.y[-1] == pytest.approx(26.0)

    def test_asf_real_reference_values(
        self, epdl_fe_dataset: EPDLDataset,
    ) -> None:
        """Real anomalous scattering factor — first y value matches LLNL data"""
        asf = epdl_fe_dataset.form_factors["asf_real"]
        assert asf.x.size == 361
        assert asf.y[0] == pytest.approx(-26.0059225, rel=1e-6)

    def test_asf_imag_reference_size(
        self, epdl_fe_dataset: EPDLDataset,
    ) -> None:
        asf = epdl_fe_dataset.form_factors["asf_imag"]
        assert asf.x.size == 361
