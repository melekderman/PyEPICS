#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Regression tests for :class:`pyepics.readers.eedl.EEDLReader`

Exercises the reader against a committed ENDF fixture (LLNL EPICS 2025
``EEDL.ZA026000``) and asserts on the resulting
:class:`~pyepics.models.records.EEDLDataset` contents.

Assertions intentionally inspect only the public dataset surface (no
``endf`` library internals) so the suite acts as a safety net for the
later native-parser migration (Phase 2).

Coverage
--------
* MF=23 cross sections populate ``cross_sections`` with non-empty,
  monotonic, non-negative arrays for the expected MT numbers.
* MF=26 distributions populate ``distributions`` and
  ``average_energy_losses`` for the manually-parsed (MT=525) and
  ``endf``-parsed (MT=527/528 plus subshell spectra) channels.
* ``bremsstrahlung_spectra`` is populated when MF=26 / MT=527 is
  present in the source file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyepics.models.records import EEDLDataset
from pyepics.readers.eedl import EEDLReader

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "endf" / "EEDL.ZA026000.endf"


@pytest.fixture(scope="module")
def eedl_fe_dataset() -> EEDLDataset:
    """Parsed EEDL dataset for Iron (Z=26) from the committed fixture"""
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
    reader = EEDLReader()
    return reader.read(FIXTURE_PATH)


class TestEEDLRealParseMetadata:
    """Top-level metadata extracted from EEDL Fe (Z=26)"""

    def test_atomic_number(self, eedl_fe_dataset: EEDLDataset) -> None:
        assert eedl_fe_dataset.Z == 26

    def test_symbol(self, eedl_fe_dataset: EEDLDataset) -> None:
        assert eedl_fe_dataset.symbol == "Fe"

    def test_atomic_weight_ratio(self, eedl_fe_dataset: EEDLDataset) -> None:
        assert eedl_fe_dataset.atomic_weight_ratio == pytest.approx(
            55.3672319, rel=1e-6,
        )

    def test_za_identifier(self, eedl_fe_dataset: EEDLDataset) -> None:
        assert eedl_fe_dataset.ZA == pytest.approx(26000.0)


class TestEEDLRealParseCrossSections:
    """MF=23 cross sections (electron interactions)"""

    expected_labels = (
        "xs_tot", "xs_ion", "xs_lge", "xs_el", "xs_brem", "xs_exc", "xs_K",
    )

    def test_cross_sections_populated(self, eedl_fe_dataset: EEDLDataset) -> None:
        assert eedl_fe_dataset.cross_sections, "cross_sections must not be empty"

    @pytest.mark.parametrize("label", expected_labels)
    def test_expected_label_present(
        self, eedl_fe_dataset: EEDLDataset, label: str,
    ) -> None:
        assert label in eedl_fe_dataset.cross_sections, (
            f"{label!r} missing from cross_sections "
            f"(have: {sorted(eedl_fe_dataset.cross_sections)})"
        )

    def test_arrays_aligned_and_non_empty(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        for label, rec in eedl_fe_dataset.cross_sections.items():
            assert rec.energy.size > 0, f"{label}: empty energy grid"
            assert rec.energy.shape == rec.cross_section.shape, (
                f"{label}: energy / xs shape mismatch"
            )

    def test_xs_non_negative(self, eedl_fe_dataset: EEDLDataset) -> None:
        for label, rec in eedl_fe_dataset.cross_sections.items():
            assert np.all(rec.cross_section >= 0), f"{label} has negative values"

    def test_xs_energy_monotonic(self, eedl_fe_dataset: EEDLDataset) -> None:
        for label, rec in eedl_fe_dataset.cross_sections.items():
            assert np.all(np.diff(rec.energy) >= 0), (
                f"{label}: energy grid not monotonic"
            )

    def test_xs_total_reference_values(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """Total electron xs for Fe: 471 points, starts at first-shell BE"""
        xs = eedl_fe_dataset.cross_sections["xs_tot"]
        assert xs.energy.size == 471
        assert xs.energy[0] == pytest.approx(7.87)
        assert xs.energy[-1] == pytest.approx(1.0e11)

    def test_xs_k_shell_binding_energy(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """K-shell electroionisation xs starts at the K binding energy"""
        xs_k = eedl_fe_dataset.cross_sections["xs_K"]
        assert xs_k.energy[0] == pytest.approx(7117.0)


class TestEEDLRealParseDistributions:
    """MF=26 distributions, average energy losses, bremsstrahlung spectra"""

    def test_ang_lge_present(self, eedl_fe_dataset: EEDLDataset) -> None:
        """MT=525 (large-angle elastic angular distribution) is manually parsed"""
        assert "ang_lge" in eedl_fe_dataset.distributions

    def test_ang_lge_arrays_aligned(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        rec = eedl_fe_dataset.distributions["ang_lge"]
        assert rec.inc_energy.shape == rec.value.shape == rec.probability.shape
        assert rec.inc_energy.size > 0

    def test_ang_lge_cosine_range(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """Cosine μ values must lie within [-1, +1]"""
        rec = eedl_fe_dataset.distributions["ang_lge"]
        assert rec.value.min() >= -1.0 - 1e-9
        assert rec.value.max() <= 1.0 + 1e-9

    def test_excitation_loss_present(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """MT=528 yields an entry in average_energy_losses"""
        assert "loss_exc" in eedl_fe_dataset.average_energy_losses

    def test_excitation_loss_arrays_aligned(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        rec = eedl_fe_dataset.average_energy_losses["loss_exc"]
        assert rec.energy.shape == rec.avg_loss.shape
        assert rec.energy.size > 0

    def test_bremsstrahlung_spectra_present(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """MT=527 photon spectrum populates bremsstrahlung_spectra"""
        assert eedl_fe_dataset.bremsstrahlung_spectra is not None

    def test_bremsstrahlung_spectra_arrays_aligned(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        rec = eedl_fe_dataset.bremsstrahlung_spectra
        assert rec is not None
        assert rec.inc_energy.shape == rec.value.shape == rec.probability.shape
        assert rec.inc_energy.size > 0

    def test_subshell_spectrum_present(
        self, eedl_fe_dataset: EEDLDataset,
    ) -> None:
        """MT=534 (K-shell energy spectrum) is populated"""
        assert "spec_K" in eedl_fe_dataset.distributions
