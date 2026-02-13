#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Shared pytest fixtures for PyEPICS tests

Provides synthetic ENDF-like data for testing parsers, validators,
and HDF5 converters without requiring real ENDF data files.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture
def sample_eedl_dataset() -> EEDLDataset:
    """Minimal synthetic EEDL dataset for Hydrogen (Z=1)"""
    energy = np.array([10.0, 100.0, 1000.0, 10000.0], dtype="f8")
    xs_vals = np.array([1.0e-20, 5.0e-21, 1.0e-21, 2.0e-22], dtype="f8")

    return EEDLDataset(
        Z=1,
        symbol="H",
        atomic_weight_ratio=0.9992,
        ZA=1000.0,
        cross_sections={
            "xs_tot": CrossSectionRecord(
                label="xs_tot",
                energy=energy,
                cross_section=xs_vals,
            ),
            "xs_el": CrossSectionRecord(
                label="xs_el",
                energy=energy,
                cross_section=xs_vals * 0.5,
            ),
            "xs_lge": CrossSectionRecord(
                label="xs_lge",
                energy=energy,
                cross_section=xs_vals * 0.3,
            ),
            "xs_brem": CrossSectionRecord(
                label="xs_brem",
                energy=energy,
                cross_section=xs_vals * 0.1,
            ),
            "xs_exc": CrossSectionRecord(
                label="xs_exc",
                energy=energy,
                cross_section=xs_vals * 0.05,
            ),
            "xs_ion": CrossSectionRecord(
                label="xs_ion",
                energy=energy,
                cross_section=xs_vals * 0.05,
            ),
        },
        distributions={
            "ang_lge": DistributionRecord(
                label="ang_lge",
                inc_energy=np.array([100.0, 100.0, 1000.0, 1000.0], dtype="f8"),
                value=np.array([-1.0, 0.0, -0.5, 0.5], dtype="f8"),
                probability=np.array([0.3, 0.7, 0.4, 0.6], dtype="f8"),
            ),
        },
        average_energy_losses={
            "loss_exc": AverageEnergyLoss(
                label="loss_exc",
                energy=energy,
                avg_loss=np.array([5.0, 10.0, 20.0, 50.0], dtype="f8"),
            ),
            "loss_brem_spec": AverageEnergyLoss(
                label="loss_brem_spec",
                energy=energy,
                avg_loss=np.array([2.0, 8.0, 30.0, 100.0], dtype="f8"),
            ),
        },
    )


@pytest.fixture
def sample_epdl_dataset() -> EPDLDataset:
    """Minimal synthetic EPDL dataset for Hydrogen (Z=1)"""
    energy = np.array([100.0, 1000.0, 10000.0, 100000.0], dtype="f8")
    xs_vals = np.array([5.0e-22, 3.0e-22, 1.0e-22, 5.0e-23], dtype="f8")

    return EPDLDataset(
        Z=1,
        symbol="H",
        atomic_weight_ratio=0.9992,
        ZA=1000.0,
        cross_sections={
            "xs_tot": CrossSectionRecord(
                label="xs_tot",
                energy=energy,
                cross_section=xs_vals,
            ),
            "xs_coherent": CrossSectionRecord(
                label="xs_coherent",
                energy=energy,
                cross_section=xs_vals * 0.3,
            ),
            "xs_incoherent": CrossSectionRecord(
                label="xs_incoherent",
                energy=energy,
                cross_section=xs_vals * 0.5,
            ),
            "xs_photoelectric": CrossSectionRecord(
                label="xs_photoelectric",
                energy=energy,
                cross_section=xs_vals * 0.2,
            ),
            "xs_pair_total": CrossSectionRecord(
                label="xs_pair_total",
                energy=energy,
                cross_section=np.zeros(4, dtype="f8"),
            ),
            "xs_pair_nuclear": CrossSectionRecord(
                label="xs_pair_nuclear",
                energy=energy,
                cross_section=np.zeros(4, dtype="f8"),
            ),
            "xs_pair_electron": CrossSectionRecord(
                label="xs_pair_electron",
                energy=energy,
                cross_section=np.zeros(4, dtype="f8"),
            ),
        },
        form_factors={
            "ff_coherent": FormFactorRecord(
                label="ff_coherent",
                x=np.array([0.0, 1.0, 2.0, 5.0], dtype="f8"),
                y=np.array([1.0, 0.8, 0.5, 0.1], dtype="f8"),
            ),
            "sf_incoherent": FormFactorRecord(
                label="sf_incoherent",
                x=np.array([0.0, 1.0, 2.0, 5.0], dtype="f8"),
                y=np.array([0.0, 0.3, 0.6, 0.95], dtype="f8"),
            ),
        },
    )


@pytest.fixture
def sample_eadl_dataset() -> EADLDataset:
    """Minimal synthetic EADL dataset for Iron (Z=26)"""
    return EADLDataset(
        Z=26,
        symbol="Fe",
        atomic_weight_ratio=55.845,
        ZA=26000.0,
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
                    SubshellTransition(
                        origin_designator=2,
                        origin_label="L1",
                        secondary_designator=3,
                        secondary_label="L2",
                        energy_eV=5900.0,
                        probability=0.658,
                        is_radiative=False,
                    ),
                ],
            ),
            "L1": SubshellRelaxation(
                designator=2,
                name="L1",
                binding_energy_eV=844.6,
                n_electrons=2.0,
                transitions=[
                    SubshellTransition(
                        origin_designator=5,
                        origin_label="M1",
                        secondary_designator=0,
                        secondary_label="radiative",
                        energy_eV=700.0,
                        probability=0.12,
                        is_radiative=True,
                    ),
                    SubshellTransition(
                        origin_designator=5,
                        origin_label="M1",
                        secondary_designator=6,
                        secondary_label="M2",
                        energy_eV=650.0,
                        probability=0.88,
                        is_radiative=False,
                    ),
                ],
            ),
        },
    )
