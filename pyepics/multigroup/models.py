#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Typed dataclass models for collapsed EPICS datasets

Every model is a frozen-safe ``dataclass`` carrying scalar metadata and
NumPy arrays.  Models are the sole output of the physics/collapse layers and
the sole input of the converter layer, enforcing strict separation of concerns.

Hierarchy
---------
::
    GroupStructure     : Descending energy group bounds in eV
    GroupXS            : Cross sections for a single reaction, all groups
    ScatteringMatrix   : Scattering matrix, group-to-group, within species
    BFPCoefficients    : Stopping power, energy straggling, and momentum transfer cross section, all groups
    CutoffSpec         : CEPXS or RADIANT convention. String, for provenance
    ParticleMG         : Contains all data for a single particle
    MGElement          : Contains electron and/or photon ParticleMG data
    MGLibrary          : A library of multiple MGElements

Units
-----
* Energies are in **eV** unless explicitly noted otherwise.
* Cross sections are in **barns** (ENDF convention).
* Stopping power is in **eV*barns**, energy straggling coefficient is in **barns*eV^2**
* Momentum-transfer values are in **1/Å** (inverse ångström).
"""