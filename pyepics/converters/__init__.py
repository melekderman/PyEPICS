#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
HDF5 converters for EPICS datasets

Provides three conversion APIs:

* :func:`~pyepics.converters.hdf5.convert_dataset_to_hdf5`
    Legacy one-step converter (reads ENDF, writes MCDC-style HDF5).
* :func:`~pyepics.converters.hdf5.create_raw_hdf5`
    Writes a "raw" HDF5 preserving original grids and breakpoints.
* :func:`~pyepics.converters.hdf5.create_mcdc_hdf5`
    Writes an MCDC-optimised HDF5 with common energy grid and PDFs.
* :func:`~pyepics.converters.hdf5.create_combined_mcdc_hdf5`
    Creates a single MCDC HDF5 per element with electron, photon,
    and atomic data combined.
"""

from __future__ import annotations

from pyepics.converters.hdf5 import (
    convert_dataset_to_hdf5,
    create_combined_mcdc_hdf5,
    create_mcdc_hdf5,
    create_raw_hdf5,
)

__all__ = [
    "convert_dataset_to_hdf5",
    "create_raw_hdf5",
    "create_mcdc_hdf5",
    "create_combined_mcdc_hdf5",
]
