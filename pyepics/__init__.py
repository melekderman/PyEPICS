#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
PyEPICS - Python library for reading and converting EPICS nuclear data

Parse EEDL, EADL, and EPDL files from the IAEA EPICS (Electron Photon
Interaction Cross Sections) database and convert them into structured
HDF5 format suitable for Monte Carlo transport codes.

Pipeline
--------
1. **Download** ENDF files from IAEA:
   ``python -m pyepics.cli download``

2. **Raw HDF5** (full-fidelity, original grids):
   ``python -m pyepics.cli raw``

3. **MCDC HDF5** (common grid, interpolated, transport-ready):
   ``python -m pyepics.cli mcdc``

4. **Both raw + MCDC** in one step:
   ``python -m pyepics.cli all``

Modules
-------
readers
    ENDF-format file readers for EEDL, EADL, and EPDL datasets.
models
    Typed dataclass records returned by the readers.
converters
    HDF5 converters: raw (full-fidelity) and MCDC (transport-optimised).
io
    Dataset downloader.
utils
    Shared parsing helpers and validation logic.

Examples
--------
>>> from pyepics import EEDLReader, create_raw_hdf5, create_mcdc_hdf5
>>> create_raw_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "raw_data/Fe.h5")
>>> create_mcdc_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "mcdc_data/Fe.h5")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Melek Derman"

from pyepics.readers.eedl import EEDLReader
from pyepics.readers.eadl import EADLReader
from pyepics.readers.epdl import EPDLReader
from pyepics.converters.hdf5 import (
    convert_dataset_to_hdf5,
    create_raw_hdf5,
    create_mcdc_hdf5,
)
from pyepics.exceptions import (
    PyEPICSError,
    ParseError,
    ValidationError,
    FileFormatError,
    ConversionError,
    DownloadError,
)

__all__ = [
    # Version
    "__version__",
    # Readers
    "EEDLReader",
    "EADLReader",
    "EPDLReader",
    # Converter
    "convert_dataset_to_hdf5",
    "create_raw_hdf5",
    "create_mcdc_hdf5",
    # Exceptions
    "PyEPICSError",
    "ParseError",
    "ValidationError",
    "FileFormatError",
    "ConversionError",
    "DownloadError",
]
