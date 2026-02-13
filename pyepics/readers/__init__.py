#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
ENDF-format readers for EPICS datasets

This sub-package provides three concrete reader classes, one for each EPICS
evaluation library:

* :class:`~pyepics.readers.eedl.EEDLReader` — Evaluated Electron Data Library
* :class:`~pyepics.readers.eadl.EADLReader` — Evaluated Atomic Data Library
* :class:`~pyepics.readers.epdl.EPDLReader` — Evaluated Photon Data Library

All readers share the :class:`~pyepics.readers.base.BaseReader` interface.
"""

from __future__ import annotations

from pyepics.readers.eedl import EEDLReader
from pyepics.readers.eadl import EADLReader
from pyepics.readers.epdl import EPDLReader

__all__ = ["EEDLReader", "EADLReader", "EPDLReader"]
