#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Typed dataclass models for EPICS parsed data

All models are plain ``dataclasses`` carrying NumPy arrays and scalar
metadata.  They are the sole output format of the reader layer and the
sole input format accepted by the converter layer.
"""

from __future__ import annotations

from pyepics.models.records import (
    CrossSectionRecord,
    DistributionRecord,
    FormFactorRecord,
    SubshellTransition,
    SubshellRelaxation,
    EEDLDataset,
    EPDLDataset,
    EADLDataset,
)

__all__ = [
    "CrossSectionRecord",
    "DistributionRecord",
    "FormFactorRecord",
    "SubshellTransition",
    "SubshellRelaxation",
    "EEDLDataset",
    "EPDLDataset",
    "EADLDataset",
]
