#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Abstract base class for all EPICS dataset readers

Every concrete reader (EEDL, EADL, EPDL) inherits from
:class:`BaseReader` and implements the :meth:`read` method,
which returns a typed dataclass model from :mod:`pyepics.models`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from pyepics.models.records import EEDLDataset, EPDLDataset, EADLDataset

logger = logging.getLogger(__name__)

DatasetModel = Union[EEDLDataset, EPDLDataset, EADLDataset]
"""Type alias for the union of all dataset model types."""


class BaseReader(ABC):
    """Abstract base for ENDF-format EPICS dataset readers

    Subclasses must override :meth:`read` to open a specific ENDF file
    type, delegate parsing to :mod:`pyepics.utils.parsing`, validate
    results via :mod:`pyepics.utils.validation`, and return the
    appropriate dataclass model.

    The *validate* keyword argument controls whether post-parse
    validation is performed.  When ``False`` the reader skips
    monotonicity and non-negativity checks, which can be useful for
    exploratory work with non-standard data.

    Notes
    -----
    Readers must never call HDF5 writing functions — that is the
    responsibility of the converter layer.  The dependency direction is::

        utils ← models ← readers ← converters
    """

    @abstractmethod
    def read(
        self,
        path: Path | str,
        *,
        validate: bool = True,
    ) -> DatasetModel:
        """Parse an ENDF-format EPICS file and return a typed dataset model

        Parameters
        ----------
        path : Path | str
            Filesystem path to the ENDF source file.
        validate : bool, optional
            If ``True`` (default), run post-parse validation checks on
            energies, cross sections, and probabilities.

        Returns
        -------
        DatasetModel
            One of :class:`~pyepics.models.records.EEDLDataset`,
            :class:`~pyepics.models.records.EPDLDataset`, or
            :class:`~pyepics.models.records.EADLDataset`.

        Raises
        ------
        FileFormatError
            If the file cannot be identified as the expected EPICS type.
        ParseError
            If the file content is malformed.
        ValidationError
            If *validate* is ``True`` and any post-parse check fails.
        """
        ...
