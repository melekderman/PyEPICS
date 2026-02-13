#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Custom exception hierarchy for the PyEPICS package

All exceptions raised by PyEPICS inherit from :class:`PyEPICSError`, making it
possible to catch every library-specific error with a single ``except`` clause
while still allowing fine-grained handling when needed.

Exception Hierarchy
-------------------
::

    PyEPICSError
    ├── ParseError          # Malformed file content
    ├── ValidationError     # Failed format or range checks
    ├── FileFormatError     # Wrong file type or header
    ├── ConversionError     # HDF5 write failures
    └── DownloadError       # Network errors (future)
"""

from __future__ import annotations


class PyEPICSError(Exception):
    """Base exception for all PyEPICS errors

    Every exception raised by PyEPICS is a subclass of this type.
    Catching ``PyEPICSError`` therefore catches any library-specific failure
    while still allowing standard Python exceptions (``KeyError``,
    ``TypeError``, etc.) to propagate normally.
    """


class ParseError(PyEPICSError):
    """Raised when an ENDF-format file contains malformed or unparseable content

    This includes unexpected column widths, non-numeric data in numeric
    fields, truncated records, or any deviation from the ENDF-6 fixed-width
    format that prevents successful extraction of physical data.

    Parameters
    ----------
    message : str
        Human-readable description of the parse failure, including the
        file path and approximate line number when available.
    """


class ValidationError(PyEPICSError):
    """Raised when parsed data fails post-parse validation checks

    Validation checks include energy monotonicity, non-negative cross
    sections, probability normalization, and physically meaningful
    atomic-number ranges.  A ``ValidationError`` means the file was
    *parseable* but the resulting data violates expected constraints.

    Parameters
    ----------
    message : str
        Description of the failed check, including the field name,
        expected constraint, and actual value.
    """


class FileFormatError(PyEPICSError):
    """Raised when a file does not match the expected EPICS/ENDF format

    This is raised *before* full parsing begins—for example when the
    filename pattern does not match ``ZA{ZZZ}000`` or when the file
    cannot be opened with the ``endf`` library.

    Parameters
    ----------
    message : str
        Description of the format mismatch, including the expected
        pattern and actual file characteristics.
    """


class ConversionError(PyEPICSError):
    """Raised when HDF5 conversion fails

    This covers any error during HDF5 file creation: permission denied,
    disk full, incompatible dataset shapes, or a missing prerequisite
    dataset that should have been written in an earlier step.

    Parameters
    ----------
    message : str
        Description of the conversion failure and the target HDF5 path.
    """


class DownloadError(PyEPICSError):
    """Raised when dataset download from IAEA fails

    Reserved for future use by the ``io.download`` module.  Covers HTTP
    errors, connection timeouts, and checksum mismatches.

    Parameters
    ----------
    message : str
        Description of the network failure, including the URL attempted.
    """
