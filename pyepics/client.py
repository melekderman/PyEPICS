#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
High-level user API for querying element properties

This module provides a convenient, user-friendly interface to query,
compare, and visualise atomic / electron / photon properties from the
EPICS (EEDL / EPDL / EADL) datasets.

Usage
-----
>>> from pyepics.client import EPICSClient
>>> client = EPICSClient("data/endf")
>>> props = client.get_properties("Fe")
>>> props["Z"]
26
>>> df = client.compare(["Fe", "Cu", "Au"], properties=["Z", "binding_energies"])
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pyepics.exceptions import PyEPICSError, ValidationError
from pyepics.models.records import (
    EADLDataset,
    EEDLDataset,
    EPDLDataset,
)
from pyepics.readers.eadl import EADLReader
from pyepics.readers.eedl import EEDLReader
from pyepics.readers.epdl import EPDLReader
from pyepics.utils.constants import PERIODIC_TABLE

logger = logging.getLogger(__name__)

# Type alias for element identifiers
ElementID = int | str

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYMBOL_TO_Z: dict[str, int] = {
    v["symbol"].lower(): k for k, v in PERIODIC_TABLE.items()
}
_NAME_TO_Z: dict[str, int] = {v["name"].lower(): k for k, v in PERIODIC_TABLE.items()}


def _resolve_element(element: ElementID) -> tuple[int, str]:
    """Resolve an element identifier to (Z, symbol).

    Parameters
    ----------
    element : int or str
        Atomic number, element symbol (case-insensitive), or element name.

    Returns
    -------
    tuple[int, str]
        ``(Z, symbol)``

    Raises
    ------
    ValidationError
        If the element cannot be resolved.
    """
    if isinstance(element, (int, np.integer)):
        z = int(element)
        if z not in PERIODIC_TABLE:
            raise ValidationError(
                f"Atomic number Z={z} is outside the valid range [1, 118]."
            )
        return z, PERIODIC_TABLE[z]["symbol"]

    if isinstance(element, str):
        key = element.strip().lower()
        # Try symbol first
        if key in _SYMBOL_TO_Z:
            z = _SYMBOL_TO_Z[key]
            return z, PERIODIC_TABLE[z]["symbol"]
        # Try full name
        if key in _NAME_TO_Z:
            z = _NAME_TO_Z[key]
            return z, PERIODIC_TABLE[z]["symbol"]
        raise ValidationError(
            f"Unknown element identifier: {element!r}. "
            f"Provide an atomic number (1–118), symbol (e.g. 'Fe'), "
            f"or element name (e.g. 'Iron')."
        )

    raise ValidationError(
        f"Element must be an int or str, got {type(element).__name__}."
    )


def _find_endf_file(data_dir: Path, library: str, z: int) -> Path | None:
    """Locate an ENDF file for the given library and atomic number."""
    lib_upper = library.upper()
    subdir_map = {"EEDL": "eedl", "EPDL": "epdl", "EADL": "eadl"}
    subdir = subdir_map.get(lib_upper)
    if subdir is None:
        return None
    folder = data_dir / subdir
    if not folder.is_dir():
        return None
    # Pattern: EEDL.ZA026000.endf
    za_str = f"{z:03d}000"
    fname = f"{lib_upper}.ZA{za_str}.endf"
    path = folder / fname
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Element result container
# ---------------------------------------------------------------------------


class ElementProperties:
    """Container for all properties of a single element.

    This object behaves like a dictionary but also provides attribute
    access for convenience.

    Attributes
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element symbol.
    name : str
        Element name.
    electron : EEDLDataset or None
        Parsed EEDL data (if available).
    photon : EPDLDataset or None
        Parsed EPDL data (if available).
    atomic : EADLDataset or None
        Parsed EADL data (if available).
    """

    def __init__(
        self,
        z: int,
        symbol: str,
        name: str,
        *,
        electron: EEDLDataset | None = None,
        photon: EPDLDataset | None = None,
        atomic: EADLDataset | None = None,
    ) -> None:
        """Initialise an ElementProperties container."""
        self.Z = z
        self.symbol = symbol
        self.name = name
        self.electron = electron
        self.photon = photon
        self.atomic = atomic

    # -- Derived scalar helpers --

    @property
    def binding_energies(self) -> dict[str, float]:
        """Subshell binding energies (eV) from EADL data.

        Returns
        -------
        dict[str, float]
            Mapping of subshell label → binding energy.  Empty if no
            EADL data is loaded.
        """
        if self.atomic is None:
            return {}
        return {
            name: sub.binding_energy_eV for name, sub in self.atomic.subshells.items()
        }

    @property
    def electron_cross_section_labels(self) -> list[str]:
        """List of available electron cross-section keys."""
        if self.electron is None:
            return []
        return list(self.electron.cross_sections.keys())

    @property
    def photon_cross_section_labels(self) -> list[str]:
        """List of available photon cross-section keys."""
        if self.photon is None:
            return []
        return list(self.photon.cross_sections.keys())

    @property
    def subshells(self) -> list[str]:
        """List of subshell labels from EADL data."""
        if self.atomic is None:
            return []
        return list(self.atomic.subshells.keys())

    @property
    def n_subshells(self) -> int:
        """Number of subshells from EADL data."""
        if self.atomic is None:
            return 0
        return self.atomic.n_subshells

    def to_dict(self) -> dict[str, Any]:
        """Export a flat summary dictionary.

        Returns
        -------
        dict[str, Any]
            Includes scalar metadata, binding energies, and lists of
            available cross-section keys.  Array data is *not* included
            to keep the output concise.
        """
        d: dict[str, Any] = {
            "Z": self.Z,
            "symbol": self.symbol,
            "name": self.name,
            "n_subshells": self.n_subshells,
            "binding_energies": self.binding_energies,
            "electron_cross_sections": self.electron_cross_section_labels,
            "photon_cross_sections": self.photon_cross_section_labels,
            "subshells": self.subshells,
        }
        if self.electron is not None:
            d["atomic_weight_ratio"] = self.electron.atomic_weight_ratio
        elif self.photon is not None:
            d["atomic_weight_ratio"] = self.photon.atomic_weight_ratio
        elif self.atomic is not None:
            d["atomic_weight_ratio"] = self.atomic.atomic_weight_ratio
        return d

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        libs = []
        if self.electron:
            libs.append("EEDL")
        if self.photon:
            libs.append("EPDL")
        if self.atomic:
            libs.append("EADL")
        return (
            f"ElementProperties(Z={self.Z}, symbol={self.symbol!r}, "
            f"name={self.name!r}, libraries=[{', '.join(libs)}])"
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to :meth:`to_dict` keys."""
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Support ``key in props`` membership tests."""
        return key in self.to_dict()


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class EPICSClient:
    """High-level interface for querying EPICS element data.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ``eedl/``, ``epdl/``, ``eadl/``
        sub-folders with ENDF files.  Defaults to ``"data/endf"``
        relative to the current working directory.

    Examples
    --------
    >>> client = EPICSClient("data/endf")
    >>> fe = client.get_element("Fe")
    >>> fe.Z
    26
    >>> fe.binding_energies  # doctest: +SKIP
    {'K': 7112.0, 'L1': 844.6, ...}
    """

    def __init__(self, data_dir: str | Path = "data/endf") -> None:
        """Initialise the client with the path to ENDF data."""
        self._data_dir = Path(data_dir)
        self._eedl_reader = EEDLReader()
        self._epdl_reader = EPDLReader()
        self._eadl_reader = EADLReader()
        # Cache: Z -> (eedl, epdl, eadl)
        self._cache: dict[
            int, tuple[EEDLDataset | None, EPDLDataset | None, EADLDataset | None]
        ] = {}

    # -- Internal loading --

    def _load(
        self, z: int, *, libraries: Sequence[str] = ("EEDL", "EPDL", "EADL")
    ) -> tuple[EEDLDataset | None, EPDLDataset | None, EADLDataset | None]:
        """Load and cache datasets for element *z*."""
        if z in self._cache:
            cached = self._cache[z]
            # If all requested libraries are already cached, return
            eedl, epdl, eadl = cached
            need_eedl = "EEDL" in libraries and eedl is None
            need_epdl = "EPDL" in libraries and epdl is None
            need_eadl = "EADL" in libraries and eadl is None
            if not (need_eedl or need_epdl or need_eadl):
                return cached
            # Otherwise, load missing ones
        else:
            eedl, epdl, eadl = None, None, None

        lib_upper = {lib.upper() for lib in libraries}

        if "EEDL" in lib_upper and eedl is None:
            path = _find_endf_file(self._data_dir, "EEDL", z)
            if path is not None:
                try:
                    eedl = self._eedl_reader.read(path)
                except Exception as exc:
                    logger.warning("Failed to read EEDL for Z=%d: %s", z, exc)

        if "EPDL" in lib_upper and epdl is None:
            path = _find_endf_file(self._data_dir, "EPDL", z)
            if path is not None:
                try:
                    epdl = self._epdl_reader.read(path)
                except Exception as exc:
                    logger.warning("Failed to read EPDL for Z=%d: %s", z, exc)

        if "EADL" in lib_upper and eadl is None:
            path = _find_endf_file(self._data_dir, "EADL", z)
            if path is not None:
                try:
                    eadl = self._eadl_reader.read(path)
                except Exception as exc:
                    logger.warning("Failed to read EADL for Z=%d: %s", z, exc)

        result = (eedl, epdl, eadl)
        self._cache[z] = result
        return result

    # -- Public API --

    def get_element(
        self,
        element: ElementID,
        *,
        libraries: Sequence[str] = ("EEDL", "EPDL", "EADL"),
    ) -> ElementProperties:
        """Retrieve all available data for an element.

        Parameters
        ----------
        element : int or str
            Atomic number, symbol (e.g. ``"Fe"``), or full name
            (e.g. ``"Iron"``).
        libraries : sequence of str, optional
            Which EPICS libraries to load.  Defaults to all three.

        Returns
        -------
        ElementProperties
            Container with parsed datasets and derived properties.

        Raises
        ------
        ValidationError
            If *element* cannot be resolved.

        Examples
        --------
        >>> client = EPICSClient("data/endf")
        >>> fe = client.get_element("Fe")
        >>> fe.symbol
        'Fe'
        """
        z, symbol = _resolve_element(element)
        name = PERIODIC_TABLE[z]["name"]
        eedl, epdl, eadl = self._load(z, libraries=libraries)
        return ElementProperties(
            z, symbol, name, electron=eedl, photon=epdl, atomic=eadl
        )

    def get_properties(
        self,
        element: ElementID,
        *,
        libraries: Sequence[str] = ("EEDL", "EPDL", "EADL"),
    ) -> dict[str, Any]:
        """Return a flat summary dictionary for an element.

        This is a convenience wrapper around :meth:`get_element` that
        returns a plain ``dict`` rather than an :class:`ElementProperties`
        object.

        Parameters
        ----------
        element : int or str
            Element identifier.
        libraries : sequence of str, optional
            Libraries to load.

        Returns
        -------
        dict[str, Any]
            See :meth:`ElementProperties.to_dict`.
        """
        return self.get_element(element, libraries=libraries).to_dict()

    def compare(
        self,
        elements: Sequence[ElementID],
        *,
        properties: Sequence[str] | None = None,
        libraries: Sequence[str] = ("EEDL", "EPDL", "EADL"),
    ) -> list[dict[str, Any]]:
        """Compare properties across multiple elements.

        Parameters
        ----------
        elements : sequence of int or str
            Elements to compare.
        properties : sequence of str or None, optional
            If given, only include these keys in each row.
            Defaults to all scalar properties.
        libraries : sequence of str, optional
            Libraries to load.

        Returns
        -------
        list[dict[str, Any]]
            One dict per element.  If ``pandas`` is available, call
            :meth:`compare_df` instead for a DataFrame.

        Examples
        --------
        >>> client = EPICSClient("data/endf")
        >>> rows = client.compare(["H", "He", "Li"])
        >>> [r["symbol"] for r in rows]
        ['H', 'He', 'Li']
        """
        rows: list[dict[str, Any]] = []
        for elem in elements:
            d = self.get_properties(elem, libraries=libraries)
            if properties is not None:
                d = {k: d[k] for k in properties if k in d}
            rows.append(d)
        return rows

    def compare_df(
        self,
        elements: Sequence[ElementID],
        *,
        properties: Sequence[str] | None = None,
        libraries: Sequence[str] = ("EEDL", "EPDL", "EADL"),
    ):
        """Compare elements and return a pandas DataFrame.

        Requires ``pandas`` to be installed.

        Parameters
        ----------
        elements : sequence of int or str
            Elements to compare.
        properties : sequence of str or None, optional
            Subset of properties to include.
        libraries : sequence of str, optional
            Libraries to load.

        Returns
        -------
        pandas.DataFrame
            One row per element, columns are property names.

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for compare_df(). "
                "Install it with: pip install pandas"
            ) from None

        rows = self.compare(elements, properties=properties, libraries=libraries)
        return pd.DataFrame(rows)

    def binding_energy_table(
        self,
        elements: Sequence[ElementID],
    ):
        """Build a binding-energy table (requires pandas).

        Parameters
        ----------
        elements : sequence of int or str
            Elements to include.

        Returns
        -------
        pandas.DataFrame
            Rows = elements, columns = subshell labels.
            Missing subshells are ``NaN``.

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for binding_energy_table(). "
                "Install it with: pip install pandas"
            ) from None

        records = []
        for elem in elements:
            ep = self.get_element(elem, libraries=["EADL"])
            row: dict[str, Any] = {"Z": ep.Z, "symbol": ep.symbol, "name": ep.name}
            row.update(ep.binding_energies)
            records.append(row)
        return pd.DataFrame(records).set_index("symbol")

    def get_cross_section(
        self,
        element: ElementID,
        label: str,
        *,
        library: str = "EEDL",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a specific cross-section array.

        Parameters
        ----------
        element : int or str
            Element identifier.
        label : str
            Cross-section label (e.g. ``"xs_tot"``).
        library : str
            ``"EEDL"`` or ``"EPDL"``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(energy_eV, cross_section_barns)``

        Raises
        ------
        KeyError
            If the label does not exist for this element.
        """
        ep = self.get_element(element, libraries=[library])
        lib_upper = library.upper()
        if lib_upper == "EEDL":
            ds = ep.electron
        elif lib_upper == "EPDL":
            ds = ep.photon
        else:
            raise ValidationError(
                f"Cross sections are only in EEDL/EPDL, not {library!r}."
            )

        if ds is None:
            raise PyEPICSError(
                f"No {lib_upper} data found for {ep.symbol} (Z={ep.Z}). "
                f"Check that the ENDF files exist in {self._data_dir}."
            )

        if label not in ds.cross_sections:
            available = list(ds.cross_sections.keys())
            raise KeyError(
                f"Cross-section label {label!r} not found for "
                f"{ep.symbol}. Available: {available}"
            )

        rec = ds.cross_sections[label]
        return rec.energy.copy(), rec.cross_section.copy()

    # -- Cache management --

    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        self._cache.clear()

    @property
    def available_elements(self) -> list[int]:
        """Return sorted list of atomic numbers with ENDF data on disk.

        Scans the data directory for EEDL files (as the primary indicator).
        """
        eedl_dir = self._data_dir / "eedl"
        if not eedl_dir.is_dir():
            return []
        zs = []
        for f in sorted(eedl_dir.iterdir()):
            if f.suffix == ".endf" and f.stem.startswith("EEDL.ZA"):
                try:
                    za_part = f.stem.split("ZA")[1]
                    z = int(za_part[:3])
                    zs.append(z)
                except (IndexError, ValueError):
                    pass
        return sorted(zs)
