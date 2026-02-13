#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
EADL (Evaluated Atomic Data Library) reader

Parses ENDF-format EADL files and returns a strongly-typed
:class:`~pyepics.models.records.EADLDataset` instance containing atomic
relaxation data (binding energies, transition probabilities, fluorescence
yields, and Auger yields).

Supported ENDF sections
-----------------------
* **MF=28, MT=533** — Atomic relaxation data (transition arrays per
  subshell, including radiative and non-radiative channels).

File Format Assumptions
-----------------------
* Standard ENDF-6 fixed-width format.
* The ``endf`` Python package handles the MF=28 section, exposing a
  ``subshells`` list of dicts with keys ``SUBI``, ``EBI``, ``ELN``,
  ``NTR``, and ``transitions`` (each with ``SUBJ``, ``SUBK``, ``ETR``,
  ``FTR``).

References
----------
.. [1] ENDF-6 Formats Manual (ENDF-102, BNL-90365-2009 Rev. 2), §28.
.. [2] IAEA Nuclear Data Services — EPICS 2023.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import endf
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'endf' package is required by EADLReader.  "
        "Install it with: pip install endf"
    ) from _exc

from pyepics.exceptions import FileFormatError
from pyepics.models.records import (
    EADLDataset,
    SubshellRelaxation,
    SubshellTransition,
)
from pyepics.readers.base import BaseReader
from pyepics.utils.constants import (
    PERIODIC_TABLE,
    SUBSHELL_DESIGNATORS,
)
from pyepics.utils.parsing import extract_atomic_number_from_path
from pyepics.utils.validation import validate_atomic_number

logger = logging.getLogger(__name__)


class EADLReader(BaseReader):
    """Reader for EADL (Evaluated Atomic Data Library) ENDF files

    Extracts atomic relaxation data from MF=28 / MT=533.  Each subshell
    entry includes the binding energy, electron count, and a list of
    transitions (radiative X-ray emission and non-radiative Auger /
    Coster-Kronig).

    Notes
    -----
    A radiative transition has ``SUBK == 0`` in the ENDF record; a
    non-radiative transition has ``SUBK > 0``.  This convention is
    preserved in the :class:`SubshellTransition` model.

    Examples
    --------
    >>> reader = EADLReader()
    >>> dataset = reader.read("eadl/EADL.ZA026000.endf")
    >>> dataset.Z
    26
    >>> "K" in dataset.subshells
    True
    """

    def read(
        self,
        path: Path | str,
        *,
        validate: bool = True,
    ) -> EADLDataset:
        """Parse an EADL ENDF file and return a typed dataset model

        Parameters
        ----------
        path : Path | str
            Path to the EADL ENDF file.  The filename must contain
            ``ZA{ZZZ}000``.
        validate : bool, optional
            Run post-parse validation on the atomic number.  Default
            ``True``.

        Returns
        -------
        EADLDataset
            Fully populated atomic relaxation dataset model.

        Raises
        ------
        FileFormatError
            If the file is missing or cannot be parsed.
        ParseError
            If the ENDF content is malformed.
        ValidationError
            If *validate* is ``True`` and the atomic number is out of
            range.
        """
        filepath = Path(path)
        logger.debug("Opening EADL file: %s", filepath)

        if not filepath.is_file():
            raise FileFormatError(f"EADL file not found: {filepath}")

        Z = extract_atomic_number_from_path(filepath)
        if validate:
            validate_atomic_number(Z)

        entry = PERIODIC_TABLE.get(Z, {})
        symbol = entry.get("symbol", f"Z{Z:03d}")

        try:
            mat = endf.Material(str(filepath))
        except Exception as exc:
            raise FileFormatError(
                f"Failed to open {filepath} with endf library: {exc}"
            ) from exc

        logger.debug("Loaded ENDF material for Z=%d (%s)", Z, symbol)

        # Extract AWR / ZA
        awr = 0.0
        za = float(Z * 1000)
        for sec in mat.section_data.values():
            if isinstance(sec, dict) and "AWR" in sec:
                awr = float(sec["AWR"])
                za = float(sec.get("ZA", za))
                break

        # MF=28 / MT=533 — atomic relaxation
        key = (28, 533)
        n_subshells = 0
        subshells: dict[str, SubshellRelaxation] = {}

        if key in mat.section_data:
            sec = mat.section_data[key]
            n_subshells = int(sec.get("NSS", 0))

            for subshell_rec in sec.get("subshells", []):
                subi = int(subshell_rec.get("SUBI", 0))
                shell_name = SUBSHELL_DESIGNATORS.get(subi, f"S{subi}")

                transitions: list[SubshellTransition] = []
                for trans in subshell_rec.get("transitions", []):
                    subj = int(trans.get("SUBJ", 0))
                    subk = int(trans.get("SUBK", 0))

                    transitions.append(
                        SubshellTransition(
                            origin_designator=subj,
                            origin_label=SUBSHELL_DESIGNATORS.get(subj, f"S{subj}"),
                            secondary_designator=subk,
                            secondary_label=(
                                "radiative"
                                if subk == 0
                                else SUBSHELL_DESIGNATORS.get(subk, f"S{subk}")
                            ),
                            energy_eV=float(trans.get("ETR", 0.0)),
                            probability=float(trans.get("FTR", 0.0)),
                            is_radiative=(subk == 0),
                        )
                    )

                subshells[shell_name] = SubshellRelaxation(
                    designator=subi,
                    name=shell_name,
                    binding_energy_eV=float(subshell_rec.get("EBI", 0.0)),
                    n_electrons=float(subshell_rec.get("ELN", 0.0)),
                    transitions=transitions,
                )
                logger.debug(
                    "  Subshell %s: BE=%.2f eV, %d transitions",
                    shell_name,
                    subshell_rec.get("EBI", 0.0),
                    len(transitions),
                )
        else:
            logger.warning("No MF=28/MT=533 section found for Z=%d", Z)

        dataset = EADLDataset(
            Z=Z,
            symbol=symbol,
            atomic_weight_ratio=awr,
            ZA=za,
            n_subshells=n_subshells,
            subshells=subshells,
        )

        logger.debug(
            "EADL parse complete for Z=%d: %d subshells",
            Z,
            len(subshells),
        )
        return dataset
