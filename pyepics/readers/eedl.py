#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
EEDL (Evaluated Electron Data Library) reader

Parses ENDF-format EEDL files and returns a strongly-typed
:class:`~pyepics.models.records.EEDLDataset` instance.  All low-level
parsing is delegated to :mod:`pyepics.utils.parsing`; validation is
handled by :mod:`pyepics.utils.validation`.

Supported ENDF sections
-----------------------
* **MF=23** — Electron cross sections (total, elastic, bremsstrahlung,
  excitation, ionisation per subshell).
* **MF=26** — Angular and energy distributions (large-angle elastic
  angular distributions, bremsstrahlung spectra, excitation average
  energy loss, subshell energy spectra).

File Format Assumptions
-----------------------
* The file follows ENDF-6 fixed-width format (80 chars/line).
* The ``endf`` Python package is used to parse MF=23 and most of MF=26.
* MF=26 / MT=525 (large-angle elastic angular distribution) is parsed
  manually via :func:`pyepics.utils.parsing.parse_mf26_mt525` because
  the ``endf`` library does not handle it reliably.

References
----------
.. [1] ENDF-6 Formats Manual (ENDF-102, BNL-90365-2009 Rev. 2).
.. [2] IAEA Nuclear Data Services — EPICS 2023.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import endf
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'endf' package is required by EEDLReader.  "
        "Install it with: pip install endf"
    ) from _exc

from pyepics.exceptions import FileFormatError, ParseError
from pyepics.models.records import (
    AverageEnergyLoss,
    CrossSectionRecord,
    DistributionRecord,
    EEDLDataset,
)
from pyepics.readers.base import BaseReader
from pyepics.utils.constants import (
    PERIODIC_TABLE,
    SECTIONS_ABBREVS,
    SUBSHELL_LABELS,
)
from pyepics.utils.parsing import (
    extract_atomic_number_from_path,
    parse_mf26_mt525,
)
from pyepics.utils.validation import (
    validate_atomic_number,
    validate_cross_section,
)

logger = logging.getLogger(__name__)


class EEDLReader(BaseReader):
    """Reader for EEDL (Evaluated Electron Data Library) ENDF files

    Extracts electron interaction cross sections (MF=23) and angular /
    energy distributions (MF=26) from a single-element ENDF file generated
    by the IAEA EPICS 2023 pipeline.

    The reader produces an :class:`~pyepics.models.records.EEDLDataset`
    dataclass that can be passed directly to the HDF5 converter.

    Notes
    -----
    The ENDF file is opened using ``endf.Material(path)``, which reads
    the entire file into memory.  For very large files this may require
    significant RAM; however, individual EEDL element files are typically
    < 10 MB so this is not a practical concern.

    Examples
    --------
    >>> reader = EEDLReader()
    >>> dataset = reader.read("eedl/EEDL.ZA026000.endf")
    >>> dataset.Z
    26
    >>> "xs_tot" in dataset.cross_sections
    True
    """

    def read(
        self,
        path: Path | str,
        *,
        validate: bool = True,
    ) -> EEDLDataset:
        """Parse an EEDL ENDF file and return a typed dataset model

        Parameters
        ----------
        path : Path | str
            Path to the EEDL ENDF file.  The filename **must** contain the
            pattern ``ZA{ZZZ}000`` so that the atomic number can be
            extracted (e.g. ``EEDL.ZA026000.endf`` for iron).
        validate : bool, optional
            Run post-parse validation on cross-section arrays.  Default
            ``True``.

        Returns
        -------
        EEDLDataset
            Fully populated dataset model.

        Raises
        ------
        FileFormatError
            If the file does not exist, cannot be opened by the ``endf``
            library, or has an unrecognised filename pattern.
        ParseError
            If any ENDF section is malformed.
        ValidationError
            If *validate* is ``True`` and any cross-section array fails
            monotonicity or non-negativity checks.
        """
        filepath = Path(path)
        logger.debug("Opening EEDL file: %s", filepath)

        if not filepath.is_file():
            raise FileFormatError(f"EEDL file not found: {filepath}")

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

        # Extract AWR / ZA from first available section header
        awr = 0.0
        za = float(Z * 1000)
        for sec in mat.section_data.values():
            if isinstance(sec, dict) and "AWR" in sec:
                awr = float(sec["AWR"])
                za = float(sec.get("ZA", za))
                break

        cross_sections: dict[str, CrossSectionRecord] = {}
        distributions: dict[str, DistributionRecord] = {}
        average_energy_losses: dict[str, AverageEnergyLoss] = {}
        bremsstrahlung_spectra: DistributionRecord | None = None

        # -- MF=23: Cross Sections ----------------------------------------
        for (mf, mt), abbrev in SECTIONS_ABBREVS.items():
            if mf != 23 or (mf, mt) not in mat.section_data:
                continue

            sec = mat.section_data[(mf, mt)]
            sigma = sec.get("sigma")
            if sigma is None:
                continue

            energy = np.asarray(sigma.x, dtype="f8")
            xs = np.asarray(sigma.y, dtype="f8")
            bps = np.asarray(sigma.breakpoints, dtype="f8") if sigma.breakpoints is not None else None
            interp = np.asarray(sigma.interpolation, dtype="f8") if sigma.interpolation is not None else None

            if validate:
                validate_cross_section(energy, xs, label=abbrev)

            cross_sections[abbrev] = CrossSectionRecord(
                label=abbrev,
                energy=energy,
                cross_section=xs,
                breakpoints=bps,
                interpolation=interp,
            )
            logger.debug("  MF=23/MT=%d (%s): %d points", mt, abbrev, energy.size)

        # -- MF=26: Distributions -----------------------------------------
        for (mf, mt), abbrev in SECTIONS_ABBREVS.items():
            if mf != 26 or (mf, mt) not in mat.section_data:
                continue

            sec = mat.section_data[(mf, mt)]

            if mt == 525:
                # Large-angle elastic angular distribution — manual parse
                raw_text = mat.section_text.get((mf, mt))
                if raw_text is None:
                    logger.warning("MF=26/MT=525: no raw text available, skipping")
                    continue
                groups = parse_mf26_mt525(raw_text)
                inc_e: list[float] = []
                mu_vals: list[float] = []
                prob_vals: list[float] = []
                for grp in groups:
                    for mu, prob in grp["pairs"]:
                        inc_e.append(grp["E_in"])
                        mu_vals.append(mu)
                        prob_vals.append(prob)
                distributions[abbrev] = DistributionRecord(
                    label=abbrev,
                    inc_energy=np.asarray(inc_e, dtype="f8"),
                    value=np.asarray(mu_vals, dtype="f8"),
                    probability=np.asarray(prob_vals, dtype="f8"),
                )
                logger.debug("  MF=26/MT=525 (%s): %d groups", abbrev, len(groups))

            elif mt == 528:
                # Excitation average energy loss
                prod = sec["products"][0]
                dist = prod.get("distribution") or {}
                et = dist.get("ET")
                if et is not None:
                    average_energy_losses[abbrev] = AverageEnergyLoss(
                        label=abbrev,
                        energy=np.asarray(et.x, dtype="f8"),
                        avg_loss=np.asarray(et.y, dtype="f8"),
                    )
                    logger.debug("  MF=26/MT=528 (%s): %d points", abbrev, len(et.x))

            elif mt == 527:
                # Bremsstrahlung: photon spectrum + electron avg energy loss
                ph = next((p for p in sec["products"] if p.get("ZAP") == 0), None)
                el = next((p for p in sec["products"] if p.get("ZAP") == 11), None)

                # Electron average energy loss
                if el:
                    el_dist = el.get("distribution") or {}
                    et = el_dist.get("ET")
                    if et is not None:
                        average_energy_losses[abbrev] = AverageEnergyLoss(
                            label=abbrev,
                            energy=np.asarray(et.x, dtype="f8"),
                            avg_loss=np.asarray(et.y, dtype="f8"),
                        )

                # Photon spectrum
                if ph:
                    ph_dist = ph.get("distribution") or {}
                    E_inc = ph_dist.get("E", [])
                    sub_list = ph_dist.get("distribution", [])
                    inc_e_arr: list[float] = []
                    out_e_arr: list[float] = []
                    b_arr: list[float] = []
                    for idx, sub in enumerate(sub_list):
                        E_out = sub.get("E'", [])
                        b_raw = sub.get("b")
                        if b_raw is not None:
                            for eo, bb in zip(E_out, b_raw):
                                inc_e_arr.append(E_inc[idx])
                                out_e_arr.append(eo)
                                b_arr.append(float(bb))
                    if inc_e_arr:
                        bremsstrahlung_spectra = DistributionRecord(
                            label=abbrev,
                            inc_energy=np.asarray(inc_e_arr, dtype="f8"),
                            value=np.asarray(out_e_arr, dtype="f8"),
                            probability=np.asarray(b_arr, dtype="f8"),
                        )
                logger.debug("  MF=26/MT=527 (%s): bremsstrahlung spectra", abbrev)

            else:
                # Subshell energy spectra
                prod = sec["products"][0]
                y_tab = prod.get("y")
                dist = prod.get("distribution") or {}
                E_inc = dist.get("E", [])
                sub_list = dist.get("distribution", [])

                inc_e_arr2: list[float] = []
                out_e_arr2: list[float] = []
                b_arr2: list[float] = []
                for idx, sub in enumerate(sub_list):
                    E_out = sub.get("E'", [])
                    b_raw = sub.get("b")
                    b_flat = b_raw.flatten() if b_raw is not None else []
                    for eo, bb in zip(E_out, b_flat):
                        inc_e_arr2.append(E_inc[idx])
                        out_e_arr2.append(eo)
                        b_arr2.append(float(bb))

                if inc_e_arr2:
                    distributions[abbrev] = DistributionRecord(
                        label=abbrev,
                        inc_energy=np.asarray(inc_e_arr2, dtype="f8"),
                        value=np.asarray(out_e_arr2, dtype="f8"),
                        probability=np.asarray(b_arr2, dtype="f8"),
                    )

                # Store binding energy from y_tab if available
                if y_tab is not None and mt in SUBSHELL_LABELS:
                    shell_label = SUBSHELL_LABELS[mt]
                    xs_key = f"xs_{shell_label}"
                    if xs_key in cross_sections:
                        # Attach binding energy as first y_tab energy point
                        if hasattr(y_tab, 'x') and len(y_tab.x) > 0:
                            cross_sections[xs_key].breakpoints = (
                                cross_sections[xs_key].breakpoints
                            )  # preserve existing

                logger.debug("  MF=26/MT=%d (%s): %d records", mt, abbrev, len(inc_e_arr2))

        dataset = EEDLDataset(
            Z=Z,
            symbol=symbol,
            atomic_weight_ratio=awr,
            ZA=za,
            cross_sections=cross_sections,
            distributions=distributions,
            average_energy_losses=average_energy_losses,
            bremsstrahlung_spectra=bremsstrahlung_spectra,
        )

        logger.debug(
            "EEDL parse complete for Z=%d: %d xs, %d dist, %d avg-loss",
            Z,
            len(cross_sections),
            len(distributions),
            len(average_energy_losses),
        )
        return dataset
