#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
HDF5 converter for EPICS parsed datasets

Writes deterministic, self-documenting HDF5 files from the typed
dataclass models returned by the reader layer.

HDF5 Layout
-----------
All three dataset types share a common metadata block and then branch
into library-specific groups::

    /metadata/
        Z                   int64    — atomic number
        symbol              string   — element symbol
        ZA                  float64  — ZA identifier (Z × 1000 + A)
        AWR                 float64  — atomic weight ratio

    /EEDL/
        Z_{ZZZ}/
            total/
                energy              float64[]   units: eV
                cross_section       float64[]   units: barns
            elastic_scatter/
                total/  ...
                large_angle/ ...
            ionization/
                total/  ...
                subshells/
                    K/  ...
            bremsstrahlung/ ...
            excitation/ ...

    /EPDL/
        Z_{ZZZ}/
            total/ ...
            coherent_scattering/ ...
            incoherent_scattering/ ...
            photoelectric/ ...
            pair_production/ ...
            form_factors/ ...

    /EADL/
        Z_{ZZZ}/
            subshells/
                K/
                    binding_energy_eV   float64
                    n_electrons         float64
                    radiative/ ...
                    non_radiative/ ...

Physical units are stored as HDF5 dataset attributes
(``ds.attrs["units"] = "eV"``).  Element-level metadata (Z, AWR, etc.)
are stored as group attributes on the ``Z_{ZZZ}`` group.

References
----------
.. [1] HDF5 best practices, The HDF Group.
.. [2] ENDF-6 Formats Manual (ENDF-102).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import h5py
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'h5py' package is required by the HDF5 converter.  "
        "Install it with: pip install h5py"
    ) from _exc

from pyepics.exceptions import ConversionError
from pyepics.models.records import (
    EADLDataset,
    EEDLDataset,
    EPDLDataset,
)
from pyepics.readers.base import DatasetModel
from pyepics.utils.constants import SUBSHELL_LABELS
from pyepics.utils.parsing import (
    build_pdf,
    linear_interpolation,
    small_angle_scattering_cosine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal writers
# ---------------------------------------------------------------------------

def _write_metadata(
    h5f: h5py.File,
    dataset: DatasetModel,
) -> None:
    """Write the ``/metadata`` group to an HDF5 file

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : DatasetModel
        Parsed dataset providing Z, symbol, ZA, and AWR.
    """
    meta = h5f.create_group("metadata")
    meta.create_dataset("Z", data=np.int64(dataset.Z))
    meta.create_dataset("symbol", data=dataset.symbol)
    meta.create_dataset("ZA", data=np.float64(dataset.ZA))
    meta.create_dataset("AWR", data=np.float64(dataset.atomic_weight_ratio))


def _create_xs_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    units: str,
) -> h5py.Dataset:
    """Create a float64 dataset with a ``units`` attribute

    Parameters
    ----------
    group : h5py.Group
        Parent group.
    name : str
        Dataset name.
    data : numpy.ndarray
        Data array.
    units : str
        Physical units string stored as ``ds.attrs["units"]``.

    Returns
    -------
    h5py.Dataset
        The created dataset.
    """
    ds = group.create_dataset(name, data=np.asarray(data, dtype="f8"))
    ds.attrs["units"] = units
    return ds


def _write_eedl(h5f: h5py.File, dataset: EEDLDataset) -> None:
    """Write an EEDL dataset to the ``/EEDL/Z_{ZZZ}`` group

    Reproduces the MCDC-compatible layout used by the original PyEEDL
    pipeline, including interpolation of all cross sections onto a
    common energy grid and computation of small-angle scattering
    cosine distributions.

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file.
    dataset : EEDLDataset
        Parsed EEDL dataset.
    """
    Z = dataset.Z
    root = h5f.create_group(f"EEDL/Z_{Z:03d}")
    root.attrs["Z"] = Z
    root.attrs["symbol"] = dataset.symbol
    root.attrs["AWR"] = dataset.atomic_weight_ratio

    xs = dataset.cross_sections

    # -- Common energy grid from total cross section ---------------------
    if "xs_tot" not in xs:
        logger.warning("No total cross section (xs_tot) for Z=%d", Z)
        return

    xs_energy_grid = xs["xs_tot"].energy
    total_xs = xs["xs_tot"].cross_section

    # Helper to interpolate onto grid
    def interp(key: str) -> np.ndarray:
        if key in xs:
            return linear_interpolation(xs_energy_grid, xs[key].energy, xs[key].cross_section)
        return np.zeros_like(xs_energy_grid)

    xs_sc_total = interp("xs_el")
    xs_sc_la = interp("xs_lge")
    xs_brem = interp("xs_brem")
    xs_exc = interp("xs_exc")
    xs_ion_total = interp("xs_ion")
    xs_sc_sa = xs_sc_total - xs_sc_la

    # Write common grid
    _create_xs_dataset(root, "xs_energy_grid", xs_energy_grid, "eV")

    # Total
    total_grp = root.create_group("total")
    _create_xs_dataset(total_grp, "xs", total_xs, "barns")

    # Elastic scattering
    es_grp = root.create_group("elastic_scattering")
    _create_xs_dataset(es_grp, "xs", xs_sc_total, "barns")

    # Large angle
    la_grp = es_grp.create_group("large_angle")
    _create_xs_dataset(la_grp, "xs", xs_sc_la, "barns")
    if "ang_lge" in dataset.distributions:
        d = dataset.distributions["ang_lge"]
        eg, off, val, PDF = build_pdf(d.inc_energy, d.value, d.probability)
        sc_grp = la_grp.create_group("scattering_cosine")
        _create_xs_dataset(sc_grp, "energy_grid", eg, "eV")
        sc_grp.create_dataset("energy_offset", data=off)
        sc_grp.create_dataset("value", data=val)
        sc_grp.create_dataset("PDF", data=PDF)

    # Small angle
    sa_grp = es_grp.create_group("small_angle")
    _create_xs_dataset(sa_grp, "xs", xs_sc_sa, "barns")
    mask_sa = xs_sc_sa > 0.0
    if np.any(mask_sa):
        eg_sa, off_sa, val_sa, pdf_sa = small_angle_scattering_cosine(
            Z, xs_energy_grid[mask_sa], n_mu=200,
        )
        sc_grp_sa = sa_grp.create_group("scattering_cosine")
        _create_xs_dataset(sc_grp_sa, "energy_grid", eg_sa, "eV")
        sc_grp_sa.create_dataset("energy_offset", data=off_sa)
        sc_grp_sa.create_dataset("value", data=val_sa)
        sc_grp_sa.create_dataset("PDF", data=pdf_sa)

    # Bremsstrahlung
    brem_grp = root.create_group("bremsstrahlung")
    _create_xs_dataset(brem_grp, "xs", xs_brem, "barns")
    if "loss_brem_spec" in dataset.average_energy_losses:
        ael = dataset.average_energy_losses["loss_brem_spec"]
        el_grp = brem_grp.create_group("energy_loss")
        _create_xs_dataset(el_grp, "energy", ael.energy, "eV")
        _create_xs_dataset(el_grp, "value", ael.avg_loss, "eV")

    # Excitation
    exc_grp = root.create_group("excitation")
    _create_xs_dataset(exc_grp, "xs", xs_exc, "barns")
    if "loss_exc" in dataset.average_energy_losses:
        ael = dataset.average_energy_losses["loss_exc"]
        el_grp = exc_grp.create_group("energy_loss")
        _create_xs_dataset(el_grp, "energy", ael.energy, "eV")
        _create_xs_dataset(el_grp, "value", ael.avg_loss, "eV")

    # Ionization
    ion_grp = root.create_group("ionization")
    _create_xs_dataset(ion_grp, "xs", xs_ion_total, "barns")
    subs_grp = ion_grp.create_group("subshells")

    subshell_mts = set(SUBSHELL_LABELS.keys())
    for mt, shell_label in SUBSHELL_LABELS.items():
        xs_key = f"xs_{shell_label}"
        spec_key = f"spec_{shell_label}"

        if xs_key not in xs:
            continue

        shell_xs = linear_interpolation(
            xs_energy_grid, xs[xs_key].energy, xs[xs_key].cross_section,
        )
        sg = subs_grp.create_group(shell_label)
        _create_xs_dataset(sg, "xs", shell_xs, "barns")

        if spec_key in dataset.distributions:
            d = dataset.distributions[spec_key]
            egp, offp, valp, PDFp = build_pdf(d.inc_energy, d.value, d.probability)
            pg = sg.create_group("product")
            _create_xs_dataset(pg, "energy_grid", egp, "eV")
            pg.create_dataset("energy_offset", data=offp)
            pg.create_dataset("value", data=valp)
            pg.create_dataset("PDF", data=PDFp)

    logger.debug("Wrote EEDL data for Z=%d", Z)


def _write_epdl(h5f: h5py.File, dataset: EPDLDataset) -> None:
    """Write an EPDL dataset to the ``/EPDL/Z_{ZZZ}`` group

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file.
    dataset : EPDLDataset
        Parsed EPDL dataset.
    """
    Z = dataset.Z
    root = h5f.create_group(f"EPDL/Z_{Z:03d}")
    root.attrs["Z"] = Z
    root.attrs["symbol"] = dataset.symbol
    root.attrs["AWR"] = dataset.atomic_weight_ratio

    xs = dataset.cross_sections
    ff = dataset.form_factors

    if "xs_tot" not in xs:
        logger.warning("No total cross section for EPDL Z=%d", Z)
        return

    xs_energy_grid = xs["xs_tot"].energy

    def interp(key: str) -> np.ndarray:
        if key in xs:
            return linear_interpolation(xs_energy_grid, xs[key].energy, xs[key].cross_section)
        return np.zeros_like(xs_energy_grid)

    _create_xs_dataset(root, "xs_energy_grid", xs_energy_grid, "eV")

    # Total
    tot_grp = root.create_group("total")
    _create_xs_dataset(tot_grp, "xs", xs["xs_tot"].cross_section, "barns")

    # Coherent scattering
    coh_grp = root.create_group("coherent_scattering")
    _create_xs_dataset(coh_grp, "xs", interp("xs_coherent"), "barns")
    if "ff_coherent" in ff:
        ff_grp = coh_grp.create_group("form_factor")
        _create_xs_dataset(ff_grp, "momentum_transfer", ff["ff_coherent"].x, "1/angstrom")
        ff_grp.create_dataset("value", data=ff["ff_coherent"].y)

    # Incoherent scattering
    inc_grp = root.create_group("incoherent_scattering")
    _create_xs_dataset(inc_grp, "xs", interp("xs_incoherent"), "barns")
    if "sf_incoherent" in ff:
        sf_grp = inc_grp.create_group("scattering_function")
        _create_xs_dataset(sf_grp, "momentum_transfer", ff["sf_incoherent"].x, "1/angstrom")
        sf_grp.create_dataset("value", data=ff["sf_incoherent"].y)

    # Photoelectric
    pe_grp = root.create_group("photoelectric")
    _create_xs_dataset(pe_grp, "xs", interp("xs_photoelectric"), "barns")
    pe_subs = pe_grp.create_group("subshells")
    for mt, shell_label in SUBSHELL_LABELS.items():
        key = f"xs_pe_{shell_label}"
        if key not in xs:
            continue
        sg = pe_subs.create_group(shell_label)
        shell_xs = linear_interpolation(xs_energy_grid, xs[key].energy, xs[key].cross_section)
        _create_xs_dataset(sg, "xs", shell_xs, "barns")

    # Pair production
    pp_grp = root.create_group("pair_production")
    _create_xs_dataset(pp_grp, "xs", interp("xs_pair_total"), "barns")
    nuc_grp = pp_grp.create_group("nuclear")
    _create_xs_dataset(nuc_grp, "xs", interp("xs_pair_nuclear"), "barns")
    ele_grp = pp_grp.create_group("electron")
    _create_xs_dataset(ele_grp, "xs", interp("xs_pair_electron"), "barns")

    logger.debug("Wrote EPDL data for Z=%d", Z)


def _write_eadl(h5f: h5py.File, dataset: EADLDataset) -> None:
    """Write an EADL dataset to the ``/EADL/Z_{ZZZ}`` group

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file.
    dataset : EADLDataset
        Parsed EADL dataset.
    """
    Z = dataset.Z
    root = h5f.create_group(f"EADL/Z_{Z:03d}")
    root.attrs["Z"] = Z
    root.attrs["symbol"] = dataset.symbol
    root.attrs["AWR"] = dataset.atomic_weight_ratio

    root.create_dataset("n_subshells", data=dataset.n_subshells)
    subs_grp = root.create_group("subshells")

    # Summary arrays
    shell_names: list[str] = []
    binding_energies: list[float] = []
    n_electrons_arr: list[float] = []

    for name, shell in dataset.subshells.items():
        shell_names.append(name)
        binding_energies.append(shell.binding_energy_eV)
        n_electrons_arr.append(shell.n_electrons)

        sg = subs_grp.create_group(name)
        sg.attrs["designator"] = shell.designator
        ds_be = sg.create_dataset("binding_energy_eV", data=shell.binding_energy_eV)
        ds_be.attrs["units"] = "eV"
        sg.create_dataset("n_electrons", data=shell.n_electrons)

        if not shell.transitions:
            continue

        # Separate radiative / non-radiative
        rad_trans = [t for t in shell.transitions if t.is_radiative]
        auger_trans = [t for t in shell.transitions if not t.is_radiative]

        if rad_trans:
            rg = sg.create_group("radiative")
            rg.create_dataset(
                "origin_designator",
                data=np.array([t.origin_designator for t in rad_trans], dtype="i4"),
            )
            ds_e = rg.create_dataset(
                "energy_eV",
                data=np.array([t.energy_eV for t in rad_trans], dtype="f8"),
            )
            ds_e.attrs["units"] = "eV"
            rg.create_dataset(
                "probability",
                data=np.array([t.probability for t in rad_trans], dtype="f8"),
            )
            fy = sum(t.probability for t in rad_trans)
            rg.create_dataset("fluorescence_yield", data=fy)

        if auger_trans:
            ag = sg.create_group("non_radiative")
            ag.create_dataset(
                "origin_designator",
                data=np.array([t.origin_designator for t in auger_trans], dtype="i4"),
            )
            ag.create_dataset(
                "secondary_designator",
                data=np.array([t.secondary_designator for t in auger_trans], dtype="i4"),
            )
            ds_e = ag.create_dataset(
                "energy_eV",
                data=np.array([t.energy_eV for t in auger_trans], dtype="f8"),
            )
            ds_e.attrs["units"] = "eV"
            ag.create_dataset(
                "probability",
                data=np.array([t.probability for t in auger_trans], dtype="f8"),
            )
            ay = sum(t.probability for t in auger_trans)
            ag.create_dataset("auger_yield", data=ay)

    # Summary datasets at root level
    if shell_names:
        root.create_dataset(
            "shell_names",
            data=np.array(shell_names, dtype="S8"),
        )
        ds_be = root.create_dataset(
            "binding_energies_eV",
            data=np.array(binding_energies, dtype="f8"),
        )
        ds_be.attrs["units"] = "eV"
        root.create_dataset(
            "n_electrons",
            data=np.array(n_electrons_arr, dtype="f8"),
        )

    logger.debug("Wrote EADL data for Z=%d (%d subshells)", Z, len(shell_names))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_WRITERS = {
    "EEDL": (_write_eedl, EEDLDataset),
    "EADL": (_write_eadl, EADLDataset),
    "EPDL": (_write_epdl, EPDLDataset),
}


def convert_dataset_to_hdf5(
    dataset_type: Literal["EEDL", "EADL", "EPDL"],
    source_path: Path | str,
    output_path: Path | str,
    *,
    validate: bool = True,
    overwrite: bool = False,
) -> None:
    """Read an ENDF source file and write a structured HDF5 file

    This is the main convenience function of the converter layer.  It
    instantiates the appropriate reader, parses the source file, and
    writes the result to an HDF5 file with a deterministic,
    self-documenting group layout.

    Parameters
    ----------
    dataset_type : ``"EEDL"`` | ``"EADL"`` | ``"EPDL"``
        Which EPICS library the source file belongs to.
    source_path : Path | str
        Path to the ENDF source file.
    output_path : Path | str
        Path for the output HDF5 file.  Parent directories are created
        automatically.
    validate : bool, optional
        Run post-parse validation.  Default ``True``.
    overwrite : bool, optional
        If ``True``, overwrite an existing HDF5 file.  If ``False``
        (default), raise :class:`~pyepics.exceptions.ConversionError`
        when the output file already exists.

    Raises
    ------
    ConversionError
        If *overwrite* is ``False`` and *output_path* exists, or if
        any HDF5 write operation fails.
    FileFormatError
        If the source file cannot be opened.
    ParseError
        If the source file content is malformed.
    ValidationError
        If validation is enabled and fails.
    ValueError
        If *dataset_type* is not one of the supported types.

    Examples
    --------
    >>> convert_dataset_to_hdf5(
    ...     "EEDL",
    ...     "eedl/EEDL.ZA026000.endf",
    ...     "output/Fe.h5",
    ...     overwrite=True,
    ... )
    """
    if dataset_type not in _WRITERS:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}.  "
            f"Must be one of: {sorted(_WRITERS.keys())}"
        )

    src = Path(source_path)
    out = Path(output_path)

    if out.exists() and not overwrite:
        raise ConversionError(
            f"Output file {out} already exists and overwrite=False."
        )

    # Select reader
    from pyepics.readers.eedl import EEDLReader
    from pyepics.readers.eadl import EADLReader
    from pyepics.readers.epdl import EPDLReader

    reader_map = {
        "EEDL": EEDLReader,
        "EADL": EADLReader,
        "EPDL": EPDLReader,
    }

    reader = reader_map[dataset_type]()
    logger.debug("Parsing %s from %s", dataset_type, src)
    dataset = reader.read(src, validate=validate)

    # Write HDF5
    out.parent.mkdir(parents=True, exist_ok=True)
    writer_fn, expected_type = _WRITERS[dataset_type]

    if not isinstance(dataset, expected_type):
        raise ConversionError(
            f"Reader returned {type(dataset).__name__}, expected {expected_type.__name__}."
        )

    try:
        mode = "w" if overwrite else "w-"
        with h5py.File(str(out), mode) as h5f:
            _write_metadata(h5f, dataset)
            writer_fn(h5f, dataset)
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(
            f"Failed to write HDF5 file {out}: {exc}"
        ) from exc

    logger.info("Wrote %s HDF5 file: %s", dataset_type, out)


# ---------------------------------------------------------------------------
# Two-step pipeline: raw + MCDC
# ---------------------------------------------------------------------------

def _get_reader(dataset_type: str):
    """Return the correct reader class for a dataset type."""
    from pyepics.readers.eedl import EEDLReader
    from pyepics.readers.eadl import EADLReader
    from pyepics.readers.epdl import EPDLReader
    return {"EEDL": EEDLReader, "EADL": EADLReader, "EPDL": EPDLReader}[dataset_type]


def create_raw_hdf5(
    dataset_type: Literal["EEDL", "EADL", "EPDL"],
    source_path: Path | str,
    output_path: Path | str,
    *,
    validate: bool = True,
    overwrite: bool = False,
) -> None:
    """Parse an ENDF file and write a **raw** HDF5 file

    Raw files preserve the original energy grids, breakpoints, and
    interpolation info exactly as they appear in the ENDF evaluation.
    They are suitable for external users who want full-fidelity data.

    Parameters
    ----------
    dataset_type : ``"EEDL"`` | ``"EADL"`` | ``"EPDL"``
        Which EPICS library.
    source_path : Path | str
        Path to the ENDF source file.
    output_path : Path | str
        Path for the output HDF5 file.
    validate : bool, optional
        Post-parse validation.  Default ``True``.
    overwrite : bool, optional
        Overwrite existing file.  Default ``False``.

    Examples
    --------
    >>> create_raw_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "raw_data/Fe.h5")
    """
    from pyepics.converters.raw_hdf5 import (
        write_raw_eedl,
        write_raw_epdl,
        write_raw_eadl,
    )

    writers = {"EEDL": write_raw_eedl, "EPDL": write_raw_epdl, "EADL": write_raw_eadl}
    if dataset_type not in writers:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    src = Path(source_path)
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise ConversionError(f"Output file {out} already exists and overwrite=False.")

    reader = _get_reader(dataset_type)()
    dataset = reader.read(src, validate=validate)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = "w" if overwrite else "w-"
        with h5py.File(str(out), mode) as h5f:
            writers[dataset_type](h5f, dataset)
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(f"Failed to write raw HDF5 {out}: {exc}") from exc

    logger.info("Wrote raw %s HDF5: %s", dataset_type, out)


def create_mcdc_hdf5(
    dataset_type: Literal["EEDL", "EADL", "EPDL"],
    source_path: Path | str,
    output_path: Path | str,
    *,
    validate: bool = True,
    overwrite: bool = False,
) -> None:
    """Parse an ENDF file and write an **MCDC-format** HDF5 file

    MCDC files have cross sections interpolated onto a common energy
    grid, compressed distribution tables, and analytically computed
    small-angle scattering. They are optimised for transport codes.

    Parameters
    ----------
    dataset_type : ``"EEDL"`` | ``"EADL"`` | ``"EPDL"``
        Which EPICS library.
    source_path : Path | str
        Path to the ENDF source file.
    output_path : Path | str
        Path for the output HDF5 file.
    validate : bool, optional
        Post-parse validation.  Default ``True``.
    overwrite : bool, optional
        Overwrite existing file.  Default ``False``.

    Examples
    --------
    >>> create_mcdc_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "mcdc_data/Fe.h5")
    """
    from pyepics.converters.mcdc_hdf5 import (
        write_mcdc_eedl,
        write_mcdc_epdl,
        write_mcdc_eadl,
    )

    writers = {"EEDL": write_mcdc_eedl, "EPDL": write_mcdc_epdl, "EADL": write_mcdc_eadl}
    if dataset_type not in writers:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    src = Path(source_path)
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise ConversionError(f"Output file {out} already exists and overwrite=False.")

    reader = _get_reader(dataset_type)()
    dataset = reader.read(src, validate=validate)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = "w" if overwrite else "w-"
        with h5py.File(str(out), mode) as h5f:
            writers[dataset_type](h5f, dataset)
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(f"Failed to write MCDC HDF5 {out}: {exc}") from exc

    logger.info("Wrote MCDC %s HDF5: %s", dataset_type, out)
