#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
MCDC-format HDF5 writer for EPICS datasets

Writes HDF5 files optimised for the MC/DC (Monte Carlo / Dynamic Code)
transport code.  These differ from the "raw" format in several ways:

* All cross sections are **interpolated onto a common energy grid**.
* Angular distributions are converted to (energy_grid, energy_offset,
  value, PDF) compressed tables via :func:`~pyepics.utils.parsing.build_pdf`.
* Small-angle elastic scattering cosine PDFs are **analytically computed**
  from screened Rutherford via
  :func:`~pyepics.utils.parsing.small_angle_scattering_cosine`.
* Atomic relaxation transitions are split into radiative / non-radiative
  groups with pre-computed fluorescence and Auger yields.

.. important::
   **This file is intentionally kept separate** from the raw HDF5 writer
   so that the MCDC data layout can be changed frequently without
   affecting the raw-data pipeline.  If the transport code changes its
   expected input format, edit this file only.

Output Directories
------------------
* ``mcdc_data/``             — EEDL (electron)
* ``mcdc_data_photon/``      — EPDL (photon)
* ``mcdc_data_atomic/``      — EADL (atomic)

See Also
--------
pyepics.converters.raw_hdf5 : raw (full-fidelity) writer
pyepics.converters.hdf5      : high-level convenience API
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as _exc:
    raise ImportError(
        "The 'h5py' package is required.  Install with: pip install h5py"
    ) from _exc

from pyepics.exceptions import ConversionError
from pyepics.models.records import (
    EADLDataset,
    EEDLDataset,
    EPDLDataset,
)
from pyepics.utils.constants import SUBSHELL_LABELS
from pyepics.utils.parsing import (
    build_pdf,
    linear_interpolation,
    small_angle_scattering_cosine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_xs_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    units: str,
) -> h5py.Dataset:
    """Create a float64 dataset with a ``units`` attribute."""
    ds = group.create_dataset(name, data=np.asarray(data, dtype="f8"))
    ds.attrs["units"] = units
    return ds


def _write_mcdc_metadata(h5f: h5py.File, dataset) -> None:
    """Write top-level metadata expected by MC/DC."""
    h5f.create_dataset("atomic_weight_ratio", data=np.float64(dataset.atomic_weight_ratio))
    h5f.create_dataset("atomic_number", data=np.int64(dataset.Z))
    h5f.create_dataset("element_name", data=dataset.symbol)


# ---------------------------------------------------------------------------
# EEDL MCDC writer
# ---------------------------------------------------------------------------

def write_mcdc_eedl(h5f: h5py.File, dataset: EEDLDataset) -> None:
    """Write an MCDC-format EEDL HDF5 file

    All cross sections are resampled onto the total-xs energy grid.
    Angular distributions are compressed into (grid, offset, value, PDF)
    tables.  Small-angle elastic scattering cosine PDFs are computed
    analytically from screened Rutherford theory.

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EEDLDataset
        Parsed EEDL dataset.
    """
    _write_mcdc_metadata(h5f, dataset)

    Z = dataset.Z
    xs = dataset.cross_sections
    dist = dataset.distributions
    ael = dataset.average_energy_losses

    if "xs_tot" not in xs:
        logger.warning("No total cross section (xs_tot) for Z=%d", Z)
        return

    root = h5f.create_group("electron_reactions")
    xs_energy_grid = xs["xs_tot"].energy
    total_xs = xs["xs_tot"].cross_section

    # Interpolation helper
    def interp(key: str) -> np.ndarray:
        if key in xs:
            return linear_interpolation(
                xs_energy_grid, xs[key].energy, xs[key].cross_section,
            )
        return np.zeros_like(xs_energy_grid)

    xs_sc_total = interp("xs_el")
    xs_sc_la = interp("xs_lge")
    xs_brem = interp("xs_brem")
    xs_exc = interp("xs_exc")
    xs_ion_total = interp("xs_ion")
    xs_sc_sa = xs_sc_total - xs_sc_la

    # Common grid
    _create_xs_dataset(root, "xs_energy_grid", xs_energy_grid, "eV")

    # --- Total ---
    total_grp = root.create_group("total")
    _create_xs_dataset(total_grp, "xs", total_xs, "barns")

    # --- Elastic scattering ---
    es_grp = root.create_group("elastic_scattering")
    _create_xs_dataset(es_grp, "xs", xs_sc_total, "barns")

    # Large angle
    la_grp = es_grp.create_group("large_angle")
    _create_xs_dataset(la_grp, "xs", xs_sc_la, "barns")
    if "ang_lge" in dist:
        d = dist["ang_lge"]
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

    # --- Bremsstrahlung ---
    brem_grp = root.create_group("bremsstrahlung")
    _create_xs_dataset(brem_grp, "xs", xs_brem, "barns")
    if "loss_brem_spec" in ael:
        a = ael["loss_brem_spec"]
        el_grp = brem_grp.create_group("energy_loss")
        _create_xs_dataset(el_grp, "energy", a.energy, "eV")
        _create_xs_dataset(el_grp, "value", a.avg_loss, "eV")

    # --- Excitation ---
    exc_grp = root.create_group("excitation")
    _create_xs_dataset(exc_grp, "xs", xs_exc, "barns")
    if "loss_exc" in ael:
        a = ael["loss_exc"]
        el_grp = exc_grp.create_group("energy_loss")
        _create_xs_dataset(el_grp, "energy", a.energy, "eV")
        _create_xs_dataset(el_grp, "value", a.avg_loss, "eV")

    # --- Ionization ---
    ion_grp = root.create_group("ionization")
    _create_xs_dataset(ion_grp, "xs", xs_ion_total, "barns")
    subs_grp = ion_grp.create_group("subshells")

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

        # Binding energy (last point of the shell xs energy grid)
        sg.create_dataset(
            "binding_energy",
            data=np.float64(xs[xs_key].energy[0]),
        )

        if spec_key in dist:
            d = dist[spec_key]
            egp, offp, valp, PDFp = build_pdf(d.inc_energy, d.value, d.probability)
            pg = sg.create_group("product")
            _create_xs_dataset(pg, "energy_grid", egp, "eV")
            pg.create_dataset("energy_offset", data=offp)
            pg.create_dataset("value", data=valp)
            pg.create_dataset("PDF", data=PDFp)

    logger.debug("Wrote MCDC EEDL for Z=%d", Z)


# ---------------------------------------------------------------------------
# EPDL MCDC writer
# ---------------------------------------------------------------------------

def write_mcdc_epdl(h5f: h5py.File, dataset: EPDLDataset) -> None:
    """Write an MCDC-format EPDL HDF5 file

    All cross sections are resampled onto the total-xs energy grid.
    Form factors are stored as-is (no interpolation needed).

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EPDLDataset
        Parsed EPDL dataset.
    """
    _write_mcdc_metadata(h5f, dataset)

    xs = dataset.cross_sections
    ff = dataset.form_factors

    if "xs_tot" not in xs:
        logger.warning("No total cross section for EPDL Z=%d", dataset.Z)
        return

    root = h5f.create_group("photon_reactions")
    xs_energy_grid = xs["xs_tot"].energy

    def interp(key: str) -> np.ndarray:
        if key in xs:
            return linear_interpolation(
                xs_energy_grid, xs[key].energy, xs[key].cross_section,
            )
        return np.zeros_like(xs_energy_grid)

    _create_xs_dataset(root, "xs_energy_grid", xs_energy_grid, "eV")

    # Total
    tot_grp = root.create_group("total")
    _create_xs_dataset(tot_grp, "xs", xs["xs_tot"].cross_section, "barns")

    # Coherent scattering
    coh_grp = root.create_group("coherent_scattering")
    _create_xs_dataset(coh_grp, "xs", interp("xs_coherent"), "barns")
    if "ff_coherent" in ff:
        ffg = coh_grp.create_group("form_factor")
        _create_xs_dataset(ffg, "momentum_transfer", ff["ff_coherent"].x, "1/angstrom")
        ffg.create_dataset("value", data=ff["ff_coherent"].y)

    # Incoherent scattering
    inc_grp = root.create_group("incoherent_scattering")
    _create_xs_dataset(inc_grp, "xs", interp("xs_incoherent"), "barns")
    if "sf_incoherent" in ff:
        sfg = inc_grp.create_group("scattering_function")
        _create_xs_dataset(sfg, "momentum_transfer", ff["sf_incoherent"].x, "1/angstrom")
        sfg.create_dataset("value", data=ff["sf_incoherent"].y)

    # Photoelectric
    pe_grp = root.create_group("photoelectric")
    _create_xs_dataset(pe_grp, "xs", interp("xs_photoelectric"), "barns")
    pe_subs = pe_grp.create_group("subshells")
    for mt, shell_label in SUBSHELL_LABELS.items():
        key = f"xs_pe_{shell_label}"
        if key not in xs:
            continue
        sg = pe_subs.create_group(shell_label)
        shell_xs = linear_interpolation(
            xs_energy_grid, xs[key].energy, xs[key].cross_section,
        )
        _create_xs_dataset(sg, "xs", shell_xs, "barns")

    # Pair production
    pp_grp = root.create_group("pair_production")
    _create_xs_dataset(pp_grp, "xs", interp("xs_pair_total"), "barns")
    nuc_grp = pp_grp.create_group("nuclear")
    _create_xs_dataset(nuc_grp, "xs", interp("xs_pair_nuclear"), "barns")
    ele_grp = pp_grp.create_group("electron")
    _create_xs_dataset(ele_grp, "xs", interp("xs_pair_electron"), "barns")

    logger.debug("Wrote MCDC EPDL for Z=%d", dataset.Z)


# ---------------------------------------------------------------------------
# EADL MCDC writer
# ---------------------------------------------------------------------------

def write_mcdc_eadl(h5f: h5py.File, dataset: EADLDataset) -> None:
    """Write an MCDC-format EADL HDF5 file

    Splits transitions into radiative / non-radiative groups with
    pre-computed fluorescence and Auger yields.

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EADLDataset
        Parsed EADL dataset.
    """
    _write_mcdc_metadata(h5f, dataset)

    root = h5f.create_group("atomic_relaxation")
    root.create_dataset("n_subshells", data=np.int64(dataset.n_subshells))

    binding_energies: list[float] = []
    n_electrons_arr: list[float] = []

    subs_grp = root.create_group("subshells")
    for name, shell in dataset.subshells.items():
        binding_energies.append(shell.binding_energy_eV)
        n_electrons_arr.append(shell.n_electrons)

        sg = subs_grp.create_group(name)
        ds_be = sg.create_dataset("binding_energy_eV", data=shell.binding_energy_eV)
        ds_be.attrs["units"] = "eV"
        sg.create_dataset("n_electrons", data=shell.n_electrons)

        if not shell.transitions:
            continue

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
            rg.create_dataset("fluorescence_yield", data=np.float64(fy))

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
            ag.create_dataset("auger_yield", data=np.float64(ay))

    if binding_energies:
        ds_be = root.create_dataset(
            "binding_energies_eV",
            data=np.array(binding_energies, dtype="f8"),
        )
        ds_be.attrs["units"] = "eV"
        root.create_dataset(
            "n_electrons",
            data=np.array(n_electrons_arr, dtype="f8"),
        )

    logger.debug("Wrote MCDC EADL for Z=%d (%d subshells)", dataset.Z, dataset.n_subshells)
