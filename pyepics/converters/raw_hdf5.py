#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Raw HDF5 writer for EPICS parsed datasets

Writes a "raw" HDF5 file that preserves every piece of information from
the ENDF source file: original energy grids, breakpoints, interpolation
law codes, and distribution tables.  No resampling or merging is done.

These files are intended for **external users** who need the full
fidelity of the ENDF evaluation — users can inspect, re-interpolate,
or convert the data with their own tools.

Output Directories
------------------
* ``raw_data/``           — EEDL (electron) raw files
* ``raw_data_photon/``    — EPDL (photon)  raw files
* ``raw_data_atomic/``    — EADL (atomic)  raw files

HDF5 Layout — EEDL
-------------------
::

    /metadata/
        Z, symbol, ZA, AWR

    /total_xs/cross_section/
        energy, cross_section, breakpoints, interpolation

    /elastic_scatter/
        cross_section/total/    ...
        cross_section/large_angle/  ...
        distributions/large_angle/
            inc_energy, mu, probability
            y_inc_energy, y_yield

    /bremsstrahlung/
        cross_section/ ...
        distributions/
            inc_energy, out_energy, probability
            loss_inc_energy, avg_loss

    /excitation/
        cross_section/ ...
        distributions/
            loss_inc_energy, avg_loss

    /ionization/
        cross_section/total/ ...
        cross_section/{shell}/ ...
        distributions/{shell}/
            inc_energy, out_energy, probability
            y_inc_energy, y_yield
            binding_energy
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
    CrossSectionRecord,
    DistributionRecord,
    EADLDataset,
    EEDLDataset,
    EPDLDataset,
    FormFactorRecord,
)
from pyepics.utils.constants import SUBSHELL_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_raw_metadata(h5f: h5py.File, dataset) -> None:
    """Write ``/metadata`` group."""
    meta = h5f.create_group("metadata")
    meta.create_dataset("Z", data=np.int64(dataset.Z))
    meta.create_dataset("Sym", data=dataset.symbol)
    meta.create_dataset("ZA", data=np.float64(dataset.ZA))
    meta.create_dataset("AWR", data=np.float64(dataset.atomic_weight_ratio))


def _write_xs_record(grp: h5py.Group, rec: CrossSectionRecord) -> None:
    """Write a single cross-section record with breakpoint/interpolation info."""
    ds_e = grp.create_dataset("energy", data=rec.energy)
    ds_e.attrs["units"] = "eV"
    ds_xs = grp.create_dataset("cross_section", data=rec.cross_section)
    ds_xs.attrs["units"] = "barns"
    if rec.breakpoints is not None:
        grp.create_dataset("breakpoints", data=rec.breakpoints)
    if rec.interpolation is not None:
        grp.create_dataset("interpolation", data=rec.interpolation)


def _write_ff_record(grp: h5py.Group, rec: FormFactorRecord) -> None:
    """Write a form-factor record with breakpoint/interpolation info."""
    ds_x = grp.create_dataset("momentum_transfer", data=rec.x)
    ds_x.attrs["units"] = "1/angstrom"
    grp.create_dataset("form_factor" if "ff_" in rec.label else "scattering_function", data=rec.y)
    if rec.breakpoints is not None:
        grp.create_dataset("breakpoints", data=rec.breakpoints)
    if rec.interpolation is not None:
        grp.create_dataset("interpolation", data=rec.interpolation)


# ---------------------------------------------------------------------------
# EEDL raw writer
# ---------------------------------------------------------------------------

def write_raw_eedl(h5f: h5py.File, dataset: EEDLDataset) -> None:
    """Write a raw EEDL dataset preserving all original data

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EEDLDataset
        Parsed EEDL dataset.
    """
    _write_raw_metadata(h5f, dataset)
    xs = dataset.cross_sections
    dist = dataset.distributions
    ael = dataset.average_energy_losses

    # --- Total cross section ---
    if "xs_tot" in xs:
        _write_xs_record(h5f.create_group("total_xs/cross_section"), xs["xs_tot"])

    # --- Elastic scattering ---
    es = h5f.create_group("elastic_scatter")
    if "xs_el" in xs:
        _write_xs_record(es.create_group("cross_section/total"), xs["xs_el"])
    if "xs_lge" in xs:
        _write_xs_record(es.create_group("cross_section/large_angle"), xs["xs_lge"])
    if "ang_lge" in dist:
        d = dist["ang_lge"]
        dg = es.create_group("distributions/large_angle")
        dg.create_dataset("inc_energy", data=d.inc_energy)
        dg.create_dataset("mu", data=d.value)
        dg.create_dataset("probability", data=d.probability)

    # --- Bremsstrahlung ---
    bg = h5f.create_group("bremsstrahlung")
    if "xs_brem" in xs:
        _write_xs_record(bg.create_group("cross_section"), xs["xs_brem"])
    if dataset.bremsstrahlung_spectra is not None:
        bd = dataset.bremsstrahlung_spectra
        bdg = bg.create_group("distributions")
        bdg.create_dataset("inc_energy", data=bd.inc_energy)
        bdg.create_dataset("out_energy", data=bd.value)
        bdg.create_dataset("b", data=bd.probability)
    if "loss_brem_spec" in ael:
        a = ael["loss_brem_spec"]
        lg = bg.require_group("distributions")
        lg.create_dataset("loss_inc_energy", data=a.energy)
        lg.create_dataset("avg_loss", data=a.avg_loss)

    # --- Excitation ---
    eg = h5f.create_group("excitation")
    if "xs_exc" in xs:
        _write_xs_record(eg.create_group("cross_section"), xs["xs_exc"])
    if "loss_exc" in ael:
        a = ael["loss_exc"]
        edg = eg.create_group("distributions")
        edg.create_dataset("loss_inc_energy", data=a.energy)
        edg.create_dataset("avg_loss", data=a.avg_loss)

    # --- Ionization ---
    ig = h5f.create_group("ionization")
    if "xs_ion" in xs:
        _write_xs_record(ig.create_group("cross_section/total"), xs["xs_ion"])

    for mt, shell_label in SUBSHELL_LABELS.items():
        xs_key = f"xs_{shell_label}"
        spec_key = f"spec_{shell_label}"
        if xs_key not in xs:
            continue
        _write_xs_record(ig.create_group(f"cross_section/{shell_label}"), xs[xs_key])
        if spec_key in dist:
            d = dist[spec_key]
            dg = ig.create_group(f"distributions/{shell_label}")
            dg.create_dataset("inc_energy", data=d.inc_energy)
            dg.create_dataset("out_energy", data=d.value)
            dg.create_dataset("b", data=d.probability)

    logger.debug("Wrote raw EEDL for Z=%d", dataset.Z)


# ---------------------------------------------------------------------------
# EPDL raw writer
# ---------------------------------------------------------------------------

def write_raw_epdl(h5f: h5py.File, dataset: EPDLDataset) -> None:
    """Write a raw EPDL dataset preserving all original data

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EPDLDataset
        Parsed EPDL dataset.
    """
    _write_raw_metadata(h5f, dataset)
    xs = dataset.cross_sections
    ff = dataset.form_factors

    # --- Total cross section ---
    if "xs_tot" in xs:
        _write_xs_record(h5f.create_group("total_xs/cross_section"), xs["xs_tot"])

    # --- Coherent scattering ---
    cg = h5f.create_group("coherent_scattering")
    if "xs_coherent" in xs:
        _write_xs_record(cg.create_group("cross_section"), xs["xs_coherent"])
    if "ff_coherent" in ff:
        _write_ff_record(cg.create_group("form_factor"), ff["ff_coherent"])

    # --- Incoherent scattering ---
    icg = h5f.create_group("incoherent_scattering")
    if "xs_incoherent" in xs:
        _write_xs_record(icg.create_group("cross_section"), xs["xs_incoherent"])
    if "sf_incoherent" in ff:
        _write_ff_record(icg.create_group("scattering_function"), ff["sf_incoherent"])

    # --- Photoelectric ---
    pg = h5f.create_group("photoelectric")
    if "xs_photoelectric" in xs:
        _write_xs_record(pg.create_group("cross_section/total"), xs["xs_photoelectric"])
    for mt, shell_label in SUBSHELL_LABELS.items():
        key = f"xs_pe_{shell_label}"
        if key in xs:
            _write_xs_record(pg.create_group(f"cross_section/{shell_label}"), xs[key])

    # --- Pair production ---
    ppg = h5f.create_group("pair_production")
    if "xs_pair_total" in xs:
        _write_xs_record(ppg.create_group("cross_section/total"), xs["xs_pair_total"])
    if "xs_pair_nuclear" in xs:
        _write_xs_record(ppg.create_group("cross_section/nuclear"), xs["xs_pair_nuclear"])
    if "xs_pair_electron" in xs:
        _write_xs_record(ppg.create_group("cross_section/electron"), xs["xs_pair_electron"])

    # --- Form factors (anomalous) ---
    if "ff_anomalous_imag" in ff or "ff_anomalous_real" in ff:
        ag = h5f.create_group("form_factors/anomalous")
        if "ff_anomalous_imag" in ff:
            r = ff["ff_anomalous_imag"]
            ag.create_dataset("energy", data=r.x)
            ag.create_dataset("imaginary", data=r.y)
        if "ff_anomalous_real" in ff:
            r = ff["ff_anomalous_real"]
            if "energy" not in ag:
                ag.create_dataset("energy", data=r.x)
            ag.create_dataset("real", data=r.y)

    logger.debug("Wrote raw EPDL for Z=%d", dataset.Z)


# ---------------------------------------------------------------------------
# EADL raw writer
# ---------------------------------------------------------------------------

def write_raw_eadl(h5f: h5py.File, dataset: EADLDataset) -> None:
    """Write a raw EADL dataset preserving all original data

    Parameters
    ----------
    h5f : h5py.File
        Open HDF5 file handle (write mode).
    dataset : EADLDataset
        Parsed EADL dataset.
    """
    _write_raw_metadata(h5f, dataset)

    root = h5f.create_group("atomic_relaxation")
    root.create_dataset("n_subshells", data=np.int64(dataset.n_subshells))

    shell_names: list[str] = []
    binding_energies: list[float] = []
    n_electrons_arr: list[float] = []

    subs = root.create_group("subshells")
    for name, shell in dataset.subshells.items():
        shell_names.append(name)
        binding_energies.append(shell.binding_energy_eV)
        n_electrons_arr.append(shell.n_electrons)

        sg = subs.create_group(name)
        sg.create_dataset("designator", data=np.int32(shell.designator))
        ds_be = sg.create_dataset("binding_energy_eV", data=shell.binding_energy_eV)
        ds_be.attrs["units"] = "eV"
        sg.create_dataset("n_electrons", data=shell.n_electrons)

        if not shell.transitions:
            continue

        tg = sg.create_group("transitions")
        tg.create_dataset(
            "origin_designator",
            data=np.array([t.origin_designator for t in shell.transitions], dtype="i4"),
        )
        tg.create_dataset(
            "secondary_designator",
            data=np.array([t.secondary_designator for t in shell.transitions], dtype="i4"),
        )
        ds_e = tg.create_dataset(
            "energy_eV",
            data=np.array([t.energy_eV for t in shell.transitions], dtype="f8"),
        )
        ds_e.attrs["units"] = "eV"
        tg.create_dataset(
            "probability",
            data=np.array([t.probability for t in shell.transitions], dtype="f8"),
        )
        tg.create_dataset(
            "is_radiative",
            data=np.array([t.is_radiative for t in shell.transitions], dtype="bool"),
        )

    if shell_names:
        root.create_dataset("shell_names", data=np.array(shell_names, dtype="S8"))
        ds_be = root.create_dataset(
            "binding_energies_eV",
            data=np.array(binding_energies, dtype="f8"),
        )
        ds_be.attrs["units"] = "eV"
        root.create_dataset(
            "n_electrons",
            data=np.array(n_electrons_arr, dtype="f8"),
        )

    logger.debug("Wrote raw EADL for Z=%d (%d subshells)", dataset.Z, len(shell_names))
