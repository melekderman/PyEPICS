#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
PyEPICS Regression Test Report Generator

Runs every analysis defined in the regression test suite and produces a
self-contained, multi-page PDF report.  The report includes:

  * Unit-test summary (pytest)
  * Electron / photon cross-section plots (EEDL, EPDL)
  * Binding-energy comparison (EADL vs NIST reference)
  * Transition-energy plots (K -> L2, K -> L3, L2-L3 splitting)
  * HDF5 round-trip validation
  * Data-dictionary completeness check (PyEEDL <-> PyEPICS)
  * Docstring coverage audit
  * Physical-constant verification
  * Overall pass / fail summary

Usage
-----
    python tests/generate_report.py                    # default output
    python tests/generate_report.py -o my_report.pdf   # custom path
"""

from __future__ import annotations

import argparse
import ast
import datetime
import glob
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PYEPICS_ROOT = SCRIPT_DIR.parent
PYEEDL_ROOT = PYEPICS_ROOT.parent / "PyEEDL"

# Ensure pyepics is importable
if str(PYEPICS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYEPICS_ROOT))


def _lazy_imports():
    """Import heavy deps only when needed (keeps --help fast)."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    import h5py

    from pyepics.readers.eedl import EEDLReader
    from pyepics.readers.epdl import EPDLReader
    from pyepics.readers.eadl import EADLReader
    from pyepics.converters.hdf5 import (
        convert_dataset_to_hdf5,
        _write_eedl,
        _write_metadata,
    )
    from pyepics.models.records import (
        AverageEnergyLoss,
        EEDLDataset,
        CrossSectionRecord,
        DistributionRecord,
    )
    from pyepics.utils.constants import (
        PERIODIC_TABLE,
        MF_MT,
        PHOTON_MF_MT,
        ATOMIC_MF_MT,
        MF23,
        MF26,
        MF27,
        MF28,
        SECTIONS_ABBREVS,
        PHOTON_SECTIONS_ABBREVS,
        ATOMIC_SECTIONS_ABBREVS,
        SUBSHELL_LABELS,
        SUBSHELL_DESIGNATORS,
        FINE_STRUCTURE,
        ELECTRON_MASS,
        BARN_TO_CM2,
        PLANCK_CONSTANT,
        SPEED_OF_LIGHT,
        ELECTRON_CHARGE,
    )

    return SimpleNamespace(**locals())


from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Utility: render a text page into the PDF
# ---------------------------------------------------------------------------

def _text_page(pdf, lines: list[str], *, title: str = "", fontsize: int = 9):
    """Render a page of monospaced text into *pdf*."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    body = "\n".join(lines)
    ax.text(
        0.02,
        0.95,
        body,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily="monospace",
        verticalalignment="top",
    )
    pdf.savefig(fig)
    plt.close(fig)


def _section_title_page(pdf, title: str, subtitle: str = ""):
    """Full-page section divider."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.55, title, transform=ax.transAxes, fontsize=24,
            fontweight="bold", ha="center", va="center")
    if subtitle:
        ax.text(0.5, 0.42, subtitle, transform=ax.transAxes, fontsize=14,
                ha="center", va="center", color="gray")
    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# Section runners — each appends pages to *pdf* and returns a dict
# with at minimum {"passed": bool}
# ===================================================================

def section_cover(pdf, ctx):
    """Title / cover page."""
    import matplotlib.pyplot as plt

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.65, "PyEPICS Regression Test Report", fontsize=28,
            fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.50, f"Generated: {now}", fontsize=14,
            ha="center", va="center", color="gray", transform=ax.transAxes)
    ax.text(0.5, 0.42, f"PyEPICS root: {PYEPICS_ROOT}", fontsize=10,
            ha="center", va="center", color="gray", transform=ax.transAxes)
    ax.text(0.5, 0.36, f"PyEEDL root:  {PYEEDL_ROOT}", fontsize=10,
            ha="center", va="center", color="gray", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)
    return {"passed": True}


# -------------------------------------------------------------------
def section_unit_tests(pdf, ctx):
    """Run pytest and report results."""
    _section_title_page(pdf, "1. Unit Tests", "pytest execution summary")

    test_dir = str(PYEPICS_ROOT / "tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    lines = (result.stdout + result.stderr).splitlines()

    # Paginate long output (max ~55 lines per page)
    PAGE = 55
    for i in range(0, len(lines), PAGE):
        _text_page(pdf, lines[i : i + PAGE], title="pytest output")

    passed = result.returncode == 0
    ctx["pytest_exit"] = result.returncode
    ctx["pytest_output"] = result.stdout
    return {"passed": passed}


# -------------------------------------------------------------------
def section_eedl_plots(pdf, ctx):
    """Parse EEDL files and produce cross-section plots."""
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(pdf, "2. EEDL Electron Cross-Section Plots")

    eedl_dir = PYEEDL_ROOT / "eedl"
    eedl_files = sorted(eedl_dir.glob("*EEDL*.endf")) if eedl_dir.exists() else []

    if not eedl_files:
        _text_page(pdf, [
            "No EEDL ENDF files found.",
            f"Searched: {eedl_dir}",
            "",
            "Download from: https://www-nds.iaea.org/epics/",
        ], title="EEDL — skipped")
        return {"passed": None}  # skipped

    reader = M.EEDLReader()
    styles = {
        "xs_tot":  {"color": "black",  "ls": "--", "lw": 2,   "label": "Total"},
        "xs_el":   {"color": "blue",   "ls": "-",  "lw": 1.5, "label": "Elastic"},
        "xs_lge":  {"color": "purple", "ls": ":",  "lw": 1,   "label": "Large Angle"},
        "xs_brem": {"color": "red",    "ls": "-",  "lw": 1.5, "label": "Bremsstrahlung"},
        "xs_exc":  {"color": "green",  "ls": "-",  "lw": 1.5, "label": "Excitation"},
        "xs_ion":  {"color": "orange", "ls": "-",  "lw": 1.5, "label": "Ionization"},
    }

    for fpath in eedl_files[:5]:
        ds = reader.read(str(fpath))
        fig, ax = plt.subplots(figsize=(11, 7))
        for abbrev, xs in ds.cross_sections.items():
            sty = dict(styles.get(abbrev, {"color": "gray", "ls": "-", "lw": 0.8, "label": abbrev}))
            if len(xs.energy) > 0:
                ax.loglog(xs.energy, xs.cross_section,
                          label=sty.pop("label", abbrev), **sty)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Cross Section (barns)")
        ax.set_title(f"EEDL Electron Cross Sections — {ds.symbol} (Z={ds.Z})")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return {"passed": True}


# -------------------------------------------------------------------
def section_epdl_plots(pdf, ctx):
    """Parse EPDL files and produce photon cross-section plots."""
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(pdf, "3. EPDL Photon Cross-Section Plots")

    epdl_files = sorted((PYEEDL_ROOT / "eedl").glob("*EPDL*.endf")) if (PYEEDL_ROOT / "eedl").exists() else []
    if not epdl_files:
        _text_page(pdf, ["No EPDL ENDF files found — skipped."], title="EPDL")
        return {"passed": None}

    reader = M.EPDLReader()
    colors = ["black", "blue", "red", "green", "orange", "purple", "cyan", "magenta"]

    for fpath in epdl_files[:5]:
        ds = reader.read(str(fpath))
        fig, ax = plt.subplots(figsize=(11, 7))
        for i, (abbrev, xs) in enumerate(ds.cross_sections.items()):
            c = colors[i % len(colors)]
            ls = "--" if "tot" in abbrev else "-"
            if len(xs.energy) > 0:
                ax.loglog(xs.energy, xs.cross_section, label=abbrev, color=c, ls=ls)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Cross Section (barns)")
        ax.set_title(f"EPDL Photon Cross Sections — {ds.symbol} (Z={ds.Z})")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return {"passed": True}


# -------------------------------------------------------------------
def section_binding_energy(pdf, ctx):
    """Compare EADL binding energies against NIST reference data."""
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(pdf, "4. Binding-Energy Validation",
                        "EADL vs NIST reference (K-shell)")

    mcdc_dir = PYEEDL_ROOT / "mcdc_data"
    h5_files = sorted(mcdc_dir.glob("*.h5")) if mcdc_dir.exists() else []

    if not h5_files:
        _text_page(pdf, ["No MC/DC HDF5 files found — skipped."], title="Binding energy")
        return {"passed": None}

    h5py = M.h5py
    PERIODIC_TABLE = M.PERIODIC_TABLE

    be_rows = []
    for h5path in h5_files:
        try:
            with h5py.File(str(h5path), "r") as f:
                Z = int(f["atomic_number"][()])
                sym = PERIODIC_TABLE.get(Z, {}).get("symbol", "?")
                if "electron_reactions/ionization/subshells" in f:
                    for shell, grp in f["electron_reactions/ionization/subshells"].items():
                        be_rows.append({"Z": Z, "symbol": sym, "subshell": shell,
                                        "be_eV": float(grp["binding_energy"][()])})
        except Exception:
            pass

    if not be_rows:
        _text_page(pdf, ["No subshell data found in HDF5 files."], title="Binding energy")
        return {"passed": None}

    df_be = pd.DataFrame(be_rows)
    ctx["df_be"] = df_be

    # Load NIST reference
    ref_csv = PYEPICS_ROOT / "reference_data" / "reference_binding_energies.csv"
    if not ref_csv.exists():
        _text_page(pdf, [f"Reference CSV not found: {ref_csv}"], title="Binding energy")
        return {"passed": None}

    df_ref = pd.read_csv(ref_csv)
    df_ref_k = df_ref[df_ref["subshell"] == "K"].copy()
    df_k = df_be[df_be["subshell"] == "K"].sort_values("Z").copy()

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.semilogy(df_k["Z"], df_k["be_eV"], "o-", color="blue", ms=4,
                 label="EADL (parsed)", alpha=0.8)
    ax1.semilogy(df_ref_k["Z"], df_ref_k["binding_energy_eV"], "s", color="red",
                 ms=8, label="NIST Reference", zorder=5)
    ax1.set_ylabel("K-Shell Binding Energy (eV)")
    ax1.set_title("K-Shell Binding Energy: EADL vs NIST Reference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    df_comp = pd.merge(df_k[["Z", "be_eV"]], df_ref_k[["Z", "binding_energy_eV"]], on="Z")
    df_comp["rel_error_pct"] = 100 * abs(
        df_comp["be_eV"] - df_comp["binding_energy_eV"]
    ) / df_comp["binding_energy_eV"]

    ax2.bar(df_comp["Z"], df_comp["rel_error_pct"], color="orange", alpha=0.7, width=1.0)
    ax2.axhline(5.0, color="red", ls="--", label="5% threshold")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_xlabel("Atomic Number (Z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    max_err = df_comp["rel_error_pct"].max()
    ctx["be_max_err"] = max_err

    lines = [
        f"Compared {len(df_comp)} elements (K-shell)",
        f"Max relative error: {max_err:.3f}%",
        "",
        "PASS" if max_err < 5.0 else "FAIL — some values exceed 5% threshold",
    ]
    _text_page(pdf, lines, title="Binding-energy numerical summary")

    return {"passed": max_err < 5.0}


# -------------------------------------------------------------------
def section_transition_energies(pdf, ctx):
    """Plot K->L2, K->L3 transition energies and L2-L3 splitting."""
    import matplotlib.pyplot as plt

    _section_title_page(pdf, "5. Transition Energies",
                        "K -> L2/L3 and L2-L3 splitting")

    df_be = ctx.get("df_be")
    if df_be is None or df_be.empty:
        _text_page(pdf, ["No binding-energy data available — skipped."])
        return {"passed": None}

    df_k_z = df_be[df_be["subshell"] == "K"].set_index("Z")["be_eV"]
    df_l2_z = df_be[df_be["subshell"] == "L2"].set_index("Z")["be_eV"]
    df_l3_z = df_be[df_be["subshell"] == "L3"].set_index("Z")["be_eV"]

    common_kl2 = df_k_z.index.intersection(df_l2_z.index)
    common_kl3 = df_k_z.index.intersection(df_l3_z.index)

    if len(common_kl2) == 0:
        _text_page(pdf, ["Insufficient data for transition-energy plots."])
        return {"passed": None}

    trans_kl2 = (df_k_z.loc[common_kl2] - df_l2_z.loc[common_kl2]).reset_index()
    trans_kl2.columns = ["Z", "transition_eV"]
    trans_kl3 = (df_k_z.loc[common_kl3] - df_l3_z.loc[common_kl3]).reset_index()
    trans_kl3.columns = ["Z", "transition_eV"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    ax1.loglog(trans_kl2["Z"], trans_kl2["transition_eV"], "o-", color="blue",
               ms=4, label="K -> L2")
    ax1.loglog(trans_kl3["Z"], trans_kl3["transition_eV"], "s-", color="red",
               ms=4, label="K -> L3", alpha=0.7)
    ax1.set_ylabel("Transition Energy (eV)")
    ax1.set_title("Atomic Subshell Transition Energies from EADL")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(4, 100)

    splitting = (df_l2_z.loc[common_kl3] - df_l3_z.loc[common_kl3]).reset_index()
    splitting.columns = ["Z", "splitting_eV"]
    ax2.semilogy(splitting["Z"], splitting["splitting_eV"], "^-", color="green",
                 ms=4, label="L2 - L3 splitting")
    ax2.set_xlabel("Atomic Number (Z)")
    ax2.set_ylabel("L2-L3 Splitting (eV)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    lines = [
        f"K -> L2 transition energies: {len(trans_kl2)} elements",
        f"K -> L3 transition energies: {len(trans_kl3)} elements",
        f"L2-L3 splitting data points: {len(splitting)} elements",
    ]
    _text_page(pdf, lines, title="Transition-energy summary")
    return {"passed": True}


# -------------------------------------------------------------------
def section_h5_cross_sections(pdf, ctx):
    """Plot cross sections read from MC/DC HDF5 files."""
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(pdf, "6. MC/DC HDF5 Cross-Section Plots")

    mcdc_dir = PYEEDL_ROOT / "mcdc_data"
    h5py = M.h5py
    PERIODIC_TABLE = M.PERIODIC_TABLE

    styles = {
        "Total":          {"color": "black",  "ls": "--", "lw": 2},
        "Elastic":        {"color": "blue",   "ls": "-",  "lw": 1.5},
        "Large Angle":    {"color": "purple", "ls": ":",  "lw": 1.2},
        "Small Angle":    {"color": "cyan",   "ls": ":",  "lw": 1.2},
        "Bremsstrahlung": {"color": "red",    "ls": "-",  "lw": 1.5},
        "Excitation":     {"color": "green",  "ls": "-",  "lw": 1.5},
        "Ionization":     {"color": "orange", "ls": "-",  "lw": 1.5},
    }
    reaction_paths = [
        ("Total",          "electron_reactions/total/xs"),
        ("Elastic",        "electron_reactions/elastic_scattering/xs"),
        ("Large Angle",    "electron_reactions/elastic_scattering/large_angle/xs"),
        ("Small Angle",    "electron_reactions/elastic_scattering/small_angle/xs"),
        ("Bremsstrahlung", "electron_reactions/bremsstrahlung/xs"),
        ("Excitation",     "electron_reactions/excitation/xs"),
        ("Ionization",     "electron_reactions/ionization/xs"),
    ]

    plotted = 0
    for elem in ["H", "Fe", "Al", "Cu", "Au", "Pb"]:
        h5 = mcdc_dir / f"{elem}.h5"
        if not h5.exists():
            continue
        with h5py.File(str(h5), "r") as f:
            Z = int(f["atomic_number"][()])
            e_grid = f["electron_reactions/xs_energy_grid"][()]
            fig, ax = plt.subplots(figsize=(11, 7))
            for name, path in reaction_paths:
                if path in f:
                    ax.loglog(e_grid, f[path][()], label=name,
                              **styles.get(name, {"color": "gray", "ls": "-", "lw": 1}))
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Cross Section (barns)")
            ax.set_title(f"MC/DC Electron Cross Sections — {elem} (Z={Z})")
            ax.legend(loc="upper right")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            plotted += 1

    if plotted == 0:
        _text_page(pdf, ["No MC/DC HDF5 files found."], title="HDF5 cross sections")
        return {"passed": None}

    return {"passed": True}


# -------------------------------------------------------------------
def section_hdf5_roundtrip(pdf, ctx):
    """Write a synthetic dataset to HDF5 and read it back."""
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(pdf, "7. HDF5 Round-Trip Validation")

    energy = np.logspace(1, 7, 200)
    xs_total = 1e6 * energy ** (-0.8)
    xs_elastic = 0.9e6 * energy ** (-0.8)
    xs_brem = 0.01 * energy ** 0.2

    test_ds = M.EEDLDataset(
        Z=26,
        symbol="Fe",
        atomic_weight_ratio=55.345,
        ZA=26000.0,
        cross_sections={
            "xs_tot": M.CrossSectionRecord(label="xs_tot", energy=energy, cross_section=xs_total),
            "xs_el": M.CrossSectionRecord(label="xs_el", energy=energy, cross_section=xs_elastic),
            "xs_brem": M.CrossSectionRecord(label="xs_brem", energy=energy, cross_section=xs_brem),
        },
        distributions={},
        average_energy_losses={},
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    # Use internal writers (convert_dataset_to_hdf5 takes a source ENDF path)
    with M.h5py.File(tmp_path, "w") as h5f:
        M._write_metadata(h5f, test_ds)
        M._write_eedl(h5f, test_ds)

    z_grp = f"EEDL/Z_{test_ds.Z:03d}"
    checks = []
    with M.h5py.File(tmp_path, "r") as f:
        checks.append(("Z == 26", int(f["metadata/Z"][()]) == 26))
        checks.append(("symbol == Fe", f["metadata/symbol"].asstr()[()] == "Fe"))
        checks.append(("EEDL/ group exists", "EEDL" in f))
        checks.append((f"{z_grp}/total exists", f"{z_grp}/total" in f))
        checks.append((f"{z_grp}/elastic_scattering exists", f"{z_grp}/elastic_scattering" in f))
        eg = f[f"{z_grp}/xs_energy_grid"][:]
        rt_xs = f[f"{z_grp}/total/xs"][:]
        checks.append(("energy grid shape", eg.shape == energy.shape))
        checks.append(("energy grid allclose", np.allclose(eg, energy)))
        checks.append(("total xs allclose", np.allclose(rt_xs, xs_total)))
        ds_eg = f[f"{z_grp}/xs_energy_grid"]
        checks.append(("energy unit == eV", ds_eg.attrs.get("units") == "eV"))

    os.unlink(tmp_path)

    n_pass = sum(1 for _, ok in checks if ok)
    lines = [f"{'Check':<35} Result", "=" * 50]
    for name, ok in checks:
        lines.append(f"{name:<35} {'PASS' if ok else 'FAIL'}")
    lines += ["", f"{n_pass}/{len(checks)} checks passed"]

    ctx["hdf5_checks"] = checks
    _text_page(pdf, lines, title="HDF5 round-trip results")

    return {"passed": n_pass == len(checks)}


# -------------------------------------------------------------------
def section_data_dictionaries(pdf, ctx):
    """Compare PyEEDL and PyEPICS data-mapping dictionaries."""
    M = ctx["M"]
    _section_title_page(pdf, "8. Data-Dictionary Completeness")

    pyeedl_data_path = PYEEDL_ROOT / "pyeedl" / "data.py"
    if not pyeedl_data_path.exists():
        _text_page(pdf, [f"PyEEDL data.py not found: {pyeedl_data_path}"])
        return {"passed": None}

    spec = importlib.util.spec_from_file_location("pyeedl_data", str(pyeedl_data_path))
    pyeedl_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pyeedl_data)

    dict_pairs = [
        ("MF_MT", getattr(pyeedl_data, "MF_MT", {}), M.MF_MT),
        ("SECTIONS_ABBREVS", getattr(pyeedl_data, "SECTIONS_ABBREVS", {}), M.SECTIONS_ABBREVS),
        ("PHOTON_MF_MT", getattr(pyeedl_data, "PHOTON_MF_MT", {}), M.PHOTON_MF_MT),
        ("PHOTON_SECTIONS_ABBREVS", getattr(pyeedl_data, "PHOTON_SECTIONS_ABBREVS", {}), M.PHOTON_SECTIONS_ABBREVS),
        ("ATOMIC_MF_MT", getattr(pyeedl_data, "ATOMIC_MF_MT", {}), M.ATOMIC_MF_MT),
        ("SUBSHELL_LABELS", getattr(pyeedl_data, "SUBSHELL_LABELS", {}), M.SUBSHELL_LABELS),
        ("SUBSHELL_DESIGNATORS", getattr(pyeedl_data, "SUBSHELL_DESIGNATORS", {}), M.SUBSHELL_DESIGNATORS),
        ("PERIODIC_TABLE", getattr(pyeedl_data, "PERIODIC_TABLE", {}), M.PERIODIC_TABLE),
        ("MF23", getattr(pyeedl_data, "MF23", {}), M.MF23),
        ("MF26", getattr(pyeedl_data, "MF26", {}), M.MF26),
        ("MF27", getattr(pyeedl_data, "MF27", {}), M.MF27),
        ("MF28", getattr(pyeedl_data, "MF28", {}), M.MF28),
    ]

    lines = [
        f"{'Dictionary':<28} {'PyEEDL':>7} {'PyEPICS':>8} {'Match':>6} {'Miss':>5} {'Extra':>6} Status",
        "=" * 80,
    ]
    all_ok = True
    for name, old_d, new_d in dict_pairs:
        old_k, new_k = set(old_d.keys()), set(new_d.keys())
        miss = len(old_k - new_k)
        extra = len(new_k - old_k)
        match = len(old_k & new_k)
        status = "OK" if miss == 0 else f"MISSING {miss}"
        if miss > 0:
            all_ok = False
        lines.append(
            f"{name:<28} {len(old_k):>7} {len(new_k):>8} {match:>6} {miss:>5} {extra:>6} {status}"
        )

    _text_page(pdf, lines, title="Data-dictionary comparison")

    # Physical constants
    const_names = [
        "FINE_STRUCTURE", "ELECTRON_MASS", "BARN_TO_CM2",
        "PLANCK_CONSTANT", "SPEED_OF_LIGHT", "ELECTRON_CHARGE",
    ]
    const_lines = [f"{'Constant':<25} {'PyEEDL':>20} {'PyEPICS':>20} Match", "=" * 72]
    const_ok = True
    for name in const_names:
        old_v = getattr(pyeedl_data, name, None)
        new_v = getattr(M, name, None)
        ok = old_v == new_v if (old_v is not None and new_v is not None) else False
        if not ok:
            const_ok = False
        const_lines.append(f"{name:<25} {str(old_v):>20} {str(new_v):>20} {'OK' if ok else 'MISMATCH'}")

    const_lines += ["", "All constants match." if const_ok else "MISMATCH detected!"]
    _text_page(pdf, const_lines, title="Physical-constant verification")

    ctx["const_ok"] = const_ok
    return {"passed": all_ok and const_ok}


# -------------------------------------------------------------------
def section_docstring_audit(pdf, ctx):
    """Audit docstring coverage across pyepics."""
    _section_title_page(pdf, "9. Docstring Coverage Audit")

    rows = []
    py_files = sorted((PYEPICS_ROOT / "pyepics").rglob("*.py"))
    for fpath in py_files:
        rel = fpath.relative_to(PYEPICS_ROOT)
        try:
            tree = ast.parse(fpath.read_text())
        except SyntaxError:
            continue
        rows.append({"file": str(rel), "type": "module", "name": str(rel),
                      "has_docstring": ast.get_docstring(tree) is not None})
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append({"file": str(rel), "type": "function", "name": node.name,
                              "has_docstring": ast.get_docstring(node) is not None})
            elif isinstance(node, ast.ClassDef):
                rows.append({"file": str(rel), "type": "class", "name": node.name,
                              "has_docstring": ast.get_docstring(node) is not None})

    total = len(rows)
    with_doc = sum(1 for r in rows if r["has_docstring"])
    without_doc = total - with_doc
    pct = 100 * with_doc / total if total else 0

    lines = [
        f"Total items scanned:  {total}",
        f"With docstring:       {with_doc} ({pct:.0f}%)",
        f"Without docstring:    {without_doc} ({100-pct:.0f}%)",
        "",
    ]
    missing = [r for r in rows if not r["has_docstring"]]
    if missing:
        lines.append("Items missing docstrings:")
        lines.append(f"  {'File':<45} {'Type':<12} Name")
        lines.append("  " + "-" * 75)
        for r in missing:
            lines.append(f"  {r['file']:<45} {r['type']:<12} {r['name']}")
    else:
        lines.append("All modules, classes, and functions have docstrings.")

    _text_page(pdf, lines, title="Docstring coverage report")

    ctx["docstring_pct"] = pct
    return {"passed": pct == 100}


# -------------------------------------------------------------------
def section_summary(pdf, ctx):
    """Final pass/fail summary page."""
    import matplotlib.pyplot as plt

    results = ctx["results"]

    lines = [
        "=" * 60,
        "  PyEPICS REGRESSION TEST — OVERALL SUMMARY",
        "=" * 60,
        "",
    ]
    all_pass = True
    for name, res in results.items():
        p = res["passed"]
        if p is None:
            status = "SKIPPED (data unavailable)"
        elif p:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        lines.append(f"  {name:<45} {status}")

    lines += [
        "",
        "=" * 60,
        f"  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}",
        "=" * 60,
    ]

    _text_page(pdf, lines, title="Overall Summary", fontsize=11)
    return {"passed": all_pass}


# ===================================================================
# Main
# ===================================================================

def generate_report(output_path: str | Path) -> bool:
    """Run all analyses and write the PDF report.

    Returns True if all tests passed.
    """
    M = _lazy_imports()
    from matplotlib.backends.backend_pdf import PdfPages

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx: dict = {"M": M, "results": {}}

    sections = [
        ("Cover", section_cover),
        ("Unit tests", section_unit_tests),
        ("EEDL cross-section plots", section_eedl_plots),
        ("EPDL cross-section plots", section_epdl_plots),
        ("Binding-energy validation", section_binding_energy),
        ("Transition energies", section_transition_energies),
        ("MC/DC HDF5 cross-section plots", section_h5_cross_sections),
        ("HDF5 round-trip", section_hdf5_roundtrip),
        ("Data-dictionary completeness", section_data_dictionaries),
        ("Docstring coverage", section_docstring_audit),
    ]

    with PdfPages(str(output_path)) as pdf:
        for name, func in sections:
            print(f"  Running: {name} ...", end=" ", flush=True)
            try:
                res = func(pdf, ctx)
            except Exception as exc:
                res = {"passed": False, "error": str(exc)}
                _text_page(pdf, [
                    f"ERROR in section: {name}",
                    "",
                    str(exc),
                ], title=f"Error — {name}")
                print(f"ERROR: {exc}")
            else:
                p = res.get("passed")
                print("PASS" if p else ("SKIP" if p is None else "FAIL"))
            ctx["results"][name] = res

        # Final summary page
        section_summary(pdf, ctx)

    all_pass = all(
        r["passed"] is not False for r in ctx["results"].values()
    )
    print()
    print(f"Report written to: {output_path}")
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyEPICS regression-test PDF report.",
    )
    default_out = PYEPICS_ROOT / "reports" / "regression_report.pdf"
    parser.add_argument(
        "-o", "--output",
        default=str(default_out),
        help=f"Output PDF path (default: {default_out})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PyEPICS Regression Test Report Generator")
    print("=" * 60)
    print()

    ok = generate_report(args.output)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
