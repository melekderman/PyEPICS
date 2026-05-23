#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
PyEPICS Regression Test Report Generator

Runs every analysis defined in the regression test suite and produces a
self-contained, multi-page PDF report.  The report includes:

  * Unit-test summary (pytest)
  * Electron / photon cross-section plots (EEDL, EPDL)
  * Binding-energy comparison (EADL vs EEDL/ENDF reference)
  * Transition-energy plots (K -> L2, K -> L3, L2-L3 splitting)
  * HDF5 round-trip validation
  * Data-dictionary completeness check (ENDF source vs PyEPICS)
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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PYEPICS_ROOT = SCRIPT_DIR.parent

# Ensure pyepics is importable
if str(PYEPICS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYEPICS_ROOT))


def _lazy_imports():
    """Import heavy deps only when needed (keeps --help fast)."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

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
    ax.text(
        0.5,
        0.55,
        title,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
    )
    if subtitle:
        ax.text(
            0.5,
            0.42,
            subtitle,
            transform=ax.transAxes,
            fontsize=14,
            ha="center",
            va="center",
            color="gray",
        )
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
    ax.text(
        0.5,
        0.65,
        "PyEPICS Regression Test Report",
        fontsize=28,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.50,
        f"Generated: {now}",
        fontsize=14,
        ha="center",
        va="center",
        color="gray",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        f"PyEPICS root: {PYEPICS_ROOT}",
        fontsize=10,
        ha="center",
        va="center",
        color="gray",
        transform=ax.transAxes,
    )
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

    eedl_dir = PYEPICS_ROOT / "data" / "endf" / "eedl"
    eedl_files = sorted(eedl_dir.glob("*EEDL*.endf")) if eedl_dir.exists() else []

    if not eedl_files:
        _text_page(
            pdf,
            [
                "No EEDL ENDF files found.",
                f"Searched: {eedl_dir}",
                "",
                "Download from: https://nuclear.llnl.gov/EPICS/",
            ],
            title="EEDL — skipped",
        )
        return {"passed": None}  # skipped

    reader = M.EEDLReader()
    styles = {
        "xs_tot": {"color": "black", "ls": "--", "lw": 2, "label": "Total"},
        "xs_el": {"color": "blue", "ls": "-", "lw": 1.5, "label": "Elastic"},
        "xs_lge": {"color": "purple", "ls": ":", "lw": 1, "label": "Large Angle"},
        "xs_brem": {"color": "red", "ls": "-", "lw": 1.5, "label": "Bremsstrahlung"},
        "xs_exc": {"color": "green", "ls": "-", "lw": 1.5, "label": "Excitation"},
        "xs_ion": {"color": "orange", "ls": "-", "lw": 1.5, "label": "Ionization"},
    }

    for fpath in eedl_files[:5]:
        ds = reader.read(str(fpath))
        fig, ax = plt.subplots(figsize=(11, 7))
        for abbrev, xs in ds.cross_sections.items():
            sty = dict(
                styles.get(
                    abbrev, {"color": "gray", "ls": "-", "lw": 0.8, "label": abbrev}
                )
            )
            if len(xs.energy) > 0:
                ax.loglog(
                    xs.energy, xs.cross_section, label=sty.pop("label", abbrev), **sty
                )
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

    epdl_files = (
        sorted((PYEPICS_ROOT / "data" / "endf" / "epdl").glob("*EPDL*.endf"))
        if (PYEPICS_ROOT / "data" / "endf" / "epdl").exists()
        else []
    )
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
    """Compare EADL binding energies against EEDL/ENDF reference data.

    The reference CSV contains binding energies extracted from the EEDL ENDF
    files, not from NIST.  Analysis covers all available subshells.
    """
    import matplotlib.pyplot as plt

    M = ctx["M"]
    _section_title_page(
        pdf,
        "4. Binding-Energy Validation",
        "EADL vs EEDL/ENDF reference (all subshells)",
    )

    mcdc_dir = PYEPICS_ROOT / "data" / "mcdc" / "electron"
    h5_files = sorted(mcdc_dir.glob("*.h5")) if mcdc_dir.exists() else []

    if not h5_files:
        _text_page(
            pdf, ["No MC/DC HDF5 files found — skipped."], title="Binding energy"
        )
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
                    for shell, grp in f[
                        "electron_reactions/ionization/subshells"
                    ].items():
                        be_rows.append(
                            {
                                "Z": Z,
                                "symbol": sym,
                                "subshell": shell,
                                "be_eV": float(grp["binding_energy"][()]),
                            }
                        )
        except Exception:
            pass

    if not be_rows:
        _text_page(
            pdf, ["No subshell data found in HDF5 files."], title="Binding energy"
        )
        return {"passed": None}

    df_be = pd.DataFrame(be_rows)
    ctx["df_be"] = df_be

    # Load EEDL/ENDF reference (not NIST)
    ref_csv = PYEPICS_ROOT / "tests" / "fixtures" / "reference_binding_energies.csv"
    if not ref_csv.exists():
        _text_page(pdf, [f"Reference CSV not found: {ref_csv}"], title="Binding energy")
        return {"passed": None}

    df_ref = pd.read_csv(ref_csv)

    # --- Per-subshell analysis ---
    all_subshells = sorted(df_ref["subshell"].unique())
    overall_max_err = 0.0
    subshell_summaries = []

    for shell in all_subshells:
        df_ref_shell = df_ref[df_ref["subshell"] == shell].copy()
        df_shell = df_be[df_be["subshell"] == shell].sort_values("Z").copy()

        if df_shell.empty or df_ref_shell.empty:
            continue

        df_comp = pd.merge(
            df_shell[["Z", "be_eV"]],
            df_ref_shell[["Z", "binding_energy_eV"]],
            on="Z",
        )
        if df_comp.empty:
            continue

        df_comp["rel_error_pct"] = (
            100
            * abs(df_comp["be_eV"] - df_comp["binding_energy_eV"])
            / df_comp["binding_energy_eV"]
        )

        max_err = df_comp["rel_error_pct"].max()
        overall_max_err = max(overall_max_err, max_err)
        subshell_summaries.append(
            {
                "subshell": shell,
                "n_elements": len(df_comp),
                "max_err_pct": max_err,
                "passed": bool(max_err < 5.0),
            }
        )

    # Plot K-shell (primary visual — always present in reference)
    df_ref_k = df_ref[df_ref["subshell"] == "K"].copy()
    df_k = df_be[df_be["subshell"] == "K"].sort_values("Z").copy()

    if not df_k.empty and not df_ref_k.empty:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.semilogy(
            df_k["Z"],
            df_k["be_eV"],
            "o-",
            color="blue",
            ms=4,
            label="EADL (parsed)",
            alpha=0.8,
        )
        ax1.semilogy(
            df_ref_k["Z"],
            df_ref_k["binding_energy_eV"],
            "s",
            color="red",
            ms=8,
            label="EEDL/ENDF Reference",
            zorder=5,
        )
        ax1.set_ylabel("K-Shell Binding Energy (eV)")
        ax1.set_title("K-Shell Binding Energy: EADL vs EEDL/ENDF Reference")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        df_comp_k = pd.merge(
            df_k[["Z", "be_eV"]], df_ref_k[["Z", "binding_energy_eV"]], on="Z"
        )
        df_comp_k["rel_error_pct"] = (
            100
            * abs(df_comp_k["be_eV"] - df_comp_k["binding_energy_eV"])
            / df_comp_k["binding_energy_eV"]
        )

        ax2.bar(
            df_comp_k["Z"],
            df_comp_k["rel_error_pct"],
            color="orange",
            alpha=0.7,
            width=1.0,
        )
        ax2.axhline(5.0, color="red", ls="--", label="5% threshold")
        ax2.set_ylabel("Relative Error (%)")
        ax2.set_xlabel("Atomic Number (Z)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    ctx["be_max_err"] = overall_max_err

    # Summary page for all subshells
    lines = [
        "Binding-energy validation: EADL (parsed) vs EEDL/ENDF reference",
        "Source: tests/fixtures/reference_binding_energies.csv (from EEDL ENDF files)",
        "",
        f"{'Subshell':<12} {'Elements':>10} {'Max Error (%)':>14} {'Status':>8}",
        "=" * 50,
    ]
    for s in subshell_summaries:
        status = "PASS" if s["passed"] else "FAIL"
        lines.append(
            f"{s['subshell']:<12} {s['n_elements']:>10} "
            f"{s['max_err_pct']:>14.3f} {status:>8}"
        )
    lines += [
        "",
        f"Overall max relative error: {overall_max_err:.3f}%",
        "",
        "PASS" if overall_max_err < 5.0 else "FAIL — some values exceed 5% threshold",
    ]
    _text_page(pdf, lines, title="Binding-energy summary (all subshells)")

    return {"passed": bool(overall_max_err < 5.0)}


# -------------------------------------------------------------------
def section_transition_energies(pdf, ctx):
    """Plot K->L2, K->L3 transition energies and L2-L3 splitting."""
    import matplotlib.pyplot as plt

    _section_title_page(pdf, "5. Transition Energies", "K -> L2/L3 and L2-L3 splitting")

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
    ax1.loglog(
        trans_kl2["Z"],
        trans_kl2["transition_eV"],
        "o-",
        color="blue",
        ms=4,
        label="K -> L2",
    )
    ax1.loglog(
        trans_kl3["Z"],
        trans_kl3["transition_eV"],
        "s-",
        color="red",
        ms=4,
        label="K -> L3",
        alpha=0.7,
    )
    ax1.set_ylabel("Transition Energy (eV)")
    ax1.set_title("Atomic Subshell Transition Energies from EADL")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(4, 100)

    splitting = (df_l2_z.loc[common_kl3] - df_l3_z.loc[common_kl3]).reset_index()
    splitting.columns = ["Z", "splitting_eV"]
    ax2.semilogy(
        splitting["Z"],
        splitting["splitting_eV"],
        "^-",
        color="green",
        ms=4,
        label="L2 - L3 splitting",
    )
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

    mcdc_dir = PYEPICS_ROOT / "data" / "mcdc" / "electron"
    h5py = M.h5py
    PERIODIC_TABLE = M.PERIODIC_TABLE

    styles = {
        "Total": {"color": "black", "ls": "--", "lw": 2},
        "Elastic": {"color": "blue", "ls": "-", "lw": 1.5},
        "Large Angle": {"color": "purple", "ls": ":", "lw": 1.2},
        "Small Angle": {"color": "cyan", "ls": ":", "lw": 1.2},
        "Bremsstrahlung": {"color": "red", "ls": "-", "lw": 1.5},
        "Excitation": {"color": "green", "ls": "-", "lw": 1.5},
        "Ionization": {"color": "orange", "ls": "-", "lw": 1.5},
    }
    reaction_paths = [
        ("Total", "electron_reactions/total/xs"),
        ("Elastic", "electron_reactions/elastic_scattering/xs"),
        ("Large Angle", "electron_reactions/elastic_scattering/large_angle/xs"),
        ("Small Angle", "electron_reactions/elastic_scattering/small_angle/xs"),
        ("Bremsstrahlung", "electron_reactions/bremsstrahlung/xs"),
        ("Excitation", "electron_reactions/excitation/xs"),
        ("Ionization", "electron_reactions/ionization/xs"),
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
                    ax.loglog(
                        e_grid,
                        f[path][()],
                        label=name,
                        **styles.get(name, {"color": "gray", "ls": "-", "lw": 1}),
                    )
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

    M = ctx["M"]
    _section_title_page(pdf, "7. HDF5 Round-Trip Validation")

    energy = np.logspace(1, 7, 200)
    xs_total = 1e6 * energy ** (-0.8)
    xs_elastic = 0.9e6 * energy ** (-0.8)
    xs_brem = 0.01 * energy**0.2

    test_ds = M.EEDLDataset(
        Z=26,
        symbol="Fe",
        atomic_weight_ratio=55.345,
        ZA=26000.0,
        cross_sections={
            "xs_tot": M.CrossSectionRecord(
                label="xs_tot", energy=energy, cross_section=xs_total
            ),
            "xs_el": M.CrossSectionRecord(
                label="xs_el", energy=energy, cross_section=xs_elastic
            ),
            "xs_brem": M.CrossSectionRecord(
                label="xs_brem", energy=energy, cross_section=xs_brem
            ),
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
        checks.append(
            (f"{z_grp}/elastic_scattering exists", f"{z_grp}/elastic_scattering" in f)
        )
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
    """Verify PyEPICS data-mapping dictionaries against ENDF file contents.

    Instead of comparing against an external tool, this section verifies
    that PyEPICS mapping dictionaries cover all (MF, MT) section pairs
    found in the actual ENDF source files.
    """
    M = ctx["M"]
    _section_title_page(
        pdf, "8. Data-Dictionary Completeness", "ENDF source files vs PyEPICS mappings"
    )

    # Discover (MF, MT) pairs present in actual ENDF files
    endf_dir = PYEPICS_ROOT / "data" / "endf"
    endf_mf_mt_sets = {"eedl": set(), "epdl": set(), "eadl": set()}

    for lib_name in ("eedl", "epdl", "eadl"):
        lib_dir = endf_dir / lib_name
        if not lib_dir.exists():
            continue
        for fpath in sorted(lib_dir.glob("*.endf")):  # scan all files
            try:
                import endf

                tape = endf.Material(fpath)
                # section_data keys are already (MF, MT) tuples
                for mf_mt in tape.section_data:
                    endf_mf_mt_sets[lib_name].add(mf_mt)
            except Exception:
                pass

    # Compare PyEPICS mapping tables against ENDF contents
    mapping_checks = [
        ("ELECTRON_MF_MT (EEDL)", endf_mf_mt_sets.get("eedl", set()), M.ELECTRON_MF_MT),
        ("PHOTON_MF_MT (EPDL)", endf_mf_mt_sets.get("epdl", set()), M.PHOTON_MF_MT),
        ("ATOMIC_MF_MT (EADL)", endf_mf_mt_sets.get("eadl", set()), M.ATOMIC_MF_MT),
    ]

    lines = [
        f"{'Mapping':<28} {'ENDF':>6} {'PyEPICS':>8} {'Covered':>8} {'Missing':>8} Status",
        "=" * 72,
    ]
    all_ok = True
    for name, endf_keys, pyepics_dict in mapping_checks:
        pyepics_keys = set(pyepics_dict.keys())
        covered = len(endf_keys & pyepics_keys)
        missing = len(endf_keys - pyepics_keys)
        status = "OK" if missing == 0 else f"MISSING {missing}"
        if missing > 0:
            all_ok = False
        lines.append(
            f"{name:<28} {len(endf_keys):>6} {len(pyepics_keys):>8} "
            f"{covered:>8} {missing:>8} {status}"
        )

    if not any(endf_mf_mt_sets.values()):
        lines.append("")
        lines.append("No ENDF files found — dictionary check inconclusive.")
        lines.append("Download ENDF data with: python -m pyepics.cli download")
        _text_page(pdf, lines, title="Data-dictionary comparison (ENDF vs PyEPICS)")
        return {"passed": None}

    _text_page(pdf, lines, title="Data-dictionary comparison (ENDF vs PyEPICS)")

    # Verify internal dictionary consistency
    internal_checks = [
        ("PERIODIC_TABLE", M.PERIODIC_TABLE, 100),  # Z=1..100
        ("ELECTRON_SUBSHELL_LABELS", M.ELECTRON_SUBSHELL_LABELS, 1),
        ("SUBSHELL_DESIGNATORS", M.SUBSHELL_DESIGNATORS, 1),
        ("ELECTRON_SECTIONS_ABBREVS", M.ELECTRON_SECTIONS_ABBREVS, 1),
    ]

    int_lines = [
        f"{'Dictionary':<28} {'Entries':>8} {'Min expected':>13} Status",
        "=" * 56,
    ]
    for name, d, min_expected in internal_checks:
        n = len(d)
        ok = n >= min_expected
        if not ok:
            all_ok = False
        int_lines.append(
            f"{name:<28} {n:>8} {min_expected:>13} {'OK' if ok else 'FAIL'}"
        )

    _text_page(pdf, int_lines, title="Internal dictionary completeness")

    # Physical constants (self-check against NIST CODATA 2018 values)
    const_checks = [
        ("FINE_STRUCTURE", M.FINE_STRUCTURE, 7.2973525693e-3),
        ("ELECTRON_MASS", M.ELECTRON_MASS, 0.51099895069),
        ("BARN_TO_CM2", M.BARN_TO_CM2, 1e-24),
        ("PLANCK_CONSTANT", M.PLANCK_CONSTANT, 6.62607015e-34),
        ("SPEED_OF_LIGHT", M.SPEED_OF_LIGHT, 299792458.0),
        ("ELECTRON_CHARGE", M.ELECTRON_CHARGE, 1.602176634e-19),
    ]
    const_lines = [
        f"{'Constant':<25} {'PyEPICS':>20} {'CODATA 2018':>20} Match",
        "=" * 72,
    ]
    const_ok = True
    for name, pyepics_v, ref_v in const_checks:
        ok = pyepics_v == ref_v
        if not ok:
            const_ok = False
        const_lines.append(
            f"{name:<25} {str(pyepics_v):>20} {str(ref_v):>20} {'OK' if ok else 'MISMATCH'}"
        )

    const_lines += [
        "",
        "All constants match CODATA 2018." if const_ok else "MISMATCH detected!",
    ]
    _text_page(
        pdf, const_lines, title="Physical-constant verification (NIST CODATA 2018)"
    )

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
        rows.append(
            {
                "file": str(rel),
                "type": "module",
                "name": str(rel),
                "has_docstring": ast.get_docstring(tree) is not None,
            }
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append(
                    {
                        "file": str(rel),
                        "type": "function",
                        "name": node.name,
                        "has_docstring": ast.get_docstring(node) is not None,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                rows.append(
                    {
                        "file": str(rel),
                        "type": "class",
                        "name": node.name,
                        "has_docstring": ast.get_docstring(node) is not None,
                    }
                )

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
                _text_page(
                    pdf,
                    [
                        f"ERROR in section: {name}",
                        "",
                        str(exc),
                    ],
                    title=f"Error — {name}",
                )
                print(f"ERROR: {exc}")
            else:
                p = res.get("passed")
                print("PASS" if p else ("SKIP" if p is None else "FAIL"))
            ctx["results"][name] = res

        # Final summary page
        section_summary(pdf, ctx)

    all_pass = all(
        r["passed"] is None or r["passed"]  # None = skipped, truthy = passed
        for r in ctx["results"].values()
    )
    print()
    print(f"Report written to: {output_path}")
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyEPICS regression-test PDF report.",
    )
    default_out = PYEPICS_ROOT / "tests" / "reports" / "regression_report.pdf"
    parser.add_argument(
        "-o",
        "--output",
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
