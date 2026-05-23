#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Optional plotting helpers for PyEPICS

All functions in this module require ``matplotlib``.  The core library
works without it — these are convenience wrappers for quick
visualisation.

Usage
-----
>>> from pyepics.plotting import plot_cross_sections, plot_binding_energies
>>> plot_cross_sections(client, "Fe", labels=["xs_tot", "xs_el"])  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyepics.client import ElementID, EPICSClient


def _import_matplotlib():
    """Import matplotlib or raise a helpful error."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from None


def plot_cross_sections(
    client: EPICSClient,
    element: ElementID,
    *,
    labels: Sequence[str] | None = None,
    library: str = "EEDL",
    logx: bool = True,
    logy: bool = True,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Plot cross sections for a single element.

    Parameters
    ----------
    client : EPICSClient
        Initialised client instance.
    element : int or str
        Element identifier.
    labels : sequence of str or None, optional
        Cross-section labels to plot.  Defaults to all available.
    library : str
        ``"EEDL"`` or ``"EPDL"``.
    logx, logy : bool
        Use logarithmic axes.
    title : str or None
        Plot title.  Auto-generated if ``None``.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on.  Creates a new figure if ``None``.
    show : bool
        Call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    plt = _import_matplotlib()

    ep = client.get_element(element, libraries=[library])
    lib_upper = library.upper()
    ds = ep.electron if lib_upper == "EEDL" else ep.photon
    if ds is None:
        raise ValueError(f"No {lib_upper} data for {ep.symbol} (Z={ep.Z}).")

    xs_dict = ds.cross_sections
    if labels is None:
        labels = list(xs_dict.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for label in labels:
        if label not in xs_dict:
            continue
        rec = xs_dict[label]
        ax.plot(rec.energy, rec.cross_section, label=label)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross Section (barns)")
    ax.set_title(title or f"{ep.symbol} (Z={ep.Z}) — {lib_upper} Cross Sections")
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def compare_cross_sections(
    client: EPICSClient,
    elements: Sequence[ElementID],
    label: str,
    *,
    library: str = "EEDL",
    logx: bool = True,
    logy: bool = True,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Compare a single cross-section type across multiple elements.

    Parameters
    ----------
    client : EPICSClient
        Initialised client instance.
    elements : sequence of int or str
        Elements to compare.
    label : str
        Cross-section label (e.g. ``"xs_tot"``).
    library : str
        ``"EEDL"`` or ``"EPDL"``.
    logx, logy : bool
        Use logarithmic axes.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Existing axes.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for elem in elements:
        try:
            energy, xs = client.get_cross_section(elem, label, library=library)
        except (KeyError, Exception):
            continue
        from pyepics.client import _resolve_element

        z, sym = _resolve_element(elem)
        ax.plot(energy, xs, label=f"{sym} (Z={z})")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross Section (barns)")
    ax.set_title(title or f"{label} — Element Comparison")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_binding_energies(
    client: EPICSClient,
    elements: Sequence[ElementID],
    *,
    subshell: str | None = None,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Plot binding energies vs. atomic number.

    Parameters
    ----------
    client : EPICSClient
        Initialised client instance.
    elements : sequence of int or str
        Elements to plot.
    subshell : str or None
        If given, plot only this subshell (e.g. ``"K"``).
        Otherwise, plot all subshells with connected lines.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Existing axes.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Collect data: subshell -> [(Z, BE)]
    data: dict[str, list[tuple[int, float]]] = {}
    for elem in elements:
        ep = client.get_element(elem, libraries=["EADL"])
        if ep.atomic is None:
            continue
        z = ep.Z
        for name, sub in ep.atomic.subshells.items():
            if subshell is not None and name != subshell:
                continue
            data.setdefault(name, []).append((z, sub.binding_energy_eV))

    for name, points in sorted(data.items()):
        points.sort()
        zs = [p[0] for p in points]
        bes = [p[1] for p in points]
        ax.plot(zs, bes, "o-", label=name, markersize=3)

    ax.set_xlabel("Atomic Number (Z)")
    ax.set_ylabel("Binding Energy (eV)")
    ax.set_yscale("log")
    ax.set_title(title or "Subshell Binding Energies vs. Z")
    ax.legend(fontsize="x-small", ncol=3, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_shell_binding_energies(
    client: EPICSClient,
    element: ElementID,
    *,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
) -> Any:
    """Bar chart of binding energies by subshell for a single element.

    Parameters
    ----------
    client : EPICSClient
        Initialised client instance.
    element : int or str
        Element identifier.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Existing axes.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_matplotlib()

    ep = client.get_element(element, libraries=["EADL"])
    if ep.atomic is None:
        raise ValueError(f"No EADL data for {ep.symbol} (Z={ep.Z}).")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    names = list(ep.atomic.subshells.keys())
    bes = [ep.atomic.subshells[n].binding_energy_eV for n in names]

    ax.bar(names, bes, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Subshell")
    ax.set_ylabel("Binding Energy (eV)")
    ax.set_yscale("log")
    ax.set_title(title or f"{ep.symbol} (Z={ep.Z}) — Subshell Binding Energies")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, which="both", axis="y", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax
