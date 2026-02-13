#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Shared ENDF-format parsing helpers for the PyEPICS package

All low-level text parsing, numeric conversion, and record flattening
routines live here so that none of the three reader modules duplicates
format-specific logic.

ENDF-6 Fixed-Width Format
-------------------------
Every ENDF record line is exactly 80 characters wide:

* **Columns 0–65** (66 chars): six data fields, each 11 characters wide.
* **Columns 66–69** (4 chars):  MAT number.
* **Columns 70–71** (2 chars):  MF number.
* **Columns 72–74** (3 chars):  MT number.
* **Columns 75–79** (5 chars):  line sequence number.

Floating-point values use Fortran-style notation where the exponent
marker ``E`` may be replaced by ``D`` or omitted entirely (implicit
exponent, e.g. ``" 1.23456-03"``).

References
----------
.. [1] ENDF-6 Formats Manual (ENDF-102), BNL-90365-2009 Rev. 2, §0.6.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from pyepics.exceptions import ParseError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENDF-6 fixed-width column layout constants
# ---------------------------------------------------------------------------

ENDF_LINE_WIDTH: int = 80
"""Total width of an ENDF record line in characters."""

ENDF_DATA_WIDTH: int = 66
"""Width of the data portion of an ENDF record line (first 66 chars)."""

ENDF_FIELD_WIDTH: int = 11
"""Width of a single numeric field inside the data portion."""

ENDF_FIELDS_PER_LINE: int = 6
"""Number of numeric fields per data line (66 / 11)."""

ENDF_HEADER_LINES_MT525: int = 8
"""Number of header lines to skip when parsing MF=26 / MT=525 raw text."""

# ENDF filename pattern: captures the 3-digit zero-padded atomic number.
ENDF_FILENAME_PATTERN: re.Pattern[str] = re.compile(r"ZA(\d{3})000")
"""Compiled regex that extracts the atomic number *Z* from an ENDF file name.

The pattern matches the ``ZA{ZZZ}000`` portion of canonical EPICS file names
such as ``EEDL.ZA026000.endf`` (iron, Z = 26).
"""


# ---------------------------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------------------------

def float_endf(s: str) -> float:
    """Convert an ENDF-6 formatted string to a Python float

    ENDF floating-point fields may use ``D`` instead of ``E`` for the
    exponent marker, or omit the marker entirely when a sign character
    (``+`` or ``-``) is embedded inside the mantissa string.  This
    function handles all three conventions.

    Parameters
    ----------
    s : str
        An 11-character (or shorter, if stripped) string taken from an
        ENDF data field.  May contain leading/trailing whitespace.

    Returns
    -------
    float
        The converted numeric value.  Returns ``0.0`` for blank fields.

    Raises
    ------
    ParseError
        If the string cannot be converted to a float after all
        transformations.

    Notes
    -----
    Conversion strategy:

    1. Replace ``D`` with ``E`` (Fortran double-precision marker).
    2. If no ``E`` is present but a ``+`` or ``-`` appears after the
       first character, insert ``E`` immediately before that sign.
    3. Call ``float()`` on the result.

    Examples
    --------
    >>> float_endf(" 1.23456+03")
    1234.56
    >>> float_endf(" 1.23456D-03")
    0.00123456
    >>> float_endf("  ")
    0.0
    """
    t = s.replace("D", "E").strip()
    if not t:
        return 0.0

    # Insert missing exponent marker when sign is embedded
    if ("+" in t[1:] or "-" in t[1:]) and "E" not in t.upper():
        idx = max(t.rfind("+", 1), t.rfind("-", 1))
        if idx > 0:
            t = t[:idx] + "E" + t[idx:]

    try:
        return float(t)
    except ValueError as exc:
        raise ParseError(
            f"Cannot convert ENDF field {s!r} to float (after transform: {t!r})"
        ) from exc


def int_endf(s: str) -> int:
    """Convert an ENDF-6 formatted string to a Python int

    Parameters
    ----------
    s : str
        An 11-character (or shorter) string taken from an ENDF integer
        field.  May contain leading/trailing whitespace.

    Returns
    -------
    int
        The converted integer value.  Returns ``0`` for blank or
        non-numeric fields.

    Examples
    --------
    >>> int_endf("         42")
    42
    >>> int_endf("           ")
    0
    """
    t = s.strip()
    if t.lstrip("-").isdigit():
        return int(t)
    return 0


# ---------------------------------------------------------------------------
# MF=26 / MT=525 raw-text parser
# ---------------------------------------------------------------------------

def parse_mf26_mt525(raw: str) -> list[dict]:
    """Parse MF=26, MT=525 large-angle elastic angular distribution data

    This section cannot be reliably parsed by the ``endf`` Python library
    and is therefore read directly from the raw section text.  The first
    eight lines are header/control records and are skipped.

    Each subsequent group begins with a CONT line containing the incident
    energy, a secondary energy value, and the word-count / pair-count
    descriptors ``NW`` and ``NL``.  The CONT line is followed by data
    lines carrying ``NL`` (μ, probability) pairs in 11-character-wide
    fields (six fields per line, yielding three pairs per full line).

    Parameters
    ----------
    raw : str
        Complete raw text of the ``(MF=26, MT=525)`` section, including
        the eight header lines.

    Returns
    -------
    list[dict]
        Each dict contains:

        * ``'E_loss'``  — float, first CONT field (often labelled E_in
          in the ENDF manual; stored here as energy loss).
        * ``'E_in'``    — float, second CONT field (incident energy).
        * ``'NW'``      — int, total words in the LIST record.
        * ``'NL'``      — int, number of μ–probability pairs.
        * ``'pairs'``   — list[tuple[float, float]], the (μ, probability)
          pairs extracted from the data lines.

    Raises
    ------
    ParseError
        If a data line is shorter than expected or a numeric field
        cannot be converted.

    Notes
    -----
    The ENDF-6 fixed-width column layout for CONT and LIST records is:

    * Columns 0–10:  field 1 (11 chars)
    * Columns 11–21: field 2
    * Columns 22–32: field 3
    * Columns 33–43: field 4
    * Columns 44–54: field 5 (NW)
    * Columns 55–65: field 6 (NL)

    Only the first 66 characters of each data line are parsed; the
    remaining 14 characters hold MAT/MF/MT/line-number identifiers that
    are not needed here.

    References
    ----------
    .. [1] ENDF-6 Formats Manual, §26.2 — File 26 Secondary Distributions.

    Examples
    --------
    >>> groups = parse_mf26_mt525(section_text)
    >>> groups[0]['E_in']
    10000.0
    >>> len(groups[0]['pairs'])
    32
    """
    lines = raw.splitlines()[ENDF_HEADER_LINES_MT525:]
    groups: list[dict] = []
    i = 0

    while i < len(lines):
        cont = lines[i]
        if len(cont) < ENDF_DATA_WIDTH:
            logger.warning(
                "CONT line %d shorter than %d chars, padding with spaces",
                i, ENDF_DATA_WIDTH,
            )
            cont = cont.ljust(ENDF_DATA_WIDTH)

        e_loss = float_endf(cont[0:ENDF_FIELD_WIDTH])
        e_in = float_endf(cont[ENDF_FIELD_WIDTH : 2 * ENDF_FIELD_WIDTH])
        nw = int_endf(cont[4 * ENDF_FIELD_WIDTH : 5 * ENDF_FIELD_WIDTH])
        nl = int_endf(cont[5 * ENDF_FIELD_WIDTH : ENDF_DATA_WIDTH])
        i += 1

        pairs: list[tuple[float, float]] = []
        while len(pairs) < nl and i < len(lines):
            ln = lines[i][:ENDF_DATA_WIDTH]
            vals = [
                float_endf(ln[j : j + ENDF_FIELD_WIDTH])
                for j in range(0, ENDF_DATA_WIDTH, ENDF_FIELD_WIDTH)
            ]
            for k in range(0, len(vals), 2):
                if len(pairs) < nl:
                    pairs.append((vals[k], vals[k + 1]))
            i += 1

        groups.append(
            {
                "E_loss": e_loss,
                "E_in": e_in,
                "NW": nw,
                "NL": nl,
                "pairs": pairs,
            }
        )

    logger.debug("Parsed %d angular-distribution groups from MF=26/MT=525", len(groups))
    return groups


# ---------------------------------------------------------------------------
# Interpolation and PDF helpers
# ---------------------------------------------------------------------------

def linear_interpolation(
    target_grid: np.ndarray,
    energy_ref: np.ndarray,
    values_ref: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate reference data onto a new energy grid

    Wrapper around :func:`numpy.interp` with a warning when the target
    grid extends beyond the reference grid.

    Parameters
    ----------
    target_grid : numpy.ndarray
        Energy points at which interpolated values are needed (eV).
    energy_ref : numpy.ndarray
        Reference energy grid (eV), must be monotonically increasing.
    values_ref : numpy.ndarray
        Reference values (e.g. cross sections in barns) corresponding
        to *energy_ref*.

    Returns
    -------
    numpy.ndarray
        Interpolated values on *target_grid*, same shape and dtype
        ``float64``.

    Notes
    -----
    Points in *target_grid* that exceed ``energy_ref[-1]`` are
    extrapolated as constant (``numpy.interp`` default), and a
    ``WARNING`` log message is emitted.

    Examples
    --------
    >>> grid = np.array([1.0, 2.0, 3.0])
    >>> ref_e = np.array([0.5, 1.5, 2.5, 3.5])
    >>> ref_v = np.array([10.0, 20.0, 30.0, 40.0])
    >>> linear_interpolation(grid, ref_e, ref_v)
    array([15., 25., 35.])
    """
    target = np.asarray(target_grid, dtype="f8")
    ref_e = np.asarray(energy_ref, dtype="f8")
    ref_v = np.asarray(values_ref, dtype="f8")

    if ref_e.size > 0 and np.any(target > ref_e[-1]):
        logger.warning(
            "Target grid has values (max=%.6e) beyond reference grid (max=%.6e); "
            "extrapolating as constant.",
            float(target.max()),
            float(ref_e[-1]),
        )

    return np.interp(target, ref_e, ref_v)


def build_pdf(
    inc_energy: np.ndarray,
    value: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert flat ENDF distribution arrays into grouped PDF format

    ENDF distribution data is stored as three aligned flat arrays:
    incident energy, outgoing value, and probability.  Multiple
    consecutive entries share the same incident energy.  This function
    groups the data by incident energy and returns offset arrays
    suitable for efficient HDF5 storage and look-up.

    Parameters
    ----------
    inc_energy : numpy.ndarray
        Flat array of incident energies (eV).  Consecutive equal values
        indicate entries that belong to the same distribution group.
    value : numpy.ndarray
        Flat array of outgoing values (e.g. cosine μ or energy E').
    probability : numpy.ndarray
        Flat array of probability-density values corresponding to
        *value*.

    Returns
    -------
    energy_grid : numpy.ndarray, float64
        Unique incident energies, one per group.
    energy_offset : numpy.ndarray, int64
        Start index of each group within *value* / *probability*.
    value_out : numpy.ndarray, float64
        Copy of *value* (contiguous float64).
    pdf_out : numpy.ndarray, float64
        Copy of *probability* (contiguous float64).

    Notes
    -----
    The offset array has the same length as *energy_grid*.  The
    number of entries in group *i* is
    ``energy_offset[i+1] - energy_offset[i]`` (with a sentinel
    equal to the total length appended internally during
    construction, then dropped before returning).

    Examples
    --------
    >>> inc = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    >>> val = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> pdf = np.array([0.5, 0.5, 0.3, 0.4, 0.3])
    >>> eg, eo, v, p = build_pdf(inc, val, pdf)
    >>> eg
    array([1., 2.])
    >>> eo
    array([0, 2])
    """
    inc = np.asarray(inc_energy, dtype="f8")
    val = np.asarray(value, dtype="f8")
    pdf = np.asarray(probability, dtype="f8")

    energy_vals: list[float] = []
    offsets: list[int] = [0]

    if inc.size:
        cur = inc[0]
        cnt = 1
        for e in inc[1:]:
            if e == cur:
                cnt += 1
            else:
                energy_vals.append(float(cur))
                offsets.append(offsets[-1] + cnt)
                cur = e
                cnt = 1
        energy_vals.append(float(cur))
        offsets.append(offsets[-1] + cnt)
    else:
        offsets = [0]

    energy_grid = np.asarray(energy_vals, dtype="f8")
    energy_offset = np.asarray(offsets[:-1], dtype="i8")

    return energy_grid, energy_offset, val, pdf


# ---------------------------------------------------------------------------
# Screened Rutherford small-angle scattering
# ---------------------------------------------------------------------------

def small_angle_eta(Z: int, energy_eV: np.ndarray) -> np.ndarray:
    """Compute the screened Rutherford parameter η for small-angle scattering

    The screening parameter η governs the forward-peaked Coulomb
    scattering distribution.  It is computed as:

    .. math::

        \\eta = \\frac{1}{4}
               \\left(\\frac{\\alpha\\,m_e c^2}{0.885\\,pc}\\right)^2
               Z^{2/3}
               \\left(1.13 + 3.76\\left(\\frac{\\alpha Z}{\\beta}\\right)^2\\right)
               \\sqrt{\\frac{\\tau}{\\tau + 1}}

    where α is the fine-structure constant, *p* is the relativistic
    momentum, β = v/c, and τ = T/(m_e c²).

    Parameters
    ----------
    Z : int
        Atomic number of the target.
    energy_eV : numpy.ndarray
        Kinetic energies of the incident electron (eV).

    Returns
    -------
    numpy.ndarray
        η values, same shape as *energy_eV*.

    Notes
    -----
    Physical constants are taken from NIST CODATA 2018.

    References
    ----------
    .. [1] Salvat, F., Jablonski, A., & Powell, C. J. (2005).
       ELSEPA—Dirac partial-wave calculation of elastic scattering of
       electrons and positrons by atoms, positive ions and molecules.
       *Computer Physics Communications*, 165(2), 157–190.
    """
    from pyepics.utils.constants import FINE_STRUCTURE, ELECTRON_MASS

    alpha = FINE_STRUCTURE
    mec2 = ELECTRON_MASS  # MeV
    T = np.asarray(energy_eV, dtype="f8") / 1e6  # MeV
    pc = np.sqrt(T * (T + 2.0 * mec2))  # MeV
    E = T + mec2  # MeV
    beta = pc / E
    tau = T / mec2

    term = (alpha * mec2 / (0.885 * pc)) ** 2
    corr = 1.13 + 3.76 * (alpha * Z / beta) ** 2

    return 0.25 * term * (Z ** (2.0 / 3.0)) * corr * np.sqrt(tau / (tau + 1.0))


def small_angle_scattering_cosine(
    Z: int,
    energy_eV: np.ndarray,
    n_mu: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate small-angle scattering cosine distributions

    For each incident energy, produce a discrete probability-density
    function (PDF) over ``n_mu`` cosine bins in the range
    [0.999999, 1.0), using the screened Rutherford formula:

    .. math::

        f(\\mu) = \\frac{1}{(\\eta + 1 - \\mu)^2}

    The PDF is normalised by trapezoidal-rule integration over the
    bin widths.

    Parameters
    ----------
    Z : int
        Atomic number of the target.
    energy_eV : numpy.ndarray
        Kinetic energies for which distributions are needed (eV).
        Only energies where the small-angle cross section is positive
        should be passed.
    n_mu : int, optional
        Number of cosine grid points.  Default is 200.

    Returns
    -------
    energy_grid : numpy.ndarray
        Unique incident energies (eV), shape ``(N,)``.
    energy_offset : numpy.ndarray
        Flat-array offsets for each energy group, shape ``(N+1,)``.
    value : numpy.ndarray
        Tiled cosine grid, shape ``(N * n_mu,)``.
    PDF : numpy.ndarray
        Normalised PDF values, shape ``(N * n_mu,)``.

    Notes
    -----
    The returned arrays use the same grouped-flat layout as
    :func:`build_pdf`.

    Examples
    --------
    >>> import numpy as np
    >>> eg, eo, val, pdf = small_angle_scattering_cosine(26, np.array([1e6]))
    >>> eg.shape
    (1,)
    >>> val.shape
    (200,)
    """
    energy_grid = np.asarray(energy_eV, dtype="f8").ravel()
    if energy_grid.size == 0:
        return (
            np.array([], dtype="f8"),
            np.array([0], dtype="i8"),
            np.array([], dtype="f8"),
            np.array([], dtype="f8"),
        )

    mu = np.linspace(0.999999, 1.0, n_mu, endpoint=False, dtype="f8")
    eta = small_angle_eta(Z, energy_grid)

    N, M = energy_grid.size, mu.size
    value = np.empty(N * M, dtype="f8")
    PDF = np.empty(N * M, dtype="f8")

    dmu = np.diff(mu)
    widths = np.empty(M, dtype="f8")
    if dmu.size:
        widths[:-1] = dmu
        widths[-1] = dmu[-1]
    else:
        widths[:] = 1.0

    for i, et in enumerate(eta):
        s = slice(i * M, (i + 1) * M)
        value[s] = mu
        f = 1.0 / (et + (1.0 - mu)) ** 2
        denom = np.dot(f, widths)
        PDF[s] = f / denom if denom > 0 else 0.0

    energy_offset = np.arange(0, (N + 1) * M, M, dtype="i8")
    return energy_grid, energy_offset, value, PDF


# ---------------------------------------------------------------------------
# File-path helpers
# ---------------------------------------------------------------------------

def extract_atomic_number_from_path(path: Path) -> int:
    """Extract the atomic number Z from an EPICS/ENDF file path

    Parameters
    ----------
    path : pathlib.Path
        Path to an ENDF file whose **stem** (or full name) contains the
        pattern ``ZA{ZZZ}000``, e.g. ``EEDL.ZA026000.endf``.

    Returns
    -------
    int
        Atomic number *Z*.

    Raises
    ------
    FileFormatError
        If the filename does not match the expected pattern.

    Examples
    --------
    >>> from pathlib import Path
    >>> extract_atomic_number_from_path(Path("EEDL.ZA001000.endf"))
    1
    """
    from pyepics.exceptions import FileFormatError

    m = ENDF_FILENAME_PATTERN.search(path.name)
    if m is None:
        raise FileFormatError(
            f"File name {path.name!r} does not match the expected EPICS "
            f"pattern 'ZA{{ZZZ}}000'.  Cannot determine atomic number."
        )
    return int(m.group(1))
