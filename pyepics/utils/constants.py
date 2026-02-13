#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Physical constants and data-mapping tables used across PyEPICS

All constants are sourced from NIST CODATA 2018 [1]_.  Mapping
dictionaries use ``(MF, MT)`` integer tuples as keys so that look-ups
from ENDF section identifiers are O(1).

References
----------
.. [1] NIST, "The 2018 CODATA Recommended Values of the Fundamental
   Physical Constants", https://physics.nist.gov/cuu/pdf/wallet_2018.pdf
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants  (NIST CODATA 2018)
# ---------------------------------------------------------------------------

FINE_STRUCTURE: float = 7.2973525693e-3
"""Fine-structure constant α (dimensionless)."""

ELECTRON_MASS: float = 0.51099895069
"""Electron rest-mass energy m_e c² (MeV)."""

BARN_TO_CM2: float = 1e-24
"""Conversion factor from barns to cm²."""

PLANCK_CONSTANT: float = 6.62607015e-34
"""Planck constant h (J·s, exact by SI definition)."""

SPEED_OF_LIGHT: float = 299792458.0
"""Speed of light in vacuum c (m/s, exact by SI definition)."""

ELECTRON_CHARGE: float = 1.602176634e-19
"""Elementary charge e (C, exact by SI definition)."""


# ---------------------------------------------------------------------------
# Periodic table  (Z = 1 … 118)
# ---------------------------------------------------------------------------

PERIODIC_TABLE: dict[int, dict[str, str]] = {
    1:   {"name": "Hydrogen",      "symbol": "H"},
    2:   {"name": "Helium",        "symbol": "He"},
    3:   {"name": "Lithium",       "symbol": "Li"},
    4:   {"name": "Beryllium",     "symbol": "Be"},
    5:   {"name": "Boron",         "symbol": "B"},
    6:   {"name": "Carbon",        "symbol": "C"},
    7:   {"name": "Nitrogen",      "symbol": "N"},
    8:   {"name": "Oxygen",        "symbol": "O"},
    9:   {"name": "Fluorine",      "symbol": "F"},
    10:  {"name": "Neon",          "symbol": "Ne"},
    11:  {"name": "Sodium",        "symbol": "Na"},
    12:  {"name": "Magnesium",     "symbol": "Mg"},
    13:  {"name": "Aluminium",     "symbol": "Al"},
    14:  {"name": "Silicon",       "symbol": "Si"},
    15:  {"name": "Phosphorus",    "symbol": "P"},
    16:  {"name": "Sulfur",        "symbol": "S"},
    17:  {"name": "Chlorine",      "symbol": "Cl"},
    18:  {"name": "Argon",         "symbol": "Ar"},
    19:  {"name": "Potassium",     "symbol": "K"},
    20:  {"name": "Calcium",       "symbol": "Ca"},
    21:  {"name": "Scandium",      "symbol": "Sc"},
    22:  {"name": "Titanium",      "symbol": "Ti"},
    23:  {"name": "Vanadium",      "symbol": "V"},
    24:  {"name": "Chromium",      "symbol": "Cr"},
    25:  {"name": "Manganese",     "symbol": "Mn"},
    26:  {"name": "Iron",          "symbol": "Fe"},
    27:  {"name": "Cobalt",        "symbol": "Co"},
    28:  {"name": "Nickel",        "symbol": "Ni"},
    29:  {"name": "Copper",        "symbol": "Cu"},
    30:  {"name": "Zinc",          "symbol": "Zn"},
    31:  {"name": "Gallium",       "symbol": "Ga"},
    32:  {"name": "Germanium",     "symbol": "Ge"},
    33:  {"name": "Arsenic",       "symbol": "As"},
    34:  {"name": "Selenium",      "symbol": "Se"},
    35:  {"name": "Bromine",       "symbol": "Br"},
    36:  {"name": "Krypton",       "symbol": "Kr"},
    37:  {"name": "Rubidium",      "symbol": "Rb"},
    38:  {"name": "Strontium",     "symbol": "Sr"},
    39:  {"name": "Yttrium",       "symbol": "Y"},
    40:  {"name": "Zirconium",     "symbol": "Zr"},
    41:  {"name": "Niobium",       "symbol": "Nb"},
    42:  {"name": "Molybdenum",    "symbol": "Mo"},
    43:  {"name": "Technetium",    "symbol": "Tc"},
    44:  {"name": "Ruthenium",     "symbol": "Ru"},
    45:  {"name": "Rhodium",       "symbol": "Rh"},
    46:  {"name": "Palladium",     "symbol": "Pd"},
    47:  {"name": "Silver",        "symbol": "Ag"},
    48:  {"name": "Cadmium",       "symbol": "Cd"},
    49:  {"name": "Indium",        "symbol": "In"},
    50:  {"name": "Tin",           "symbol": "Sn"},
    51:  {"name": "Antimony",      "symbol": "Sb"},
    52:  {"name": "Tellurium",     "symbol": "Te"},
    53:  {"name": "Iodine",        "symbol": "I"},
    54:  {"name": "Xenon",         "symbol": "Xe"},
    55:  {"name": "Caesium",       "symbol": "Cs"},
    56:  {"name": "Barium",        "symbol": "Ba"},
    57:  {"name": "Lanthanum",     "symbol": "La"},
    58:  {"name": "Cerium",        "symbol": "Ce"},
    59:  {"name": "Praseodymium",  "symbol": "Pr"},
    60:  {"name": "Neodymium",     "symbol": "Nd"},
    61:  {"name": "Promethium",    "symbol": "Pm"},
    62:  {"name": "Samarium",      "symbol": "Sm"},
    63:  {"name": "Europium",      "symbol": "Eu"},
    64:  {"name": "Gadolinium",    "symbol": "Gd"},
    65:  {"name": "Terbium",       "symbol": "Tb"},
    66:  {"name": "Dysprosium",    "symbol": "Dy"},
    67:  {"name": "Holmium",       "symbol": "Ho"},
    68:  {"name": "Erbium",        "symbol": "Er"},
    69:  {"name": "Thulium",       "symbol": "Tm"},
    70:  {"name": "Ytterbium",     "symbol": "Yb"},
    71:  {"name": "Lutetium",      "symbol": "Lu"},
    72:  {"name": "Hafnium",       "symbol": "Hf"},
    73:  {"name": "Tantalum",      "symbol": "Ta"},
    74:  {"name": "Tungsten",      "symbol": "W"},
    75:  {"name": "Rhenium",       "symbol": "Re"},
    76:  {"name": "Osmium",        "symbol": "Os"},
    77:  {"name": "Iridium",       "symbol": "Ir"},
    78:  {"name": "Platinum",      "symbol": "Pt"},
    79:  {"name": "Gold",          "symbol": "Au"},
    80:  {"name": "Mercury",       "symbol": "Hg"},
    81:  {"name": "Thallium",      "symbol": "Tl"},
    82:  {"name": "Lead",          "symbol": "Pb"},
    83:  {"name": "Bismuth",       "symbol": "Bi"},
    84:  {"name": "Polonium",      "symbol": "Po"},
    85:  {"name": "Astatine",      "symbol": "At"},
    86:  {"name": "Radon",         "symbol": "Rn"},
    87:  {"name": "Francium",      "symbol": "Fr"},
    88:  {"name": "Radium",        "symbol": "Ra"},
    89:  {"name": "Actinium",      "symbol": "Ac"},
    90:  {"name": "Thorium",       "symbol": "Th"},
    91:  {"name": "Protactinium",  "symbol": "Pa"},
    92:  {"name": "Uranium",       "symbol": "U"},
    93:  {"name": "Neptunium",     "symbol": "Np"},
    94:  {"name": "Plutonium",     "symbol": "Pu"},
    95:  {"name": "Americium",     "symbol": "Am"},
    96:  {"name": "Curium",        "symbol": "Cm"},
    97:  {"name": "Berkelium",     "symbol": "Bk"},
    98:  {"name": "Californium",   "symbol": "Cf"},
    99:  {"name": "Einsteinium",   "symbol": "Es"},
    100: {"name": "Fermium",       "symbol": "Fm"},
    101: {"name": "Mendelevium",   "symbol": "Md"},
    102: {"name": "Nobelium",      "symbol": "No"},
    103: {"name": "Lawrencium",    "symbol": "Lr"},
    104: {"name": "Rutherfordium", "symbol": "Rf"},
    105: {"name": "Dubnium",       "symbol": "Db"},
    106: {"name": "Seaborgium",    "symbol": "Sg"},
    107: {"name": "Bohrium",       "symbol": "Bh"},
    108: {"name": "Hassium",       "symbol": "Hs"},
    109: {"name": "Meitnerium",    "symbol": "Mt"},
    110: {"name": "Darmstadtium",  "symbol": "Ds"},
    111: {"name": "Roentgenium",   "symbol": "Rg"},
    112: {"name": "Copernicium",   "symbol": "Cn"},
    113: {"name": "Nihonium",      "symbol": "Nh"},
    114: {"name": "Flerovium",     "symbol": "Fl"},
    115: {"name": "Moscovium",     "symbol": "Mc"},
    116: {"name": "Livermorium",   "symbol": "Lv"},
    117: {"name": "Tennessine",    "symbol": "Ts"},
    118: {"name": "Oganesson",     "symbol": "Og"},
}


# ---------------------------------------------------------------------------
# Subshell mappings
# ---------------------------------------------------------------------------

SUBSHELL_LABELS: dict[int, str] = {
    534: "K",   535: "L1",  536: "L2",  537: "L3",
    538: "M1",  539: "M2",  540: "M3",  541: "M4",  542: "M5",
    543: "N1",  544: "N2",  545: "N3",  546: "N4",  547: "N5",
    548: "N6",  549: "N7",
    550: "O1",  551: "O2",  552: "O3",  553: "O4",  554: "O5",
    555: "O6",  556: "O7",  557: "O8",  558: "O9",
    559: "P1",  560: "P2",  561: "P3",  562: "P4",  563: "P5",
    564: "P6",  565: "P7",  566: "P8",  567: "P9",  568: "P10",
    569: "P11",
    570: "Q1",  571: "Q2",  572: "Q3",
}
"""Mapping from ENDF MT number to subshell label string.

Used to identify specific electron-ionisation or photoelectric
subshell cross-section sections.  MT 534 corresponds to the K shell,
535–537 to L sub-shells, and so on through the Q shell.
"""

SUBSHELL_DESIGNATORS: dict[int, str] = {
    1: "K",
    2: "L1",   3: "L2",   4: "L3",
    5: "M1",   6: "M2",   7: "M3",   8: "M4",   9: "M5",
    10: "N1",  11: "N2",  12: "N3",  13: "N4",  14: "N5",
    15: "N6",  16: "N7",
    17: "O1",  18: "O2",  19: "O3",  20: "O4",  21: "O5",
    22: "O6",  23: "O7",  24: "O8",  25: "O9",
    26: "P1",  27: "P2",  28: "P3",  29: "P4",  30: "P5",
    31: "P6",  32: "P7",  33: "P8",  34: "P9",  35: "P10",
    36: "P11",
    37: "Q1",  38: "Q2",  39: "Q3",
}
"""Mapping from EADL subshell-designator index to orbital label.

Index 1 = K (1s½), 2 = L1 (2s½), …, 39 = Q3 (7p³⁄₂).
"""

SUBSHELL_DESIGNATORS_INV: dict[str, int] = {
    v: k for k, v in SUBSHELL_DESIGNATORS.items()
}
"""Reverse mapping: subshell label → EADL designator index."""


# ---------------------------------------------------------------------------
# Electron (EEDL) MF/MT tables
# ---------------------------------------------------------------------------

MF_MT: dict[tuple[int, int], str] = {
    # MF=23 : Electron Cross Sections
    (23, 501): "Total Electron Cross Sections",
    (23, 522): "Ionization (sum of subshells)",
    (23, 525): "Large Angle Elastic Scattering Cross Section",
    (23, 526): "Elastic Scatter (Total) Cross Sections",
    (23, 527): "Bremsstrahlung Cross Sections",
    (23, 528): "Excitation Cross Sections",
    (23, 534): "K (1S1/2) Electroionization Subshell Cross Sections",
    (23, 535): "L1 (2s1/2) Electroionization Subshell Cross Sections",
    (23, 536): "L2 (2p1/2) Electroionization Subshell Cross Sections",
    (23, 537): "L3 (2p3/2) Electroionization Subshell Cross Sections",
    (23, 538): "M1 (3s1/2) Electroionization Subshell Cross Sections",
    (23, 539): "M2 (3p1/2) Electroionization Subshell Cross Sections",
    (23, 540): "M3 (3p3/2) Electroionization Subshell Cross Sections",
    (23, 541): "M4 (3d3/2) Electroionization Subshell Cross Sections",
    (23, 542): "M5 (3d5/2) Electroionization Subshell Cross Sections",
    (23, 543): "N1 (4s1/2) Electroionization Subshell Cross Sections",
    (23, 544): "N2 (4p1/2) Electroionization Subshell Cross Sections",
    (23, 545): "N3 (4p3/2) Electroionization Subshell Cross Sections",
    (23, 546): "N4 (4d3/2) Electroionization Subshell Cross Sections",
    (23, 547): "N5 (4d5/2) Electroionization Subshell Cross Sections",
    (23, 548): "N6 (4f5/2) Electroionization Subshell Cross Sections",
    (23, 549): "N7 (4f7/2) Electroionization Subshell Cross Sections",
    (23, 550): "O1 (5s1/2) Electroionization Subshell Cross Sections",
    (23, 551): "O2 (5p1/2) Electroionization Subshell Cross Sections",
    (23, 552): "O3 (5p3/2) Electroionization Subshell Cross Sections",
    (23, 553): "O4 (5d3/2) Electroionization Subshell Cross Sections",
    (23, 554): "O5 (5d5/2) Electroionization Subshell Cross Sections",
    (23, 555): "O6 (5f5/2) Electroionization Subshell Cross Sections",
    (23, 556): "O7 (5f7/2) Electroionization Subshell Cross Sections",
    (23, 559): "P1 (6s1/2) Electroionization Subshell Cross Sections",
    (23, 560): "P2 (6p1/2) Electroionization Subshell Cross Sections",
    (23, 561): "P3 (6p3/2) Electroionization Subshell Cross Sections",
    (23, 570): "Q1 (7s1/2) Electroionization Subshell Cross Sections",
    # MF=26 : Angular and Energy Distributions
    (26, 525): "Large Angle Elastic Angular Distributions",
    (26, 527): "Bremsstrahlung Photon Energy Spectra and Electron Average Energy Loss",
    (26, 528): "Excitation Electron Average Energy Loss",
    (26, 534): "K (1S1/2) Electroionization Subshell Energy Spectra",
    (26, 535): "L1 (2s1/2) Electroionization Subshell Energy Spectra",
    (26, 536): "L2 (2p1/2) Electroionization Subshell Energy Spectra",
    (26, 537): "L3 (2p3/2) Electroionization Subshell Energy Spectra",
    (26, 538): "M1 (3s1/2) Electroionization Subshell Energy Spectra",
    (26, 539): "M2 (3p1/2) Electroionization Subshell Energy Spectra",
    (26, 540): "M3 (3p3/2) Electroionization Subshell Energy Spectra",
    (26, 541): "M4 (3d3/2) Electroionization Subshell Energy Spectra",
    (26, 542): "M5 (3d5/2) Electroionization Subshell Energy Spectra",
    (26, 543): "N1 (4s1/2) Electroionization Subshell Energy Spectra",
    (26, 544): "N2 (4p1/2) Electroionization Subshell Energy Spectra",
    (26, 545): "N3 (4p3/2) Electroionization Subshell Energy Spectra",
    (26, 546): "N4 (4d3/2) Electroionization Subshell Energy Spectra",
    (26, 547): "N5 (4d5/2) Electroionization Subshell Energy Spectra",
    (26, 548): "N6 (4f5/2) Electroionization Subshell Energy Spectra",
    (26, 549): "N7 (4f7/2) Electroionization Subshell Energy Spectra",
    (26, 550): "O1 (5s1/2) Electroionization Subshell Energy Spectra",
    (26, 551): "O2 (5p1/2) Electroionization Subshell Energy Spectra",
    (26, 552): "O3 (5p3/2) Electroionization Subshell Energy Spectra",
    (26, 553): "O4 (5d3/2) Electroionization Subshell Energy Spectra",
    (26, 554): "O5 (5d5/2) Electroionization Subshell Energy Spectra",
    (26, 555): "O6 (5f5/2) Electroionization Subshell Energy Spectra",
    (26, 556): "O7 (5f7/2) Electroionization Subshell Energy Spectra",
    (26, 557): "O8 (5g7/2) Electroionization Subshell Energy Spectra",
    (26, 558): "O9 (5g9/2) Electroionization Subshell Energy Spectra",
    (26, 559): "P1 (6s1/2) Electroionization Subshell Energy Spectra",
    (26, 560): "P2 (6p1/2) Electroionization Subshell Energy Spectra",
    (26, 561): "P3 (6p3/2) Electroionization Subshell Energy Spectra",
    (26, 562): "P4 (6d3/2) Electroionization Subshell Energy Spectra",
    (26, 563): "P5 (6d5/2) Electroionization Subshell Energy Spectra",
    (26, 564): "P6 (6f5/2) Electroionization Subshell Energy Spectra",
    (26, 565): "P7 (6f7/2) Electroionization Subshell Energy Spectra",
    (26, 566): "P8 (6g7/2) Electroionization Subshell Energy Spectra",
    (26, 567): "P9 (6g9/2) Electroionization Subshell Energy Spectra",
    (26, 568): "P10 (6h7/2) Electroionization Subshell Energy Spectra",
    (26, 569): "P11 (6h9/2) Electroionization Subshell Energy Spectra",
    (26, 570): "Q1 (7s1/2) Electroionization Subshell Energy Spectra",
    (26, 571): "Q2 (7p1/2) Electroionization Subshell Energy Spectra",
    (26, 572): "Q3 (7p3/2) Electroionization Subshell Energy Spectra",
}
"""Human-readable descriptions for every EEDL (MF, MT) section pair."""

SECTIONS_ABBREVS: dict[tuple[int, int], str] = {
    # MF=23 cross sections
    (23, 501): "xs_tot",   (23, 522): "xs_ion",
    (23, 525): "xs_lge",   (23, 526): "xs_el",
    (23, 527): "xs_brem",  (23, 528): "xs_exc",
    (23, 534): "xs_K",     (23, 535): "xs_L1",   (23, 536): "xs_L2",
    (23, 537): "xs_L3",    (23, 538): "xs_M1",   (23, 539): "xs_M2",
    (23, 540): "xs_M3",    (23, 541): "xs_M4",   (23, 542): "xs_M5",
    (23, 543): "xs_N1",    (23, 544): "xs_N2",   (23, 545): "xs_N3",
    (23, 546): "xs_N4",    (23, 547): "xs_N5",   (23, 548): "xs_N6",
    (23, 549): "xs_N7",
    (23, 550): "xs_O1",    (23, 551): "xs_O2",   (23, 552): "xs_O3",
    (23, 553): "xs_O4",    (23, 554): "xs_O5",   (23, 555): "xs_O6",
    (23, 556): "xs_O7",    (23, 557): "xs_O8",   (23, 558): "xs_O9",
    (23, 559): "xs_P1",    (23, 560): "xs_P2",   (23, 561): "xs_P3",
    (23, 562): "xs_P4",    (23, 563): "xs_P5",   (23, 564): "xs_P6",
    (23, 565): "xs_P7",    (23, 566): "xs_P8",   (23, 567): "xs_P9",
    (23, 568): "xs_P10",   (23, 569): "xs_P11",
    (23, 570): "xs_Q1",    (23, 571): "xs_Q2",   (23, 572): "xs_Q3",
    # MF=26 distributions
    (26, 525): "ang_lge",   (26, 527): "loss_brem_spec",
    (26, 528): "loss_exc",
    (26, 534): "spec_K",    (26, 535): "spec_L1",  (26, 536): "spec_L2",
    (26, 537): "spec_L3",   (26, 538): "spec_M1",  (26, 539): "spec_M2",
    (26, 540): "spec_M3",   (26, 541): "spec_M4",  (26, 542): "spec_M5",
    (26, 543): "spec_N1",   (26, 544): "spec_N2",  (26, 545): "spec_N3",
    (26, 546): "spec_N4",   (26, 547): "spec_N5",  (26, 548): "spec_N6",
    (26, 549): "spec_N7",
    (26, 550): "spec_O1",   (26, 551): "spec_O2",  (26, 552): "spec_O3",
    (26, 553): "spec_O4",   (26, 554): "spec_O5",  (26, 555): "spec_O6",
    (26, 556): "spec_O7",   (26, 557): "spec_O8",  (26, 558): "spec_O9",
    (26, 559): "spec_P1",   (26, 560): "spec_P2",  (26, 561): "spec_P3",
    (26, 562): "spec_P4",   (26, 563): "spec_P5",  (26, 564): "spec_P6",
    (26, 565): "spec_P7",   (26, 566): "spec_P8",  (26, 567): "spec_P9",
    (26, 568): "spec_P10",  (26, 569): "spec_P11",
    (26, 570): "spec_Q1",   (26, 571): "spec_Q2",  (26, 572): "spec_Q3",
}
"""Short mnemonic abbreviations for each EEDL (MF, MT) section."""


# ---------------------------------------------------------------------------
# Photon (EPDL) MF/MT tables
# ---------------------------------------------------------------------------

PHOTON_MF_MT: dict[tuple[int, int], str] = {
    (23, 501): "Total Photon Cross Section",
    (23, 502): "Coherent (Rayleigh) Scattering Cross Section",
    (23, 504): "Incoherent (Compton) Scattering Cross Section",
    (23, 516): "Pair Production Cross Section (Total)",
    (23, 517): "Pair Production Cross Section (Nuclear Field)",
    (23, 518): "Pair Production Cross Section (Electron Field - Triplet)",
    (23, 522): "Total Photoelectric Cross Section",
    (23, 534): "K (1S1/2) Photoelectric Subshell Cross Section",
    (23, 535): "L1 (2s1/2) Photoelectric Subshell Cross Section",
    (23, 536): "L2 (2p1/2) Photoelectric Subshell Cross Section",
    (23, 537): "L3 (2p3/2) Photoelectric Subshell Cross Section",
    (23, 538): "M1 (3s1/2) Photoelectric Subshell Cross Section",
    (23, 539): "M2 (3p1/2) Photoelectric Subshell Cross Section",
    (23, 540): "M3 (3p3/2) Photoelectric Subshell Cross Section",
    (23, 541): "M4 (3d3/2) Photoelectric Subshell Cross Section",
    (23, 542): "M5 (3d5/2) Photoelectric Subshell Cross Section",
    (23, 543): "N1 (4s1/2) Photoelectric Subshell Cross Section",
    (23, 544): "N2 (4p1/2) Photoelectric Subshell Cross Section",
    (23, 545): "N3 (4p3/2) Photoelectric Subshell Cross Section",
    (23, 546): "N4 (4d3/2) Photoelectric Subshell Cross Section",
    (23, 547): "N5 (4d5/2) Photoelectric Subshell Cross Section",
    (23, 548): "N6 (4f5/2) Photoelectric Subshell Cross Section",
    (23, 549): "N7 (4f7/2) Photoelectric Subshell Cross Section",
    (23, 550): "O1 (5s1/2) Photoelectric Subshell Cross Section",
    (23, 551): "O2 (5p1/2) Photoelectric Subshell Cross Section",
    (23, 552): "O3 (5p3/2) Photoelectric Subshell Cross Section",
    (23, 553): "O4 (5d3/2) Photoelectric Subshell Cross Section",
    (23, 554): "O5 (5d5/2) Photoelectric Subshell Cross Section",
    (23, 555): "O6 (5f5/2) Photoelectric Subshell Cross Section",
    (23, 556): "O7 (5f7/2) Photoelectric Subshell Cross Section",
    (23, 557): "O8 (5g7/2) Photoelectric Subshell Cross Section",
    (23, 558): "O9 (5g9/2) Photoelectric Subshell Cross Section",
    (23, 559): "P1 (6s1/2) Photoelectric Subshell Cross Section",
    (23, 560): "P2 (6p1/2) Photoelectric Subshell Cross Section",
    (23, 561): "P3 (6p3/2) Photoelectric Subshell Cross Section",
    (23, 570): "Q1 (7s1/2) Photoelectric Subshell Cross Section",
    (27, 502): "Coherent Scattering Form Factor",
    (27, 504): "Incoherent Scattering Function",
    (27, 505): "Imaginary Anomalous Scattering Factor",
    (27, 506): "Real Anomalous Scattering Factor",
}

PHOTON_SECTIONS_ABBREVS: dict[tuple[int, int], str] = {
    (23, 501): "xs_tot",
    (23, 502): "xs_coherent",      (23, 504): "xs_incoherent",
    (23, 516): "xs_pair_total",    (23, 517): "xs_pair_nuclear",
    (23, 518): "xs_pair_electron", (23, 522): "xs_photoelectric",
    (23, 534): "xs_pe_K",   (23, 535): "xs_pe_L1",  (23, 536): "xs_pe_L2",
    (23, 537): "xs_pe_L3",  (23, 538): "xs_pe_M1",  (23, 539): "xs_pe_M2",
    (23, 540): "xs_pe_M3",  (23, 541): "xs_pe_M4",  (23, 542): "xs_pe_M5",
    (23, 543): "xs_pe_N1",  (23, 544): "xs_pe_N2",  (23, 545): "xs_pe_N3",
    (23, 546): "xs_pe_N4",  (23, 547): "xs_pe_N5",  (23, 548): "xs_pe_N6",
    (23, 549): "xs_pe_N7",
    (23, 550): "xs_pe_O1",  (23, 551): "xs_pe_O2",  (23, 552): "xs_pe_O3",
    (23, 553): "xs_pe_O4",  (23, 554): "xs_pe_O5",  (23, 555): "xs_pe_O6",
    (23, 556): "xs_pe_O7",  (23, 557): "xs_pe_O8",  (23, 558): "xs_pe_O9",
    (23, 559): "xs_pe_P1",  (23, 560): "xs_pe_P2",  (23, 561): "xs_pe_P3",
    (23, 570): "xs_pe_Q1",
    (27, 502): "ff_coherent",   (27, 504): "sf_incoherent",
    (27, 505): "asf_imag",      (27, 506): "asf_real",
}


# ---------------------------------------------------------------------------
# Atomic (EADL) MF/MT tables
# ---------------------------------------------------------------------------

ATOMIC_MF_MT: dict[tuple[int, int], str] = {
    (28, 533): "Atomic Relaxation Data",
}

ATOMIC_SECTIONS_ABBREVS: dict[tuple[int, int], str] = {
    (28, 533): "atomic_relax",
}


# ---------------------------------------------------------------------------
# ENDF field-description dictionaries (documentation only)
# ---------------------------------------------------------------------------

MF23: dict[str, str] = {
    "sigma.x":             "Array of incident-energy grid points (eV)",
    "sigma.y":             "Array of cross-section values (e.g. barns)",
    "sigma.breakpoints":   "NBT: number of points in each interpolation region",
    "sigma.interpolation": "INT: interpolation law code for each region",
    "ZA":                  "ZA identifier of the target (Z × 1000 + A)",
    "AWR":                 "Atomic weight ratio of the target",
}

MF26: dict[str, str] = {
    "ZAP":                              "Product identifier (11=electron, 0=photon)",
    "AWI":                              "Atomic weight ratio of the incident particle",
    "LAW":                              "Representation law (1=continuum, 2=angular, 8=ET)",
    "y":                                "Yield y(E) vs incident energy (Tabulated1D)",
    "distribution.LANG":                "Angular representation selector",
    "distribution.LEP":                 "Secondary-energy interpolation scheme",
    "distribution.NR":                  "Number of interpolation regions",
    "distribution.NE":                  "Number of incident-energy points",
    "distribution.E":                   "Array of incident energies (eV)",
    "distribution.distribution.ND":     "Number of discrete outgoing-energy points",
    "distribution.distribution.NA":     "Number of angular parameters",
    "distribution.distribution.NW":     "Total words in the LIST record",
    "distribution.distribution.NEP":    "Number of secondary energy points",
    "distribution.distribution.E'":     "Array of outgoing energies E' (eV)",
    "distribution.distribution.b":      "PDF or coefficient array",
}

MF27: dict[str, str] = {
    "sigma.x":             "Momentum transfer (1/Å) or energy (eV)",
    "sigma.y":             "Form factor or scattering function values",
    "sigma.breakpoints":   "NBT: interpolation region breakpoints",
    "sigma.interpolation": "INT: interpolation law codes",
    "ZA":                  "ZA identifier of the target",
    "AWR":                 "Atomic weight ratio of the target",
}

MF28: dict[str, str] = {
    "NSS":  "Number of subshells with data",
    "SUBI": "Subshell designator (ENDF-6)",
    "EBI":  "Subshell binding energy (eV)",
    "ELN":  "Number of electrons in subshell when neutral",
    "NTR":  "Number of transitions for this subshell",
    "SUBJ": "Designator for subshell from which electron originates",
    "SUBK": "Designator for fill-subshell (0=radiative)",
    "ETR":  "Transition energy (eV)",
    "FTR":  "Fractional probability of transition",
}
