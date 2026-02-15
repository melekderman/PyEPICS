Data Sources
============

PyEPICS processes evaluated nuclear data from the LLNL EPICS 2025 database.
This page documents the authoritative sources for each data type.

ENDF Libraries
--------------

All cross-section, relaxation, and transition data are parsed from ENDF-6
format files provided by LLNL:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Library
     - Description
     - Source
   * - **EEDL**
     - Evaluated Electron Data Library
     - `LLNL EPICS 2025 <https://nuclear.llnl.gov/EPICS/>`_
   * - **EPDL**
     - Evaluated Photon Data Library
     - `LLNL EPICS 2025 <https://nuclear.llnl.gov/EPICS/>`_
   * - **EADL**
     - Evaluated Atomic Data Library
     - `LLNL EPICS 2025 <https://nuclear.llnl.gov/EPICS/>`_

Binding Energies
----------------

Binding energies are sourced from the **EADL** (Evaluated Atomic Data Library),
parsed from the ENDF-6 format files. They are **not** sourced from the NIST
X-Ray Transition Energies database.

The reference validation data in ``tests/fixtures/reference_binding_energies.csv``
contains binding energies extracted from the EEDL ENDF files themselves. These
values are used for round-trip validation: the report generator compares
PyEPICS-parsed values against the ENDF source values to verify parsing
correctness.

The analysis covers all available subshells (K, L1, L2, L3, M1, …) for
elements Z=1 through Z=100, not only the K-shell.

Physical Constants
------------------

Physical constants are sourced from NIST CODATA 2018:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Constant
     - Value
     - Source
   * - Fine-structure constant (α)
     - 7.2973525693 × 10⁻³
     - NIST CODATA 2018
   * - Electron rest-mass energy (m_e c²)
     - 0.51099895069 MeV
     - NIST CODATA 2018
   * - Barn to cm² conversion
     - 1 × 10⁻²⁴
     - Definition
   * - Planck constant (h)
     - 6.62607015 × 10⁻³⁴ J·s
     - SI definition (exact)
   * - Speed of light (c)
     - 299 792 458 m/s
     - SI definition (exact)
   * - Elementary charge (e)
     - 1.602176634 × 10⁻¹⁹ C
     - SI definition (exact)

MF/MT Mapping Tables
---------------------

Every section in an ENDF file is identified by a pair of integers
**(MF, MT)** — *MF* selects the data type (cross sections, distributions,
form factors, …) and *MT* selects the reaction or subshell.  PyEPICS
keeps three mapping dictionaries in ``pyepics/utils/constants.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Dictionary
     - Library
     - Purpose
   * - ``MF_MT`` / ``SECTIONS_ABBREVS``
     - EEDL
     - Electron cross-section & distribution sections
   * - ``PHOTON_MF_MT`` / ``PHOTON_SECTIONS_ABBREVS``
     - EPDL
     - Photon cross-section & form-factor sections
   * - ``ATOMIC_MF_MT`` / ``ATOMIC_SECTIONS_ABBREVS``
     - EADL
     - Atomic relaxation sections

Each ``*_MF_MT`` dict maps ``(MF, MT)`` → human-readable description; each
``*_SECTIONS_ABBREVS`` dict maps the same keys → short mnemonic used as
HDF5 group names.

**Adding a new (MF, MT) pair** — If a new evaluation introduces a
previously unseen section:

1. Add the ``(MF, MT)`` key and its description to the appropriate
   ``*_MF_MT`` dictionary, following the ENDF-6 Formats Manual [1]_ for
   the correct meaning.
2. Add the same key with a short mnemonic to the matching
   ``*_SECTIONS_ABBREVS`` dictionary.
3. Run ``pytest tests/test_mapping_completeness.py`` — it will flag any
   ENDF (MF, MT) pair that lacks a mapping.

.. [1] A. Trkov *et al.*, "ENDF-6 Formats Manual", BNL-90365-2009 Rev. 2,
   https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf

Reference
---------

`NIST CODATA 2018 — Wallet Card (PDF) <https://physics.nist.gov/cuu/pdf/wallet_2018.pdf>`_
