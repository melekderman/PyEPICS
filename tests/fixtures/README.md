# PyEPICS Reference Data

This directory contains reference data for regression testing and validation
of PyEPICS against the LLNL EPICS 2025 database.

## Files

| File | Description |
|---|---|
| `reference_binding_energies.csv` | Binding energies extracted from EEDL ENDF files for select elements (eV) |
| `reference_cross_sections.csv` | Spot-check cross-section values from EPICS 2025 documentation |
| `.gitkeep` | Placeholder for HDF5 output files produced during test runs |

## Sources

- **Binding Energies**: Extracted from EEDL (Evaluated Electron Data Library)
  ENDF-6 format files. These are **not** from NIST X-Ray Transition Energies.
  The values are used for round-trip validation of PyEPICS parsing.
- **Cross Sections**: LLNL EPICS 2025 evaluated data
  <https://nuclear.llnl.gov/EPICS/>
- **Atomic Relaxation**: EADL (Evaluated Atomic Data Library) via ENDF-6 format
