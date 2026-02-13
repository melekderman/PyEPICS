# PyEPICS Reference Data

This directory contains reference data for regression testing and validation
of PyEPICS against the IAEA EPICS 2023 database.

## Files

| File | Description |
|---|---|
| `reference_binding_energies.csv` | NIST X-Ray Transition Energies binding energies for select elements (eV) |
| `reference_cross_sections.csv` | Spot-check cross-section values from EPICS 2023 documentation |
| `.gitkeep` | Placeholder for HDF5 output files produced during test runs |

## Sources

- **Binding Energies**: NIST X-Ray Transition Energies Database
  <https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html>
- **Cross Sections**: IAEA EPICS 2023 evaluated data
  <https://www-nds.iaea.org/epics/>
- **Atomic Relaxation**: EADL (Evaluated Atomic Data Library) via ENDF-6 format
