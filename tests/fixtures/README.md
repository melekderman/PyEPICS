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

## ENDF fixtures for the real-parse tests

`tests/test_epdl_real_parse.py`, `tests/test_eadl_real_parse.py`, and
`tests/test_eedl_real_parse.py` exercise the readers against real LLNL EPICS
2025 Iron (Z=26) ENDF files. Those data files are **not committed**; the tests
`skip` automatically unless the files are present at:

```
tests/fixtures/endf/EPDL.ZA026000.endf
tests/fixtures/endf/EADL.ZA026000.endf
tests/fixtures/endf/EEDL.ZA026000.endf
```

To enable (un-skip) those tests, download the Fe files from LLNL EPICS 2025:

```bash
mkdir -p tests/fixtures/endf
base=https://nuclear.llnl.gov/EPICS/ENDF2025
for lib in EPDL EADL EEDL; do
  curl -fsSL "$base/$lib.ELEMENTS/ZA026000" -o "tests/fixtures/endf/$lib.ZA026000.endf"
done
```

The full element set can also be fetched with the CLI
(`python -m pyepics.cli download`), which writes to `data/endf/<lib>/` and also
populates the inputs for `tests/test_mapping_completeness.py`.
