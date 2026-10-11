# PyEPICS Test Fixtures

`tests/test_epdl.py`, `tests/test_eadl.py`, `tests/test_eedl.py`, and
`tests/test_hdf5.py` exercise the readers against real LLNL EPICS
2025 Iron (Z=26) ENDF files. Those data files are **not committed**; the tests
`skip` automatically unless the files are present at:

```
tests/fixtures/endf/EPDL.ZA026000.endf
tests/fixtures/endf/EADL.ZA026000.endf
tests/fixtures/endf/EEDL.ZA026000.endf
```

CI downloads them before running the tests (`.github/workflows/test.yml`).
To enable (un-skip) those tests locally, download the Fe files from LLNL
EPICS 2025:

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
