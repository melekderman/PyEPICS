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
(`python -m pyepics.cli download`), which writes to `data/2025/endf/<lib>/` and
also populates the inputs for `tests/test_mapping_completeness.py` (which scans
`data/2025/endf/<lib>/`). The CLI accepts `--release {2025,2023,2017}`
(default `2025`) and writes everything under `<data-dir>/data/<release>/`; the
fixtures above and the mapping-completeness test use EPICS 2025.

## Optional local April 2026 GFP reference

`tests/fixtures/april_gfp_Fe.npz` is a local-only reference that is not
authorized for distribution. Its exact path is ignored by Git. Do not add,
copy, regenerate, upload, or commit this file, or embed its contents in code
or another fixture.

When an existing local reference is available, `tests/test_mcdc_gfp.py`
uses it for the optional `test_parameters_match_april`, `test_fe_layout`,
and `test_fe_values` comparisons. These tests skip with an explicit reason
when the reference is absent; a present but unreadable or invalid archive
is an error. The Fe comparisons also require the optional EEDL file above.

Mandatory tests use synthetic inputs and independent mathematical references
to check the model-output schema, model fitting, and screened Rutherford
moments. They do not require the April archive or external ENDF files. The
independent Fe moment checks remain separate from the April comparisons.
