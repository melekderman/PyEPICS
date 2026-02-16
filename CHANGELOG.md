# Changelog

All notable changes to PyEPICS are documented in this file.

## [1.0.0] — 2026-02-15

### Added

- **Combined MCDC HDF5 output**: Each element now produces a single HDF5
  file (e.g. `Fe.h5`) containing `electron_reactions`, `photon_reactions`,
  and `atomic_relaxation` groups together.
- `create_combined_mcdc_hdf5()` — new public API for writing combined files.
- `write_mcdc_combined()` — low-level writer accepting EEDL/EPDL/EADL datasets.
- Full **EPDL** (photon) support: reader, raw HDF5, MCDC HDF5 — cross sections,
  form factors, photoelectric subshells, pair production.
- Full **EADL** (atomic relaxation) support: reader, raw HDF5, MCDC HDF5 —
  subshell binding energies, radiative/non-radiative transitions,
  fluorescence and Auger yields.
- `EPICSClient` — high-level API for querying element properties across
  all three libraries (EEDL, EPDL, EADL).
- `ElementProperties` — container with binding energies, cross sections,
  fluorescence data, and transition energies.
- CLI tool (`epics`) for download, raw, mcdc, and full pipeline execution.
- Sphinx documentation with API reference, data-sources guide, and pipeline docs.
- 164 unit tests covering readers, converters, mapping completeness, and pipeline.
- PDF regression-test report generator.

### Changed

- **PyPI package name** changed from `pyepics-data` to `epics`.
  Import name remains `pyepics` (`import pyepics`).
- Electron-specific constants renamed for clarity:
  `MF_MT` → `ELECTRON_MF_MT`,
  `SECTIONS_ABBREVS` → `ELECTRON_SECTIONS_ABBREVS`,
  `SUBSHELL_LABELS` → `ELECTRON_SUBSHELL_LABELS`.
  Old names kept as backward-compatible aliases.
- MCDC CLI now produces one combined file per element in `data/mcdc/`
  (previously separated into `data/mcdc/electron/`, `photon/`, `atomic/`).

## [0.1.0] — 2026-12-18

### Added

- Initial release with EEDL (electron) reader and HDF5 converter.
- Basic constant dictionaries and utility functions.
- PyEEDL backward-compatibility layer (`pyepics.pyeedl_compat`).
