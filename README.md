# PyEPICS

[![CI](https://github.com/melekderman/PyEPICS/actions/workflows/ci.yml/badge.svg)](https://github.com/melekderman/PyEPICS/actions/workflows/ci.yml)
[![Documentation](https://github.com/melekderman/PyEPICS/actions/workflows/docs.yml/badge.svg)](https://github.com/melekderman/PyEPICS/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![ReadTheDocs](https://readthedocs.org/projects/pyepics/badge/?version=latest)](https://pyepics.readthedocs.io/en/latest/)

> Python library for reading and converting EPICS (Electron Photon Interaction Cross Sections) nuclear data.

PyEPICS parses EEDL, EPDL, and EADL files from the [IAEA EPICS 2023](https://www-nds.iaea.org/epics/) database (in ENDF-6 format) and converts them into structured HDF5 files suitable for Monte Carlo transport codes such as [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC).

---

## Package Structure

```
PyEPICS/
├── pyepics/
│   ├── __init__.py              # Public API
│   ├── cli.py                   # Batch processing CLI
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── pyeedl_compat.py         # Backward-compatibility shim for legacy PyEEDL code
│   ├── readers/
│   │   ├── base.py              # Abstract BaseReader
│   │   ├── eedl.py              # EEDLReader — electron data
│   │   ├── epdl.py              # EPDLReader — photon data
│   │   └── eadl.py              # EADLReader — atomic relaxation data
│   ├── models/
│   │   └── records.py           # Typed dataclass models (EEDLDataset, EPDLDataset, EADLDataset)
│   ├── converters/
│   │   ├── hdf5.py              # High-level API (create_raw_hdf5, create_mcdc_hdf5)
│   │   ├── raw_hdf5.py          # Raw HDF5 writer (original grids, breakpoints)
│   │   └── mcdc_hdf5.py         # MCDC HDF5 writer (common grid, PDFs, interpolated)
│   ├── utils/
│   │   ├── constants.py         # Physical constants, periodic table, MF/MT tables
│   │   ├── parsing.py           # ENDF format parsing helpers
│   │   └── validation.py        # Post-parse validation routines
│   └── io/
│       └── download.py          # Dataset downloader from IAEA
└── tests/
    ├── conftest.py              # Shared pytest fixtures
    ├── test_eedl.py             # EEDL reader + parsing + validation tests
    ├── test_epdl.py             # EPDL reader tests
    ├── test_eadl.py             # EADL reader tests
    ├── test_hdf5.py             # Legacy HDF5 converter tests
    ├── test_pipeline.py         # Raw + MCDC pipeline tests
    └── generate_report.py       # PDF regression-test report generator
```

## Architecture

The package follows a strict layered dependency graph:

```
utils ← models ← readers ← converters (raw_hdf5 / mcdc_hdf5)
```

| Layer | Responsibility |
|---|---|
| **utils** | ENDF parsing helpers (`float_endf`, `parse_mf26_mt525`, …), validation routines, physical constants, MF/MT mapping tables |
| **models** | Typed `dataclass` records (`EEDLDataset`, `EPDLDataset`, `EADLDataset`) — the sole output of readers and sole input to converters |
| **readers** | `EEDLReader`, `EPDLReader`, `EADLReader` — parse ENDF files via the `endf` library and return model instances |
| **converters** | Two-step conversion: `raw_hdf5` (full-fidelity) and `mcdc_hdf5` (transport-optimised) |
| **io** | Dataset download from IAEA |
| **cli** | Batch processing for the full pipeline |

## Installation

```bash
pip install numpy h5py endf
# For downloading data from IAEA:
pip install requests beautifulsoup4
```

---

## Data Pipeline

PyEPICS follows a three-step pipeline, mirroring the PyEEDL workflow:

```
IAEA website                    download
    │                           ─────────────────────►
    ▼
eedl/ epdl/ eadl/              raw ENDF files (.endf)
    │                           ─────────────────────►
    ▼
raw_data/ raw_data_photon/     raw HDF5 (original grids, breakpoints)
raw_data_atomic/                for external users
    │                           ─────────────────────►
    ▼
mcdc_data/ mcdc_data_photon/   MCDC HDF5 (common grid, PDFs)
mcdc_data_atomic/               for transport codes
```

### Step 1: Download ENDF Data from IAEA

Download all three EPICS libraries (EEDL, EPDL, EADL) from the IAEA Nuclear Data Services:

```bash
# Download all libraries
python -m pyepics.cli download

# Download only electron data (EEDL)
python -m pyepics.cli download --libraries electron

# Download to a custom directory
python -m pyepics.cli download --data-dir /path/to/data
```

This creates three directories with `.endf` files:

```
eedl/   ← EEDL.ZA001000.endf, EEDL.ZA002000.endf, ... (Z=1–100)
epdl/   ← EPDL.ZA001000.endf, ... 
eadl/   ← EADL.ZA001000.endf, ...
```

### Step 2: Create Raw HDF5 Files

Raw files preserve every piece of information from the ENDF source: original energy grids, breakpoints, and interpolation law codes. These are intended for **external users** who need full-fidelity data.

```bash
# Create raw HDF5 for all libraries (Z=1–100)
python -m pyepics.cli raw

# Only electron data
python -m pyepics.cli raw --libraries electron

# Process a specific Z range
python -m pyepics.cli raw --z-min 1 --z-max 30

# Overwrite existing files
python -m pyepics.cli raw --overwrite
```

Output directories:

```
raw_data/          ← H.h5, He.h5, ..., Fe.h5, ... (electron)
raw_data_photon/   ← H.h5, He.h5, ...              (photon)
raw_data_atomic/   ← H.h5, He.h5, ...              (atomic relaxation)
```

### Step 3: Create MCDC HDF5 Files

MCDC files are optimised for transport codes. All cross sections are interpolated onto a common energy grid, angular distributions are compressed into (grid, offset, value, PDF) tables, and small-angle elastic scattering cosine PDFs are analytically computed.

```bash
# Create MCDC HDF5 for all libraries
python -m pyepics.cli mcdc

# Only electron data
python -m pyepics.cli mcdc --libraries electron

# Specific Z range
python -m pyepics.cli mcdc --z-min 26 --z-max 26   # Fe only
```

Output directories:

```
mcdc_data/          ← H.h5, He.h5, ..., Fe.h5, ... (electron)
mcdc_data_photon/   ← H.h5, He.h5, ...              (photon)
mcdc_data_atomic/   ← H.h5, He.h5, ...              (atomic relaxation)
```

### Full Pipeline (Raw + MCDC in One Step)

```bash
# Run raw + MCDC for all libraries, all elements
python -m pyepics.cli all

# Run everything but continue if an element fails
python -m pyepics.cli all --continue-on-error

# Full pipeline for first 30 elements only
python -m pyepics.cli all --z-min 1 --z-max 30 --overwrite
```

### Python API

You can also use the pipeline functions directly from Python:

```python
from pyepics import create_raw_hdf5, create_mcdc_hdf5

# Step 2: Raw HDF5
create_raw_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "raw_data/Fe.h5", overwrite=True)

# Step 3: MCDC HDF5
create_mcdc_hdf5("EEDL", "eedl/EEDL.ZA026000.endf", "mcdc_data/Fe.h5", overwrite=True)

# Download programmatically
from pyepics.io.download import download_library, download_all
download_library("eedl")      # downloads to ./eedl/
download_all()                 # downloads all three
```

---

## Quick Start

```python
from pyepics import EEDLReader

# Parse an EEDL file
reader = EEDLReader()
dataset = reader.read("eedl/EEDL.ZA026000.endf")
print(dataset.Z, dataset.symbol)  # 26, "Fe"
print(list(dataset.cross_sections.keys()))  # ['xs_tot', 'xs_el', 'xs_lge', ...]
```

## Exception Hierarchy

All library exceptions inherit from `PyEPICSError`:

```
PyEPICSError
├── ParseError          # Malformed ENDF content
├── ValidationError     # Failed physics checks (e.g. negative cross sections)
├── FileFormatError     # Wrong file type or unrecognised filename
├── ConversionError     # HDF5 write failures
└── DownloadError       # Network errors (future)
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Backward Compatibility

A `pyeedl_compat` shim re-exports all legacy `pyeedl` symbols:

```python
from pyepics.pyeedl_compat import PERIODIC_TABLE, float_endf, SUBSHELL_LABELS
```

## License

MIT — see [LICENSE](LICENSE) for details.
An open-source tool that converts EPICS datasets (EEDL, EADL, and EPDL) into structured HDF5 files, making evaluated electron-photon coupled physics data easy to use in Python-based simulations, analysis, and modern scientific workflows.
