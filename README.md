# PyEPICS

[![CI](https://github.com/melekderman/PyEPICS/actions/workflows/ci.yml/badge.svg)](https://github.com/melekderman/PyEPICS/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![ReadTheDocs](https://readthedocs.org/projects/pyepics/badge/?version=latest)](https://pyepics.readthedocs.io/en/latest/)

> Python library for reading and converting EPICS (Electron Photon Interaction Cross Sections) nuclear data by LLNL.

PyEPICS parses EEDL, EPDL, and EADL files from the [LLNL EPICS 2025](https://nuclear.llnl.gov/EPICS/) database (in ENDF-6 format) and converts them into structured HDF5 files suitable for Monte Carlo transport codes such as [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC).

---

## Package Structure

```
PyEPICS/
├── pyepics/                         # Source code
│   ├── __init__.py              # Public API
│   ├── client.py                # High-level element query API (EPICSClient)
│   ├── plotting.py              # Optional plotting helpers (matplotlib)
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
│       └── download.py          # Dataset downloader from LLNL
├── data/                            # All data (gitignored)
│   ├── endf/                    # Downloaded ENDF source files
│   │   ├── eedl/                # EEDL electron data
│   │   ├── epdl/                # EPDL photon data
│   │   └── eadl/                # EADL atomic relaxation data
│   ├── raw/                     # Generated raw HDF5 files
│   │   ├── electron/
│   │   ├── photon/
│   │   └── atomic/
│   └── mcdc/                    # Generated MCDC HDF5 files
│       ├── electron/
│       ├── photon/
│       └── atomic/
└── tests/
    ├── conftest.py              # Shared pytest fixtures
    ├── test_eedl.py             # EEDL reader + parsing + validation tests
    ├── test_epdl.py             # EPDL reader tests
    ├── test_eadl.py             # EADL reader tests
    ├── test_hdf5.py             # Legacy HDF5 converter tests
    ├── test_pipeline.py         # Raw + MCDC pipeline tests
    ├── generate_report.py       # PDF regression-test report generator
    ├── fixtures/                # Reference validation data
    └── reports/                 # Generated regression reports
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
| **client** | High-level `EPICSClient` for querying/comparing element properties |
| **plotting** | Optional visualisation helpers (requires `matplotlib`) |
| **io** | Dataset download from LLNL |
| **cli** | Batch processing for the full pipeline |

## Installation

```bash
# From PyPI (when published)
pip install epics

# With all optional dependencies
pip install "epics[all]"

# From source (editable, for development)
git clone https://github.com/melekderman/PyEPICS.git
cd PyEPICS
pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) for full details on optional extras, developer
setup, and platform notes.

---

## Data Pipeline

PyEPICS follows a three-step pipeline:

```
LLNL website                    download
    │                           ─────────────────────►
    ▼
data/endf/{eedl,epdl,eadl}/    raw ENDF files (.endf)
    │                           ─────────────────────►
    ▼
data/raw/{electron,photon,     raw HDF5 (original grids, breakpoints)
          atomic}/              for external users
    │                           ─────────────────────►
    ▼
data/mcdc/{electron,photon,    MCDC HDF5 (common grid, PDFs)
           atomic}/             for transport codes
```

### Step 1: Download ENDF Data from LLNL

Download all three EPICS libraries (EEDL, EPDL, EADL) from LLNL Nuclear Data:

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
data/endf/eedl/   ← EEDL.ZA001000.endf, EEDL.ZA002000.endf, ... (Z=1–100)
data/endf/epdl/   ← EPDL.ZA001000.endf, ... 
data/endf/eadl/   ← EADL.ZA001000.endf, ...
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
data/raw/electron/   ← H.h5, He.h5, ..., Fe.h5, ... (electron)
data/raw/photon/     ← H.h5, He.h5, ...              (photon)
data/raw/atomic/     ← H.h5, He.h5, ...              (atomic relaxation)
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
data/mcdc/electron/   ← H.h5, He.h5, ..., Fe.h5, ... (electron)
data/mcdc/photon/     ← H.h5, He.h5, ...              (photon)
data/mcdc/atomic/     ← H.h5, He.h5, ...              (atomic relaxation)
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
create_raw_hdf5("EEDL", "data/endf/eedl/EEDL.ZA026000.endf", "data/raw/electron/Fe.h5", overwrite=True)

# Step 3: MCDC HDF5
create_mcdc_hdf5("EEDL", "data/endf/eedl/EEDL.ZA026000.endf", "data/mcdc/electron/Fe.h5", overwrite=True)

# Download programmatically
from pyepics.io.download import download_library, download_all
download_library("eedl")      # downloads to ./data/endf/eedl/
download_all()                 # downloads all three
```

---

## Quick Start

### High-Level Client API

```python
from pyepics import EPICSClient

client = EPICSClient("data/endf")

# Query a single element (by symbol, name, or Z)
fe = client.get_element("Fe")
print(fe.Z, fe.symbol)            # 26, "Fe"
print(fe.binding_energies)         # {'K': 7112.0, 'L1': 844.6, ...}
print(fe.electron_cross_section_labels)  # ['xs_tot', 'xs_el', ...]

# Compare multiple elements
rows = client.compare(["Fe", "Cu", "Au"])

# DataFrame output (requires pandas)
df = client.compare_df(["Fe", "Cu", "Au"])

# Get raw cross-section arrays
energy, xs = client.get_cross_section("Fe", "xs_tot")
```

### Plotting (requires matplotlib)

```python
from pyepics.plotting import plot_cross_sections, compare_cross_sections

plot_cross_sections(client, "Fe")
compare_cross_sections(client, ["C", "Fe", "Au"], "xs_tot")
```

### Low-Level Reader API

```python
from pyepics import EEDLReader

# Parse an EEDL file directly
reader = EEDLReader()
dataset = reader.read("data/endf/eedl/EEDL.ZA026000.endf")
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
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Backward Compatibility

A `pyeedl_compat` shim re-exports legacy API symbols for backward compatibility:

```python
from pyepics.pyeedl_compat import PERIODIC_TABLE, float_endf, ELECTRON_SUBSHELL_LABELS
```

## Data Sources

PyEPICS uses the following authoritative data sources:

| Data Type | Source | Reference |
|---|---|---|
| **Electron cross sections** | EEDL (Evaluated Electron Data Library) | [LLNL EPICS 2025](https://nuclear.llnl.gov/EPICS/) |
| **Photon cross sections** | EPDL (Evaluated Photon Data Library) | [LLNL EPICS 2025](https://nuclear.llnl.gov/EPICS/) |
| **Atomic relaxation** | EADL (Evaluated Atomic Data Library) | [LLNL EPICS 2025](https://nuclear.llnl.gov/EPICS/) |
| **Binding energies** | EADL via ENDF-6 format (not NIST) | Parsed from EADL `.endf` files |
| **Physical constants** | NIST CODATA 2018 | [NIST CODATA](https://physics.nist.gov/cuu/pdf/wallet_2018.pdf) |

> **Note:** Binding energies are sourced from EADL (parsed from ENDF files), not from
> the NIST X-Ray Transition Energies database. The reference validation data in
> `tests/fixtures/reference_binding_energies.csv` is extracted from EEDL ENDF files
> and compared against the PyEPICS-parsed values to ensure round-trip consistency.

## Acknowledgements

This work was supported by the Center for Advancing the Radiation Resilience of Electronics (CARRE), a PSAAP-IV project funded by the Department of Energy, grant number: DE-NA0004268.

## License

Please see [LICENSE](LICENSE) for details.
