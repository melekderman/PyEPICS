# Installing PyEPICS

## Supported Python Versions

PyEPICS requires **Python 3.11 or later** (3.11, 3.12, 3.13).

## Quick Install from PyPI

```bash
pip install pyepics-data
```

This installs the core package with the minimum required dependencies
(`numpy`, `h5py`, `endf`).

## Optional Extras

PyEPICS defines optional dependency groups you can install as needed:

| Extra      | What it adds                        | Install command                          |
|------------|-------------------------------------|------------------------------------------|
| `download` | `requests`, `beautifulsoup4`        | `pip install "pyepics-data[download]"`   |
| `pandas`   | `pandas`                            | `pip install "pyepics-data[pandas]"`     |
| `plot`     | `matplotlib`                        | `pip install "pyepics-data[plot]"`       |
| `all`      | All optional dependencies           | `pip install "pyepics-data[all]"`        |
| `dev`      | Testing + linting + docs tooling    | `pip install "pyepics-data[dev]"`        |

### Examples

```bash
# Core only (reading ENDF files and converting to HDF5)
pip install pyepics-data

# With plotting and pandas for interactive exploration
pip install "pyepics-data[plot,pandas]"

# Everything (including download support)
pip install "pyepics-data[all]"
```

## Developer Install

Clone the repository and install in editable mode with development
dependencies:

```bash
git clone https://github.com/melekderman/PyEPICS.git
cd PyEPICS

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Editable install with dev dependencies
pip install -e ".[dev]"
```

This gives you:

- `pytest` for running tests
- `ruff` for linting
- `sphinx`, `sphinx-rtd-theme`, `myst-parser` for building docs
- All optional runtime dependencies (`pandas`, `matplotlib`, etc.)

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only the client API tests
python -m pytest tests/test_client.py -v

# Run with coverage (if pytest-cov is installed)
python -m pytest tests/ --cov=pyepics --cov-report=term-missing
```

## Building Documentation

```bash
cd docs
pip install -r requirements.txt   # if not using [dev] extra
make html
```

The built HTML will be in `docs/_build/html/`.

## Building Distributions

To build source and wheel distributions for publishing:

```bash
pip install build
python -m build
```

This produces:

```
dist/
├── pyepics_data-0.1.0.tar.gz      # sdist
└── pyepics_data-0.1.0-py3-none-any.whl  # wheel
```

## Verifying the Install

After installation, verify everything works:

```python
import pyepics
print(pyepics.__version__)  # "0.1.0"

# Check that the client API is available
from pyepics import EPICSClient
print("EPICSClient loaded successfully")
```

## Platform Notes

- **macOS / Linux / Windows**: All supported via pure-Python code.
  Binary dependencies (`numpy`, `h5py`) provide pre-built wheels for
  all major platforms.
- **Conda**: You can also install dependencies via conda-forge:

  ```bash
  conda install numpy h5py
  pip install pyepics-data
  ```
