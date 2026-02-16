# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add parent directory to path so autodoc can import pyepics
sys.path.insert(0, os.path.abspath(".."))

project = "PyEPICS"
copyright = "2026, Melek Derman"
author = "Melek Derman"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon settings (Google/NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# MyST-Parser settings (allows .md files as source)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
