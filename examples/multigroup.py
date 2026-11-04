"""
Multigroup data processing example script

Collapses the EEDL electron data for one element onto an energy group
structure with the BFP transport model (RADIANT soft/catastrophic cutoff)
and writes the result to a multigroup HDF5 file.

Run from the repository root, after downloading the EEDL data::

    python examples/multigroup.py

.. warning::
   Work in progress.  The weighting spectrum and the BFP generation
   (``bfp.generate``) are implemented and the script builds the electron
   ``ParticleMG``.  ``mg_hdf5.write_mg_hdf5`` is not implemented yet, so the
   final steps (element, library and output file) are still commented out and
   no file is written.
"""

import sys
import pathlib
from pathlib import Path

import numpy as np

from pyepics import EEDLReader
from pyepics.multigroup.models import (
    GroupStructure,
    MGElement,
    MGLibrary,
    Particle,
    ParticleMG,
    TransportCorrection,
    TransportModel,
)
from pyepics.multigroup import weighting
from pyepics.multigroup import bfp
# from pyepics.multigroup.mg_hdf5 import write_mg_hdf5



EEDL_FILE = current_dir = Path(__file__).resolve().parent / "data" / "endf" / "eedl" / "EEDL.ZA026000.endf" # iron
OUT_FILE = Path("data/mg/Fe.h5")



if not EEDL_FILE.is_file():
    sys.exit(
        f"{EEDL_FILE} not found.  Download the data first:\n"
        f'    python -c "from pyepics.io.download import download_library; '
        f"download_library('eedl', 'data/endf/eedl')\""
    )

structure = GroupStructure("electron_40g", np.geomspace(10e6, 1e3, 41)) # bounds in eV
weight = weighting.uniform

# load CE (pointwise) data
dataset = EEDLReader().read(EEDL_FILE)

bfp_data = bfp.generate(dataset, structure, weight, cutoff="RADIANT")

electron = ParticleMG(
    Particle.ELECTRON,
    TransportModel.BFP,
    structure,
    cross_sections=bfp_data.cross_sections,
    scattering=bfp_data.scattering,
    coefficients=bfp_data.coefficients,
    cutoff="RADIANT",
    transport_correction=TransportCorrection.LANDESMAN_MOREL,
)
# element = MGElement(
#     dataset.Z, dataset.symbol, dataset.atomic_weight_ratio, electron=electron
# )
# library = MGLibrary("EPICS2025 electron, 40 groups", {element.Z: element})
# OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# write_mg_hdf5(library, OUT_FILE, overwrite=True)
# print(f"Wrote {OUT_FILE}")

