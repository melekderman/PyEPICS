User Guide
==========

This guide covers practical examples for querying element data, comparing
elements, plotting, and working with generated HDF5 files.

.. contents:: Table of Contents
   :local:
   :depth: 2


Querying a Single Element
--------------------------

The :class:`~pyepics.client.EPICSClient` is the main entry point for
interactive exploration.  Point it at the directory containing your ENDF
files:

.. code-block:: python

   from pyepics import EPICSClient

   client = EPICSClient("data/endf")

   # Look up iron by symbol, name, or atomic number — all equivalent:
   fe = client.get_element("Fe")
   fe = client.get_element("Iron")
   fe = client.get_element(26)

   # Inspect scalar metadata
   print(fe.Z)       # 26
   print(fe.symbol)   # "Fe"
   print(fe.name)     # "Iron"

   # List available electron (EEDL) cross-section keys
   print(fe.electron_cross_section_labels)
   # ['xs_tot', 'xs_el', 'xs_lge', 'xs_brem', 'xs_exc', 'xs_ion', 'xs_K', ...]

   # List available photon (EPDL) cross-section keys
   print(fe.photon_cross_section_labels)
   # ['xs_tot', 'xs_coherent', 'xs_incoherent', 'xs_photoelectric', ...]

   # Subshell binding energies from EADL
   print(fe.binding_energies)
   # {'K': 7112.0, 'L1': 844.6, 'L2': 719.9, 'L3': 706.8, ...}

   # Number of subshells and their labels
   print(fe.n_subshells, fe.subshells)

   # Get everything as a flat dictionary
   summary = fe.to_dict()
   print(summary.keys())


Retrieving Cross-Section Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the raw energy/cross-section NumPy arrays:

.. code-block:: python

   energy, xs = client.get_cross_section("Fe", "xs_tot", library="EEDL")
   print(energy.shape, xs.shape)  # e.g. (92,) (92,)

   # Same for photon cross sections
   energy, xs = client.get_cross_section("Fe", "xs_tot", library="EPDL")


Comparing Multiple Elements
----------------------------

Compare scalar properties across elements:

.. code-block:: python

   # Returns a list of dicts
   rows = client.compare(["Fe", "Cu", "Au"])
   for r in rows:
       print(f"{r['symbol']:>3s}  Z={r['Z']:>3d}  subshells={r['n_subshells']}")

   # Filter to specific properties
   rows = client.compare(
       ["H", "He", "Li", "Be", "B", "C"],
       properties=["Z", "symbol", "n_subshells"],
   )

If ``pandas`` is installed, get a DataFrame directly:

.. code-block:: python

   df = client.compare_df(["Fe", "Cu", "Au"])
   print(df[["symbol", "Z", "n_subshells"]])
   #   symbol   Z  n_subshells
   # 0     Fe  26           11
   # 1     Cu  29           12
   # 2     Au  79           25


Binding Energy Table
~~~~~~~~~~~~~~~~~~~~~

Build a binding-energy table across elements (requires ``pandas``):

.. code-block:: python

   df = client.binding_energy_table(range(26, 31))
   print(df[["Z", "K", "L1", "L2", "L3"]])


Plotting
--------

The :mod:`pyepics.plotting` module provides convenience wrappers for
common plots.  Requires ``matplotlib``:

.. code-block:: bash

   pip install matplotlib

Cross Sections for One Element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyepics.plotting import plot_cross_sections

   # All electron cross sections for iron
   plot_cross_sections(client, "Fe")

   # Specific labels only
   plot_cross_sections(client, "Fe", labels=["xs_tot", "xs_el", "xs_brem"])

   # Photon cross sections
   plot_cross_sections(client, "Fe", library="EPDL")


Compare a Cross Section Across Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyepics.plotting import compare_cross_sections

   compare_cross_sections(client, ["C", "Al", "Fe", "Cu", "Au"], "xs_tot")


Binding Energies vs. Atomic Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyepics.plotting import plot_binding_energies

   # All subshells
   plot_binding_energies(client, range(1, 100))

   # Only K shell
   plot_binding_energies(client, range(1, 100), subshell="K")


Shell-by-Shell Bar Chart
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyepics.plotting import plot_shell_binding_energies

   plot_shell_binding_energies(client, "Au")


Using Custom Axes
~~~~~~~~~~~~~~~~~~~

All plotting functions accept an ``ax`` parameter so you can compose
multi-panel figures:

.. code-block:: python

   import matplotlib.pyplot as plt
   from pyepics.plotting import plot_cross_sections

   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
   plot_cross_sections(client, "Fe", ax=axes[0], show=False)
   plot_cross_sections(client, "Au", ax=axes[1], show=False)
   plt.tight_layout()
   plt.show()


.. _user_guide_hdf5:

Reading Generated HDF5 Files
-----------------------------

After running the pipeline (see :doc:`pipeline`), PyEPICS produces HDF5
files in two output directories:

.. code-block:: text

   data/
   ├── raw/
   │   ├── electron/   ← H.h5, He.h5, … (EEDL)
   │   ├── photon/     ← H.h5, He.h5, … (EPDL)
   │   └── atomic/     ← H.h5, He.h5, … (EADL)
   └── mcdc/
       ├── electron/   ← H.h5, He.h5, … (EEDL)
       ├── photon/     ← H.h5, He.h5, … (EPDL)
       └── atomic/     ← H.h5, He.h5, … (EADL)

Each file is named ``<Symbol>.h5`` (e.g. ``Fe.h5`` for iron, ``Au.h5`` for
gold).


Locating Files
~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path

   raw_electron_dir = Path("data/raw/electron")
   files = sorted(raw_electron_dir.glob("*.h5"))
   print(files)  # [PosixPath('data/raw/electron/Ac.h5'), ...]


Inspecting HDF5 Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``h5py`` to inspect the internal structure:

.. code-block:: python

   import h5py

   with h5py.File("data/raw/electron/Fe.h5", "r") as f:
       # Print all top-level groups
       print(list(f.keys()))
       # e.g. ['metadata', 'total_cross_section', 'elastic_scatter', ...]

       # Walk every group and dataset
       def print_tree(name, obj):
           indent = "  " * name.count("/")
           if isinstance(obj, h5py.Dataset):
               print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
           else:
               print(f"{indent}{name}/")
       f.visititems(print_tree)


Raw Electron (EEDL) HDF5 Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /metadata
     Z (int), symbol (str), library (str="EEDL"), format (str="raw")

   /total_cross_section
     energy   (N,)   float64   [eV]
     xs       (N,)   float64   [barns]

   /elastic_scatter/total
     energy, xs

   /elastic_scatter/large_angle
     energy, xs

   /elastic_scatter/large_angle_angular_distribution
     inc_energy, cosine, probability

   /bremsstrahlung/cross_section
     energy, xs

   /bremsstrahlung/spectra
     inc_energy, photon_energy, probability

   /bremsstrahlung/average_energy_loss
     energy, avg_loss

   /excitation/cross_section
     energy, xs

   /excitation/average_energy_loss
     energy, avg_loss

   /ionization/total
     energy, xs

   /ionization/<subshell>/cross_section        (e.g. /ionization/K/cross_section)
     energy, xs

   /ionization/<subshell>/energy_spectrum
     inc_energy, secondary_energy, probability


MCDC Electron (EEDL) HDF5 Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /metadata
     Z, symbol, library="EEDL", format="mcdc"

   /electron_reactions/energy_grid      (M,)  float64  [eV]

   /electron_reactions/total_cross_section        (M,)
   /electron_reactions/elastic_scatter             (M,)
   /electron_reactions/large_angle_elastic         (M,)
   /electron_reactions/bremsstrahlung              (M,)
   /electron_reactions/excitation                  (M,)
   /electron_reactions/ionization                  (M,)

   /electron_reactions/large_angle_scattering_cosine/
     grid, offset, cosine, pdf

   /electron_reactions/small_angle_scattering_cosine/
     grid, offset, cosine, pdf

   /electron_reactions/bremsstrahlung_spectra/
     grid, offset, photon_energy, pdf

   /electron_reactions/excitation_average_energy_loss      (M,)
   /electron_reactions/bremsstrahlung_average_energy_loss  (M,)

   /electron_reactions/<subshell>/cross_section     (M,)
   /electron_reactions/<subshell>/binding_energy    scalar
   /electron_reactions/<subshell>/energy_spectrum/
     grid, offset, secondary_energy, pdf


Reading Cross Sections into a DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("data/raw/electron/Fe.h5", "r") as f:
       energy = f["total_cross_section/energy"][:]
       xs = f["total_cross_section/xs"][:]

   # If pandas is available:
   import pandas as pd

   df = pd.DataFrame({"energy_eV": energy, "xs_barns": xs})
   print(df.head())

   # Plot directly
   df.plot(x="energy_eV", y="xs_barns", logx=True, logy=True,
           title="Fe Total Electron Cross Section")


Reading MCDC Data
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("data/mcdc/electron/Fe.h5", "r") as f:
       energy = f["electron_reactions/energy_grid"][:]
       xs_tot = f["electron_reactions/total_cross_section"][:]
       xs_el  = f["electron_reactions/elastic_scatter"][:]

       # All cross sections are on the same energy grid
       assert energy.shape == xs_tot.shape == xs_el.shape


Reading Atomic Relaxation Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py

   with h5py.File("data/raw/atomic/Fe.h5", "r") as f:
       for subshell in f.keys():
           if subshell == "metadata":
               continue
           grp = f[subshell]
           be = grp.attrs.get("binding_energy_eV", "N/A")
           n  = grp.attrs.get("n_electrons", "N/A")
           nt = grp["transition_energy"].shape[0] if "transition_energy" in grp else 0
           print(f"{subshell}: BE={be} eV, electrons={n}, transitions={nt}")


Reading Photon Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py

   with h5py.File("data/raw/photon/Fe.h5", "r") as f:
       # Cross sections
       energy = f["total_cross_section/energy"][:]
       xs = f["total_cross_section/xs"][:]

       # Form factors
       x = f["form_factors/coherent/x"][:]
       y = f["form_factors/coherent/y"][:]

   # MCDC format
   with h5py.File("data/mcdc/photon/Fe.h5", "r") as f:
       energy = f["photon_reactions/energy_grid"][:]
       xs_tot = f["photon_reactions/total_cross_section"][:]


Converting to pandas and Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import pandas as pd
   import matplotlib.pyplot as plt

   with h5py.File("data/raw/electron/Fe.h5", "r") as f:
       df = pd.DataFrame({
           "energy": f["total_cross_section/energy"][:],
           "total": f["total_cross_section/xs"][:],
       })

       # Add more cross sections if available
       if "elastic_scatter/total" in f:
           e = f["elastic_scatter/total/energy"][:]
           xs = f["elastic_scatter/total/xs"][:]
           df_el = pd.DataFrame({"energy": e, "elastic": xs})
           df = df.merge(df_el, on="energy", how="outer").sort_values("energy")

   fig, ax = plt.subplots(figsize=(8, 5))
   ax.loglog(df["energy"], df["total"], label="Total")
   if "elastic" in df.columns:
       ax.loglog(df["energy"], df["elastic"], label="Elastic")
   ax.set_xlabel("Energy (eV)")
   ax.set_ylabel("Cross Section (barns)")
   ax.set_title("Fe — Electron Cross Sections (raw)")
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()
