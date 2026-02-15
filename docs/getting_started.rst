Getting Started
===============

Installation
------------

From PyPI (when published):

.. code-block:: bash

   pip install pyepics-data

With optional extras:

.. code-block:: bash

   # Plotting and pandas support
   pip install "pyepics-data[plot,pandas]"

   # All optional dependencies
   pip install "pyepics-data[all]"

From source (development):

.. code-block:: bash

   git clone https://github.com/melekderman/PyEPICS.git
   cd PyEPICS
   pip install -e ".[dev]"

See `INSTALL.md <https://github.com/melekderman/PyEPICS/blob/main/INSTALL.md>`_
for full details.

Quick Start
-----------

High-level client API:

.. code-block:: python

   from pyepics import EPICSClient

   client = EPICSClient("data/endf")

   # Query an element by symbol, name, or atomic number
   fe = client.get_element("Fe")
   print(fe.Z, fe.symbol)            # 26, "Fe"
   print(fe.binding_energies)         # {'K': 7112.0, 'L1': 844.6, ...}
   print(fe.electron_cross_section_labels)

   # Compare multiple elements
   df = client.compare_df(["Fe", "Cu", "Au"])

Low-level reader API:

.. code-block:: python

   from pyepics import EEDLReader

   reader = EEDLReader()
   dataset = reader.read("data/endf/eedl/EEDL.ZA026000.endf")
   print(dataset.Z, dataset.symbol)  # 26, "Fe"

See :doc:`user_guide` for more examples including plotting and reading
HDF5 output files.
