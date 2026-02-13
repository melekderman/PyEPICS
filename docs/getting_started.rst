Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install numpy h5py endf

   # For downloading data from IAEA:
   pip install requests beautifulsoup4

Quick Start
-----------

.. code-block:: python

   from pyepics import EEDLReader

   reader = EEDLReader()
   dataset = reader.read("eedl/EEDL.ZA026000.endf")
   print(dataset.Z, dataset.symbol)  # 26, "Fe"
