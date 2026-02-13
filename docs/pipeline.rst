Data Pipeline
=============

PyEPICS follows a three-step pipeline:

1. **Download** ENDF files from IAEA
2. **Raw HDF5** — full-fidelity, original grids
3. **MCDC HDF5** — transport-code optimised

Step 1: Download
----------------

.. code-block:: bash

   python -m pyepics.cli download
   python -m pyepics.cli download --libraries electron

Step 2: Raw HDF5
-----------------

.. code-block:: bash

   python -m pyepics.cli raw
   python -m pyepics.cli raw --libraries electron --z-min 1 --z-max 30

Step 3: MCDC HDF5
------------------

.. code-block:: bash

   python -m pyepics.cli mcdc
   python -m pyepics.cli mcdc --libraries electron

Full Pipeline
-------------

.. code-block:: bash

   python -m pyepics.cli all --overwrite
