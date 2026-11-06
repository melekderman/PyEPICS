Multigroup HDF5 Format
======================

Overview
--------

The :code:`multigroup` module contains functionality for collapsing pointwise data into group-average cross sections and Boltzmann Fokker-Planck coefficients.
The cross section data can be written to an HDF5 file.

A single HDF5 file contains data from multiple elements, called a "Library".

File Layout
-----------

.. code-block::
   :caption: Library example layout

    .
    ├── @title           # string
    ├── @comment         # string
    ├── @format          # "mg-pyepics-data"
    ├── @format-version  # string: major.minor.patch
    ├── @pyepics-version # string
    ├── @source          # String, EPICS
    ├── @date-time       # YYYY-MM-DDTHH:MM:SS date and time of generation
    ├── particles
    │   ├── electron
    │   │   ├── @model     # string: "bfp"
    │   │   ├── @cutoff    # string: "radiant", "cepxs", or custom
    │   │   ├── @weighting # string: "uniform", "inverse-s", "inverse-e"
    │   │   ├── @legendre-order # int, >= 0, max. Legendre order
    │   │   ├── @transport-correction # string: "none", "extended", "landesman-morel"
    │   │   └── energy-grid
    │   └── photon
    │       ├── @model   # string: must be "boltzmann"
    │       ├── @weighting
    │       ├── @legendre-order
    │       └── energy-grid
    └── elements
        ├── Al
        │   ├── @z-number
        │   ├── @atomic-weight-ratio
        │   ├── electron
        │   │   ├── cross-sections
        │   │   │   ├── total
        │   │   │   └── absorption
        │   │   ├── scattering 
        │   │   │   ├── inelastic       # [l, g_from, g_to]
        │   │   │   ├── elastic
        │   │   │   └── bremsstrahlung
        │   │   ├── soft-collisions
        │   │   │   ├── stopping-power
        │   │   │   │   ├── group-average
        │   │   │   │   └── boundary
        │   │   │   ├── energy-straggling
        │   │   │   │   ├── group-average
        │   │   │   │   └── boundary
        │   │   │   └── momentum-transfer
        │   │   └── production  # production matrices account for both soft and catastrophic interactions
        │   │       └── photon
        │   │           └── bremsstrahlung  # [l, g_electron_in, g_photon_out]
        │   └── photon
        └── Fe


Library Attributes
------------------

Each library has the following top-level attributes. Each one is required unless marked as optional:

- :code:`title`: library title
- :code:`comment`: optional, specified by the user who generated it
- :code:`format`: "mg-mcdc-data"
- :code:`format-version`
- :code:`pyepics-version`
- :code:`source`: source library for cross sections (e.g. EPICS2025)
- :code:`date-time`: date and time of generation

Particle Groups
---------------

Particle groups contain the parameters that each particle's reactions depend on:

+------------------------------+---------------------+-----------------------------------------------------------------------------------------+
| Attribute                    | Type                | Meaning                                                                                 |
+==============================+=====================+=========================================================================================+
| :code:`model`                | string, required    | Physics model: "bfp" for Boltzmann Fokker-Planck, or "boltzmann" for standard Boltzmann |
+------------------------------+---------------------+-----------------------------------------------------------------------------------------+
| :code:`cutoff`               | string, conditional | Only if :code:`model` is "bfp". Defines energy cutoff standard; "cepxs" or "radiant"    |
+------------------------------+---------------------+-----------------------------------------------------------------------------------------+
| :code:`weighting`            | string, required    | Weighting spectrum. "uniform", "inverse-e", or "inverse-s" (bfp only)                   |
+------------------------------+---------------------+-----------------------------------------------------------------------------------------+
| :code:`legendre-order`       | int, required       | Maximum Legendre moment for scattering.                                                 |
+------------------------------+---------------------+-----------------------------------------------------------------------------------------+
| :code:`transport-correction` | string, conditional | Electron only; photons are never corrected. "none", "extended", or "landesman-morel"    |
+------------------------------+---------------------+-----------------------------------------------------------------------------------------+

Each particle group must also contain :code:`energy-grid`, the energy group boundaries. 
These values are descending and in eV.

The particle group must be present in order to define data for that particle, or cross-species production data.

Element Groups
--------------

Each element is a group under :code:`elements`, named by its element symbol with its natural capitalization (e.g. :code:`Fe`).
An element group has the following attributes, both required:

- :code:`z-number`: atomic number, an integer from 1 to 118. Must match the group name.
- :code:`atomic-weight-ratio`: ratio of the atomic mass to the neutron mass (ENDF AWR), a positive float. Needed to convert the per-atom cross sections to macroscopic ones.

An element group contains an :code:`electron` and/or :code:`photon` group, and at least one of the two.
Each one may only be present if the matching group exists under :code:`particles`.
All of an element's arrays are defined on that particle's :code:`energy-grid`, so every element in a library shares the same group structure, transport model, cutoff and transport correction for each particle.

The contents of a particle group within an element depend on the particle's :code:`model`:

+--------------------------------------+--------------+--------------+--------------+------------------+
| Member                               | bfp          | bcsd         | fp           | boltzmann        |
+======================================+==============+==============+==============+==================+
| :code:`cross-sections`               | required     | required     | required     | required         |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`scattering`                   | required     | required     | forbidden    | required         |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`soft-collisions`              | required     | required     | required     | forbidden        |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`stopping-power`               | required     | required     | required     | n/a              |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`energy-straggling`            | optional     | forbidden    | required     | n/a              |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`momentum-transfer`            | required     | forbidden    | required     | n/a              |
+--------------------------------------+--------------+--------------+--------------+------------------+
| :code:`production`                   | optional     | optional     | optional     | optional         |
+--------------------------------------+--------------+--------------+--------------+------------------+
| particle :code:`cutoff` attribute    | required     | required     | forbidden    | forbidden        |
+--------------------------------------+--------------+--------------+--------------+------------------+

The :code:`stopping-power`, :code:`energy-straggling` and :code:`momentum-transfer` rows refer to members of :code:`soft-collisions`.
Photons only support the "boltzmann" model, so a photon group contains :code:`cross-sections` and :code:`scattering`, and optionally :code:`production`.

Datasets and Units
------------------

Every dataset is stored as :code:`float64` and carries a :code:`units` attribute holding the unit string given below.
A dataset holds a plain array.
Quantities that need several arrays are groups of datasets (e.g. :code:`stopping-power`).
In the shapes below, :code:`G` is the number of groups of the particle's :code:`energy-grid`, and :code:`L` is the particle's :code:`legendre-order`.

+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| Path (within a particle group of an element)   | Shape               | Units            | Meaning                                      |
+================================================+=====================+==================+==============================================+
| :code:`cross-sections/total`                   | :code:`(G,)`        | barn             | Total cross section (transport-corrected     |
|                                                |                     |                  | if :code:`transport-correction` is not       |
|                                                |                     |                  | "none")                                      |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`cross-sections/absorption`              | :code:`(G,)`        | barn             | Absorption cross section                     |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`scattering/<reaction>`                  | :code:`(L+1, G, G)` | barn             | Group-to-group scattering, indexed           |
|                                                |                     |                  | :code:`[l, g_from, g_to]`. For "bfp" and     |
|                                                |                     |                  | "bcsd" only catastrophic collisions          |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`soft-collisions/stopping-power/`        | :code:`(G,)`        | eV*barn          | Restricted stopping power averaged over each |
| :code:`group-average`                          |                     |                  | group                                        |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`soft-collisions/stopping-power/`        | :code:`(G+1,)`      | eV*barn          | Restricted stopping power at the group       |
| :code:`boundary`                               |                     |                  | boundaries                                   |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`soft-collisions/energy-straggling/`     | :code:`(G,)`        | barn*eV^2        | Energy-straggling coefficient averaged over  |
| :code:`group-average`                          |                     |                  | each group                                   |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`soft-collisions/energy-straggling/`     | :code:`(G+1,)`      | barn*eV^2        | Energy-straggling coefficient at the group   |
| :code:`boundary`                               |                     |                  | boundaries                                   |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`soft-collisions/momentum-transfer`      | :code:`(G,)`        | barn             | Restricted momentum-transfer cross section   |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+
| :code:`production/<secondary>/<reaction>`      | :code:`(L+1, G_in,` | barn             | Production of the secondary particle,        |
|                                                | :code:`G_out)`      |                  | indexed :code:`[l, g_in, g_out]`             |
+------------------------------------------------+---------------------+------------------+----------------------------------------------+

The particle's :code:`energy-grid` has shape :code:`(G+1,)` and units eV.
Further reaction labels may be added under :code:`cross-sections`, :code:`scattering` and :code:`production`.

Scattering
^^^^^^^^^^

Group :code:`0` is the highest-energy group, so down-scatter has :code:`g_to >= g_from` (the upper triangle, including the diagonal).
Codes that use the transposed convention must transpose when they read the file.
The :code:`l = 0` moment is non-negative, and higher moments may be negative.
For the "bfp" and "bcsd" models the scattering matrices contain only the catastrophic collisions, those above the cutoff.
For "boltzmann" they contain all collisions.

Production
^^^^^^^^^^

A production matrix is stored under the incident particle's group, in :code:`production/<secondary>/<reaction>`.
Its first axis uses the incident particle's :code:`energy-grid` (:code:`G_in` groups) and its second axis uses the secondary particle's :code:`energy-grid` (:code:`G_out` groups).
The two grids may differ, so the matrix is not square and has no down-scatter triangle.
Production matrices include both soft and catastrophic interactions, since the secondary particle must be transported either way.
The secondary particle's group must exist under :code:`particles`.

Conventions
-----------

- Cross sections are in barns
- Energy is in eV
- Groups are descending in energy (group :code:`0` is the highest-energy group)

Versioning
----------

The :code:`format-version` attribute is a string of the form :code:`major.minor.patch`.

- A change of the **major** version breaks compatibility: a layout change, or a change in the meaning of an existing attribute or dataset. Readers must check it and refuse files with a major version they do not support.
- A change of the **minor** version is backward compatible: new optional attributes, datasets or reaction labels. A reader written for the same major version can ignore what it does not recognize.
- A change of the **patch** version clarifies the specification or fixes a bug in a writer, with no change to the layout.

The :code:`pyepics-version` attribute records the version of PyEPICS that wrote the file for provenance only.
It is not used to decide how to read the file.
