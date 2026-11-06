Group Collapse
==============

This page describes how pointwise (continuous-energy) cross sections and
Boltzmann-Fokker-Planck coefficients are averaged onto energy groups.
The pointwise quantities themselves, including the split into soft and catastrophic
collisions, are described in :doc:`bfp`.

Overview
--------

A multigroup value is the average of the pointwise quantity over an energy group,
weighted by an approximation of the particle flux in that group.
The collapse needs three inputs:

- the pointwise data, e.g. a cross section :math:`\Sigma(E)`;
- an energy group structure (see `Energy Group Structure`_);
- a weighting spectrum, :math:`w(E)` (see `Weighting Spectrum`_).

It produces:

- group-averaged total and reaction cross sections (see
  `Group-Averaged Cross Sections`_);
- group-to-group scattering and production matrices (see `Scattering Matrices`_);
- the restricted stopping power and, optionally, the restricted energy straggling,
  averaged over each group and evaluated at the group boundaries.

The integrals are evaluated numerically, with Gauss-Legendre quadrature on each
segment between the energies of the pointwise table [1]_.

Energy Group Structure
----------------------

The :math:`G` energy groups are defined by :math:`G+1` bounds in eV,
:math:`E_0 > E_1 > \dots > E_G \ge 0`.
Group :math:`g` spans :math:`E_g < E \le E_{g-1}` and has width
:math:`\Delta E_g = E_{g-1} - E_g`.
Groups are numbered from the highest energy: group :math:`g = 1` is the highest-energy
group (index 0 in the HDF5 file).

The lowest bound, :math:`E_G`, is the lower energy limit of the data.
Particles that fall below it are considered absorbed.

The group structure also affects the pointwise data, not only the averaging: the
soft/catastrophic cutoff :math:`E_c(E)` is defined from the group bounds, so the
restricted stopping power, straggling, momentum transfer and the catastrophic
scattering all depend on it (see :ref:`bfp-cutoff`).

Weighting Spectrum
------------------

The group average weights each energy by :math:`w(E)`, a function that approximates
the scalar flux :math:`\phi(E)` within the group.
Only its shape within each group matters: multiplying :math:`w` by a constant changes
nothing.
The choice has the largest effect for wide groups and for quantities that vary quickly
with energy, and it vanishes as the groups become narrow.

Pre-defined spectra:

- **Uniform**, :math:`w(E) = 1`: constant flux in each group. This is the assumption
  of the group averages in [1]_.
- **Inverse energy**, :math:`w(E) = 1/E`: the flux of a particle slowing down in a
  medium. It is singular at zero, so it cannot be used for a structure whose lowest
  bound is 0.
- **Inverse stopping power**, :math:`w(E) = 1/S(E)`: the flux of a particle that
  slows down continuously and is not scattered.

A user-supplied spectrum can be any function of energy, or a tabulated flux (for
example, from an earlier transport calculation).
The weight must be finite and non-negative, and must integrate to a positive value in
every group.

The weighting is recorded with the library for provenance (see :doc:`/mg_hdf5_format`).

Group-Averaged Cross Sections
-----------------------------

For a pointwise cross section :math:`\Sigma(E)`, the group value is

.. math::

   \Sigma_g = \frac{\displaystyle\int_{E_g}^{E_{g-1}} \Sigma(E)\,w(E)\,dE}
                   {\displaystyle\int_{E_g}^{E_{g-1}} w(E)\,dE}.

The total cross section of a particle is the sum over all interactions :math:`x`
of the collapsed total cross sections of each interaction,
:math:`\Sigma_{t,g} = \sum_x \Sigma_{t,g}^x`.
The pointwise tables are interpolated linearly in energy and taken to be zero outside
their energy range.

The restricted stopping power :math:`S` and the restricted energy straggling
:math:`T` are averaged in the same way,

.. math::

   S_g = \frac{\int_g S(E)\,w(E)\,dE}{\int_g w(E)\,dE}, \qquad
   T_g = \frac{\int_g T(E)\,w(E)\,dE}{\int_g w(E)\,dE}.

The group-averaged stopping power is used to compute the energy deposited in each
group.
The stopping power and straggling are also evaluated, without averaging, at the group
boundaries,

.. math::

   S(E_g), \qquad T(E_g), \qquad g = 0, \dots, G,

since some discretizations of the Fokker-Planck operator require them.

The restricted momentum transfer :math:`\sigma_{\mathrm{tr},g}` is *not* averaged from pointwise values.
It follows from the group scattering moments of the transport correction (see
:ref:`bfp-transport-correction`).

Scattering Matrices
-------------------

The group-to-group scattering cross section of Legendre order :math:`\ell`, for a
particle going from group :math:`g'` to group :math:`g`, averages over the energy
:math:`E` in the source group and integrates over the final energy :math:`E'` in the
destination group:

.. math::

   \Sigma_{s,\ell,g'\to g} =
   \frac{\displaystyle\int_{g'} dE\; w(E) \int_{g} dE'\;
   \Sigma_{s,\ell}(E \to E')}
   {\displaystyle\int_{g'} w(E)\,dE}

where :math:`\Sigma_{s,\ell}(E\to E')` is the :math:`\ell`-th Legendre moment of the
differential scattering cross section,

.. math::

   \Sigma_{s,\ell}(E \to E') = 2\pi \int_{-1}^{1} P_\ell(\mu)\,
   \Sigma_s(E \to E', \mu)\, d\mu.

The same expression gives the production of a different particle (e.g. a
bremsstrahlung photon from an electron), with :math:`g'` in the incident
particle's group structure and :math:`g` in the group structure of the produced
particle.

Some conventions apply:

- The matrix is indexed ``[l, g_from, g_to]``.
  Since group 1 has the highest energy, down-scatter lies in the upper triangle, and
  codes that use the opposite convention must transpose it.
- For the BFP model, only the catastrophic collisions contribute, those that leave the
  particle below :math:`E_c(E)`. The soft collisions are in the stopping power,
  straggling and momentum transfer instead.
- Elastic scattering does not change the energy, so its matrix is diagonal
  (:math:`g' = g`).
- Production matrices include soft and catastrophic interactions, and need not be
  square, since the two particles may use different group structures.

The transport correction is applied *after* the collapse, to the within-group
(:math:`g \to g`) elastic moments and the total cross section of each group (see
:doc:`bfp`).

References
----------

.. [1] C. Bienvenue, A. Naceur, A. Hébert, J.-F. Carrier,
   "Toward highly accurate multigroup coupled photon-electron-positron
   cross-sections for the Boltzmann Fokker-Planck equation",
   J. Comput. Phys. 524 (2025) 113740.
   https://doi.org/10.1016/j.jcp.2025.113740
