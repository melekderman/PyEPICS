Boltzmann-Fokker-Planck Moments
===============================

This page describes the electron data that PyEPICS generates for the multigroup
Boltzmann-Fokker-Planck (BFP) transport equation: what the equation needs, how
collisions are divided into *soft* and *catastrophic* ones, how each EEDL reaction
contributes, and how the elastic scattering is transport-corrected.
The methodology follows Bienvenue et al. [1]_ (below, "the paper"),
whose cross-section production code is RADIANT [2]_.

Overview
--------

Charged-particle scattering is strongly forward-peaked and loses little energy per
collision.
The BFP equation [3]_ divides collisions into two domains:

- **Catastrophic** collisions, with large energy loss or large angular deflection.
  They remain in the Boltzmann scattering operator.
- **Soft** collisions, with small energy loss and deflection. They are replaced by
  their leading moments, a continuous energy loss (the *stopping power*) and a
  continuous angular deflection (the *momentum transfer*), which form a
  Fokker-Planck operator.

For a particle with energy :math:`E`, direction :math:`\boldsymbol{\Omega} = (\mu, \theta)`
(polar cosine and azimuthal angle) and angular flux :math:`\psi`, the steady-state
equation is

.. math::

   \boldsymbol{\Omega}\cdot\nabla\psi + \Sigma_t \psi
   = Q^{\mathrm{FP}} + Q^{\mathrm{B}}

with the continuous slowing-down and angular Fokker-Planck operator

.. math::

   Q^{\mathrm{FP}} =
   \frac{\partial}{\partial E}\left[S(E)\,\psi\right]
   + \sigma_{\mathrm{tr}}(E)\left[\frac{\partial}{\partial\mu}\left(1-\mu^2\right)\frac{\partial}{\partial\mu}
   + \frac{1}{1-\mu^2}\frac{\partial^2}{\partial\theta^2}\right]\psi

and the Boltzmann operator for the catastrophic collisions

.. math::

   Q^{\mathrm{B}} = \sum_{p'} \int_0^\infty dE' \sum_{\ell=0}^{L}
   \frac{2\ell+1}{4\pi}\,\Sigma_{s,\ell}^{p'\to p}(E' \to E)
   \sum_{m=-\ell}^{\ell} Y_\ell^m(\boldsymbol{\Omega})
   \int_{4\pi} d^2\Omega'\, Y_\ell^m(\boldsymbol{\Omega}')\,\psi_{p'}(\mathbf{r},\boldsymbol{\Omega}',E')

where :math:`S` is the restricted stopping power, :math:`\sigma_{\mathrm{tr}}` is the restricted momentum
transfer, :math:`\Sigma_t` is the total cross section, :math:`\Sigma_{s,\ell}` is the
:math:`\ell`-th Legendre moment of the scattering cross section, :math:`P_\ell` are
Legendre polynomials and :math:`Y_\ell^m` are real spherical harmonics.

.. note::
   "Restricted" means that the quantity only includes *soft* collisions, those whose
   energy loss is below a cutoff.
   *Catastrophic* collisions, with a larger energy loss, are not part of the restricted
   stopping power, energy straggling and momentum transfer, and are instead in the
   scattering cross sections of the Boltzmann operator.
   See :ref:`bfp-cutoff` for how soft and catastrophic collisions are defined.

.. note::
   The paper [1]_ writes energies in units of :math:`m_e c^2` and uses macroscopic
   cross sections in cm\ :sup:`-1`.
   PyEPICS works per atom, with energies in eV and cross sections in barns, so that
   the data is independent of the material.
   Stopping power is therefore in eV\ ·barn, energy straggling in barn\ ·eV\ :sup:`2`
   and momentum transfer in barn.
   Multiplying by the atom density (atoms/(barn·cm)) gives the macroscopic values,
   in eV/cm, eV\ :sup:`2`/cm and cm\ :sup:`-1`.

The multigroup data that a BFP solver needs are, per group :math:`g`:

- the total cross section :math:`\Sigma_{t,g}` and the catastrophic scattering matrices
  :math:`\Sigma_{s,\ell,g'\to g}`, both transport-corrected (see
  `Transport Correction`_);
- the restricted stopping power :math:`S_g`, both group-averaged and at the group boundaries;
- optionally the restricted energy straggling :math:`T_g`, likewise;
- the restricted momentum transfer :math:`\sigma_{\mathrm{tr},g}`.

How pointwise data are averaged onto groups is described in :doc:`collapse`.

.. _bfp-cutoff:

Soft/Catastrophic Cutoff
------------------------

Whether a collision is soft or catastrophic is decided by a cutoff energy
:math:`E_c(E)` that depends on the energy :math:`E` of the incident electron.
A collision is **soft** if the electron leaves it with an energy of at least
:math:`E_c(E)`, so its energy loss is at most :math:`E - E_c(E)`.
Otherwise it is **catastrophic**.

The limiting cases show what the cutoff controls:

- :math:`E_c(E) = 0` makes every collision soft, and the BFP equation reduces to the
  Fokker-Planck equation.
- :math:`E_c(E) = E` makes every collision catastrophic, and it reduces to the
  Boltzmann equation.

The cutoff must be chosen consistently with the energy groups.
Two conventions are supported.

CEPXS
^^^^^

In CEPXS [4]_ a collision is soft if the particle scatters from group
:math:`g` into group :math:`g` or the adjacent group :math:`g+1`, and catastrophic if
it scatters into a non-adjacent group.
The cutoff is therefore the lower bound of the adjacent group:

.. math::

   E_c(E) = E_{g+1} \qquad \text{for } E_g < E \le E_{g-1}.

The cutoff is piecewise constant and jumps at every group boundary.
It makes the restricted stopping power at a boundary ambiguous, because the value depends on
whether the boundary is regarded as the top of group :math:`g+1` or the bottom of
group :math:`g`.
The paper reports a relative difference between the two definitions of between 3 %
and 25 % for impact ionization in aluminum, and between 72 % and 862 % for
bremsstrahlung (with 80 logarithmically spaced groups between 1 keV and 10.58 MeV).

RADIANT
^^^^^^^

RADIANT [1]_ removes the ambiguity by making the cutoff continuous.
It passes linearly through the points :math:`(E_{g-1},\,E_g)` and
:math:`(E_g,\,E_{g+1})`, so for :math:`E_g < E \le E_{g-1}`

.. math::

   E_c(E) = \left(\frac{E_g - E_{g+1}}{E_{g-1} - E_g}\right) E
   - \frac{E_g^2 - E_{g-1}\,E_{g+1}}{E_{g-1} - E_g}

with the additional definition :math:`E_{G+1} = 0`.
Below :math:`E_G` the cutoff is 0, so every collision is soft, which is also
the limit of both conventions as :math:`E \to E_G`.

Both conventions put the cutoff one group below the current energy.
RADIANT keeps the stopping power at group boundaries the same whichever neighbouring
group is considered.

Mapping of EEDL Reactions
-------------------------

The EEDL electron library gives pointwise cross sections and spectra for each
interaction.
Each one is divided into soft and catastrophic parts with :math:`E_c(E)`, as
summarized below.
The notation :math:`\Delta E` stands for the energy lost by the incident electron.

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32

   * - Reaction
     - EEDL data
     - Soft (:math:`\Delta E \le E - E_c`)
     - Catastrophic (:math:`\Delta E > E - E_c`)
   * - Elastic scattering
     - total, large-angle cosine tables
     - Forward-peaked part, through the transport correction: the momentum
       transfer :math:`\sigma_{\mathrm{tr},g}`
     - Remaining elastic moments, within-group scattering, with no energy loss
   * - Excitation
     - cross section, average energy loss
     - Stopping power and straggling, if the average loss is at most :math:`E - E_c`
     - Undeflected down-scatter by the average loss, otherwise
   * - Bremsstrahlung
     - cross section, photon energy spectra, average loss
     - Stopping power and straggling from photons with energy at most
       :math:`E - E_c`
     - Undeflected down-scatter of the electron, by the photon energy. The photon
       is produced in either case.
   * - Electroionization
     - cross section for each subshell, knock-on spectra, binding energies
     - Stopping power and straggling, if the loss (binding energy plus
       knock-on energy) is at most :math:`E - E_c`
     - Scattering of the primary electron, plus production of the knock-on electron
       in its energy group

Some points apply to all of them:

- Only electron-to-electron scattering is stored for the catastrophic collisions.
  Photons created by bremsstrahlung are *production* data and include both soft and
  catastrophic events, since the photons are transported in either case.
- The momentum transfer comes from elastic scattering only.
  Bremsstrahlung leaves the electron undeflected, and the paper neglects the
  deflection of ionization.
- The total cross section is the sum over all interactions of the full (soft plus
  catastrophic) elastic cross section and the catastrophic parts of the others. The
  soft collisions are not in it. It is then corrected as described under
  `Transport Correction`_.

Stopping Power and Energy Straggling
------------------------------------

The restricted stopping power is the mean energy lost per unit path length in soft
collisions.
It equals the total stopping power minus the catastrophic part [1]_:

.. math::

   S(E) = \sum_x \int_{E_c(E)}^{E} (E - E')\, \sigma_x(E \to E')\, dE'

where the sum runs over the interactions :math:`x` in the table above and
:math:`\sigma_x(E \to E')` is the differential cross section of interaction :math:`x`
for an electron going from energy :math:`E` to energy :math:`E'`.
The lower limit :math:`E_c(E)` keeps only the soft collisions.

The *energy straggling* coefficient is the second moment of the soft energy loss,

.. math::

   T(E) = \sum_x \int_{E_c(E)}^{E} (E - E')^2\, \sigma_x(E \to E')\, dE'.

Both are pointwise quantities; the group-averaged values are described in
:doc:`collapse`.

.. note::
   Straggling does not appear in the BFP equation of the paper, which keeps only the
   stopping power and the momentum transfer.
   It is the next term in the expansion of the soft collisions, and is provided as an
   optional addition for solvers that include it, e.g. as
   :math:`\tfrac{1}{2}\,\partial^2\!\left[T\,\psi\right]/\partial E^2` in the
   energy operator.

Momentum Transfer
-----------------

The Fokker-Planck angular operator has a single coefficient, the momentum transfer
:math:`\sigma_{\mathrm{tr}}`.
Soft collisions are those that deflect the particle by a small angle, so the
restricted momentum transfer is a property of the soft elastic scattering.
It is not computed by integrating the soft part of the differential cross section.
It follows from the Legendre moments of the *group* elastic scattering, as the
parameter :math:`\sigma_{\mathrm{tr},g}` of the transport correction (see below), so
it is only defined per group, and depends on the Legendre order used.

Elastic Scattering Legendre Moments
-----------------------------------

Deterministic transport needs the Legendre moments of the elastic cross section,

.. math::

   \Sigma_{s,\ell}(E) = \mathcal{N}\,\sigma_{s,\ell}(E), \qquad
   \sigma_{s,\ell}(E) = 2\pi\int_{-1}^{1} P_\ell(\mu)\,\sigma_s(E,\mu)\,d\mu
   = \sigma_{\mathrm{el}}(E)\,\langle P_\ell(\mu) \rangle

where :math:`\langle P_\ell \rangle` is the mean of :math:`P_\ell(\mu)` over a
collision and :math:`\sigma_{\mathrm{el}}` is the total elastic cross section.
The zeroth moment is the total elastic cross section, and the total cross section
used in transport is :math:`\Sigma_t = \Sigma_{s,0}` for elastic scattering.

RADIANT computes the angular distribution from the Mott cross section with Molière
screening and a Seltzer adjustment [1]_.
The EEDL library tabulates the angular distribution differently, and PyEPICS uses
that instead:

- the *large-angle* part, scattering cosines below :math:`1 - 10^{-6}`, comes from
  the tabulated cosine distributions;
- the *small-angle* remainder, :math:`\sigma_{\mathrm{el}} - \sigma_{\mathrm{la}}`, is
  described by a screened Rutherford distribution

  .. math::

     p(\mu) = \frac{2\eta\,(1+\eta)}{(1-\mu+2\eta)^2}

  with the Molière screening parameter :math:`\eta`.

The Legendre moments of both parts are evaluated exactly rather than by
quadrature, because the screened distribution is nearly singular at :math:`\mu = 1`
when :math:`\eta` is small.

The high-order moments become numerically unstable when scattering is nearly
isotropic.
Since an isotropic flux is characterized by vanishing moments for :math:`\ell \ge 1`,
the paper applies the following cleanup, which PyEPICS keeps:

.. math::

   \text{if } \left|\sigma_{s,\ell_x}(E)\right| > \left|\sigma_{s,0}(E)\right|
   \text{ then } \sigma_{s,\ell}(E) = 0 \quad \forall\, \ell \ge \ell_x.

.. _bfp-transport-correction:

Transport Correction
--------------------

The Boltzmann operator is meant to hold the catastrophic collisions only.
The *transport correction* removes the soft collisions from the elastic scattering in
the Boltzmann operator, leaving them to the Fokker-Planck operator, which represents
them through the momentum transfer :math:`\sigma_{\mathrm{tr},g}` [1]_.

For each group :math:`g`, let :math:`\Sigma_{s,\ell}^{g\to g}` be the elastic Legendre
moments within the group and :math:`L_{\max}` the last non-zero one.
The soft elastic moments are assumed to be

.. math::

   \Sigma_{s,\ell}^{\mathrm{soft}} = \Sigma_{s,0}^{\mathrm{soft}} - \sigma_{\mathrm{tr},g}\,\ell(\ell+1),
   \qquad \ell \in \{1, \dots, L_{\max}\},

and equating the soft and Boltzmann moments for :math:`\ell = L_{\max}-1` and
:math:`\ell = L_{\max}` (Landesman and Morel) fixes the two unknowns, the momentum
transfer

.. math::

   \sigma_{\mathrm{tr},g} = \frac{\Sigma_{s,L_{\max}-1}^{g\to g} - \Sigma_{s,L_{\max}}^{g\to g}}
              {2L_{\max}}

and the soft zeroth moment

.. math::

   \Sigma_{s,0}^{\mathrm{soft}} = \Sigma_{s,L_{\max}}^{g\to g}
   + \sigma_{\mathrm{tr},g}\,L_{\max}(L_{\max}+1).

The soft part is then removed from the data.
The transport-corrected total cross section and scattering moments are

.. math::

   \hat\Sigma_{t,g} = \Sigma_{t,g} - \Sigma_{s,L_{\max}}^{g\to g}
   - \sigma_{\mathrm{tr},g}\,L_{\max}(L_{\max}+1)

.. math::

   \hat\Sigma_{s,\ell}^{g\to g} = \Sigma_{s,\ell}^{g\to g} - \Sigma_{s,L_{\max}}^{g\to g}
   - \sigma_{\mathrm{tr},g}\left[L_{\max}(L_{\max}+1) - \ell(\ell+1)\right]

so that the corrected moment of order :math:`L_{\max}` vanishes and the corrected zeroth
moment equals the corrected elastic total.

Two simplifications of the correction exist:

- **Extended transport correction.** With :math:`\sigma_{\mathrm{tr},g} = 0`, the formulas reduce to
  :math:`\hat\Sigma_{t} = \Sigma_{t} - \Sigma_{s,L_{\max}}` and
  :math:`\hat\Sigma_{s,\ell} = \Sigma_{s,\ell} - \Sigma_{s,L_{\max}}`.
  It stays within the Boltzmann operator, and is the right choice when there is no
  Fokker-Planck operator to take up the soft part.
- **No correction.** The data keeps the physical total and moments.

A corrected total cross section is **not** the physical total, so the correction that
was applied must be recorded with the data and used consistently by the solver.
Groups whose elastic scattering is isotropic (:math:`L_{\max} < 1`) are not changed
and have :math:`\sigma_{\mathrm{tr},g} = 0`, and a negative :math:`\sigma_{\mathrm{tr},g}` (non-monotone moments) is set to
zero, which leaves the extended correction.

References
----------

.. [1] C. Bienvenue, A. Naceur, A. Hébert, J.-F. Carrier,
   "Toward highly accurate multigroup coupled photon-electron-positron
   cross-sections for the Boltzmann Fokker-Planck equation",
   J. Comput. Phys. 524 (2025) 113740.
   https://doi.org/10.1016/j.jcp.2025.113740

.. [2] C. Bienvenue, *Radiant.jl* (v1.0.0), 2024.
   https://github.com/CBienvenue/Radiant.jl

.. [3] K. Przybylski, J. Ligou, "Numerical analysis of the Boltzmann
   equation including Fokker-Planck terms", Nucl. Sci. Eng. 81 (1982) 92-109.

.. [4] L. Lorence Jr., J. E. Morel, G. D. Valdez, "Physics guide to CEPXS: a
   multigroup coupled electron-photon cross-section generating code", Technical Report,
   Sandia National Laboratories, 1989.
