"""Tests for pyepics.multigroup.bfp (cutoff energy and pointwise stopping power)."""

from __future__ import annotations

import numpy as np
import pytest

from pyepics.exceptions import ValidationError
from pyepics.models.records import (
    AverageEnergyLoss,
    CrossSectionRecord,
    DistributionRecord,
    EEDLDataset,
)
from pyepics.multigroup.bfp import (
    CutoffModel,
    cutoff_energy,
    pointwise_stopping_power,
)
from pyepics.multigroup.models import GroupStructure

# Geometric structure, ratio 0.1: e = [1e4, 1e3, 100, 10]
STRUCTURE = GroupStructure("g3", [1e4, 1e3, 100.0, 10.0])
GRID = np.array([10.0, 100.0, 1e3, 1e4])


def xs(label, sigma, energy=GRID):
    return CrossSectionRecord(
        label, np.asarray(energy, float), np.asarray(sigma, float)
    )


def uniform_spectrum(label, upper, e_in=(10.0, 1e4)):
    """Flat pdf on [0, upper] at each incident energy"""
    n = len(e_in)
    return DistributionRecord(
        label,
        np.repeat(e_in, 2).astype(float),
        np.tile([0.0, upper], n),
        np.full(2 * n, 1.0 / upper),
    )


def dataset(**parts):
    cross_sections = {"xs_tot": xs("xs_tot", np.ones(GRID.size))}
    cross_sections.update(parts.get("cross_sections", {}))
    return EEDLDataset(
        26,
        "Fe",
        55.0,
        26000.0,
        cross_sections=cross_sections,
        distributions=parts.get("distributions", {}),
        average_energy_losses=parts.get("losses", {}),
        bremsstrahlung_spectra=parts.get("brem_spectra"),
    )


class TestCutoffEnergy:
    def test_radiant_is_one_group_below_on_geometric_grid(self):
        # holds for the top groups; the last one is linear down to zero (below)
        e = np.array([100.0, 500.0, 1e3, 5e3, 1e4])
        np.testing.assert_allclose(cutoff_energy(e, STRUCTURE), 0.1 * e)

    def test_radiant_last_group_goes_to_zero_linearly(self):
        # e[G + 1] = 0: line through (100 -> 10) and (10 -> 0)
        ec = cutoff_energy(np.array([10.0, 55.0, 100.0]), STRUCTURE)
        np.testing.assert_allclose(ec, [0.0, 5.0, 10.0])

    def test_radiant_continuous_across_bounds(self):
        for bound in STRUCTURE.energy[1:-1]:
            lo, hi = cutoff_energy(
                np.array([bound * (1 - 1e-9), bound * (1 + 1e-9)]), STRUCTURE
            )
            assert lo == pytest.approx(hi, rel=1e-6)

    def test_radiant_outside_structure(self):
        below, above = cutoff_energy(np.array([5.0, 2e4]), STRUCTURE)
        assert below == 0.0
        assert above == pytest.approx(2e3)  # top group's line continued

    def test_cepxs_is_lower_bound_of_adjacent_group(self):
        # group 0 is (1e3, 1e4] -> e[2] = 100; group 1 is (100, 1e3] -> e[3] = 10;
        # group 2 is (10, 100] -> e[4] = 0; below the structure -> 0
        e = np.array([5e3, 500.0, 50.0, 5.0, 2e4])
        np.testing.assert_allclose(
            cutoff_energy(e, STRUCTURE, "CEPXS"), [100.0, 10.0, 0.0, 0.0, 100.0]
        )

    def test_cepxs_bound_belongs_to_the_group_below(self):
        # E = 100 is the top of group 2 -> e[4] = 0
        assert cutoff_energy(np.array([100.0]), STRUCTURE, CutoffModel.CEPXS)[0] == 0.0

    @pytest.mark.parametrize("model", ["RADIANT", "CEPXS"])
    def test_between_zero_and_energy(self, model):
        e = np.geomspace(1.0, 1e6, 200)
        ec = cutoff_energy(e, STRUCTURE, model)
        assert np.all(ec >= 0.0) and np.all(ec <= e)

    def test_unknown_model(self):
        with pytest.raises(ValidationError, match="cutoff"):
            cutoff_energy(np.array([1.0]), STRUCTURE, "nope")


class TestExcitation:
    def make(self, loss):
        return dataset(
            cross_sections={"xs_exc": xs("xs_exc", [2.0] * 4)},
            losses={"loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, loss))},
        )

    def test_soft_when_loss_fits_below_cutoff(self):
        energy, s = pointwise_stopping_power(self.make(5.0), STRUCTURE)
        np.testing.assert_array_equal(energy, GRID)
        # loss cutoff E - Ec = [10, 90, 900, 9000]
        np.testing.assert_allclose(s, 10.0)

    def test_catastrophic_when_loss_exceeds_cutoff(self):
        _, s = pointwise_stopping_power(self.make(20.0), STRUCTURE)
        np.testing.assert_allclose(s, [0.0, 40.0, 40.0, 40.0])

    def test_string_cutoff_accepted(self):
        _, a = pointwise_stopping_power(self.make(20.0), STRUCTURE, "RADIANT")
        _, b = pointwise_stopping_power(self.make(20.0), STRUCTURE, CutoffModel.RADIANT)
        np.testing.assert_array_equal(a, b)


class TestIonization:
    @staticmethod
    def make():
        return dataset(
            cross_sections={"xs_K": xs("xs_K", [0.0, 1.0, 1.0, 1.0])},
            distributions={"spec_K": uniform_spectrum("spec_K", 1000.0)},
        )

    def test_restricted_mean_loss(self):
        # U = 10 (first energy of xs_K).  Soft iff U + W <= E - Ec, and the
        # knock-on is capped at (E - U) / 2.  The spectrum is flat on [0, 1000].
        _, s = pointwise_stopping_power(self.make(), STRUCTURE)
        u = 10.0

        def expected(cap):  # u * P(W <= cap) + <W; W <= cap>
            return u * cap / 1000.0 + cap**2 / 2000.0

        np.testing.assert_allclose(
            s,
            [
                0.0,  # below threshold cross section
                expected(min(90.0 - u, 45.0)),  # E=100: cutoff 80, cap 45
                expected(min(900.0 - u, 495.0)),  # E=1e3: cutoff 890, cap 495
                u + 500.0,  # E=1e4: the whole spectrum is soft
            ],
        )

    def test_no_soft_collisions_when_cutoff_below_binding(self):
        # E=10 has loss cutoff 10 = U, so only W = 0 would be soft: measure zero
        ds = self.make()
        ds.cross_sections["xs_K"] = xs("xs_K", [0.0, 0.0, 0.0, 1.0])
        _, s = pointwise_stopping_power(ds, STRUCTURE)
        np.testing.assert_allclose(s[:3], 0.0)

    def test_spectrum_without_cross_section(self):
        ds = self.make()
        del ds.cross_sections["xs_K"]
        with pytest.raises(ValidationError, match="xs_K"):
            pointwise_stopping_power(ds, STRUCTURE)

    def test_incident_energy_mixing_in_log_energy(self):
        # Spectra at 10 eV and 1e4 eV with different support; at 1e2 eV the
        # weight on the 1e4 spectrum is ln(10)/ln(1e3) = 1/3.
        e_in = np.array([10.0, 10.0, 1e4, 1e4])
        spectrum = DistributionRecord(
            "spec_K",
            e_in,
            np.array([0.0, 1.0, 0.0, 3.0]),
            np.array([1.0, 1.0, 1 / 3, 1 / 3]),
        )
        ds = dataset(
            cross_sections={"xs_K": xs("xs_K", [0.0, 1.0, 1.0, 1.0])},
            distributions={"spec_K": spectrum},
        )
        big = GroupStructure("big", [1e13, 1e12])  # Ec = 0, everything soft
        _, s = pointwise_stopping_power(ds, big)
        mean_w = (1 - 1 / 3) * 0.5 + (1 / 3) * 1.5
        assert s[1] == pytest.approx(10.0 + mean_w)


class TestBremsstrahlung:
    @staticmethod
    def make(mean=450.0):
        return dataset(
            cross_sections={"xs_brem": xs("xs_brem", [1.0] * 4)},
            losses={
                "loss_brem_spec": AverageEnergyLoss(
                    "loss_brem_spec", GRID, np.full(4, mean)
                )
            },
            brem_spectra=uniform_spectrum("brem", 1000.0),
        )

    def test_soft_fraction_scales_tabulated_mean(self):
        # photon energy is capped at E; flat spectrum on [0, 1000], so
        # frac = (min(L, 1000) / min(E, 1000))^2 with L = E - Ec = [10, 90, 900, 9000]
        _, s = pointwise_stopping_power(self.make(), STRUCTURE)
        np.testing.assert_allclose(s, 450.0 * np.array([1.0, 0.81, 0.81, 1.0]))

    def test_unrestricted_limit_is_tabulated_mean_not_spectrum_mean(self):
        big = GroupStructure("big", [1e13, 1e12])
        _, s = pointwise_stopping_power(self.make(mean=450.0), big)
        # spectrum mean is 500, but the table says 450; E=10 and 100 cut the spectrum
        assert s[3] == pytest.approx(450.0)

    def test_needs_spectra_and_mean_loss(self):
        ds = self.make()
        ds.bremsstrahlung_spectra = None
        with pytest.raises(ValidationError, match="spectra"):
            pointwise_stopping_power(ds, STRUCTURE)
        ds = self.make()
        ds.average_energy_losses = {}
        with pytest.raises(ValidationError, match="loss_brem_spec"):
            pointwise_stopping_power(ds, STRUCTURE)


class TestTotals:
    def test_channels_add_and_elastic_is_ignored(self):
        grid = np.array([100.0, 1e3])
        ds = EEDLDataset(
            26,
            "Fe",
            55.0,
            26000.0,
            cross_sections={
                "xs_tot": xs("xs_tot", [1.0, 1.0], grid),
                "xs_el": xs("xs_el", [9.0, 9.0], grid),
                "xs_lge": xs("xs_lge", [9.0, 9.0], grid),
                "xs_exc": xs("xs_exc", [2.0, 2.0], grid),
                "xs_brem": xs("xs_brem", [1.0, 1.0], grid),
                "xs_K": xs("xs_K", [3.0, 3.0, 3.0], [10.0, 100.0, 1e3]),
            },
            distributions={"spec_K": uniform_spectrum("spec_K", 5.0)},
            average_energy_losses={
                "loss_exc": AverageEnergyLoss("loss_exc", grid, [3.0, 3.0]),
                "loss_brem_spec": AverageEnergyLoss("loss_brem_spec", grid, [4.0, 4.0]),
            },
            bremsstrahlung_spectra=uniform_spectrum("brem", 5.0),
        )
        big = GroupStructure("big", [1e13, 1e12])
        _, s = pointwise_stopping_power(ds, big)
        # excitation 2*3 + brem 1*4 + ionization 3*(10 + 2.5)
        np.testing.assert_allclose(s, 6.0 + 4.0 + 37.5)

    def test_soft_never_exceeds_total_and_is_non_negative(self):
        ds = dataset(
            cross_sections={
                "xs_exc": xs("xs_exc", [2.0] * 4),
                "xs_brem": xs("xs_brem", [1.0] * 4),
                "xs_K": xs("xs_K", [0.0, 1.0, 1.0, 1.0]),
            },
            distributions={"spec_K": uniform_spectrum("spec_K", 1000.0)},
            losses={
                "loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, 20.0)),
                "loss_brem_spec": AverageEnergyLoss(
                    "loss_brem_spec", GRID, np.full(4, 450.0)
                ),
            },
            brem_spectra=uniform_spectrum("brem", 1000.0),
        )
        _, total = pointwise_stopping_power(ds, GroupStructure("big", [1e13, 1e12]))
        for model in CutoffModel:
            _, soft = pointwise_stopping_power(ds, STRUCTURE, model)
            assert np.all(soft >= 0.0)
            assert np.all(soft <= total * (1 + 1e-12))

    def test_requires_total_cross_section(self):
        ds = dataset()
        del ds.cross_sections["xs_tot"]
        with pytest.raises(ValidationError, match="xs_tot"):
            pointwise_stopping_power(ds, STRUCTURE)

    def test_empty_dataset_gives_zero(self):
        _, s = pointwise_stopping_power(dataset(), STRUCTURE)
        np.testing.assert_array_equal(s, 0.0)


# ---------------------------------------------------------------------------
# Straggling, catastrophic collisions, transport correction, generate
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from pyepics.multigroup import weighting  # noqa: E402
from pyepics.multigroup.bfp import (  # noqa: E402
    BFPData,
    _bin_integrals,
    _catastrophic_bremsstrahlung,
    _catastrophic_excitation,
    _catastrophic_ionization,
    generate,
    landesman_morel_correction,
    pointwise_straggling,
)
from pyepics.multigroup.models import (  # noqa: E402
    MOMENTUM_TRANSFER_LABEL,
    Particle,
    ParticleMG,
    TransportCorrection,
    TransportModel,
)

# Bottom bound 0: every outgoing electron lands in some group, so nothing is absorbed
OPEN = GroupStructure("open", [1e4, 1e3, 100.0, 10.0, 0.0])
FE_FILE = (
    Path(__file__).resolve().parent.parent
    / "examples/data/endf/eedl/EEDL.ZA026000.endf"
)


class TestStraggling:
    def test_excitation_is_loss_squared(self):
        ds = dataset(
            cross_sections={"xs_exc": xs("xs_exc", [2.0] * 4)},
            losses={"loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, 5.0))},
        )
        _, s2 = pointwise_straggling(ds, STRUCTURE)
        np.testing.assert_allclose(s2, 2.0 * 25.0)

    def test_ionization_second_moment_of_loss(self):
        ds = dataset(
            cross_sections={"xs_K": xs("xs_K", [0.0, 1.0, 1.0, 1.0])},
            distributions={"spec_K": uniform_spectrum("spec_K", 1000.0)},
        )
        energy, s2 = pointwise_straggling(ds, STRUCTURE)
        u = 10.0
        cap = 500.0  # E = 1e4: whole flat spectrum [0, 1000] is soft
        assert s2[3] == pytest.approx(u**2 + 2 * u * 500.0 + 1000.0**2 / 3.0)
        assert s2[0] == 0.0
        np.testing.assert_array_equal(energy, GRID)
        assert cap

    def test_bremsstrahlung_uses_the_spectrum_directly(self):
        ds = dataset(
            cross_sections={"xs_brem": xs("xs_brem", [1.0] * 4)},
            losses={"loss_brem_spec": AverageEnergyLoss("l", GRID, np.full(4, 450.0))},
            brem_spectra=uniform_spectrum("brem", 1000.0),
        )
        _, s2 = pointwise_straggling(ds, GroupStructure("big", [1e13, 1e12]))
        # E=1e4: whole spectrum, <k^2> = 1000^2 / 3 (no table scaling for the second moment)
        assert s2[3] == pytest.approx(1000.0**2 / 3.0)


class TestEvaluationAtGivenEnergies:
    def test_energy_argument_replaces_the_grid(self):
        ds = dataset(
            cross_sections={"xs_exc": xs("xs_exc", [2.0] * 4)},
            losses={"loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, 5.0))},
        )
        at = np.array([50.0, 500.0])
        energy, s = pointwise_stopping_power(ds, STRUCTURE, energy=at)
        np.testing.assert_array_equal(energy, at)
        np.testing.assert_allclose(s, 10.0)


class TestBinIntegrals:
    @staticmethod
    def brute(w, p, edges, order, cosine, limit=np.inf, n=200001):
        x = np.linspace(edges[0], edges[-1], n)
        density = np.interp(x, w, p, left=0.0, right=0.0) * (x <= limit)
        leg = np.polynomial.legendre.legvander(np.clip(cosine(x), -1, 1), order)
        out = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            m = (x >= a) & (x <= b)
            out.append(np.trapezoid(density[m, None] * leg[m], x[m], axis=0))
        return np.array(out)

    def test_matches_brute_force_with_a_cosine_and_a_limit(self):
        w = np.array([0.0, 100.0, 300.0, 900.0])
        p = np.array([0.004, 0.003, 0.001, 0.0002])
        edges = np.array([0.0, 50.0, 200.0, 400.0, 1000.0])

        def cosine(
            x,
        ):  # smooth on [0, 1000]; the real cosines start at a positive bound
            return np.sqrt((x + 50.0) / (x + 350.0))

        for limit in (np.inf, 350.0):
            got = _bin_integrals(w, p, edges, 4, cosine, limit)
            # the brute-force trapezoid is only O(dx) accurate across the step at `limit`
            np.testing.assert_allclose(
                got, self.brute(w, p, edges, 4, cosine, limit), atol=5e-6
            )

    def test_zero_outside_the_table(self):
        out = _bin_integrals(
            np.array([10.0, 20.0]),
            np.array([1.0, 1.0]),
            np.array([100.0, 200.0]),
            2,
            np.ones_like,
        )
        np.testing.assert_array_equal(out, 0.0)


class TestCatastrophicBremsstrahlung:
    @staticmethod
    def make(sigma=1.0):
        return dataset(
            cross_sections={"xs_brem": xs("xs_brem", [sigma] * 4)},
            losses={"loss_brem_spec": AverageEnergyLoss("l", GRID, np.full(4, 450.0))},
            brem_spectra=uniform_spectrum("brem", 1000.0),
        )

    def test_matches_analytic_groups_for_a_flat_spectrum(self):
        # E = 800: Ec = 80, so catastrophic means E_e <= 80, i.e. k in [720, 800].
        # Flat pdf on k in [0, 1000]: group 3 (0, 10] <-> k in [790, 800), group 2
        # (10, 80] <-> k in [720, 790).
        total, matrix = _catastrophic_bremsstrahlung(
            self.make(2.0), OPEN, "RADIANT", np.array([800.0]), 3
        )
        assert total[0] == pytest.approx(2.0 * 80.0 / 1000.0)
        np.testing.assert_allclose(matrix[0, 0], 2.0 * np.array([0.0, 0.0, 0.07, 0.01]))

    def test_every_moment_is_equal_because_the_electron_is_undeflected(self):
        _, matrix = _catastrophic_bremsstrahlung(
            self.make(), OPEN, "RADIANT", np.array([800.0, 3000.0]), 5
        )
        for l_order in range(1, 6):
            np.testing.assert_array_equal(matrix[:, l_order], matrix[:, 0])

    def test_nothing_is_lost_when_no_group_is_below(self):
        energy = np.array([20.0, 150.0, 800.0, 5000.0])
        total, matrix = _catastrophic_bremsstrahlung(
            self.make(), OPEN, "RADIANT", energy, 2
        )
        np.testing.assert_allclose(matrix[:, 0].sum(axis=1), total, rtol=1e-12)

    def test_electrons_that_end_below_the_structure_are_absorbed(self):
        closed = GroupStructure("closed", [1e4, 1e3, 100.0, 10.0])
        total, matrix = _catastrophic_bremsstrahlung(
            self.make(), closed, "RADIANT", np.array([800.0]), 2
        )
        assert matrix[0, 0].sum() < total[0]


class TestCatastrophicIonization:
    @staticmethod
    def make():
        return dataset(
            cross_sections={"xs_K": xs("xs_K", [0.0, 1.0, 1.0, 1.0])},
            distributions={"spec_K": uniform_spectrum("spec_K", 1000.0)},
        )

    def test_total_and_production_for_a_flat_spectrum(self):
        energy = np.array([200.0, 2000.0, 8000.0])
        total, matrix = _catastrophic_ionization(
            self.make(), OPEN, "RADIANT", energy, 2
        )
        u = 10.0
        ec = cutoff_energy(energy, OPEN)
        cap = (energy - u) / 2.0
        all_prob = np.minimum(cap, 1000.0) / 1000.0
        soft = np.clip(np.minimum(energy - u - ec, cap), 0.0, 1000.0) / 1000.0
        sigma = np.interp(energy, GRID, [0.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(total, sigma * (all_prob - soft), rtol=1e-12)
        # per collision: one knock-on (any W up to the cap) plus the primary when catastrophic
        np.testing.assert_allclose(
            matrix[:, 0].sum(axis=1), total + sigma * all_prob, rtol=1e-10
        )

    def test_is_downscatter_only_and_zero_below_threshold(self):
        energy = np.geomspace(5.0, 9000.0, 40)
        _, matrix = _catastrophic_ionization(self.make(), OPEN, "RADIANT", energy, 4)
        group = np.searchsorted(-OPEN.energy, -energy) - 1  # source group of each node
        for q in range(energy.size):
            for g in range(OPEN.n_groups):
                if g < group[q]:
                    assert np.all(matrix[q, :, g] == 0.0)
        np.testing.assert_array_equal(matrix[energy <= 10.0], 0.0)

    def test_knock_on_moments_follow_the_kinematic_cosine(self):
        # one node, one target group: compare with direct quadrature of P_l(mu_s(W))
        e_in = 2000.0
        _, matrix = _catastrophic_ionization(
            self.make(), OPEN, "RADIANT", np.array([e_in]), 3
        )
        sigma = np.interp(e_in, GRID, [0.0, 1.0, 1.0, 1.0])
        # knock-on W in group 2 (10, 100] with the flat pdf 1e-3
        w = np.linspace(10.0, 100.0, 400001)
        e_red, w_red = e_in / 510998.95, w / 510998.95
        mu = np.sqrt(w_red * (e_red + 2) / (e_red * (w_red + 2)))
        leg = np.polynomial.legendre.legvander(mu, 3)
        reference = sigma * np.trapezoid(1e-3 * leg, w, axis=0)
        # the primary also lands in group 2? E' >= (E - U) / 2 = 995: no
        # one decade-wide bin with a 6-point rule: real groups are a factor ~1.26 wide
        np.testing.assert_allclose(matrix[0, :, 2], reference, rtol=2e-5)


class TestCatastrophicExcitation:
    def test_down_scatter_by_the_loss_when_it_exceeds_the_cutoff(self):
        ds = dataset(
            cross_sections={"xs_exc": xs("xs_exc", [3.0] * 4)},
            losses={"loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, 95.0))},
        )
        # E = 100: Ec = 10, loss cutoff 90 < 95 -> catastrophic, E_out = 5 in group 3 (0, 10]
        total, matrix = _catastrophic_excitation(
            ds, OPEN, "RADIANT", np.array([100.0, 1000.0]), 2
        )
        np.testing.assert_allclose(total, [3.0, 0.0])
        np.testing.assert_allclose(matrix[0, :, 3], 3.0)
        np.testing.assert_array_equal(matrix[1], 0.0)

    def test_absent_without_excitation_data(self):
        total, matrix = _catastrophic_excitation(
            dataset(), OPEN, "RADIANT", np.array([100.0]), 2
        )
        assert total[0] == 0.0 and not matrix.any()


class TestLandesmanMorel:
    def test_equations_62_to_66(self):
        # moments S[l] = s0 (1 - c l (l + 1)): the soft part is exactly c s0 l (l + 1)
        s0, c, order = 10.0, 0.01, 6
        ell = np.arange(order + 1)
        moments = s0 * (1.0 - c * ell * (ell + 1))
        total, diagonal, momentum = landesman_morel_correction(
            np.array([s0 + 3.0]), moments[None, :]
        )
        assert momentum[0] == pytest.approx(c * s0)
        np.testing.assert_allclose(diagonal[0], 0.0, atol=1e-12)  # all of it was soft
        assert total[0] == pytest.approx(3.0)  # only the other interactions remain

    def test_general_identities(self):
        moments = np.array(
            [[5.0, 4.0, 3.2, 2.9, 2.5, 0.0, 0.0]]
        )  # last non-zero is l = 4
        total, diagonal, momentum = landesman_morel_correction(np.array([6.0]), moments)
        last = 4
        t_g = (moments[0, 3] - moments[0, 4]) / (2 * last)
        assert momentum[0] == pytest.approx(t_g)
        assert total[0] == pytest.approx(6.0 - moments[0, 4] - t_g * last * (last + 1))
        assert diagonal[0, 0] == pytest.approx(
            total[0] - 1.0
        )  # S'[0] = total' - the rest
        assert diagonal[0, last] == pytest.approx(0.0, abs=1e-14)
        np.testing.assert_array_equal(diagonal[0, last + 1 :], 0.0)

    def test_isotropic_and_empty_groups_are_untouched(self):
        moments = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        total, diagonal, momentum = landesman_morel_correction(
            np.array([5.0, 0.0]), moments
        )
        np.testing.assert_array_equal(total, [5.0, 0.0])
        np.testing.assert_array_equal(diagonal, moments)
        np.testing.assert_array_equal(momentum, 0.0)

    def test_negative_momentum_transfer_is_set_to_zero(self):
        moments = np.array([[5.0, 3.0, 3.5]])  # S[1] < S[2]
        total, diagonal, momentum = landesman_morel_correction(np.array([5.0]), moments)
        assert momentum[0] == 0.0
        assert total[0] == pytest.approx(
            5.0 - 3.5
        )  # plain extended transport correction
        np.testing.assert_allclose(diagonal[0], moments[0] - 3.5)


class TestGenerate:
    @staticmethod
    def make():
        iso = (np.array([-1.0, 0.999999]), np.array([0.5, 0.5]))
        mu, pdf = iso
        angular = DistributionRecord(
            "ang_lge",
            np.repeat([10.0, 1e4], 2),
            np.tile(mu, 2),
            np.tile(pdf, 2),
        )
        ds = dataset(
            cross_sections={
                "xs_el": xs("xs_el", [2.0] * 4),
                "xs_lge": xs("xs_lge", [2.0] * 4),
                "xs_exc": xs("xs_exc", [1.0] * 4),
            },
            losses={"loss_exc": AverageEnergyLoss("loss_exc", GRID, np.full(4, 5.0))},
            distributions={"ang_lge": angular},
        )
        return ds

    def test_returns_a_valid_particle_mg(self):
        data = generate(
            self.make(), STRUCTURE, weighting.uniform, "RADIANT", legendre_order=3
        )
        assert isinstance(data, BFPData)
        assert set(data.cross_sections) == {"xs_tot", MOMENTUM_TRANSFER_LABEL}
        assert set(data.scattering) == {
            "elastic"
        }  # no ionization or bremsstrahlung data
        ParticleMG(
            Particle.ELECTRON,
            TransportModel.BFP,
            STRUCTURE,
            cross_sections=data.cross_sections,
            scattering=data.scattering,
            coefficients=data.coefficients,
            cutoff="RADIANT",
            transport_correction=TransportCorrection.LANDESMAN_MOREL,
        )

    def test_coefficients_and_shapes(self):
        data = generate(self.make(), STRUCTURE, "uniform", legendre_order=3)
        # excitation with a 5 eV loss is soft everywhere: S = 1 * 5, straggling = 1 * 25
        np.testing.assert_allclose(data.coefficients.stopping_power, 5.0)
        np.testing.assert_allclose(data.coefficients.boundary_stopping_power, 5.0)
        np.testing.assert_allclose(data.coefficients.straggling, 25.0)
        np.testing.assert_allclose(data.coefficients.boundary_straggling, 25.0)
        assert data.coefficients.boundary_stopping_power.shape == (
            STRUCTURE.n_groups + 1,
        )
        assert data.scattering["elastic"].matrix.shape == (4, 3, 3)

    def test_isotropic_elastic_needs_no_correction(self):
        data = generate(self.make(), STRUCTURE, "uniform", legendre_order=3)
        np.testing.assert_allclose(
            data.cross_sections["xs_tot"].cross_section, 2.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            data.cross_sections[MOMENTUM_TRANSFER_LABEL].cross_section, 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            data.scattering["elastic"].matrix[0].diagonal(), 2.0, rtol=1e-5
        )

    def test_elastic_is_diagonal_in_energy(self):
        data = generate(self.make(), STRUCTURE, "uniform", legendre_order=3)
        matrix = data.scattering["elastic"].matrix
        assert (
            np.count_nonzero(
                matrix - np.einsum("lgg->lg", matrix)[:, :, None] * np.eye(3)
            )
            == 0
        )

    def test_input_validation(self):
        with pytest.raises(ValidationError, match="legendre_order"):
            generate(self.make(), STRUCTURE, "uniform", legendre_order=0)
        with pytest.raises(ValidationError, match="cutoff"):
            generate(self.make(), STRUCTURE, "uniform", cutoff="nope")
        with pytest.raises(ValidationError, match="Unknown weighting"):
            generate(self.make(), STRUCTURE, "1/S")

    def test_weighting_object_and_name_agree(self):
        a = generate(self.make(), STRUCTURE, "inverse_energy", legendre_order=2)
        b = generate(self.make(), STRUCTURE, weighting.inverse_energy, legendre_order=2)
        np.testing.assert_array_equal(
            a.cross_sections["xs_tot"].cross_section,
            b.cross_sections["xs_tot"].cross_section,
        )


@pytest.mark.skipif(
    not FE_FILE.is_file(), reason="EEDL iron file not downloaded under examples/data"
)
class TestGenerateIron:
    @pytest.fixture(scope="class")
    def result(self):
        from pyepics import EEDLReader

        structure = GroupStructure("ten", np.geomspace(1e7, 1e3, 11))
        data = generate(
            EEDLReader().read(FE_FILE),
            structure,
            weighting.uniform,
            "RADIANT",
            legendre_order=15,
        )
        return structure, data

    def test_group_stopping_power_lies_between_its_bound_values(self, result):
        _, data = result
        c = data.coefficients
        low = np.minimum(c.boundary_stopping_power[:-1], c.boundary_stopping_power[1:])
        high = np.maximum(c.boundary_stopping_power[:-1], c.boundary_stopping_power[1:])
        assert np.all(
            (c.stopping_power >= 0.98 * low) & (c.stopping_power <= 1.02 * high)
        )

    def test_matrices_are_down_scatter_only_and_non_negative(self, result):
        _, data = result
        for label in ("elastic", "ionization", "bremsstrahlung"):
            matrix = data.scattering[label]
            assert matrix.is_downscatter_only()
            assert np.all(matrix.matrix[0] >= 0.0)

    def test_momentum_transfer_grows_toward_low_energy(self, result):
        _, data = result
        t = data.cross_sections[MOMENTUM_TRANSFER_LABEL].cross_section
        assert np.all(t > 0.0) and np.all(np.diff(t) > 0.0)

    def test_corrected_total_is_below_the_raw_elastic_total(self, result):
        structure, data = result
        from pyepics import EEDLReader

        sigma_el = EEDLReader().read(FE_FILE).cross_sections["xs_el"]
        raw = np.interp(
            0.5 * (structure.upper + structure.lower),
            sigma_el.energy,
            sigma_el.cross_section,
        )
        assert np.all(data.cross_sections["xs_tot"].cross_section < raw)
