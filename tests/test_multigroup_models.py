"""Tests for pyepics.multigroup.models (GroupStructure)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyepics.exceptions import ValidationError
from pyepics.multigroup.models import GroupStructure
from pyepics.utils.validation import validate_strictly_descending


class TestGroupStructureValid:
    def test_basic(self):
        gs = GroupStructure("g3", np.array([100.0, 10.0, 1.0, 0.0]))
        assert gs.n_groups == 3
        np.testing.assert_array_equal(gs.upper, [100.0, 10.0, 1.0])
        np.testing.assert_array_equal(gs.lower, [10.0, 1.0, 0.0])
        np.testing.assert_array_equal(gs.width, [90.0, 9.0, 1.0])

    def test_list_input_coerced_to_float(self):
        gs = GroupStructure("ints", [4, 2, 1])
        assert gs.energy.dtype == np.float64

    def test_input_is_copied_and_read_only(self):
        src = np.array([10.0, 5.0, 1.0])
        gs = GroupStructure("g", src)
        src[0] = -1.0
        assert gs.energy[0] == 10.0
        with pytest.raises(ValueError):
            gs.energy[0] = 99.0

    def test_frozen(self):
        gs = GroupStructure("g", [2.0, 1.0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            gs.label = "other"

    @pytest.mark.parametrize(
        "energy, expected",
        [(100.0, 0), (50.0, 0), (10.0, 1), (5.0, 1), (1.0, 2), (0.5, 2), (0.0, 2)],
    )
    def test_group_of(self, energy, expected):
        gs = GroupStructure("g", [100.0, 10.0, 1.0, 0.0])
        assert gs.group_of(energy) == expected

    @pytest.mark.parametrize("energy", [-1.0, 100.1])
    def test_group_of_out_of_range(self, energy):
        gs = GroupStructure("g", [100.0, 10.0, 1.0, 0.0])
        with pytest.raises(ValidationError):
            gs.group_of(energy)


class TestGroupStructureInvalid:
    @pytest.mark.parametrize(
        "energy",
        [
            [1.0, 2.0, 3.0],  # ascending
            [3.0, 3.0, 1.0],  # duplicate
            [3.0, 1.0, 2.0],  # non-monotonic
            [3.0, 1.0, -1.0],  # negative
            [3.0, np.nan, 1.0],
            [np.inf, 2.0, 1.0],
            [1.0],  # too short
            [],
            [[3.0, 2.0], [1.0, 0.5]],  # not 1-D
            ["a", "b"],  # not numeric
        ],
    )
    def test_rejected(self, energy):
        with pytest.raises(ValidationError):
            GroupStructure("bad", energy)

    def test_ascending_message_points_to_from_ascending(self):
        with pytest.raises(ValidationError, match="ascending"):
            GroupStructure("bad", [1.0, 2.0, 3.0])


class TestValidateStrictlyDescending:
    def test_ok(self):
        validate_strictly_descending(np.array([3.0, 2.0, 1.0]))

    def test_short_arrays_pass(self):
        validate_strictly_descending(np.array([1.0]))
        validate_strictly_descending(np.array([]))

    def test_violation_index_reported(self):
        with pytest.raises(ValidationError, match="index 1"):
            validate_strictly_descending(np.array([3.0, 2.0, 2.0, 1.0]))


from pyepics.multigroup.models import (  # noqa: E402
    MOMENTUM_TRANSFER_LABEL,
    BFPCoefficients,
    GroupXS,
    MGElement,
    MGLibrary,
    Particle,
    ParticleMG,
    ScatteringMatrix,
    TransportCorrection,
    TransportModel,
)


class TestGroupXS:
    def test_basic(self):
        xs = GroupXS("xs_tot", [1.0, 2.0, 0.0])
        assert xs.n_groups == 3
        assert xs.cross_section.dtype == np.float64

    def test_input_copied(self):
        src = np.array([1.0, 2.0])
        xs = GroupXS("x", src)
        src[0] = 9.0
        assert xs.cross_section[0] == 1.0

    @pytest.mark.parametrize(
        "values",
        [[1.0, -1.0], [1.0, np.nan], [np.inf], [], [[1.0, 2.0]], ["a"]],
    )
    def test_rejected(self, values):
        with pytest.raises(ValidationError):
            GroupXS("bad", values)


class TestScatteringMatrix:
    def test_basic(self):
        m = np.zeros((2, 3, 3))
        m[0] = [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]]
        sm = ScatteringMatrix("elastic", m)
        assert sm.n_groups == 3
        assert sm.order == 1

    def test_axis_convention_from_to(self):
        # one entry: from group 0 to group 2
        m = np.zeros((1, 3, 3))
        m[0, 0, 2] = 7.0
        sm = ScatteringMatrix("x", m)
        assert sm.is_downscatter_only()

    def test_upscatter_detected(self):
        m = np.zeros((1, 3, 3))
        m[0, 2, 0] = 1.0
        assert not ScatteringMatrix("x", m).is_downscatter_only()

    def test_upscatter_in_higher_moment_detected(self):
        m = np.zeros((2, 3, 3))
        m[1, 2, 0] = -1.0
        assert not ScatteringMatrix("x", m).is_downscatter_only()

    def test_rtol_ignores_tiny_upscatter(self):
        m = np.zeros((1, 2, 2))
        m[0, 0, 0] = 1.0
        m[0, 1, 0] = 1e-12
        sm = ScatteringMatrix("x", m)
        assert not sm.is_downscatter_only()
        assert sm.is_downscatter_only(rtol=1e-9)

    def test_negative_higher_moment_allowed(self):
        m = np.zeros((2, 2, 2))
        m[1, 0, 1] = -0.5
        ScatteringMatrix("x", m)

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((3, 3)),  # 2-D
            np.zeros((1, 3, 2)),  # not square
            np.zeros((0, 3, 3)),  # no moments
            np.full((1, 2, 2), np.nan),
        ],
    )
    def test_rejected_shape_or_values(self, bad):
        with pytest.raises(ValidationError):
            ScatteringMatrix("bad", bad)

    def test_negative_l0_rejected(self):
        m = np.zeros((2, 2, 2))
        m[0, 0, 1] = -1.0
        with pytest.raises(ValidationError):
            ScatteringMatrix("bad", m)


class TestBFPCoefficients:
    @staticmethod
    def make(**overrides):
        kwargs = dict(
            stopping_power=[1.0, 2.0, 3.0],
            boundary_stopping_power=[0.5, 1.5, 2.5, 3.5],
            straggling=[0.1, 0.2, 0.0],
            boundary_straggling=[0.05, 0.15, 0.1, 0.0],
        )
        kwargs.update(overrides)
        return BFPCoefficients(**kwargs)

    def test_basic(self):
        bfp = self.make()
        assert bfp.n_groups == 3
        assert bfp.has_straggling
        assert bfp.stopping_power.dtype == np.float64
        assert bfp.boundary_stopping_power.shape == (4,)
        assert bfp.boundary_straggling.shape == (4,)

    def test_without_straggling(self):
        bfp = self.make(straggling=None, boundary_straggling=None)
        assert not bfp.has_straggling
        assert bfp.straggling is None
        assert bfp.boundary_straggling is None

    def test_input_copied(self):
        src = np.array([1.0, 2.0])
        bfp = self.make(
            stopping_power=src,
            boundary_stopping_power=[1.0, 1.5, 2.0],
            straggling=None,
            boundary_straggling=None,
        )
        src[0] = 9.0
        assert bfp.stopping_power[0] == 1.0

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(stopping_power=[1.0, -1.0, 3.0]),
            dict(boundary_stopping_power=[0.5, -1.5, 2.5, 3.5]),
            dict(straggling=[0.1, -1e-3, 0.0]),
            dict(boundary_straggling=[0.05, 0.15, -0.1, 0.0]),
            dict(stopping_power=[1.0, np.nan, 3.0]),
            dict(boundary_stopping_power=[0.5, np.inf, 2.5, 3.5]),
            dict(stopping_power=[]),
            dict(stopping_power=[[1.0, 2.0, 3.0]]),
        ],
    )
    def test_rejected_values(self, overrides):
        with pytest.raises(ValidationError):
            self.make(**overrides)

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(boundary_stopping_power=[0.5, 1.5, 2.5]),  # G, not G+1
            dict(boundary_stopping_power=[0.5, 1.5, 2.5, 3.5, 4.5]),
            dict(straggling=[0.1, 0.2]),
            dict(boundary_straggling=[0.05, 0.15, 0.1]),
        ],
    )
    def test_rejected_lengths(self, overrides):
        with pytest.raises(ValidationError):
            self.make(**overrides)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(boundary_straggling=None), "without boundary_straggling"),
            (dict(straggling=None), "without straggling"),
        ],
    )
    def test_straggling_all_or_nothing(self, overrides, match):
        with pytest.raises(ValidationError, match=match):
            self.make(**overrides)


G = 3
STRUCTURE = GroupStructure("g3", [100.0, 10.0, 1.0, 0.1])


def _coeffs(straggling=False):
    return BFPCoefficients(
        [1.0] * G,
        [1.0] * (G + 1),
        straggling=[0.1] * G if straggling else None,
        boundary_straggling=[0.1] * (G + 1) if straggling else None,
    )


def _xs(label="xs_tot", n=G):
    return GroupXS(label, [1.0] * n)


def _sm(label="catastrophic", n=G):
    return ScatteringMatrix(label, np.zeros((1, n, n)))


# Minimal valid kwargs per model
VALID = {
    "bfp": dict(
        cross_sections={MOMENTUM_TRANSFER_LABEL: _xs(MOMENTUM_TRANSFER_LABEL)},
        scattering={"catastrophic": _sm()},
        coefficients=_coeffs(),
        cutoff="CEPXS",
        transport_correction="landesman_morel",
    ),
    "bcsd": dict(
        scattering={"catastrophic": _sm()},
        coefficients=_coeffs(),
        cutoff="CEPXS",
    ),
    "fp": dict(
        cross_sections={MOMENTUM_TRANSFER_LABEL: _xs(MOMENTUM_TRANSFER_LABEL)},
        coefficients=_coeffs(straggling=True),
    ),
    "boltzmann": dict(scattering={"elastic": _sm("elastic")}),
}


def make_pmg(model="bfp", particle="electron", **overrides):
    kwargs = dict(VALID[model])
    kwargs.update(overrides)
    return ParticleMG(particle, model, STRUCTURE, **kwargs)


class TestParticleMG:
    @pytest.mark.parametrize("model", list(VALID))
    def test_valid_models(self, model):
        pmg = make_pmg(model)
        assert pmg.transport_model is TransportModel(model)
        assert pmg.n_groups == G

    def test_bfp_with_straggling(self):
        make_pmg("bfp", coefficients=_coeffs(straggling=True))

    def test_enums_coerced_and_reported(self):
        pmg = make_pmg("boltzmann", particle="photon")
        assert pmg.particle == "photon"
        with pytest.raises(ValidationError, match="transport_model"):
            ParticleMG("electron", "csda", STRUCTURE)
        with pytest.raises(ValidationError, match="particle"):
            ParticleMG("positron", "boltzmann", STRUCTURE)

    @pytest.mark.parametrize("model", ["bfp", "bcsd", "fp"])
    def test_photon_requires_boltzmann(self, model):
        with pytest.raises(ValidationError, match="photons"):
            make_pmg(model, particle="photon")

    def test_dicts_are_copied(self):
        xs = {"a": _xs("a")}
        pmg = make_pmg("boltzmann", cross_sections=xs)
        xs["b"] = _xs("b")
        assert list(pmg.cross_sections) == ["a"]

    # --- consistency with the group structure -------------------------
    def test_group_count_mismatch(self):
        with pytest.raises(ValidationError, match="groups"):
            make_pmg("boltzmann", cross_sections={"xs_tot": _xs(n=G + 1)})
        with pytest.raises(ValidationError, match="groups"):
            make_pmg("boltzmann", scattering={"elastic": _sm("elastic", n=2)})
        bad = BFPCoefficients([1.0] * 2, [1.0] * 3)
        with pytest.raises(ValidationError, match="groups"):
            make_pmg("bcsd", coefficients=bad)

    def test_key_must_match_label(self):
        with pytest.raises(ValidationError, match="does not match label"):
            make_pmg("boltzmann", cross_sections={"other": _xs("xs_tot")})

    def test_wrong_leaf_type(self):
        with pytest.raises(ValidationError, match="GroupXS"):
            make_pmg("boltzmann", cross_sections={"x": _sm("x")})

    def test_group_structure_type(self):
        with pytest.raises(ValidationError, match="GroupStructure"):
            ParticleMG("electron", "boltzmann", [100.0, 1.0])

    @pytest.mark.parametrize("cutoff", ["", "  ", 3])
    def test_bad_cutoff(self, cutoff):
        with pytest.raises(ValidationError, match="cutoff"):
            make_pmg("bfp", cutoff=cutoff)

    # --- required / forbidden fields per model ------------------------
    @pytest.mark.parametrize(
        "model, overrides, match",
        [
            ("bfp", dict(coefficients=None), "requires coefficients"),
            ("bfp", dict(cutoff=None), "requires cutoff"),
            ("bfp", dict(scattering={}), "requires scattering"),
            ("bfp", dict(cross_sections={}), "requires the 'momentum_transfer'"),
            ("bcsd", dict(coefficients=None), "requires coefficients"),
            ("bcsd", dict(cutoff=None), "requires cutoff"),
            ("bcsd", dict(scattering={}), "requires scattering"),
            ("bcsd", dict(coefficients=_coeffs(straggling=True)), "straggling"),
            (
                "bcsd",
                dict(
                    cross_sections={
                        MOMENTUM_TRANSFER_LABEL: _xs(MOMENTUM_TRANSFER_LABEL)
                    }
                ),
                "does not allow the 'momentum_transfer'",
            ),
            ("fp", dict(coefficients=None), "requires coefficients"),
            ("fp", dict(coefficients=_coeffs()), "requires straggling"),
            ("fp", dict(cross_sections={}), "requires the 'momentum_transfer'"),
            ("fp", dict(cutoff="CEPXS"), "does not allow cutoff"),
            ("fp", dict(scattering={"x": _sm("x")}), "does not allow scattering"),
            ("boltzmann", dict(scattering={}), "requires scattering"),
            ("boltzmann", dict(coefficients=_coeffs()), "does not allow coefficients"),
            ("boltzmann", dict(cutoff="CEPXS"), "does not allow cutoff"),
            (
                "boltzmann",
                dict(
                    cross_sections={
                        MOMENTUM_TRANSFER_LABEL: _xs(MOMENTUM_TRANSFER_LABEL)
                    }
                ),
                "does not allow the 'momentum_transfer'",
            ),
        ],
    )
    def test_model_field_rules(self, model, overrides, match):
        with pytest.raises(ValidationError, match=match):
            make_pmg(model, **overrides)

    # --- transport correction -----------------------------------------
    def test_correction_defaults_to_none(self):
        assert make_pmg("boltzmann").transport_correction is TransportCorrection.NONE

    @pytest.mark.parametrize(
        "model, correction",
        [
            ("bfp", "landesman_morel"),
            ("bcsd", "none"),
            ("bcsd", "extended"),
            ("fp", "none"),
            ("boltzmann", "none"),
            ("boltzmann", "extended"),
        ],
    )
    def test_allowed_corrections(self, model, correction):
        pmg = make_pmg(model, transport_correction=correction)
        assert pmg.transport_correction == correction

    @pytest.mark.parametrize(
        "model, correction",
        [
            ("bfp", "none"),
            ("bfp", "extended"),
            ("bcsd", "landesman_morel"),
            ("fp", "extended"),
            ("fp", "landesman_morel"),
            ("boltzmann", "landesman_morel"),
        ],
    )
    def test_disallowed_corrections(self, model, correction):
        with pytest.raises(ValidationError, match="transport_correction"):
            make_pmg(model, transport_correction=correction)

    def test_photon_allows_no_correction(self):
        with pytest.raises(ValidationError, match="transport_correction"):
            make_pmg("boltzmann", particle="photon", transport_correction="extended")

    def test_unknown_correction(self):
        with pytest.raises(ValidationError, match="transport_correction"):
            make_pmg("boltzmann", transport_correction="transport")


import dataclasses  # noqa: E402,F811


def make_element(z=26, symbol="Fe", electron="bfp", photon="boltzmann", **kw):
    return MGElement(
        z,
        symbol,
        kw.pop("awr", 55.34),
        electron=make_pmg(electron) if electron else None,
        photon=make_pmg(photon, particle="photon") if photon else None,
        **kw,
    )


class TestMGElement:
    def test_basic(self):
        el = make_element()
        assert (el.Z, el.symbol) == (26, "Fe")
        assert list(el.particles) == [Particle.ELECTRON, Particle.PHOTON]

    def test_single_particle(self):
        assert list(make_element(photon=None).particles) == [Particle.ELECTRON]
        assert list(make_element(electron=None).particles) == [Particle.PHOTON]

    def test_numpy_int_and_awr_coerced(self):
        el = make_element(z=np.int64(26), awr=np.float32(55.0))
        assert type(el.Z) is int and type(el.atomic_weight_ratio) is float

    @pytest.mark.parametrize("z", [0, 119, -1, 26.0, "26", True, None])
    def test_bad_z(self, z):
        with pytest.raises(ValidationError):
            MGElement(z, "Fe", 55.0, electron=make_pmg("bfp"))

    def test_symbol_must_match_z(self):
        with pytest.raises(ValidationError, match="does not match"):
            MGElement(26, "Cu", 55.0, electron=make_pmg("bfp"))

    @pytest.mark.parametrize("awr", [0.0, -1.0, np.nan, np.inf, "x", None])
    def test_bad_awr(self, awr):
        with pytest.raises(ValidationError, match="atomic_weight_ratio"):
            MGElement(26, "Fe", awr, electron=make_pmg("bfp"))

    def test_needs_some_data(self):
        with pytest.raises(ValidationError, match="electron and/or photon"):
            MGElement(26, "Fe", 55.0)

    def test_slot_particle_must_match(self):
        with pytest.raises(ValidationError, match="holds photon"):
            MGElement(26, "Fe", 55.0, electron=make_pmg("boltzmann", particle="photon"))
        with pytest.raises(ValidationError, match="holds electron"):
            MGElement(26, "Fe", 55.0, photon=make_pmg("bfp"))

    def test_slot_type(self):
        with pytest.raises(ValidationError, match="ParticleMG"):
            MGElement(26, "Fe", 55.0, electron="data")


class TestMGLibrary:
    def test_basic(self):
        lib = MGLibrary("test", {26: make_element(), 1: make_element(1, "H")})
        assert len(lib) == 2 and 26 in lib and 3 not in lib
        assert lib[1].symbol == "H"
        assert lib.particles == [Particle.ELECTRON, Particle.PHOTON]
        assert lib.group_structure("electron") is STRUCTURE
        assert lib.group_structure(Particle.PHOTON).n_groups == G

    def test_empty_library(self):
        lib = MGLibrary("empty")
        assert len(lib) == 0 and lib.particles == []
        with pytest.raises(ValidationError, match="no element"):
            lib.group_structure("electron")

    def test_dict_copied(self):
        elements = {26: make_element()}
        lib = MGLibrary("t", elements)
        elements[1] = make_element(1, "H")
        assert len(lib) == 1

    @pytest.mark.parametrize("label", ["", "  ", None])
    def test_bad_label(self, label):
        with pytest.raises(ValidationError, match="label"):
            MGLibrary(label)

    def test_key_must_match_z(self):
        with pytest.raises(ValidationError, match="does not match Z"):
            MGLibrary("t", {1: make_element()})

    def test_value_type(self):
        with pytest.raises(ValidationError, match="MGElement"):
            MGLibrary("t", {26: "Fe"})

    def test_photon_only_elements_mix_with_full_ones(self):
        MGLibrary("t", {26: make_element(), 1: make_element(1, "H", electron=None)})

    def _pair(self, **electron_changes):
        a = make_element()
        b = make_element(1, "H")
        b.electron = dataclasses.replace(b.electron, **electron_changes)
        return {26: a, 1: b}

    def test_group_structure_must_agree(self):
        alt = GroupStructure("alt", [200.0, 10.0, 1.0, 0.1])
        with pytest.raises(ValidationError, match="group_structure"):
            MGLibrary("t", self._pair(group_structure=alt))

    def test_group_structure_compared_by_value_not_label(self):
        same = GroupStructure("renamed", STRUCTURE.energy)
        MGLibrary("t", self._pair(group_structure=same))

    def test_cutoff_must_agree(self):
        with pytest.raises(ValidationError, match="cutoff"):
            MGLibrary("t", self._pair(cutoff="RADIANT"))

    def test_model_must_agree(self):
        a = make_element(electron="bfp")
        b = make_element(1, "H", electron="bcsd")
        with pytest.raises(ValidationError, match="transport_model"):
            MGLibrary("t", {26: a, 1: b})

    def test_correction_must_agree(self):
        a = make_element(electron="bcsd")
        b = make_element(1, "H", electron="bcsd")
        b.electron = dataclasses.replace(b.electron, transport_correction="extended")
        with pytest.raises(ValidationError, match="transport_correction"):
            MGLibrary("t", {26: a, 1: b})

    def test_electron_and_photon_groups_may_differ(self):
        el = make_element()
        el.photon = dataclasses.replace(
            el.photon, group_structure=GroupStructure("ph", [50.0, 5.0, 1.0, 0.1])
        )
        lib = MGLibrary("t", {26: el})
        assert lib.group_structure("photon") is not lib.group_structure("electron")
