"""Tests for pyepics.multigroup.weighting."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyepics.exceptions import ValidationError
from pyepics.multigroup import weighting
from pyepics.multigroup.weighting import WeightingSpectrum


class TestPredefined:
    def test_uniform(self):
        np.testing.assert_array_equal(weighting.uniform([1.0, 5.0, 1e6]), [1.0] * 3)
        assert weighting.uniform(0.0) == 1.0  # allowed at zero

    def test_inverse_energy(self):
        np.testing.assert_allclose(weighting.inverse_energy([1.0, 4.0]), [1.0, 0.25])

    @pytest.mark.parametrize("e", [0.0, -1.0, [1.0, 0.0]])
    def test_inverse_energy_rejects_nonpositive(self, e):
        with pytest.raises(ValidationError, match="positive"):
            weighting.inverse_energy(e)

    def test_names(self):
        assert weighting.uniform.name == "uniform"
        assert weighting.inverse_energy.name == "inverse_energy"
        assert set(weighting.PREDEFINED) == {"uniform", "inverse_energy"}

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            weighting.uniform.name = "x"

    def test_scalar_in_scalar_out(self):
        assert weighting.inverse_energy(2.0).shape == ()


class TestCustomFunction:
    def test_basic(self):
        s = WeightingSpectrum("sqrt", np.sqrt)
        np.testing.assert_allclose(s([1.0, 4.0]), [1.0, 2.0])

    @pytest.mark.parametrize("name", ["", "  ", None, 3])
    def test_bad_name(self, name):
        with pytest.raises(ValidationError, match="name"):
            WeightingSpectrum(name, np.sqrt)

    def test_not_callable(self):
        with pytest.raises(ValidationError, match="callable"):
            WeightingSpectrum("x", 3.0)

    @pytest.mark.parametrize(
        "func, match",
        [
            (lambda e: -np.ones_like(e), "non-negative"),
            (lambda e: np.full_like(e, np.nan), "finite"),
            (lambda e: np.full_like(e, np.inf), "finite"),
            (lambda e: np.ones(3), "shape"),
            (lambda e: 1.0, "shape"),
        ],
    )
    def test_bad_output(self, func, match):
        with pytest.raises(ValidationError, match=match):
            WeightingSpectrum("bad", func)(np.array([1.0, 2.0]))

    @pytest.mark.parametrize("e", [[np.nan], [np.inf], [-1.0]])
    def test_bad_energies(self, e):
        with pytest.raises(ValidationError):
            weighting.uniform(e)


class TestFromTable:
    E = np.array([1.0, 10.0, 100.0])

    def test_loglog_reproduces_power_law(self):
        s = WeightingSpectrum.from_table("pl", self.E, 1.0 / self.E)
        x = np.array([2.0, 30.0, 99.0])
        np.testing.assert_allclose(s(x), 1.0 / x)

    def test_loglog_exact_at_nodes(self):
        w = np.array([3.0, 1.0, 7.0])
        s = WeightingSpectrum.from_table("t", self.E, w)
        np.testing.assert_allclose(s(self.E), w)

    def test_linear(self):
        s = WeightingSpectrum.from_table(
            "lin", self.E, [0.0, 10.0, 100.0], interpolation="linear"
        )
        np.testing.assert_allclose(s([5.5, 55.0]), [5.0, 55.0])

    def test_linear_allows_zero_and_loglog_does_not(self):
        WeightingSpectrum.from_table("z", self.E, [0.0, 1.0, 1.0], "linear")
        with pytest.raises(ValidationError, match="positive"):
            WeightingSpectrum.from_table("z", self.E, [0.0, 1.0, 1.0], "loglog")

    @pytest.mark.parametrize("x", [0.5, 100.5])
    def test_no_extrapolation(self, x):
        s = WeightingSpectrum.from_table("t", self.E, [1.0, 1.0, 1.0])
        with pytest.raises(ValidationError, match="outside the table range"):
            s(x)

    def test_table_is_copied(self):
        w = np.array([1.0, 1.0, 1.0])
        s = WeightingSpectrum.from_table("t", self.E, w)
        w[:] = 5.0
        np.testing.assert_allclose(s(10.0), 1.0)

    @pytest.mark.parametrize(
        "energy, weight",
        [
            ([1.0], [1.0]),  # too short
            ([1.0, 2.0], [1.0]),  # length mismatch
            ([2.0, 1.0], [1.0, 1.0]),  # descending
            ([1.0, 1.0], [1.0, 1.0]),  # duplicate
            ([0.0, 1.0], [1.0, 1.0]),  # non-positive energy
            ([1.0, 2.0], [1.0, -1.0]),  # negative weight
            ([1.0, 2.0], [1.0, np.nan]),
            ([[1.0, 2.0]], [[1.0, 1.0]]),  # not 1-D
            (["a", "b"], [1.0, 1.0]),
        ],
    )
    def test_bad_tables(self, energy, weight):
        with pytest.raises(ValidationError):
            WeightingSpectrum.from_table("bad", energy, weight)

    def test_bad_interpolation(self):
        with pytest.raises(ValidationError, match="interpolation"):
            WeightingSpectrum.from_table("t", self.E, [1.0] * 3, "cubic")


class TestResolve:
    def test_passthrough(self):
        assert weighting.resolve(weighting.uniform) is weighting.uniform

    def test_by_name(self):
        assert weighting.resolve("inverse_energy") is weighting.inverse_energy

    def test_unknown_name(self):
        with pytest.raises(ValidationError, match="uniform"):
            weighting.resolve("1/S")

    def test_plain_callable_wrapped_with_its_name(self):
        def my_flux(e):
            return e**0.0

        s = weighting.resolve(my_flux)
        assert s.name == "my_flux"
        np.testing.assert_array_equal(s([1.0, 2.0]), [1.0, 1.0])

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="WeightingSpectrum"):
            weighting.resolve(3)


def test_constant_weight_gives_exact_average_of_constant():
    """Weighting-independence: <c> = c for any spectrum under one quadrature rule."""
    x, w = np.polynomial.legendre.leggauss(8)
    lo, hi = 10.0, 100.0
    e = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    for s in (weighting.uniform, weighting.inverse_energy):
        wt = s(e) * w
        assert np.sum(3.0 * wt) / np.sum(wt) == pytest.approx(3.0)
