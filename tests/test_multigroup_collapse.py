"""Tests for pyepics.multigroup.collapse."""

from __future__ import annotations

import numpy as np
import pytest

from pyepics.exceptions import ValidationError
from pyepics.models.records import CrossSectionRecord
from pyepics.multigroup import weighting
from pyepics.multigroup.collapse import collapse, collapse_nodes, group_quadrature
from pyepics.multigroup.models import GroupStructure

STRUCTURE = GroupStructure("g3", [1000.0, 100.0, 10.0, 1.0])
MIDPOINT = 0.5 * (STRUCTURE.upper + STRUCTURE.lower)


class TestGroupQuadrature:
    def test_weights_sum_to_group_widths(self):
        q = group_quadrature(STRUCTURE, np.array([5.0, 50.0, 500.0, 7.0]))
        widths = np.bincount(q.group, weights=q.weight)
        np.testing.assert_allclose(widths, STRUCTURE.width)

    def test_nodes_lie_inside_their_group(self):
        q = group_quadrature(STRUCTURE, np.geomspace(1.0, 1000.0, 30))
        assert np.all(q.energy > STRUCTURE.lower[q.group])
        assert np.all(q.energy < STRUCTURE.upper[q.group])

    def test_breaks_outside_or_on_bounds_are_ignored(self):
        a = group_quadrature(STRUCTURE)
        b = group_quadrature(
            STRUCTURE, np.array([0.5, 1.0, 10.0, 100.0, 1000.0, 2000.0])
        )
        np.testing.assert_array_equal(a.energy, b.energy)

    def test_wide_segments_are_subdivided(self):
        q = group_quadrature(STRUCTURE, max_ratio=1.5)
        per_group = np.bincount(q.group)
        assert np.all(per_group >= 4 * np.ceil(np.log(10.0) / np.log(1.5)))

    def test_bad_ratio(self):
        with pytest.raises(ValidationError, match="max_ratio"):
            group_quadrature(STRUCTURE, max_ratio=1.0)

    def test_bad_points(self):
        with pytest.raises(ValidationError, match="n_points"):
            group_quadrature(STRUCTURE, n_points=0)


class TestCollapse:
    def test_constant_is_preserved_for_any_weight(self):
        xs = CrossSectionRecord("c", np.array([0.5, 2000.0]), np.array([3.0, 3.0]))
        for w in ("uniform", "inverse_energy"):
            np.testing.assert_allclose(collapse(xs, STRUCTURE, w).cross_section, 3.0)

    def test_linear_table_uniform_weight_is_midpoint(self):
        xs = CrossSectionRecord("lin", np.array([1.0, 1000.0]), np.array([1.0, 1000.0]))
        out = collapse(xs, STRUCTURE).cross_section
        np.testing.assert_allclose(out, MIDPOINT)

    def test_inverse_energy_weight_on_linear_table(self):
        # <E> with w = 1/E is (hi - lo) / ln(hi / lo)
        xs = CrossSectionRecord("lin", np.array([1.0, 1000.0]), np.array([1.0, 1000.0]))
        out = collapse(xs, STRUCTURE, weighting.inverse_energy).cross_section
        expected = STRUCTURE.width / np.log(STRUCTURE.upper / STRUCTURE.lower)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_table_nodes_split_the_groups_so_kinks_are_exact(self):
        # |E - 50| has a kink at 50 inside group 1; exact mean over (10, 100]
        xs = CrossSectionRecord(
            "kink",
            np.array([1.0, 50.0, 1000.0]),
            np.array([49.0, 0.0, 950.0]),
        )
        out = collapse(xs, STRUCTURE).cross_section[1]
        exact = (0.5 * 40.0 * 40.0 * (40.0 / 40.0) + 0.5 * 50.0 * 50.0) / 90.0
        assert out == pytest.approx(exact, rel=1e-12)

    def test_zero_outside_the_table_and_label_kept(self):
        xs = CrossSectionRecord("xs_K", np.array([100.0, 1000.0]), np.array([2.0, 2.0]))
        gxs = collapse(xs, STRUCTURE)
        assert gxs.label == "xs_K"
        assert gxs.cross_section[2] == 0.0  # group (1, 10], entirely below the table

    def test_callable_and_name_weights(self):
        xs = CrossSectionRecord("c", np.array([1.0, 1000.0]), np.array([1.0, 1.0]))
        collapse(xs, STRUCTURE, lambda e: e)  # plain callable
        collapse(xs, STRUCTURE, "uniform")

    def test_inverse_energy_needs_positive_lower_bound(self):
        zero_bottom = GroupStructure("z", [100.0, 10.0, 0.0])
        xs = CrossSectionRecord("c", np.array([0.0, 100.0]), np.array([1.0, 1.0]))
        collapse(xs, zero_bottom)  # uniform is fine
        with pytest.raises(ValidationError, match="singular"):
            collapse(xs, zero_bottom, weighting.inverse_energy)


class TestCollapseNodes:
    def test_trailing_axes_are_averaged_independently(self):
        q = group_quadrature(STRUCTURE)
        values = np.stack([np.ones(q.energy.size), q.energy], axis=1)
        out = collapse_nodes(values, q, weighting.uniform)
        assert out.shape == (3, 2)
        np.testing.assert_allclose(out[:, 0], 1.0)
        np.testing.assert_allclose(out[:, 1], MIDPOINT)

    def test_shape_mismatch(self):
        q = group_quadrature(STRUCTURE)
        with pytest.raises(ValidationError, match="nodes"):
            collapse_nodes(np.ones(3), q, weighting.uniform)

    def test_zero_weight_group(self):
        q = group_quadrature(STRUCTURE)
        only_top = weighting.WeightingSpectrum(
            "top", lambda e: (e > 100.0).astype(float)
        )
        with pytest.raises(ValidationError, match="zero"):
            collapse_nodes(np.ones(q.energy.size), q, only_top)
