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
