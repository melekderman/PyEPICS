#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Tests for EEDL reader and parsing logic

Covers cross-section extraction, MF=26 distribution parsing, validation
pass/fail, and utility functions used by the EEDL reader.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyepics.exceptions import ParseError, ValidationError, FileFormatError
from pyepics.models.records import EEDLDataset, CrossSectionRecord
from pyepics.utils.parsing import (
    float_endf,
    int_endf,
    parse_mf26_mt525,
    build_pdf,
    linear_interpolation,
    small_angle_eta,
    small_angle_scattering_cosine,
    extract_atomic_number_from_path,
)
from pyepics.utils.validation import (
    validate_atomic_number,
    validate_energy_monotonic,
    validate_non_negative,
    validate_cross_section,
    validate_probability_sum,
)


# -----------------------------------------------------------------------
# float_endf / int_endf
# -----------------------------------------------------------------------

class TestFloatEndf:
    """Tests for ENDF float conversion"""

    def test_standard_notation(self) -> None:
        assert float_endf(" 1.23456E+03") == pytest.approx(1234.56)

    def test_d_notation(self) -> None:
        assert float_endf(" 1.23456D-03") == pytest.approx(0.00123456)

    def test_implicit_exponent(self) -> None:
        assert float_endf(" 1.23456+03") == pytest.approx(1234.56)

    def test_implicit_negative_exponent(self) -> None:
        assert float_endf(" 1.23456-03") == pytest.approx(0.00123456)

    def test_blank_field(self) -> None:
        assert float_endf("           ") == 0.0

    def test_empty_string(self) -> None:
        assert float_endf("") == 0.0

    def test_invalid_raises_parse_error(self) -> None:
        with pytest.raises(ParseError):
            float_endf("not_a_number")

    def test_zero(self) -> None:
        assert float_endf(" 0.00000E+00") == 0.0

    def test_negative_value(self) -> None:
        assert float_endf("-2.50000E+01") == pytest.approx(-25.0)


class TestIntEndf:
    """Tests for ENDF integer conversion"""

    def test_padded_integer(self) -> None:
        assert int_endf("         42") == 42

    def test_blank_field(self) -> None:
        assert int_endf("           ") == 0

    def test_zero(self) -> None:
        assert int_endf("          0") == 0

    def test_non_numeric(self) -> None:
        assert int_endf("  abc  ") == 0


# -----------------------------------------------------------------------
# parse_mf26_mt525
# -----------------------------------------------------------------------

class TestParseMf26Mt525:
    """Tests for the MF=26/MT=525 manual parser"""

    def _make_raw(self) -> str:
        """Create a minimal synthetic MF=26/MT=525 section"""
        # 8 header lines + 1 CONT + 1 data line
        header = "\n".join([f"{'':66s}{'9999':>4s}{'26':>2s}{'525':>3s}{'0':>5s}"] * 8)
        # CONT: E_loss=0.0, E_in=1e4, fields 3-4 blank, NW=4, NL=2
        cont = (
            " 0.00000+00"   # field 1: E_loss
            " 1.00000+04"   # field 2: E_in
            "           "   # field 3: blank
            "           "   # field 4: blank
            "          4"   # field 5: NW
            "          2"   # field 6: NL
            "9999 26 525    1"
        )
        # Data line: mu1, p1, mu2, p2
        data = (
            "-1.0000+00"    # mu1 = -1.0  (10 chars, pad to 11)
            " "
            " 5.0000-01"    # p1  = 0.5
            " "
            " 1.0000+00"    # mu2 = 1.0
            " "
            " 5.0000-01"    # p2 = 0.5
            " "
            "           "   # field 5 (pad)
            "           "   # field 6 (pad)
        )
        return header + "\n" + cont + "\n" + data

    def test_parses_one_group(self) -> None:
        raw = self._make_raw()
        groups = parse_mf26_mt525(raw)
        assert len(groups) == 1
        assert groups[0]["E_in"] == pytest.approx(1e4)
        assert groups[0]["NL"] == 2
        assert len(groups[0]["pairs"]) == 2

    def test_pair_values(self) -> None:
        raw = self._make_raw()
        groups = parse_mf26_mt525(raw)
        mu1, p1 = groups[0]["pairs"][0]
        mu2, p2 = groups[0]["pairs"][1]
        assert p1 == pytest.approx(0.5)
        assert p2 == pytest.approx(0.5)


# -----------------------------------------------------------------------
# build_pdf
# -----------------------------------------------------------------------

class TestBuildPdf:
    """Tests for the flat-to-grouped PDF builder"""

    def test_two_groups(self) -> None:
        inc = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        val = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pdf = np.array([0.5, 0.5, 0.3, 0.4, 0.3])

        eg, eo, v, p = build_pdf(inc, val, pdf)
        assert eg.shape == (2,)
        assert eg[0] == pytest.approx(1.0)
        assert eg[1] == pytest.approx(2.0)
        assert eo[0] == 0
        assert eo[1] == 2
        np.testing.assert_array_equal(v, val)
        np.testing.assert_array_equal(p, pdf)

    def test_empty_input(self) -> None:
        eg, eo, v, p = build_pdf(
            np.array([]), np.array([]), np.array([]),
        )
        assert eg.size == 0
        assert eo.shape == (0,)


# -----------------------------------------------------------------------
# linear_interpolation
# -----------------------------------------------------------------------

class TestLinearInterpolation:
    """Tests for the interpolation wrapper"""

    def test_exact_points(self) -> None:
        grid = np.array([1.0, 2.0, 3.0])
        ref_e = np.array([1.0, 2.0, 3.0])
        ref_v = np.array([10.0, 20.0, 30.0])
        result = linear_interpolation(grid, ref_e, ref_v)
        np.testing.assert_allclose(result, ref_v)

    def test_midpoint_interpolation(self) -> None:
        grid = np.array([1.5])
        ref_e = np.array([1.0, 2.0])
        ref_v = np.array([10.0, 20.0])
        result = linear_interpolation(grid, ref_e, ref_v)
        assert result[0] == pytest.approx(15.0)


# -----------------------------------------------------------------------
# small_angle_eta / small_angle_scattering_cosine
# -----------------------------------------------------------------------

class TestSmallAngle:
    """Tests for Rutherford screening parameter and cosine distributions"""

    def test_eta_positive(self) -> None:
        eta = small_angle_eta(26, np.array([1e6]))
        assert eta[0] > 0

    def test_eta_increases_with_Z(self) -> None:
        eta_fe = small_angle_eta(26, np.array([1e6]))
        eta_pb = small_angle_eta(82, np.array([1e6]))
        assert eta_pb[0] > eta_fe[0]

    def test_cosine_output_shape(self) -> None:
        eg, eo, val, pdf = small_angle_scattering_cosine(26, np.array([1e6]), n_mu=50)
        assert eg.shape == (1,)
        assert val.shape == (50,)
        assert pdf.shape == (50,)

    def test_cosine_empty_input(self) -> None:
        eg, eo, val, pdf = small_angle_scattering_cosine(26, np.array([]))
        assert eg.size == 0


# -----------------------------------------------------------------------
# extract_atomic_number_from_path
# -----------------------------------------------------------------------

class TestExtractAtomicNumber:
    """Tests for filename pattern matching"""

    def test_hydrogen(self, tmp_path) -> None:
        p = tmp_path / "EEDL.ZA001000.endf"
        p.touch()
        assert extract_atomic_number_from_path(p) == 1

    def test_iron(self, tmp_path) -> None:
        p = tmp_path / "EPDL.ZA026000.endf"
        p.touch()
        assert extract_atomic_number_from_path(p) == 26

    def test_invalid_name(self, tmp_path) -> None:
        p = tmp_path / "random_file.txt"
        p.touch()
        with pytest.raises(FileFormatError):
            extract_atomic_number_from_path(p)


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

class TestValidation:
    """Tests for post-parse validation routines"""

    def test_valid_atomic_number(self) -> None:
        validate_atomic_number(1)
        validate_atomic_number(118)

    def test_invalid_atomic_number_zero(self) -> None:
        with pytest.raises(ValidationError):
            validate_atomic_number(0)

    def test_invalid_atomic_number_high(self) -> None:
        with pytest.raises(ValidationError):
            validate_atomic_number(200)

    def test_monotonic_energy(self) -> None:
        validate_energy_monotonic(np.array([1.0, 2.0, 3.0]))

    def test_non_monotonic_energy(self) -> None:
        with pytest.raises(ValidationError):
            validate_energy_monotonic(np.array([3.0, 1.0, 2.0]))

    def test_non_negative_pass(self) -> None:
        validate_non_negative(np.array([0.0, 1.0, 2.0]))

    def test_non_negative_fail(self) -> None:
        with pytest.raises(ValidationError):
            validate_non_negative(np.array([1.0, -0.5, 2.0]))

    def test_cross_section_shape_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            validate_cross_section(
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )

    def test_probability_sum_pass(self) -> None:
        validate_probability_sum(np.array([0.3, 0.7]))

    def test_probability_sum_fail(self) -> None:
        with pytest.raises(ValidationError):
            validate_probability_sum(np.array([0.1, 0.1]))


# -----------------------------------------------------------------------
# EEDLReader (with synthetic data)
# -----------------------------------------------------------------------

class TestEEDLReader:
    """Tests for EEDL reader (file-not-found path)"""

    def test_file_not_found(self, tmp_path) -> None:
        from pyepics.readers.eedl import EEDLReader

        reader = EEDLReader()
        with pytest.raises(FileFormatError):
            reader.read(tmp_path / "nonexistent.endf")

    def test_bad_filename_pattern(self, tmp_path) -> None:
        from pyepics.readers.eedl import EEDLReader

        bad_file = tmp_path / "bad_name.endf"
        bad_file.write_text("dummy content")
        reader = EEDLReader()
        with pytest.raises(FileFormatError):
            reader.read(bad_file)
