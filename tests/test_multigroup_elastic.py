"""Tests for pyepics.multigroup.elastic."""

from __future__ import annotations

import numpy as np
import pytest

from pyepics.exceptions import ValidationError
from pyepics.models.records import CrossSectionRecord, DistributionRecord, EEDLDataset
from pyepics.multigroup.elastic import (
    elastic_legendre_cross_sections,
    large_angle_moment_anchors,
)
from pyepics.physics.elastic import legendre_moments_screened_rutherford
from pyepics.utils.parsing import small_angle_eta

GRID = np.array([1e3, 1e4, 1e5, 1e6])


def table(energies, mu_pdf):
    """Large-angle cosine tables: the same (mu, pdf) at every energy"""
    mu, pdf = mu_pdf
    n = len(mu)
    return DistributionRecord(
        "ang_lge",
        np.repeat(energies, n).astype(float),
        np.tile(mu, len(energies)),
        np.tile(pdf, len(energies)),
    )


def dataset(sigma_el, sigma_la=None, tables=None):
    xs = {"xs_el": CrossSectionRecord("xs_el", GRID, np.asarray(sigma_el, float))}
    if sigma_la is not None:
        xs["xs_lge"] = CrossSectionRecord("xs_lge", GRID, np.asarray(sigma_la, float))
    dists = {} if tables is None else {"ang_lge": tables}
    return EEDLDataset(26, "Fe", 55.0, 26000.0, cross_sections=xs, distributions=dists)


ISOTROPIC = (np.array([-1.0, 0.999999]), np.array([0.5, 0.5]))


class TestLargeAngleAnchors:
    def test_isotropic_tables_have_no_higher_moments(self):
        e, g = large_angle_moment_anchors(table([1e3, 1e6], ISOTROPIC), 5)
        assert g.shape == (e.size, 6)
        np.testing.assert_allclose(g[:, 1:], 0.0, atol=1e-6)

    def test_tables_are_densified_between_far_energies(self):
        e, _ = large_angle_moment_anchors(table([1e3, 1e6], ISOTROPIC), 3)
        assert e.size > 2 and np.all(np.diff(e) > 0.0)
        assert np.max(e[1:] / e[:-1]) <= 1.1 + 1e-12


class TestElasticLegendre:
    def test_purely_large_angle_isotropic(self):
        ds = dataset([4.0] * 4, [4.0] * 4, table([1e3, 1e6], ISOTROPIC))
        sigma = elastic_legendre_cross_sections(ds, GRID, 4)
        np.testing.assert_allclose(sigma[:, 0], 4.0)
        np.testing.assert_allclose(sigma[:, 1:], 0.0, atol=1e-5)

    def test_purely_small_angle_is_screened_rutherford(self):
        ds = dataset([2.0] * 4)
        sigma = elastic_legendre_cross_sections(ds, GRID, 15)
        for row, e in zip(sigma, GRID, strict=True):
            eta = float(small_angle_eta(26, np.array([e]))[0])
            np.testing.assert_allclose(
                row, 2.0 * legendre_moments_screened_rutherford(eta, 15), rtol=1e-12
            )

    def test_mixture_weights_by_cross_section(self):
        ds = dataset([4.0] * 4, [1.0] * 4, table([1e3, 1e6], ISOTROPIC))
        sigma = elastic_legendre_cross_sections(ds, GRID, 3)
        eta = small_angle_eta(26, GRID)
        for row, e in zip(sigma, eta, strict=True):
            sa = legendre_moments_screened_rutherford(e, 3)
            np.testing.assert_allclose(row[1:], 3.0 * sa[1:], rtol=1e-5, atol=1e-5)

    def test_zero_elastic_gives_zero_row(self):
        ds = dataset([0.0, 3.0, 3.0, 3.0])
        sigma = elastic_legendre_cross_sections(ds, GRID, 3)
        np.testing.assert_array_equal(sigma[0], 0.0)
        assert sigma[1, 0] == 3.0

    def test_moments_are_non_increasing_in_magnitude_for_forward_peak(self):
        sigma = elastic_legendre_cross_sections(dataset([1.0] * 4), GRID, 15)
        assert np.all(np.diff(sigma, axis=1) <= 0.0)

    def test_eq60_zeros_moment_and_everything_above(self, monkeypatch):
        import pyepics.multigroup.elastic as mod

        def garbage(eta, order):
            g = np.ones(order + 1)
            g[3] = 5.0  # unphysical
            return g

        monkeypatch.setattr(mod, "legendre_moments_screened_rutherford", garbage)
        sigma = elastic_legendre_cross_sections(dataset([2.0] * 4), GRID, 6)
        np.testing.assert_allclose(sigma[:, :3], 2.0)
        np.testing.assert_array_equal(sigma[:, 3:], 0.0)

    def test_requires_xs_el(self):
        ds = dataset([1.0] * 4)
        del ds.cross_sections["xs_el"]
        with pytest.raises(ValidationError, match="xs_el"):
            elastic_legendre_cross_sections(ds, GRID, 3)

    def test_large_angle_needs_tables(self):
        with pytest.raises(ValidationError, match="ang_lge"):
            elastic_legendre_cross_sections(dataset([1.0] * 4, [1.0] * 4), GRID, 3)
