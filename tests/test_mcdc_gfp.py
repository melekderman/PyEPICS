#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Tests for the GFP elastic data written to MC/DC files (mcdc_gfp)

The April 2026 MC/DC Fe file is an optional local reference, not distributed
with the project (see tests/fixtures/README.md).  Mandatory synthetic tests
cover the output schema, model fitting, and screened Rutherford moments
against independent quadrature that resolves the forward peak.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyepics.converters.mcdc_gfp import (
    _insert_log_log_tables,
    _trapezoid,
    dk_fp_params,
    elastic_moments,
    gfp2_params,
    gfp3_params,
    gfp_parameters,
    legendre_moments_screened_rutherford,
    legendre_moments_tabulated,
    optimal_mu_star_gfp2,
)
from pyepics.converters.mcdc_hdf5 import write_mcdc_eedl
from pyepics.utils.parsing import build_pdf, linear_interpolation, small_angle_eta

FIXTURES = Path(__file__).parent / "fixtures"
APRIL_FE = FIXTURES / "april_gfp_Fe.npz"
EEDL_FE = FIXTURES / "endf" / "EEDL.ZA026000.endf"
GROUPS = ("gfp2", "dk_fp", "gfp3")
# Group names in the April file (and so in the npz keys).
APRIL_NAMES = {"gfp2": "gfp2", "dk_fp": "delta_kernel_fp", "gfp3": "gfp3"}


@pytest.fixture(scope="module")
def april():
    try:
        archive = np.load(APRIL_FE)
    except FileNotFoundError:
        pytest.skip(f"Optional local April GFP reference is absent (not distributed): {APRIL_FE}")
    with archive as z:
        return {k: z[k] for k in z.files}


def test_april_reference_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "APRIL_FE", tmp_path / "missing.npz")
    with pytest.raises(pytest.skip.Exception, match="Optional local April GFP reference is absent"):
        april.__wrapped__()


def test_april_reference_corrupt(tmp_path, monkeypatch):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a NumPy archive")
    monkeypatch.setitem(globals(), "APRIL_FE", path)
    with pytest.raises(ValueError):
        april.__wrapped__()


def test_april_reference_unreadable(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.npz"
    path.touch()
    monkeypatch.setitem(globals(), "APRIL_FE", path)

    def denied(filename):
        assert filename == path
        raise PermissionError("reference read denied")

    monkeypatch.setattr(np, "load", denied)
    with pytest.raises(PermissionError, match="reference read denied"):
        april.__wrapped__()


def _april_sigma(april):
    return np.column_stack([april[f"gfp2/Sigma_s{l}"] for l in range(4)])


# -----------------------------------------------------------------------
# Parameters from the April Sigma_sl: exact (no quadrature involved)
# -----------------------------------------------------------------------


def test_parameters_match_april(april):
    """Same Sigma_sl in, bit-identical April parameters and warnings out."""
    params = gfp_parameters(_april_sigma(april))
    for group in GROUPS:
        name = APRIL_NAMES[group]
        for key, value in params[group].items():
            ref = april[f"{name}/{key}"]
            assert value.dtype == ref.dtype, f"{group}/{key}"
            np.testing.assert_array_equal(value, ref, err_msg=f"{group}/{key}")
    assert params["gfp2"]["success"].sum() == 427
    assert params["gfp3"]["success"].sum() == 70
    assert params["dk_fp"]["success"].all()


# -----------------------------------------------------------------------
# Mandatory synthetic output schema
# -----------------------------------------------------------------------


def test_synthetic_model_output_schema(tmp_path, sample_eedl_dataset):
    """Schema expectations come from the writer contract, not reference data."""
    float_arrays = {
        "gfp2": (
            "energy_grid Sigma_s0 Sigma_s1 Sigma_s2 Sigma_s3 alpha beta beta_raw "
            "Sigma_delta0 transition_rate Sigma_tr mu_star_optimal mu_star_optimal_raw"
        ).split(),
        "dk_fp": "energy_grid Sigma_tr Sigma_delta0 mu_star mu_star_raw".split(),
        "gfp3": (
            "energy_grid Sigma_s0 Sigma_s1 Sigma_s2 Sigma_s3 alpha beta1 beta2 "
            "Sigma_delta0 transition_rate_02 transition_rate_21"
        ).split(),
    }
    scalars = {"gfp2": {"mu_star": 0.9, "beta_max": 100.0}, "dk_fp": {},
               "gfp3": {"mu_star": 0.9}}
    grid = sample_eedl_dataset.cross_sections["xs_tot"].energy
    out = tmp_path / "synthetic.h5"
    with h5py.File(out, "w") as h5f:
        write_mcdc_eedl(h5f, sample_eedl_dataset)
    with h5py.File(out, "r") as h5f:
        root = h5f["electron_reactions"]
        assert root["xs_energy_grid"].attrs["units"] == "eV"
        assert root["elastic_scattering/MT526/xs"].attrs["units"] == "barns"
        models = root["elastic_scattering/small_angle/models"]
        assert set(models) == set(GROUPS)
        for group in GROUPS:
            data = models[group]
            dtypes = {key: "f8" for key in float_arrays[group]}
            dtypes.update(success="bool", warning="S256")
            if group == "gfp2":
                dtypes.update(regime="S12", mu_star_optimal_success="bool",
                              mu_star_optimal_warning="S256")
            assert set(data) == set(dtypes) | set(scalars[group])
            for key, dtype in dtypes.items():
                assert data[key].shape == grid.shape, f"{group}/{key}"
                assert data[key].dtype == np.dtype(dtype), f"{group}/{key}"
                assert dict(data[key].attrs) == {}
            for key, value in scalars[group].items():
                assert data[key].shape == () and data[key].dtype == np.dtype("f8")
                assert data[key][()] == value
                assert dict(data[key].attrs) == {}
            np.testing.assert_array_equal(data["energy_grid"][()], grid)
            expected_attrs = (
                {"dcs_source": "eedl", "mu_grid_policy": "native", "n_angular_anchors": 26}
                if group == "gfp2" else {}
            )
            assert dict(data.attrs) == expected_attrs
        np.testing.assert_array_equal(
            models["gfp2/Sigma_s0"][()],
            sample_eedl_dataset.cross_sections["xs_el"].cross_section,
        )
        for l in range(4):
            np.testing.assert_array_equal(
                models[f"gfp2/Sigma_s{l}"][()], models[f"gfp3/Sigma_s{l}"][()]
            )


# -----------------------------------------------------------------------
# Full chain on the Fe EEDL fixture (April comparisons are optional)
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fe_written(tmp_path_factory):
    from pyepics.readers.eedl import EEDLReader

    if not EEDL_FE.is_file():
        pytest.skip(f"Fixture not found: {EEDL_FE}")
    out = tmp_path_factory.mktemp("gfp") / "Fe.h5"
    with h5py.File(str(out), "w") as h5f:
        write_mcdc_eedl(h5f, EEDLReader().read(EEDL_FE))
    models = "electron_reactions/elastic_scattering/small_angle/models"
    with h5py.File(str(out), "r") as h5f:
        return {
            group: (
                dict(h5f[f"{models}/{group}"].attrs),
                {k: v[()] for k, v in h5f[f"{models}/{group}"].items()},
            )
            for group in GROUPS
        }


def test_fe_layout(fe_written, april):
    """Same datasets, dtypes, shapes and attributes as the April file."""
    for group in GROUPS:
        attrs, data = fe_written[group]
        name = APRIL_NAMES[group]
        names = {k.split("/")[1] for k in april if k.startswith(f"{name}/")}
        assert set(data) == names
        for key, value in data.items():
            ref = april[f"{name}/{key}"]
            assert (value.dtype, value.shape) == (ref.dtype, ref.shape), key
        ref_attrs = {
            k.split("@")[1]: april[k] for k in april if k.startswith(f"{name}@")
        }
        assert attrs.keys() == ref_attrs.keys()
        for key, value in attrs.items():
            assert value == ref_attrs[key]
    attrs = fe_written["gfp2"][0]
    assert attrs["n_angular_anchors"] == 251
    assert (attrs["dcs_source"], attrs["mu_grid_policy"]) == ("eedl", "native")
    assert fe_written["gfp2"][1]["mu_star"] == 0.9
    assert fe_written["gfp2"][1]["beta_max"] == 100.0


@pytest.fixture(scope="module")
def fe_reference(fe_written):
    """Reconstruct moments from EEDL and independent SR quadrature.

    Use the evaluated MT525 cross section, before the sampler's forward
    split, and the April reference interpolation for large-angle tables.
    """
    from pyepics.readers.eedl import EEDLReader

    dataset = EEDLReader().read(EEDL_FE)
    grid = dataset.cross_sections["xs_tot"].energy
    xs_el, xs_la = [
        linear_interpolation(grid, rec.energy, rec.cross_section, rec.interpolation)
        for rec in (dataset.cross_sections[k] for k in ("xs_el", "xs_lge"))
    ]
    d = dataset.distributions["ang_lge"]
    energy, offset, mu, pdf = build_pdf(d.inc_energy, d.value, d.probability)
    ends = list(offset[1:]) + [len(mu)]
    tables = _reference_log_log_tables([
        (e, mu[a:b], pdf[a:b]) for e, a, b in zip(energy, offset, ends)
    ])
    anchor_g = np.array([
        [_trapezoid(np.polynomial.legendre.legval(m, [0] * l + [1]) * p, m)
         / _trapezoid(p, m) for l in range(4)]
        for _, m, p in tables
    ])
    g_la = np.column_stack([
        np.interp(grid, [t[0] for t in tables], anchor_g[:, l]) for l in range(4)
    ])
    xs_sa = np.maximum(xs_el - xs_la, 0.0)
    sigma = xs_la[:, None] * g_la
    for i in np.flatnonzero(xs_sa > 0.0):
        eta = float(small_angle_eta(dataset.Z, [grid[i]])[0])
        sigma[i] += xs_sa[i] * _reference_sr_moments(eta)
    sigma[:, 0] = xs_el
    sigma[xs_el <= 0.0] = 0.0
    return sigma, xs_sa > 0.0


def test_fe_values(fe_written, april, fe_reference):
    """Retain April checks where SR contributes nothing; schemas, total
    cross sections and primary success flags are unchanged everywhere.
    Corrected quantities are checked independently below.
    """
    gfp2, fp = fe_written["gfp2"][1], fe_written["dk_fp"][1]
    grid = gfp2["energy_grid"]
    np.testing.assert_array_equal(grid, april["gfp2/energy_grid"])
    hi = grid >= 10.0
    unchanged = hi & ~fe_reference[1]
    assert unchanged.sum() == 335
    np.testing.assert_array_equal(gfp2["Sigma_s0"][hi], april["gfp2/Sigma_s0"][hi])
    for l in (1, 2, 3):
        key = f"Sigma_s{l}"
        np.testing.assert_allclose(
            gfp2[key][unchanged], april[f"gfp2/{key}"][unchanged], rtol=1e-12
        )
        np.testing.assert_array_equal(gfp2[key], fe_written["gfp3"][1][key])
    np.testing.assert_array_equal(gfp2["success"][hi], april["gfp2/success"][hi])
    np.testing.assert_array_equal(
        gfp2["regime"][unchanged], april["gfp2/regime"][unchanged]
    )
    for key in ("success", "warning"):
        np.testing.assert_array_equal(fp[key][hi], april[f"delta_kernel_fp/{key}"][hi])
    for group, data, keys in (
        (
            "gfp2",
            gfp2,
            ("alpha", "beta", "Sigma_tr", "Sigma_delta0", "transition_rate"),
        ),
        ("delta_kernel_fp", fp, ("Sigma_tr", "Sigma_delta0", "mu_star")),
    ):
        for key in keys:
            np.testing.assert_allclose(
                data[key][unchanged], april[f"{group}/{key}"][unchanged],
                rtol=1e-10, err_msg=key
            )


def test_fe_corrected_moments_and_models(fe_written, fe_reference):
    sigma, corrected = fe_reference
    assert corrected.sum() == 128
    gfp2, fp = fe_written["gfp2"][1], fe_written["dk_fp"][1]
    for l in (1, 2, 3):
        np.testing.assert_allclose(
            gfp2[f"Sigma_s{l}"][corrected], sigma[corrected, l], rtol=2e-15
        )

    # Validate the fitted eigenvalues against independently integrated
    # moments.  Subtracting Sigma_sl from Sigma_s0 at high energy loses
    # relative precision; the absolute roundoff budget scales with Sigma_s0.
    a1, a2 = (sigma[corrected, 0] - sigma[corrected, l] for l in (1, 2))
    roundoff = 16.0 * np.finfo(float).eps * sigma[corrected, 0]

    def check(actual, expected):
        assert np.all(np.abs(actual - expected) <= roundoff)

    alpha, beta = gfp2["alpha"][corrected], gfp2["beta"][corrected]
    assert np.all(gfp2["success"][corrected])
    assert np.all(gfp2["regime"][corrected] == b"gfp2")
    assert np.all((beta >= 0.0) & (beta < gfp2["beta_max"]))
    np.testing.assert_array_equal(beta, gfp2["beta_raw"][corrected])
    assert np.all(gfp2["warning"][corrected] == b"")
    check(gfp2["Sigma_tr"][corrected], a1)
    check(2.0 * alpha / (1.0 + 2.0 * beta), a1)
    check(6.0 * alpha / (1.0 + 6.0 * beta), a2)
    np.testing.assert_allclose(
        gfp2["Sigma_delta0"][corrected] * (1.0 - gfp2["mu_star"]),
        2.0 * alpha, rtol=1e-10,
    )
    np.testing.assert_allclose(
        gfp2["transition_rate"][corrected] * beta, alpha, rtol=1e-10
    )
    # The independent l=3 moment exceeds the largest eigenvalue attainable
    # by the dk-GFP2 root on [-1, 1): its previous success was numerical error.
    a3 = sigma[corrected, 0] - sigma[corrected, 3]
    assert np.all(a3 > 12.0 * alpha / (1.0 + 12.0 * beta) + roundoff)
    assert not gfp2["mu_star_optimal_success"][corrected].any()
    assert np.isnan(gfp2["mu_star_optimal"][corrected]).all()
    assert np.all(gfp2["mu_star_optimal_raw"][corrected] > 1.0)

    mu = fp["mu_star"][corrected]
    assert np.all(fp["success"][corrected])
    assert np.all((-1.0 <= mu) & (mu < 1.0))
    np.testing.assert_array_equal(mu, fp["mu_star_raw"][corrected])
    check(fp["Sigma_tr"][corrected], a1)
    check(fp["Sigma_delta0"][corrected] * (1.0 - mu), a1)
    check(fp["Sigma_delta0"][corrected] * 1.5 * (1.0 - mu) * (1.0 + mu), a2)


def test_fe_below_first_elastic_energy(fe_written):
    """Elastic xs is zero below 10 eV, so no model is defined there."""
    grid = fe_written["gfp2"][1]["energy_grid"]
    lo = grid < 10.0
    assert lo.sum() == 8
    for group in GROUPS:
        data = fe_written[group][1]
        assert not data["success"][lo].any()
        assert np.all(data["warning"][lo] == b"sigma_el <= 0")
    assert np.all(fe_written["gfp2"][1]["Sigma_s0"][lo] == 0.0)
    assert np.isnan(fe_written["dk_fp"][1]["mu_star"][lo]).all()


# -----------------------------------------------------------------------
# Model behaviour
# -----------------------------------------------------------------------


def _eigenvalue_gfp2(alpha, beta, ell):
    L = ell * (ell + 1)
    return alpha * L / (1.0 + beta * L)


def test_gfp2_matches_l1_l2():
    res = gfp2_params(10.0, 9.0, 8.0)  # Sigma_a1 = 1, Sigma_a2 = 2
    assert res["success"] and res["regime"] == "gfp2" and res["warning"] == ""
    assert res["beta"] == pytest.approx(1.0 / 6.0)
    assert _eigenvalue_gfp2(res["alpha"], res["beta"], 1) == pytest.approx(1.0)
    assert _eigenvalue_gfp2(res["alpha"], res["beta"], 2) == pytest.approx(2.0)
    assert res["Sigma_delta0"] == pytest.approx(2.0 * res["alpha"] / 0.1)
    assert res["Sigma_tr"] == 1.0


def test_gfp2_capped_and_inadmissible():
    capped = gfp2_params(10.0, 9.0, 8.999)
    assert capped["regime"] == "gfp2_capped" and capped["beta"] == 100.0
    assert capped["beta_raw"] > 100.0 and "capped" in capped["warning"]

    wide = gfp2_params(10.0, 9.0, 6.5)  # Sigma_a2 > 3 Sigma_a1
    assert not wide["success"] and wide["regime"] == "boltzmann"
    assert wide["beta_raw"] < 0.0 and "Sigma_a2/Sigma_a1 = 3.5" in wide["warning"]

    none = gfp2_params(10.0, 10.0, 9.0)
    assert none["warning"] == "GFP2 inadmissible: Sigma_a1 <= 0"
    with pytest.raises(ValueError, match="mu_star"):
        gfp2_params(10.0, 9.0, 8.0, mu_star=1.0)


def test_optimal_mu_star_gfp2_matches_l3():
    alpha, beta, Sigma_a3 = 1.0, 0.1, 2.5
    mu, raw, ok, warning = optimal_mu_star_gfp2(alpha, beta, Sigma_a3)
    assert ok and mu == raw and warning == ""
    x3 = (5.0 * mu * mu + 5.0 * mu + 2.0) / 2.0  # [1 - P3(mu)] / (1 - mu)
    assert 2.0 * alpha * x3 / (1.0 + 2.0 * beta * x3) == pytest.approx(Sigma_a3)
    assert optimal_mu_star_gfp2(0.0, 0.0, 1.0)[3] == (
        "continuous GFP2 parameters are unavailable"
    )


def test_dk_fp():
    res = dk_fp_params(10.0, 9.0, 8.5)  # Sigma_a1 = 1, Sigma_a2 = 1.5
    assert res["success"]
    assert res["mu_star"] == pytest.approx(2.0 * 1.5 / 3.0 - 1.0)
    assert res["Sigma_delta0"] * (1.0 - res["mu_star"]) == pytest.approx(1.0)
    bad = dk_fp_params(10.0, 9.0, 5.0)  # mu* = 7/3 >= 1
    assert not bad["success"] and np.isnan(bad["mu_star"])
    assert bad["mu_star_raw"] == pytest.approx(7.0 / 3.0)
    assert "outside the admissible" in bad["warning"]


def test_gfp3_recovers_parameters():
    alpha, beta1, beta2 = 2.0, 0.3, 0.05
    Sigma_a = [
        alpha * L / ((1.0 + beta1 * L) * (1.0 + beta2 * L)) for L in (2.0, 6.0, 12.0)
    ]
    res = gfp3_params(10.0, *(10.0 - a for a in Sigma_a))
    assert res["success"] and res["warning"] == ""
    assert res["alpha"] == pytest.approx(alpha)
    assert res["beta1"] == pytest.approx(beta1)
    assert res["beta2"] == pytest.approx(beta2)
    assert res["transition_rate_02"] == pytest.approx(1.0 / beta2)
    assert res["Sigma_delta0"] == pytest.approx(2.0 * alpha / 0.1)


def test_no_elastic_rows():
    params = gfp_parameters(np.zeros((1, 4)))
    assert params["gfp2"]["regime"][0] == b"boltzmann"
    assert params["gfp3"]["Sigma_delta0"][0] == 0.0
    assert np.isnan(params["dk_fp"]["Sigma_delta0"][0])
    for group in GROUPS:
        assert params[group]["warning"][0] == b"sigma_el <= 0"


# -----------------------------------------------------------------------
# Legendre moments
# -----------------------------------------------------------------------


def _reference_sr_moments(eta, order=96):
    """Independent quadrature with u = log(1 + (1-mu)/eta).

    Then p(mu) |dmu| = (eta+2)/2 exp(-u) du on [0, log(1+2/eta)].
    Integrate 1-P_l as polynomials in t=1-mu to retain tiny deficits.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    length = np.log1p(2.0 / eta)
    u = 0.5 * length * (nodes + 1.0)
    t = eta * np.expm1(u)
    weights = weights * 0.5 * length * (1.0 + 0.5 * eta) * np.exp(-u)
    deficits = np.array([t, t * (3.0 - 1.5 * t), t * (6.0 - t * (7.5 - 2.5 * t))])
    return np.r_[weights.sum(), 1.0 - deficits @ weights]


@pytest.mark.parametrize("eta", [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0])
def test_screened_rutherford_g1(eta):
    """g1 = <mu> of eta(eta+2) / (2 (eta+1-mu)^2) on [-1, 1], closed form."""
    g1 = 1.0 + eta - 0.5 * eta * (eta + 2.0) * np.log(1.0 + 2.0 / eta)
    g = legendre_moments_screened_rutherford(eta)
    assert g[0] == 1.0
    assert g[1] == pytest.approx(g1, rel=1e-12)


@pytest.mark.parametrize("eta", [
    5e-16, 5e-15, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2,
    np.nextafter(0.1, 0.0), 0.1, np.nextafter(0.1, 1.0), 1.0, 10.0, 100.0, 2200.0,
])
def test_screened_rutherford_independent_quadrature(eta):
    """Cover the 10 eV--100 GeV, Z=1--100 eta range and both branches."""
    reference = _reference_sr_moments(eta)
    np.testing.assert_allclose(
        reference, _reference_sr_moments(eta, order=192), rtol=1e-12, atol=2e-14
    )
    g = legendre_moments_screened_rutherford(eta)
    assert g.shape == (4,) and g.dtype == np.dtype("f8")
    assert g[0] == 1.0
    assert np.all(np.isfinite(g))
    assert np.all((-1.0 <= g) & (g <= 1.0))
    assert g[2] >= -0.5  # minimum of P_2 on [-1, 1]
    np.testing.assert_allclose(g, reference, rtol=1e-12, atol=2e-14)


def test_screened_rutherford_forward_limit():
    moments = np.array([
        legendre_moments_screened_rutherford(eta)
        for eta in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-16)
    ])
    assert np.all(np.diff(moments[:, 1:], axis=0) > 0.0)
    assert np.all(moments <= 1.0)
    np.testing.assert_allclose(moments[-1], 1.0, rtol=0.0, atol=3e-14)


def test_tabulated_moments():
    mu = np.linspace(-1.0, 1.0, 2001)
    g = legendre_moments_tabulated(mu, 1.0 + mu)  # normalised: (1 + mu) / 2
    np.testing.assert_allclose(g, [1.0, 1.0 / 3.0, 0.0, 0.0], atol=1e-6)
    with pytest.raises(ValueError, match="non-decreasing"):
        legendre_moments_tabulated(mu[::-1], 1.0 + mu)


def test_elastic_moments_mixing():
    """Large-angle moments interpolate linearly in E; small-angle (sigma_sa)
    moments are mixed in by cross section; no elastic gives zeros."""
    anchor_E = np.array([1.0e3, 2.0e3])
    anchor_g = np.array([[1.0, 0.2, 0.1, 0.0], [1.0, 0.4, 0.3, 0.2]])
    energy = np.array([5.0, 1.5e3, 1.5e3])
    sigma = elastic_moments(
        26, energy, [0.0, 4.0, 4.0], [0.0, 4.0, 2.0], anchor_E, anchor_g
    )
    np.testing.assert_array_equal(sigma[0], 0.0)
    np.testing.assert_allclose(sigma[1], 4.0 * np.array([1.0, 0.3, 0.2, 0.1]))
    g_sa = _reference_sr_moments(float(small_angle_eta(26, [1.5e3])[0]))
    np.testing.assert_allclose(
        sigma[2], 2.0 * np.array([1.0, 0.3, 0.2, 0.1]) + 2.0 * g_sa
    )


# -----------------------------------------------------------------------
# Log-log tables for the moment anchors
# -----------------------------------------------------------------------


def _reference_log_log_tables(tables, max_ratio=1.1):
    """Reference copy of PyEEDL densify_angular_grid (April 2026)."""
    out = []
    for (e0, m0, p0), (e1, m1, p1) in zip(tables[:-1], tables[1:]):
        out.append((e0, m0, p0))
        ratio = e1 / e0
        if ratio <= max_ratio:
            continue
        n = max(1, int(np.ceil(np.log(ratio) / np.log(max_ratio))) - 1)
        mu = np.union1d(m0, m1)
        lo, hi = np.log(np.interp(mu, m0, p0)), np.log(np.interp(mu, m1, p1))
        for e in np.exp(np.linspace(np.log(e0), np.log(e1), n + 2)[1:-1]):
            t = (np.log(e) - np.log(e0)) / (np.log(e1) - np.log(e0))
            pdf = np.exp(lo + t * (hi - lo))
            out.append((e, mu, pdf / _trapezoid(pdf, mu)))
    out.append(tables[-1])
    return out


def test_inserted_tables_match_reference():
    m0, p0 = np.array([-1.0, 0.0, 1.0]), np.array([0.2, 0.5, 0.8])
    m1, p1 = np.array([-1.0, 0.5, 1.0]), np.array([0.1, 0.4, 1.4]) / 0.825
    grid, offset, value, pdf = _insert_log_log_tables(
        np.array([1.0e3, 4.0e3]),
        np.array([0, 3]),
        np.concatenate([m0, m1]),
        np.concatenate([p0, p1]),
    )
    reference = _reference_log_log_tables([(1.0e3, m0, p0), (4.0e3, m1, p1)])
    assert len(grid) == len(reference) == 16  # 14 inserted
    assert np.all(grid[1:] / grid[:-1] <= 1.1 + 1e-12)
    ends = list(offset[1:]) + [len(value)]
    for (e, mu, p), start, end, energy in zip(reference, offset, ends, grid):
        assert energy == e
        np.testing.assert_array_equal(value[start:end], mu)
        np.testing.assert_array_equal(pdf[start:end], p)


def test_inserted_tables_reject_non_positive_pdf():
    with pytest.raises(ValueError, match="log-log"):
        _insert_log_log_tables(
            np.array([1.0e3, 4.0e3]),
            np.array([0, 2]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
            np.array([0.0, 1.0, 0.5, 0.5]),
        )
