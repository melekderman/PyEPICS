#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Generalized Fokker-Planck (GFP) elastic-scattering data for MC/DC

Port of the April 2026 PyEEDL MC/DC generator (``create_mcdc_file`` with its
defaults: EEDL cross sections, native cosine grids, mu* = 0.9, beta_max = 100).
Model-fitting arithmetic follows that generator.  Screened Rutherford moments
resolve the forward peak missed by the April fixed quadrature.

The elastic Legendre cross sections Sigma_sl = sigma_el g_l (l = 0..3) mix

* the large-angle EEDL (MT525) cosine tables, whose moments g_l come from the
  trapezoid rule at each table and are interpolated linearly in energy.
  Tables are first inserted by log-log energy interpolation until adjacent
  energies differ by at most :data:`ANCHOR_MAX_ENERGY_RATIO` (PyEEDL
  ``densify_angular_grid``; FRENSIE V&V for EEDL, Kersting et al., Nucl. Sci.
  Eng. 2019, found log-log energy interpolation best).  These tables only
  provide moments; they are not written.
* the small-angle screened Rutherford part, 1/(eta + 1 - mu)^2 with the
  Moliere screening parameter eta of
  :func:`~pyepics.utils.parsing.small_angle_eta`, whose moments are evaluated
  analytically for eta <= 0.1 and by resolved 200-point Gauss-Legendre
  quadrature otherwise, always over [-1, 1].

weighted by sigma_la and sigma_sa = max(sigma_el - sigma_la, 0).  With the
transport cross sections Sigma_al = Sigma_s0 - Sigma_sl, the models match the
Boltzmann eigenvalues -Sigma_al (equation numbers follow Prinja's GFP notes,
as in the April generator):

* GFP2: alpha, beta from l = 1, 2, with a delta kernel at mu*;
* delta-kernel Fokker-Planck (``dk_fp``, April ``delta_kernel_fp``): mu*
  from l = 2 (Eqs. 17-18, 22);
* GFP3: alpha, beta1, beta2 from l = 1, 2, 3.

Corrected screened Rutherford moments change downstream model parameters
relative to April.  The fitting formulas are unchanged; ill-conditioned
solves can also amplify roundoff and change ``success`` flags.

Unlike the April generator, non-positive PDF values in the log-log tables,
cosine tables that are too short, unsorted or of zero area, and mu* outside
[-1, 1) raise ``ValueError`` instead of being floored, sorted or replaced by
no deflection.  No EPICS 2025 element reaches these cases.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from pyepics.utils.parsing import small_angle_eta

# Largest energy ratio between adjacent moment tables (April / FRENSIE).
ANCHOR_MAX_ENERGY_RATIO = 1.1
# Gauss-Legendre order for the resolved screened Rutherford moments (eta > 0.1).
SR_QUADRATURE_ORDER = 200
GFP_MU_STAR = 0.9
GFP2_BETA_MAX = 100.0

_NO_DEFLECTION = np.array([1.0, 0.0, 0.0, 0.0])

# np.trapezoid needs NumPy >= 2.0; np.trapz is the same rule in older releases.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _insert_log_log_tables(energy, offset, value, pdf):
    """Insert log-log interpolated cosine tables so that no two adjacent
    energies differ by more than :data:`ANCHOR_MAX_ENERGY_RATIO`.

    The evaluated tables are returned unchanged.
    """
    ends = list(offset[1:]) + [len(value)]
    tables = [
        (energy[i], value[offset[i] : ends[i]], pdf[offset[i] : ends[i]])
        for i in range(len(energy))
    ]
    out = []
    for (e0, mu0, p0), (e1, mu1, p1) in zip(tables[:-1], tables[1:]):
        out.append((e0, mu0, p0))
        ratio = e1 / e0
        if ratio <= ANCHOR_MAX_ENERGY_RATIO:
            continue
        n = max(1, int(np.ceil(np.log(ratio) / np.log(ANCHOR_MAX_ENERGY_RATIO))) - 1)
        mu = np.union1d(mu0, mu1)
        lo, hi = np.interp(mu, mu0, p0), np.interp(mu, mu1, p1)
        if np.any(lo <= 0.0) or np.any(hi <= 0.0):
            raise ValueError(
                f"Cosine tables at {e0} and {e1} eV have non-positive PDF values; "
                "log-log interpolation is undefined."
            )
        lo, hi = np.log(lo), np.log(hi)
        for e in np.exp(np.linspace(np.log(e0), np.log(e1), n + 2)[1:-1]):
            t = (np.log(e) - np.log(e0)) / (np.log(e1) - np.log(e0))
            p = np.exp(lo + t * (hi - lo))
            out.append((e, mu, p / _trapezoid(p, mu)))
    out.append(tables[-1])

    new_offset = np.cumsum([0] + [len(t[1]) for t in out[:-1]])
    return (
        np.array([t[0] for t in out], dtype="f8"),
        new_offset.astype(np.asarray(offset).dtype),
        np.concatenate([t[1] for t in out]),
        np.concatenate([t[2] for t in out]),
    )


def _legendre(mu):
    """P_1, P_2 and P_3 at *mu*."""
    return (mu, 0.5 * (3.0 * mu * mu - 1.0), 0.5 * (5.0 * mu * mu * mu - 3.0 * mu))


def legendre_moments_tabulated(mu, pdf):
    """Legendre moments g_0..g_3 of a lin-lin tabulated cosine PDF.

    The PDF is normalised and the moments integrated by the trapezoid rule.
    """
    mu = np.asarray(mu, dtype="f8")
    pdf = np.asarray(pdf, dtype="f8")
    if mu.size < 2 or np.any(np.diff(mu) < 0.0):
        raise ValueError("Cosine table needs at least two non-decreasing mu values.")
    norm = _trapezoid(pdf, mu)
    if norm <= 0.0:
        raise ValueError("Cosine table has a non-positive integral.")
    pdf = pdf / norm
    g = _NO_DEFLECTION.copy()
    g[1:] = [_trapezoid(p * pdf, mu) for p in _legendre(mu)]
    return g


@cache
def _sr_quadrature():
    return np.polynomial.legendre.leggauss(SR_QUADRATURE_ORDER)


def legendre_moments_screened_rutherford(eta):
    """Legendre moments g_0..g_3 of the screened Rutherford PDF
    eta (eta + 2) / (2 (eta + 1 - mu)^2) on [-1, 1].

    Integrating powers of t = 1 - mu gives m_k = <t^k> analytically.
    For eta <= 0.1 these expressions avoid large cancelling terms and
    resolve arbitrarily narrow peaks.  At larger eta the PDF is smooth;
    quadrature avoids cancellation in the analytical, near-isotropic limit.
    """
    if eta <= 0.1:
        # L = log(1 + 2/eta), without overflowing 2/eta for tiny eta.
        L = np.log(eta + 2.0) - np.log(eta)
        m1 = eta * ((1.0 + 0.5 * eta) * L - 1.0)
        m2 = 2.0 * eta * (1.0 - m1)
        m3 = 2.0 * eta - 1.5 * eta * m2
        return np.array([
            1.0,
            1.0 - m1,
            1.0 - 3.0 * m1 + 1.5 * m2,
            1.0 - 6.0 * m1 + 7.5 * m2 - 2.5 * m3,
        ], dtype="f8")
    nodes, weights = _sr_quadrature()
    pdf = (1.0 / (eta + 1.0 - nodes) ** 2) / (2.0 / (eta * (eta + 2.0)))
    g = _NO_DEFLECTION.copy()
    g[1:] = [np.dot(weights, p * pdf) for p in _legendre(nodes)]
    return g


def _interpolate_moments(energy, anchor_energy, anchor_g):
    """Moments at *energy*: linear in E between anchors, constant beyond them."""
    if energy <= anchor_energy[0]:
        return anchor_g[0]
    if energy >= anchor_energy[-1]:
        return anchor_g[-1]
    j = np.searchsorted(anchor_energy, energy, side="right") - 1
    f = (energy - anchor_energy[j]) / (anchor_energy[j + 1] - anchor_energy[j])
    return (1.0 - f) * anchor_g[j] + f * anchor_g[j + 1]


def elastic_moments(Z, energy, xs_el, xs_la, anchor_energy, anchor_g):
    """Elastic Legendre cross sections Sigma_sl = sigma_el g_l, shape (N, 4).

    g_l mixes the large-angle (*anchor_g*) and screened Rutherford moments,
    weighted by sigma_la and sigma_sa = max(sigma_el - sigma_la, 0).  Rows
    with sigma_el <= 0 are zero.
    """
    sigma = np.zeros((len(energy), 4))
    for i, E in enumerate(energy):
        sigma_el, sigma_la = float(xs_el[i]), float(xs_la[i])
        if sigma_el <= 0.0:
            continue
        sigma_sa = max(sigma_el - sigma_la, 0.0)
        g_la = (
            _interpolate_moments(E, anchor_energy, anchor_g)
            if sigma_la > 0.0
            else _NO_DEFLECTION
        )
        g_sa = (
            legendre_moments_screened_rutherford(float(small_angle_eta(Z, [E])[0]))
            if sigma_sa > 0.0
            else _NO_DEFLECTION
        )
        g = (sigma_la * g_la + sigma_sa * g_sa) / sigma_el
        g[0] = 1.0
        sigma[i] = sigma_el * g
    return sigma


def _check_mu_star(mu_star):
    if not -1.0 <= mu_star < 1.0:
        raise ValueError(f"mu_star = {mu_star} is outside [-1, 1).")


def gfp2_params(
    Sigma_s0, Sigma_s1, Sigma_s2, mu_star=GFP_MU_STAR, beta_max=GFP2_BETA_MAX
):
    """GFP2 parameters matching Sigma_al = 2 alpha x / (1 + 2 beta x), x = l(l+1)/2
    at l = 1, 2: beta = (3 Sigma_a1 - Sigma_a2) / (6 (Sigma_a2 - Sigma_a1)),
    alpha = Sigma_a1 (1 + 2 beta) / 2, Sigma_delta0 = 2 alpha / (1 - mu*).

    beta is capped at *beta_max*; beta_raw < 0 or Sigma_a1 <= 0 is
    inadmissible (``regime`` 'boltzmann').
    """
    _check_mu_star(mu_star)
    Sigma_a1 = float(Sigma_s0 - Sigma_s1)
    Sigma_a2 = float(Sigma_s0 - Sigma_s2)
    Sigma_tr = Sigma_a1

    denom = Sigma_a2 - Sigma_a1
    if abs(denom) < 1.0e-30 * max(float(Sigma_s0), 1.0):
        beta_raw = np.inf
    else:
        beta_raw = (3.0 * Sigma_a1 - Sigma_a2) / (6.0 * denom)

    if beta_raw < 0.0 or Sigma_a1 <= 0.0:
        if Sigma_a1 > 0.0:
            warning = (
                "GFP2 inadmissible: require Sigma_a1 < Sigma_a2 < 3*Sigma_a1, "
                f"got Sigma_a2/Sigma_a1 = {Sigma_a2 / Sigma_a1:.6e}"
            )
        else:
            warning = "GFP2 inadmissible: Sigma_a1 <= 0"
        return {
            "alpha": 0.0,
            "beta": float(beta_raw),
            "beta_raw": float(beta_raw),
            "Sigma_delta0": 0.0,
            "transition_rate": 0.0,
            "Sigma_tr": Sigma_tr,
            "regime": "boltzmann",
            "success": False,
            "warning": warning,
        }

    if np.isfinite(beta_raw) and beta_raw <= beta_max:
        regime, beta, warning = "gfp2", float(beta_raw), ""
    else:
        regime, beta = "gfp2_capped", float(beta_max)
        warning = f"beta_raw = {beta_raw:.6e} > beta_max = {beta_max:.6e}; capped"

    if beta <= 0.0:
        transition_rate = np.inf
    else:
        transition_rate = Sigma_tr * (1.0 + 2.0 * beta) / (2.0 * beta)
    return {
        "alpha": 0.5 * Sigma_tr * (1.0 + 2.0 * beta),
        "beta": beta,
        "beta_raw": float(beta_raw),
        "Sigma_delta0": Sigma_tr * (1.0 + 2.0 * beta) / (1.0 - float(mu_star)),
        "transition_rate": transition_rate,
        "Sigma_tr": Sigma_tr,
        "regime": regime,
        "success": True,
        "warning": warning,
    }


def _delta_kernel_mu_star(mu, warning):
    """(mu_star, mu_star_raw, success, warning) for a candidate mu*; a delta
    kernel needs a finite -1 <= mu* < 1."""
    mu = float(mu)
    if np.isfinite(mu) and -1.0 <= mu < 1.0:
        return mu, mu, True, ""
    return np.nan, mu if np.isfinite(mu) else np.nan, False, warning


def optimal_mu_star_gfp2(alpha, beta, Sigma_a3):
    """mu* of the delta-kernel GFP2 model matching the l = 3 eigenvalue
    (Prinja Eqs. 37-40): 5 mu*^2 + 5 mu* + 2 - C = 0 with
    C = Sigma_a3 / (alpha - beta Sigma_a3).

    Returns (mu_star, mu_star_raw, success, warning).
    """
    if not (np.isfinite(alpha) and np.isfinite(beta) and alpha > 0.0 and beta >= 0.0):
        return np.nan, np.nan, False, "continuous GFP2 parameters are unavailable"
    if (alpha - beta * Sigma_a3) <= 0.0:
        return (
            np.nan,
            np.nan,
            False,
            "alpha - beta*Sigma_a3 <= 0, so Eq.(ell=3) has no admissible "
            "delta-kernel root",
        )
    x3 = Sigma_a3 / (2.0 * (alpha - beta * Sigma_a3))
    disc = 20.0 * (2.0 * x3) - 15.0
    if disc < 0.0:
        return (
            np.nan,
            np.nan,
            False,
            f"GFP2 ell=3 constraint has no real mu_star root: discriminant={disc:.6e}",
        )
    mu = (-5.0 + np.sqrt(disc)) / 10.0
    return _delta_kernel_mu_star(
        mu,
        f"GFP2 ell=3 constraint gives mu_star = {mu:.6e}, outside the admissible "
        "delta-kernel range",
    )


def dk_fp_params(Sigma_s0, Sigma_s1, Sigma_s2):
    """Delta-kernel Fokker-Planck parameters with the optimal mu*.

    mu* matches the l = 2 eigenvalue, mu* = 2 Sigma_a2 / (3 Sigma_a1) - 1
    (Prinja Eq. 22), and Sigma_delta0 = Sigma_tr / (1 - mu*) (Eq. 18).
    """
    Sigma_a1 = Sigma_s0 - Sigma_s1
    Sigma_a2 = Sigma_s0 - Sigma_s2
    if Sigma_a1 <= 0.0:
        mu_star, raw, success = np.nan, np.nan, False
        warning = "Sigma_a1 <= 0, so Eq.(22) cannot define mu_star"
    else:
        mu = 2.0 * Sigma_a2 / (3.0 * Sigma_a1) - 1.0
        mu_star, raw, success, warning = _delta_kernel_mu_star(
            mu,
            f"FP ell=2 constraint gives mu_star = {mu:.6e}, outside the admissible "
            "delta-kernel range",
        )
    Sigma_tr = float(Sigma_a1)
    return {
        "Sigma_tr": Sigma_tr,
        "Sigma_delta0": Sigma_tr / (1.0 - mu_star) if success else np.nan,
        "mu_star": mu_star,
        "mu_star_raw": raw,
        "success": success,
        "warning": warning,
    }


def gfp3_params(Sigma_s0, Sigma_s1, Sigma_s2, Sigma_s3, mu_star=GFP_MU_STAR):
    """GFP3 parameters matching
    Sigma_al = alpha L / ((1 + beta1 L) (1 + beta2 L)), L = l(l+1), at l = 1, 2, 3.

    With p = beta1 + beta2 and q = beta1 beta2 the conditions are a 2x2 linear
    system; alpha follows from l = 1, Sigma_delta0 = 2 alpha / (1 - mu*), and
    the transition rates are 1/beta2 (states 0-2) and 1/beta1 (states 2-1).
    """
    _check_mu_star(mu_star)
    Sigma_a1 = float(Sigma_s0 - Sigma_s1)
    Sigma_a2 = float(Sigma_s0 - Sigma_s2)
    Sigma_a3 = float(Sigma_s0 - Sigma_s3)

    def fail(warning):
        return {
            "alpha": 0.0,
            "beta1": 0.0,
            "beta2": 0.0,
            "Sigma_delta0": np.nan,
            "transition_rate_02": np.nan,
            "transition_rate_21": np.nan,
            "success": False,
            "warning": warning,
        }

    if Sigma_a1 <= 0.0:
        return fail("Sigma_a1 <= 0")

    R2 = Sigma_a2 / Sigma_a1
    R3 = Sigma_a3 / Sigma_a1
    a11 = 6.0 * (R2 - 1.0)
    a12 = 12.0 * (3.0 * R2 - 1.0)
    b1 = 3.0 - R2
    a21 = 12.0 * (R3 - 1.0)
    a22 = 24.0 * (6.0 * R3 - 1.0)
    b2 = 6.0 - R3

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-30:
        return fail("singular system")
    p = (b1 * a22 - b2 * a12) / det
    q = (a11 * b2 - a21 * b1) / det

    disc = p * p - 4.0 * q
    if disc < 0.0:
        return fail(f"complex roots: disc={disc:.6e}")
    sqrt_disc = np.sqrt(disc)
    beta1 = (p + sqrt_disc) / 2.0
    beta2 = (p - sqrt_disc) / 2.0
    if beta1 < 0.0 or beta2 < 0.0:
        return fail(f"negative beta: beta1={beta1:.6e}, beta2={beta2:.6e}")

    alpha = Sigma_a1 * (1.0 + 2.0 * p + 4.0 * q) / 2.0
    if alpha <= 0.0:
        return fail(f"alpha <= 0: {alpha:.6e}")

    return {
        "alpha": alpha,
        "beta1": beta1,
        "beta2": beta2,
        "Sigma_delta0": alpha * 2.0 / (1.0 - float(mu_star)),
        "transition_rate_02": 1.0 / beta2 if beta2 > 0.0 else np.inf,
        "transition_rate_21": 1.0 / beta1 if beta1 > 0.0 else np.inf,
        "success": True,
        "warning": "",
    }


# Values written where there is no elastic scattering (sigma_el <= 0).
_GFP2_NO_ELASTIC = {
    "alpha": 0.0,
    "beta": 0.0,
    "beta_raw": 0.0,
    "Sigma_delta0": 0.0,
    "transition_rate": 0.0,
    "Sigma_tr": 0.0,
    "regime": "boltzmann",
    "success": False,
    "warning": "sigma_el <= 0",
}
_FP_NO_ELASTIC = {
    "Sigma_tr": 0.0,
    "Sigma_delta0": np.nan,
    "mu_star": np.nan,
    "mu_star_raw": np.nan,
    "success": False,
    "warning": "sigma_el <= 0",
}
_GFP3_NO_ELASTIC = {
    "alpha": 0.0,
    "beta1": 0.0,
    "beta2": 0.0,
    "Sigma_delta0": 0.0,
    "transition_rate_02": 0.0,
    "transition_rate_21": 0.0,
    "success": False,
    "warning": "sigma_el <= 0",
}

# String dataset widths of the April files.
_STRING_DTYPES = {"regime": "S12"}


def _columns(rows):
    """Per-energy result dicts -> one array per key (strings as fixed bytes)."""
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        if isinstance(values[0], str):
            out[key] = np.array(values, dtype=_STRING_DTYPES.get(key, "S256"))
        else:
            out[key] = np.array(values, dtype=bool if key.endswith("success") else "f8")
    return out


def gfp_parameters(sigma, mu_star=GFP_MU_STAR, beta_max=GFP2_BETA_MAX):
    """Per-energy GFP2 (with the GFP2 optimal mu*), delta-kernel FP and GFP3
    parameters from the elastic Legendre cross sections *sigma* (N, 4).

    Rows with Sigma_s0 = sigma_el <= 0 are unsuccessful with warning
    ``'sigma_el <= 0'``.  Returns ``{group: {dataset: array}}``.
    """
    _check_mu_star(mu_star)
    gfp2, fp, gfp3 = [], [], []
    for s in np.asarray(sigma, dtype="f8"):
        if s[0] <= 0.0:
            row2 = dict(_GFP2_NO_ELASTIC)
            fp.append(_FP_NO_ELASTIC)
            gfp3.append(_GFP3_NO_ELASTIC)
        else:
            row2 = gfp2_params(s[0], s[1], s[2], mu_star, beta_max)
            fp.append(dk_fp_params(s[0], s[1], s[2]))
            gfp3.append(gfp3_params(s[0], s[1], s[2], s[3], mu_star))
        keys = ("mu_star_optimal", "mu_star_optimal_raw", "mu_star_optimal_success")
        mu_opt = optimal_mu_star_gfp2(row2["alpha"], row2["beta"], s[0] - s[3])
        row2.update(zip(keys + ("mu_star_optimal_warning",), mu_opt))
        gfp2.append(row2)
    return {
        "gfp2": _columns(gfp2),
        "dk_fp": _columns(fp),
        "gfp3": _columns(gfp3),
    }


def build_gfp_data(
    Z,
    energy,
    xs_el,
    xs_la,
    table_energy,
    table_offset,
    table_mu,
    table_pdf,
    mu_star=GFP_MU_STAR,
    beta_max=GFP2_BETA_MAX,
):
    """GFP2, delta-kernel FP and GFP3 elastic data on the cross-section grid.

    Parameters
    ----------
    Z : int
        Atomic number.
    energy, xs_el, xs_la : ndarray
        Energy grid (eV) and total (MT526) and large-angle (MT525) elastic
        cross sections (b) on it.
    table_energy, table_offset, table_mu, table_pdf : ndarray
        Large-angle cosine tables as returned by
        :func:`~pyepics.utils.parsing.build_pdf`, energies increasing.
    mu_star, beta_max : float
        Delta-kernel cosine of GFP2/GFP3 and the GFP2 beta cap.

    Returns
    -------
    dict
        ``{"gfp2" | "dk_fp" | "gfp3": (attrs, datasets)}``, the
        MC/DC ``small_angle/models`` subgroups.
    """
    energy = np.asarray(energy, dtype="f8")
    if np.any(np.diff(table_energy) <= 0.0):
        raise ValueError("Large-angle cosine tables must have increasing energies.")

    anchor_energy, offset, mu, pdf = _insert_log_log_tables(
        np.asarray(table_energy, dtype="f8"),
        np.asarray(table_offset),
        np.asarray(table_mu, dtype="f8"),
        np.asarray(table_pdf, dtype="f8"),
    )
    ends = list(offset[1:]) + [len(mu)]
    anchor_g = np.array(
        [legendre_moments_tabulated(mu[a:b], pdf[a:b]) for a, b in zip(offset, ends)]
    )
    sigma = elastic_moments(Z, energy, xs_el, xs_la, anchor_energy, anchor_g)
    params = gfp_parameters(sigma, mu_star, beta_max)

    moments = {f"Sigma_s{l}": sigma[:, l].copy() for l in range(4)}
    gfp2_attrs = {
        "dcs_source": "eedl",
        "mu_grid_policy": "native",
        "n_angular_anchors": np.int64(len(anchor_energy)),
    }
    gfp2 = {
        "energy_grid": energy,
        **moments,
        **params["gfp2"],
        "mu_star": float(mu_star),
        "beta_max": float(beta_max),
    }
    fp = {"energy_grid": energy, **params["dk_fp"]}
    gfp3 = {
        "energy_grid": energy,
        **moments,
        **params["gfp3"],
        "mu_star": float(mu_star),
    }
    return {"gfp2": (gfp2_attrs, gfp2), "dk_fp": ({}, fp), "gfp3": ({}, gfp3)}
