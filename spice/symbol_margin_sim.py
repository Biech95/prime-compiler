#!/usr/bin/env python3
"""Kill-gate 2: symbol error rate vs. noise margin (X1 quantitative row).

Three fidelity levels, cheapest first:

  L1  numpy    k-bit code match in an analog CAM row with static conductance
               mismatch + thermal noise; false-accept / false-reject vs margin,
               Monte Carlo (plain + exponential-tilting importance sampling)
               against the analytic Gaussian tail.
  L2  numpy    Hopfield attractor retrieval; AGS alpha_c anchor at T=0, then
               error rate vs barrier/kT and vs synaptic mismatch.
  L3  ngspice  ONE aCAM row, k=16 lognormal-mismatched conductances into a
               finite-gain transimpedance node + behavioral comparator,
               200 paired Monte-Carlo draws shared with L1.

Pre-registration and verdicts: docs/exp_symbol_margin.md
Raw numbers: spice/out/symbol_margin_results.json

Usage:
    python3 symbol_margin_sim.py              # everything
    python3 symbol_margin_sim.py --skip-l3    # no ngspice
"""

from __future__ import annotations

import os

# Must precede numpy: 32 cores are used through multiprocessing, not BLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
WORKDIR = HERE / "out"

# ---------------------------------------------------------------------------
# Physical / design constants
# ---------------------------------------------------------------------------
KB = 1.380649e-23
T_AMB = 300.0
KT = KB * T_AMB                      # 4.1419e-21 J

C_ML = 10e-15                        # match-line capacitance [F]
V_FS = 1.0                           # full-scale match-line swing for s = k [V]

# ngspice SPICE3-legacy thermal voltage (see spice/README.md gotcha). Not used
# by the resistive summing row, kept so the constant is in one place if a
# junction thresholder is ever swapped in for the behavioral comparator.
VT_NG = 0.025864890

K_GRID = (8, 16, 64, 256)
SIGMA_G_GRID = (0.005, 0.01, 0.02, 0.05)
SIGMA_G_MOS = 0.03                   # MOS-typical corner used by K3
SIGMA_REL_TH_GRID = (0.005, 0.01, 0.02)   # sigma_th*sqrt(k)/k
EPS_TARGETS = (1e-3, 1e-6, 1e-9)

SEED = 2026


# ---------------------------------------------------------------------------
# Analytic layer
# ---------------------------------------------------------------------------
def q_tail(z):
    """Upper Gaussian tail Q(z) = P(Z > z), accurate deep into the tail."""
    return ndtr(-np.asarray(z, dtype=float))


def z_for_eps(eps):
    """z such that Q(z) = eps."""
    return -ndtri(eps)


def sigma_total(k, sigma_g, sigma_rel_th):
    """Std of the match score in score units (one bit flip = 2 units).

    Mismatch contributes sqrt(k)*sigma_g (k independent (w_i x_i)^2 = 1 terms).
    Thermal contributes sigma_th*sqrt(k), parameterised by sigma_th*sqrt(k)/k.
    """
    var_mis = k * sigma_g ** 2
    var_th = (sigma_rel_th * k) ** 2
    return float(np.sqrt(var_mis + var_th)), float(np.sqrt(var_mis)), float(np.sqrt(var_th))


def analytic_rates(k, m, sigma_tot, d_adv=1):
    """(false reject, false accept) for threshold theta = k - m.

    Match mean = k. Adversary at Hamming distance d_adv has mean k - 2*d_adv.
    """
    eps_fr = float(q_tail(m / sigma_tot))
    eps_fa = float(q_tail((2.0 * d_adv - m) / sigma_tot))
    return eps_fr, eps_fa


# ---------------------------------------------------------------------------
# L1 -- Monte Carlo
# ---------------------------------------------------------------------------
def mc_plain(k, sigma_g, sigma_th_abs, m, d_adv, n_trials, seed, lognormal=False,
             chunk=200_000):
    """Plain MC of eps_FR and eps_FA with explicit per-line mismatch draws.

    Returns (eps_fr, eps_fa, n_fr, n_fa, mean_s_match, std_s_match).
    """
    rng = np.random.default_rng(seed)
    theta_offset = -m                      # theta - k
    n_fr = 0
    n_fa = 0
    s_sum = 0.0
    s_sq = 0.0
    done = 0
    s_ln = float(np.sqrt(np.log1p(sigma_g ** 2))) if lognormal else 0.0
    while done < n_trials:
        n = min(chunk, n_trials - done)
        if lognormal:
            nu = rng.normal(0.0, s_ln, size=(n, k))
            g = np.exp(nu - 0.5 * s_ln ** 2)
        else:
            g = 1.0 + rng.normal(0.0, sigma_g, size=(n, k))
        noise_m = rng.normal(0.0, sigma_th_abs, size=n)
        # match query: c_i = +1 for all i
        s_match = g.sum(axis=1) + noise_m
        n_fr += int(np.count_nonzero(s_match - k <= theta_offset))
        s_sum += float(s_match.sum())
        s_sq += float((s_match ** 2).sum())
        # adversary at Hamming distance d_adv: flip the sign of d_adv terms
        noise_a = rng.normal(0.0, sigma_th_abs, size=n)
        s_adv = s_match - noise_m - 2.0 * g[:, :d_adv].sum(axis=1) + noise_a
        n_fa += int(np.count_nonzero(s_adv - k > theta_offset))
        done += n
    mean_s = s_sum / n_trials
    std_s = float(np.sqrt(max(s_sq / n_trials - mean_s ** 2, 0.0)))
    return (n_fr / n_trials, n_fa / n_trials, n_fr, n_fa, mean_s, std_s)


def mc_tilted(k, sigma_g, sigma_th_abs, m, d_adv, n_trials, seed, which,
              lognormal=False, chunk=200_000):
    """Importance-sampled tail estimate by exponential tilting (mean shift).

    The aggregate s = sum_i c_i g_i + n is shifted onto the threshold by giving
    each mismatch variable a shift c_i*a and the thermal variable a shift b,
    split in proportion to their variance contributions (optimal for the
    Gaussian case, merely valid for the lognormal case).

    Returns (estimate, relative standard error, effective sample size fraction).
    """
    rng = np.random.default_rng(seed)
    c = np.ones(k)
    c[:d_adv] = -1.0
    if which == "fr":
        c[:] = 1.0
        mu = float(k)
        # need s <= k - m  -> shift by Delta = -m
        delta = -m
        want_upper = False
    else:
        mu = float(k - 2 * d_adv)
        # need s > k - m -> shift by Delta = (k - m) - mu
        delta = (k - m) - mu
        want_upper = True
    thr = k - m

    s_ln = float(np.sqrt(np.log1p(sigma_g ** 2))) if lognormal else 0.0
    sg = s_ln if lognormal else sigma_g
    var_mis = k * sigma_g ** 2
    var_th = sigma_th_abs ** 2
    var_tot = var_mis + var_th
    if var_tot <= 0:
        return (float("nan"),) * 3
    a = delta * (sigma_g ** 2) / var_tot          # per-line shift magnitude
    b = delta * var_th / var_tot                  # thermal shift
    # For lognormal, shift the underlying normal by the equivalent amount.
    a_nu = a * (s_ln / sigma_g) if lognormal else a

    tot = 0.0
    tot_sq = 0.0
    done = 0
    while done < n_trials:
        n = min(chunk, n_trials - done)
        z = rng.normal(0.0, 1.0, size=(n, k))
        if lognormal:
            nu = c * a_nu + sg * z
            g = np.exp(nu - 0.5 * s_ln ** 2)
            # log density ratio p/q for N(0,s_ln^2) vs N(c*a_nu, s_ln^2)
            logw = -(2.0 * (nu * (c * a_nu)).sum(axis=1)
                     - k * a_nu ** 2) / (2.0 * s_ln ** 2)
        else:
            e = c * a + sg * z
            g = 1.0 + e
            logw = -(2.0 * (e * (c * a)).sum(axis=1)
                     - k * a ** 2) / (2.0 * sg ** 2)
        if sigma_th_abs > 0:
            nz = rng.normal(0.0, 1.0, size=n)
            nn = b + sigma_th_abs * nz
            logw = logw - (2.0 * nn * b - b ** 2) / (2.0 * sigma_th_abs ** 2)
        else:
            nn = 0.0
        s = (c * g).sum(axis=1) + nn
        hit = (s > thr) if want_upper else (s <= thr)
        w = np.where(hit, np.exp(logw), 0.0)
        tot += float(w.sum())
        tot_sq += float((w ** 2).sum())
        done += n
    est = tot / n_trials
    var = max(tot_sq / n_trials - est ** 2, 0.0)
    rse = float(np.sqrt(var / n_trials) / est) if est > 0 else float("nan")
    ess = (tot ** 2 / tot_sq / n_trials) if tot_sq > 0 else 0.0
    return est, rse, float(ess)


def _l1_worker(job):
    kind = job["kind"]
    if kind == "plain":
        fr, fa, nfr, nfa, ms, ss = mc_plain(
            job["k"], job["sigma_g"], job["sigma_th_abs"], job["m"],
            job["d_adv"], job["n"], job["seed"], job["lognormal"])
        return {**job, "eps_fr_mc": fr, "eps_fa_mc": fa,
                "n_fr": nfr, "n_fa": nfa, "mean_s": ms, "std_s": ss}
    est, rse, ess = mc_tilted(
        job["k"], job["sigma_g"], job["sigma_th_abs"], job["m"], job["d_adv"],
        job["n"], job["seed"], job["which"], job["lognormal"])
    return {**job, "est": est, "rse": rse, "ess": ess}


def run_l1(pool, n_plain=1_000_000, n_is=400_000):
    """Full L1 sweep. Returns a dict ready for JSON."""
    out = {"configs": [], "slope_fits": [], "prefactor": [],
           "k3": {}, "energy": [], "code_distance": []}

    # ---- 1) margin sweeps, analytic + IS-MC, Gaussian mismatch -------------
    jobs = []
    cfgs = []
    for k in K_GRID:
        for sg in SIGMA_G_GRID:
            for srt in SIGMA_REL_TH_GRID:
                st, st_mis, st_th = sigma_total(k, sg, srt)
                sth_abs = srt * k
                cfg = {"k": k, "sigma_g": sg, "sigma_rel_th": srt,
                       "sigma_tot": st, "sigma_mis": st_mis, "sigma_th": st_th,
                       "sigma_th_abs": sth_abs}
                cfgs.append(cfg)
                for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
                    m = z * st
                    jobs.append({"kind": "tilt", "which": "fr", "k": k,
                                 "sigma_g": sg, "sigma_th_abs": sth_abs, "m": m,
                                 "d_adv": 1, "n": n_is,
                                 "seed": abs(hash((k, sg, srt, z, "fr"))) % (2**31),
                                 "lognormal": False, "z": z})
    res = pool.map(_l1_worker, jobs, chunksize=1)
    by_cfg = {}
    for r in res:
        key = (r["k"], r["sigma_g"], r["sigma_th_abs"])
        by_cfg.setdefault(key, []).append(r)

    for cfg in cfgs:
        key = (cfg["k"], cfg["sigma_g"], cfg["sigma_th_abs"])
        rows = sorted(by_cfg[key], key=lambda r: r["z"])
        pts = []
        for r in rows:
            an_fr, an_fa = analytic_rates(cfg["k"], r["m"], cfg["sigma_tot"])
            pts.append({"z": r["z"], "m": r["m"], "u": r["z"] ** 2,
                        "eps_fr_mc": r["est"], "eps_fr_rse": r["rse"],
                        "ess": r["ess"], "eps_fr_analytic": an_fr,
                        "ratio": (r["est"] / an_fr) if an_fr > 0 else float("nan")})
        # slope of ln eps vs (m/sigma)^2 over the >=3-decade window z in [3,6]
        win = [p for p in pts if 3.0 <= p["z"] <= 6.0]
        u = np.array([p["u"] for p in win])
        le_mc = np.log(np.array([p["eps_fr_mc"] for p in win]))
        le_an = np.log(np.array([p["eps_fr_analytic"] for p in win]))
        sl_mc = float(np.polyfit(u, le_mc, 1)[0])
        sl_an = float(np.polyfit(u, le_an, 1)[0])
        dec = float((le_an[0] - le_an[-1]) / np.log(10.0))
        out["slope_fits"].append({**cfg, "slope_mc": sl_mc,
                                  "slope_analytic": sl_an, "decades": dec,
                                  "slope_dev_mc": abs(sl_mc + 0.5) / 0.5,
                                  "slope_dev_an": abs(sl_an + 0.5) / 0.5})
        out["configs"].append({**cfg, "points": pts})

    # ---- 2) plain-MC cross-check of the IS estimator ------------------------
    xjobs = []
    for k in (16, 256):
        for sg in (0.01, 0.05):
            srt = 0.01
            st, _, _ = sigma_total(k, sg, srt)
            for z in (3.0, 4.0):
                xjobs.append({"kind": "plain", "k": k, "sigma_g": sg,
                              "sigma_th_abs": srt * k, "m": z * st, "d_adv": 1,
                              "n": n_plain, "seed": 7000 + len(xjobs),
                              "lognormal": False, "z": z, "sigma_rel_th": srt,
                              "sigma_tot": st})
                xjobs.append({"kind": "tilt", "which": "fr", "k": k,
                              "sigma_g": sg, "sigma_th_abs": srt * k,
                              "m": z * st, "d_adv": 1, "n": n_is,
                              "seed": 8000 + len(xjobs), "lognormal": False,
                              "z": z, "sigma_rel_th": srt, "sigma_tot": st})
    xres = pool.map(_l1_worker, xjobs, chunksize=1)
    xcheck = {}
    for r in xres:
        key = (r["k"], r["sigma_g"], r["z"])
        d = xcheck.setdefault(key, {"k": r["k"], "sigma_g": r["sigma_g"],
                                    "z": r["z"], "sigma_tot": r["sigma_tot"]})
        if r["kind"] == "plain":
            d["plain"] = r["eps_fr_mc"]
            d["n_fr"] = r["n_fr"]
            d["mean_s"] = r["mean_s"]
            d["std_s"] = r["std_s"]
        else:
            d["is"] = r["est"]
            d["is_rse"] = r["rse"]
    for d in xcheck.values():
        d["analytic"] = float(q_tail(d["z"]))
        d["is_over_plain"] = d["is"] / d["plain"] if d.get("plain") else float("nan")
        d["std_s_over_sigma_tot"] = d["std_s"] / d["sigma_tot"]
    out["is_crosscheck"] = list(xcheck.values())

    # ---- 3) K2: lognormal mismatch prefactor --------------------------------
    pjobs = []
    for k in K_GRID:
        for sg in (0.01, SIGMA_G_MOS, 0.05):
            srt = 0.0     # isolate the mismatch tail shape
            st, _, _ = sigma_total(k, sg, srt)
            for z in (2.0, 3.0, 4.0, 5.0, 6.0):
                for which in ("fr", "fa"):
                    pjobs.append({"kind": "tilt", "which": which, "k": k,
                                  "sigma_g": sg, "sigma_th_abs": 0.0,
                                  "m": (z * st if which == "fr" else 2.0 - z * st),
                                  "d_adv": 1, "n": n_is,
                                  "seed": 9000 + len(pjobs), "lognormal": True,
                                  "z": z, "sigma_tot": st})
    pres = pool.map(_l1_worker, pjobs, chunksize=1)
    for r in pres:
        an_fr, an_fa = analytic_rates(r["k"], r["m"], r["sigma_tot"])
        an = an_fr if r["which"] == "fr" else an_fa
        out["prefactor"].append({"k": r["k"], "sigma_g": r["sigma_g"],
                                 "which": r["which"], "z": r["z"],
                                 "eps_lognormal": r["est"], "rse": r["rse"],
                                 "eps_gaussian_analytic": an,
                                 "ratio": (r["est"] / an) if an > 0 else float("nan")})

    # ---- 4) K3: required margin at the MOS-typical corner -------------------
    k3 = {"criterion_literal": "m*(eps=1e-9) > 0.25*k",
          "criterion_feasibility": "m*(eps=1e-9) > 1.0 score unit (symmetric threshold, d_min=1)",
          "rows": []}
    for k in K_GRID:
        for sg in (SIGMA_G_MOS,) + SIGMA_G_GRID:
            for srt in (0.0,) + SIGMA_REL_TH_GRID:
                st, st_mis, st_th = sigma_total(k, sg, srt)
                row = {"k": k, "sigma_g": sg, "sigma_rel_th": srt,
                       "sigma_tot": st, "sigma_mis": st_mis, "sigma_th": st_th}
                for eps in EPS_TARGETS:
                    z = float(z_for_eps(eps))
                    m_star = z * st
                    row[f"m_eps{eps:g}"] = m_star
                    row[f"m_over_k_eps{eps:g}"] = m_star / k
                    row[f"literal_fires_eps{eps:g}"] = bool(m_star > 0.25 * k)
                    row[f"feasible_eps{eps:g}"] = bool(m_star <= 1.0)
                k3["rows"].append(row)
    # largest workable k / sigma_G boundary
    bounds = []
    for eps in EPS_TARGETS:
        z = float(z_for_eps(eps))
        e = {"eps": eps, "z": z, "sigma_tot_max": 1.0 / z}
        for k in K_GRID:
            e[f"sigma_g_max_k{k}"] = (1.0 / z) / np.sqrt(k)
        e["k_max_at_sigma_g_3pct"] = ((1.0 / z) / SIGMA_G_MOS) ** 2
        for srt in SIGMA_REL_TH_GRID:
            e[f"k_max_thermal_{srt:g}"] = (1.0 / z) / srt
        bounds.append(e)
    k3["bounds"] = bounds
    out["k3"] = k3

    # ---- 5) design fix: minimum code distance -------------------------------
    for k in K_GRID:
        for d_min in (1, 2, 4, 8):
            if d_min > k // 2:
                continue
            for eps in (1e-9,):
                z = float(z_for_eps(eps))
                st_max = d_min / z
                out["code_distance"].append(
                    {"k": k, "d_min": d_min, "eps": eps,
                     "sigma_tot_max": st_max,
                     "sigma_g_max": st_max / np.sqrt(k),
                     "rate_bits_per_line": None})

    # ---- 5b) gain-accuracy requirement -------------------------------------
    # A fractional common-mode gain error g on the match line shifts the score
    # mean by g*k -- and k/sigma_tot = sqrt(k)/sigma_G is enormous, so eps is
    # exponentially sensitive to gain. dln(eps)/dg = lambda(z)*k/sigma_tot with
    # lambda(z) = phi(z)/Q(z). Requiring eps to stay within 2x of nominal gives
    # g_max = ln2*sigma_tot/(k*lambda(z)). Verified numerically below.
    gain = []
    for k in K_GRID:
        for sg in (0.01, SIGMA_G_MOS, 0.05):
            for srt in (0.0, 0.01):
                st, _, _ = sigma_total(k, sg, srt)
                for eps in EPS_TARGETS:
                    z = float(z_for_eps(eps))
                    lam = float(norm.pdf(z) / q_tail(z))
                    g_max = np.log(2.0) * st / (k * lam)
                    # numeric check: eps with the mean shifted by g_max*k
                    eps_shift = float(q_tail(z - g_max * k / st))
                    gain.append({"k": k, "sigma_g": sg, "sigma_rel_th": srt,
                                 "eps": eps, "z": z, "lambda_z": lam,
                                 "dlneps_dg": lam * k / st,
                                 "g_max_2x": float(g_max),
                                 "g_max_2x_bits": float(-np.log2(g_max)),
                                 "eps_at_g_max": eps_shift,
                                 "eps_ratio_at_g_max": eps_shift / eps})
    out["gain_sensitivity"] = gain

    # ---- 6) energy ----------------------------------------------------------
    for k in K_GRID:
        for sg in (0.01, SIGMA_G_MOS, 0.05):
            for srt in (0.0, 0.01):
                st, _, _ = sigma_total(k, sg, srt)
                for eps in EPS_TARGETS:
                    z = float(z_for_eps(eps))
                    m_star = z * st
                    dv = m_star * V_FS / k
                    e_margin = C_ML * dv ** 2
                    e_line = C_ML * V_FS ** 2
                    e_bound = KT * np.log(1.0 / eps)
                    out["energy"].append({
                        "k": k, "sigma_g": sg, "sigma_rel_th": srt, "eps": eps,
                        "m_star": m_star, "dV_margin_V": dv,
                        "E_margin_J": e_margin, "E_line_J": e_line,
                        "E_bound_J": e_bound,
                        "ratio_margin_bound": e_margin / e_bound,
                        "ratio_line_bound": e_line / e_bound,
                        "closed_form_ratio": 2.0 * C_ML * V_FS ** 2 * sg ** 2 / (k * KT)
                        if srt == 0.0 else float("nan")})
    return out


# ---------------------------------------------------------------------------
# L1b -- static mismatch: yield view
# ---------------------------------------------------------------------------
def _yield_worker(job):
    k, sg, srt, m, n_rows, seed = (job["k"], job["sigma_g"], job["sigma_rel_th"],
                                   job["m"], job["n_rows"], job["seed"])
    rng = np.random.default_rng(seed)
    sth = srt * k
    g = 1.0 + rng.normal(0.0, sg, size=(n_rows, k))
    off = g.sum(axis=1) - k                     # per-row static score offset
    if sth > 0:
        eps_row_fr = q_tail((m - off) / sth)
        eps_row_fa = q_tail((off - 2.0 * g[:, 0] + m) / sth)
    else:
        eps_row_fr = (off <= -m).astype(float)
        eps_row_fa = (off - 2.0 * g[:, 0] > -m).astype(float)
    qs = [50, 90, 99, 99.9]
    return {"k": k, "sigma_g": sg, "sigma_rel_th": srt, "m": m,
            "m_over_sigma_tot": m / sigma_total(k, sg, srt)[0],
            "fleet_mean_fr": float(eps_row_fr.mean()),
            "fleet_mean_fa": float(eps_row_fa.mean()),
            "fr_pct": {str(q): float(np.percentile(eps_row_fr, q)) for q in qs},
            "frac_rows_worse_than_1e3": float((eps_row_fr > 1e-3).mean()),
            "frac_rows_worse_than_1e6": float((eps_row_fr > 1e-6).mean())}


def run_l1_yield(pool, n_rows=200_000):
    jobs = []
    for k in K_GRID:
        for sg in (0.01, SIGMA_G_MOS):
            for srt in (0.005, 0.01):
                st, _, _ = sigma_total(k, sg, srt)
                for z in (3.0, 4.0, 6.0):
                    jobs.append({"k": k, "sigma_g": sg, "sigma_rel_th": srt,
                                 "m": z * st, "n_rows": n_rows,
                                 "seed": 12000 + len(jobs)})
    return pool.map(_yield_worker, jobs, chunksize=1)


# ---------------------------------------------------------------------------
# L2 -- Hopfield
# ---------------------------------------------------------------------------
def hopfield_J(N, P, rng, sigma_g=0.0, dtype=np.float32):
    xi = rng.integers(0, 2, size=(P, N)).astype(dtype) * 2 - 1
    J = (xi.T @ xi) / np.float32(N)
    if sigma_g > 0:
        J *= (1.0 + rng.normal(0.0, sigma_g, size=J.shape)).astype(dtype)
        J = 0.5 * (J + J.T)              # keep the energy function well defined
    np.fill_diagonal(J, 0.0)
    return J, xi


def async_zeroT(J, s, rng, max_sweeps=30):
    """Asynchronous zero-temperature updates until a fixed point."""
    N = s.shape[0]
    h = J @ s
    for _ in range(max_sweeps):
        changed = 0
        for i in rng.permutation(N):
            new = 1.0 if h[i] >= 0 else -1.0
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                h += J[:, i] * d
                changed += 1
        if changed == 0:
            break
    return s


def _ags_worker(job):
    N, alpha, rep = job["N"], job["alpha"], job["rep"]
    P = max(1, int(round(alpha * N)))
    rng = np.random.default_rng(job["seed"])
    J, xi = hopfield_J(N, P, rng, sigma_g=job.get("sigma_g", 0.0))
    s = xi[0].copy()
    s = async_zeroT(J, s, rng)
    m = float(np.dot(s, xi[0]) / N)
    # AGS retrieval state has m ~= 0.967 just below alpha_c and collapses
    # discontinuously above it; 0.9 is the standard retrieval-quality cut.
    return {"N": N, "alpha": alpha, "P": P, "rep": rep, "overlap": m,
            "sigma_g": job.get("sigma_g", 0.0), "success": bool(m > 0.9),
            "success_loose": bool(m > 0.5)}


def _l2_ags_chunk(jobs):
    return [_ags_worker(j) for j in jobs]


def _alpha_c(pts, key):
    for p, q in zip(pts[:-1], pts[1:]):
        if p[key] >= 0.5 > q[key]:
            f = (p[key] - 0.5) / (p[key] - q[key])
            return p["alpha"] + f * (q["alpha"] - p["alpha"])
    return None


def run_l2_ags(pool, Ns=(1000, 2000, 4000), reps=24, big_pool=None):
    alphas = (0.05, 0.08, 0.10, 0.115, 0.125, 0.130, 0.135, 0.138, 0.142,
              0.146, 0.150, 0.155, 0.160, 0.170, 0.185, 0.20)
    jobs = []
    big = []
    for N in Ns:
        for a in alphas:
            for r in range(reps):
                j = {"N": N, "alpha": a, "rep": r, "seed": 31000 + len(jobs) + len(big)}
                (big if (N >= 8000 and big_pool is not None) else jobs).append(j)
    res = pool.map(_l2_ags_chunk, [jobs[i::128] for i in range(128)], chunksize=1)
    res = [x for sub in res for x in sub]
    if big:
        res += [x for sub in big_pool.map(
            _l2_ags_chunk, [big[i::16] for i in range(16)], chunksize=1)
            for x in sub]
    curves = {}
    for r in res:
        curves.setdefault(r["N"], {}).setdefault(r["alpha"], []).append(r)
    summary = []
    for N, per_a in sorted(curves.items()):
        pts = []
        for a in sorted(per_a):
            rs = per_a[a]
            pts.append({"alpha": a,
                        "success_frac": float(np.mean([x["success"] for x in rs])),
                        "success_frac_loose": float(np.mean(
                            [x["success_loose"] for x in rs])),
                        "mean_overlap": float(np.mean([x["overlap"] for x in rs]))})
        ac = _alpha_c(pts, "success_frac")
        summary.append({"N": N, "points": pts, "alpha_c": ac,
                        "alpha_c_loose": _alpha_c(pts, "success_frac_loose"),
                        "rel_dev_from_AGS": (abs(ac - 0.138) / 0.138)
                        if ac is not None else None})
    # finite-size extrapolation alpha_c(N) = alpha_inf + c/sqrt(N)
    have = [s for s in summary if s["alpha_c"] is not None]
    extrap = None
    if len(have) >= 2:
        xx = np.array([1.0 / np.sqrt(s["N"]) for s in have])
        yy = np.array([s["alpha_c"] for s in have])
        c, a_inf = np.polyfit(xx, yy, 1)
        extrap = {"alpha_c_inf": float(a_inf), "slope": float(c),
                  "rel_dev_from_AGS": float(abs(a_inf - 0.138) / 0.138)}
    return {"alphas": alphas, "reps": reps, "summary": summary,
            "finite_size_extrapolation": extrap,
            "criterion": "final overlap > 0.9 after T=0 async descent from the pattern",
            "ags_reference": 0.138, "tolerance": 0.10}


def _glauber_worker(job):
    """Stationary Glauber run: error rate vs per-neuron barrier DeltaE/kT."""
    N, alpha, T = job["N"], job["alpha"], job["T"]
    P = max(1, int(round(alpha * N)))
    rng = np.random.default_rng(job["seed"])
    J, xi = hopfield_J(N, P, rng, sigma_g=job.get("sigma_g", 0.0),
                       dtype=np.float64)
    target = xi[0].astype(np.float64)
    s = target.copy()
    h = J @ s
    # equilibrate
    for _ in range(job.get("burn", 5)):
        for i in rng.permutation(N):
            p_up = 1.0 / (1.0 + np.exp(-2.0 * h[i] / T))
            new = 1.0 if rng.random() < p_up else -1.0
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                h += J[:, i] * d
    # Barrier of each stored bit, evaluated ONCE in the clean target state:
    # DeltaE_i = 2*h_i^ref*xi_i is the energy cost of flipping bit i away from
    # the pattern. Using the static reference (not the instantaneous state)
    # avoids the circularity of conditioning the barrier on the outcome.
    gap = 2.0 * (J @ target) * target
    mis_sum = np.zeros(N)
    nsamp = job.get("samples", 200)
    for _ in range(nsamp):
        for i in rng.permutation(N):
            p_up = 1.0 / (1.0 + np.exp(-2.0 * h[i] / T))
            new = 1.0 if rng.random() < p_up else -1.0
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                h += J[:, i] * d
        mis_sum += (s * target < 0).astype(float)
    mis = mis_sum / nsamp
    overlap = float(np.dot(s, target) / N)
    # bin by gap/kT
    x = gap / T
    bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                     12.0, 15.0, 20.0, 30.0])
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 30:
            continue
        binned.append({"gap_kT_lo": float(lo), "gap_kT_hi": float(hi),
                       "gap_kT_mean": float(x[sel].mean()),
                       "err_measured": float(mis[sel].mean()),
                       "err_glauber_pred": float(np.mean(
                           1.0 / (1.0 + np.exp(x[sel])))),
                       "n": int(sel.sum())})
    # Arrhenius slope: ln(err) vs gap/kT over the bins that carry statistics
    slope = None
    good = [b for b in binned if b["err_measured"] > 0]
    if len(good) >= 3:
        xx = np.array([b["gap_kT_mean"] for b in good])
        yy = np.log(np.array([b["err_measured"] for b in good]))
        slope = float(np.polyfit(xx, yy, 1)[0])
    return {"N": N, "alpha": alpha, "T": T, "sigma_g": job.get("sigma_g", 0.0),
            "overlap": overlap, "bit_error_rate": float(mis.mean()),
            "mean_gap_kT": float(x.mean()), "arrhenius_slope": slope,
            "n_samples": nsamp, "bins": binned}


def run_l2_thermal(pool, N=1500):
    jobs = []
    for alpha in (0.05, 0.10, 0.14):
        for T in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            jobs.append({"N": N, "alpha": alpha, "T": T,
                         "seed": 41000 + len(jobs)})
    return pool.map(_glauber_worker, jobs, chunksize=1)


def _retention_worker(job):
    """Time to escape the target attractor at temperature T.

    This is the direct test of the Table-4 row's t_hold term: if the retrieval
    state is a barrier-protected symbol, the escape time must be Arrhenius,
    tau = (1/f0)*exp(DeltaE/kT), so ln(tau) is linear in 1/T with slope
    DeltaE. The row's condition DeltaE/kT > ln(f0*t_hold/eps) is then a
    rearrangement of tau > t_hold/eps.
    """
    N, alpha, T = job["N"], job["alpha"], job["T"]
    P = max(1, int(round(alpha * N)))
    rng = np.random.default_rng(job["seed"])
    J, xi = hopfield_J(N, P, rng, dtype=np.float64)
    target = xi[0].astype(np.float64)
    s = target.copy()
    h = J @ s
    max_sweeps = job.get("max_sweeps", 1500)
    escaped_at = None
    for sw in range(1, max_sweeps + 1):
        for i in rng.permutation(N):
            p_up = 1.0 / (1.0 + np.exp(-2.0 * h[i] / T))
            new = 1.0 if rng.random() < p_up else -1.0
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                h += J[:, i] * d
        if np.dot(s, target) / N < 0.5:
            escaped_at = sw
            break
    return {"N": N, "alpha": alpha, "T": T, "rep": job["rep"],
            "escape_sweeps": escaped_at, "censored": escaped_at is None,
            "max_sweeps": max_sweeps}


def run_l2_retention(pool, N=1000, reps=12, max_sweeps=4000):
    # the escape window sits at a different T for each load, so each alpha
    # gets its own grid (a shared grid is either all-censored or all-instant)
    grids = {0.05: (0.50, 0.53, 0.56, 0.60, 0.64, 0.68),
             0.10: (0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.45, 0.50)}
    jobs = []
    for alpha, Ts in grids.items():
        for T in Ts:
            for r in range(reps):
                jobs.append({"N": N, "alpha": alpha, "T": T, "rep": r,
                             "max_sweeps": max_sweeps,
                             "seed": 71000 + len(jobs)})
    res = [x for sub in pool.map(_retention_worker_chunk,
                                 [jobs[i::64] for i in range(64)], chunksize=1)
           for x in sub]
    agg = {}
    for r in res:
        agg.setdefault((r["alpha"], r["T"]), []).append(r)
    rows = []
    for (a, T), rs in sorted(agg.items()):
        esc = [r["escape_sweeps"] for r in rs if r["escape_sweeps"] is not None]
        # Escape is a Poisson process, so the censoring-unbiased MLE of the
        # mean escape time is (total observed sweeps)/(number of escapes);
        # a median over only the escaped runs is badly biased when most runs
        # are censored, which is what the first pass produced.
        total_time = float(sum(r["escape_sweeps"] if r["escape_sweeps"] is not None
                               else r["max_sweeps"] for r in rs))
        rows.append({"alpha": a, "T": T, "n": len(rs), "n_escaped": len(esc),
                     "censor_frac": 1 - len(esc) / len(rs),
                     "tau_mle_sweeps": (total_time / len(esc)) if esc else None,
                     "tau_mle_rel_se": (1.0 / np.sqrt(len(esc))) if esc else None,
                     "median_escape_sweeps": float(np.median(esc)) if esc else None})
    fits = []
    for a in grids:
        pts = [r for r in rows if r["alpha"] == a
               and r["tau_mle_sweeps"] is not None and r["n_escaped"] >= 3]
        if len(pts) >= 3:
            x = np.array([1.0 / r["T"] for r in pts])
            y = np.log(np.array([r["tau_mle_sweeps"] for r in pts]))
            sl, ic = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - (sl * x + ic)) ** 2) / np.sum((y - y.mean()) ** 2)
            fits.append({"alpha": a, "n_points": len(pts),
                         "barrier_DeltaE": float(sl), "ln_prefactor": float(ic),
                         "r2": float(r2)})
    return {"rows": rows, "arrhenius_fits": fits,
            "note": "escape_sweeps is in units of one full async sweep = 1/f0"}


def _retention_worker_chunk(jobs):
    return [_retention_worker(j) for j in jobs]


def run_l2_mismatch(pool, N=1500, reps=8):
    jobs = []
    for alpha in (0.05, 0.10, 0.14):
        for sg in (0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.40):
            for r in range(reps):
                jobs.append({"N": N, "alpha": alpha, "rep": r, "sigma_g": sg,
                             "seed": 51000 + len(jobs)})
    res = pool.map(_ags_worker, jobs, chunksize=1)
    agg = {}
    for r in res:
        key = (r["alpha"], r["sigma_g"])
        agg.setdefault(key, []).append(r)
    out = []
    for (a, sg), rs in sorted(agg.items()):
        out.append({"alpha": a, "sigma_g": sg, "N": N,
                    "success_frac": float(np.mean([x["success"] for x in rs])),
                    "mean_overlap": float(np.mean([x["overlap"] for x in rs])),
                    "bit_error_rate": float(np.mean(
                        [(1 - x["overlap"]) / 2 for x in rs]))})
    return out


def run_l2_alpha_renorm(pool, N=4000, reps=16):
    """Test the law suggested by run_l2_mismatch: multiplicative synaptic
    mismatch sigma_G renormalises the load, alpha_eff = alpha*(1+sigma_G^2),
    so the measured capacity should scale as alpha_c(sigma_G) =
    alpha_c(0)/(1+sigma_G^2) rather than collapsing.
    """
    alphas = (0.10, 0.115, 0.125, 0.135, 0.142, 0.150, 0.160, 0.170, 0.185)
    sigmas = (0.0, 0.20, 0.40, 0.60)
    jobs = []
    for sg in sigmas:
        for a in alphas:
            for r in range(reps):
                jobs.append({"N": N, "alpha": a, "rep": r, "sigma_g": sg,
                             "seed": 61000 + len(jobs)})
    res = [x for sub in pool.map(_l2_ags_chunk,
                                 [jobs[i::96] for i in range(96)], chunksize=1)
           for x in sub]
    agg = {}
    for r in res:
        agg.setdefault(r["sigma_g"], {}).setdefault(r["alpha"], []).append(r)
    rows = []
    base = None
    for sg in sigmas:
        pts = [{"alpha": a,
                "success_frac": float(np.mean([x["success"] for x in agg[sg][a]]))}
               for a in sorted(agg[sg])]
        ac = _alpha_c(pts, "success_frac")
        if sg == 0.0:
            base = ac
        rows.append({"sigma_g": sg, "N": N, "alpha_c": ac, "points": pts,
                     "alpha_c_predicted": (base / (1 + sg ** 2))
                     if base is not None else None,
                     "rel_dev": (abs(ac - base / (1 + sg ** 2))
                                 / (base / (1 + sg ** 2)))
                     if (ac is not None and base is not None) else None})
    return {"law": "alpha_c(sigma_G) = alpha_c(0)/(1+sigma_G^2)", "rows": rows}


# ---------------------------------------------------------------------------
# L3 -- ngspice aCAM row
# ---------------------------------------------------------------------------
K_L3 = 16
SIGMA_G_L3 = 0.03
N_MC_L3 = 200
R0 = 100e3          # nominal cell resistance -> 1 uA per cell at V_REF
V_REF = 0.1         # per-line drive [V]  (keeps cell currents at uA scale)
RF = 100e3          # transimpedance feedback resistor
A0 = 1e4            # finite amplifier gain (deliberate non-ideality)


def build_l3_netlist(gmult, queries, m_margin):
    """gmult: (n_mc, k) conductance multipliers. queries: (n_mc, k) in {+-1}.

    Each row sums 16 currents V_i/R_i into a virtual ground formed by a
    finite-gain VCVS with feedback resistor RF. Output V_out = -RF*I_sum
    (up to the 1/A0 gain error), so the score s = -V_out/(V_REF*RF/R0).
    A behavioral comparator flags a match against theta = k - m.
    """
    L = [".title aCAM row: k=16 lognormal-mismatch match line",
         ".options gmin=1e-12 reltol=1e-10 abstol=1e-15 vntol=1e-12"]
    scale = V_REF * RF / R0             # volts per score unit at the output
    for r in range(gmult.shape[0]):
        n_in = f"in{r}_"
        n_vg = f"vg{r}"
        n_o = f"o{r}"
        for i in range(K_L3):
            v = V_REF * queries[r, i]
            L.append(f"V{n_in}{i} {n_in}{i} 0 DC {v:.10e}")
            rr = R0 / gmult[r, i]
            L.append(f"R{n_in}{i} {n_in}{i} {n_vg} {rr:.10e}")
        # finite-gain inverting amp: Vout = -A0 * V(vg)
        L.append(f"E{n_o} {n_o} 0 0 {n_vg} {A0:.6e}")
        L.append(f"Rf{r} {n_vg} {n_o} {RF:.10e}")
        # score node (1 V == 1 score unit) and comparator
        L.append(f"Bs{r} score{r} 0 V = -V({n_o})/{scale:.10e}")
        theta = K_L3 - m_margin
        L.append(f"Bc{r} cmp{r} 0 V = (V(score{r}) > {theta:.10e}) ? 1 : 0")
        L.append(f"Rl{r} score{r} 0 1e12")
        L.append(f"Rc{r} cmp{r} 0 1e12")
    L.append(".control")
    L.append("op")
    cols = " ".join(f"v(score{r}) v(cmp{r})" for r in range(gmult.shape[0]))
    L.append(f"wrdata l3_scores.txt {cols}")
    L.append(".endc")
    L.append(".end")
    return "\n".join(L) + "\n"


def run_ngspice(netlist, tag, timeout=600):
    WORKDIR.mkdir(exist_ok=True, parents=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=timeout, cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-4000:])
        sys.stderr.write(res.stderr[-4000:])
        raise RuntimeError(f"ngspice failed for {tag}")
    return res.stdout


def run_l3():
    rng = np.random.default_rng(SEED)
    s_ln = float(np.sqrt(np.log1p(SIGMA_G_L3 ** 2)))
    nu = rng.normal(0.0, s_ln, size=(N_MC_L3, K_L3))
    gmult = np.exp(nu - 0.5 * s_ln ** 2)          # E[g]=1, std[g]=3 %

    st, _, _ = sigma_total(K_L3, SIGMA_G_L3, 0.0)
    # margin chosen so the Hamming-1 false-accept rate is ~0.1, i.e. ~20
    # counted events in 200 draws (deep tails are not reachable at n=200 and
    # are reported analytically instead, as pre-registered)
    m_margin = float(2.0 - z_for_eps(0.1) * st)

    # every mismatch draw is instantiated twice: once queried with the stored
    # word (true match) and once with its Hamming-1 neighbour (adversary)
    gmult = np.vstack([gmult, gmult])
    n_row = gmult.shape[0]
    queries = np.ones((n_row, K_L3))
    adv = np.zeros(n_row, dtype=bool)
    adv[N_MC_L3:] = True
    queries[adv, 0] = -1.0

    net = build_l3_netlist(gmult, queries, m_margin)
    t0 = time.time()
    run_ngspice(net, "symbol_margin_l3")
    dt = time.time() - t0

    data = np.loadtxt(WORKDIR / "l3_scores.txt")
    row = data if data.ndim == 1 else data[0]
    vals = row[1::2]
    assert vals.size == 2 * n_row, vals.size
    score_spice = vals[0::2]
    cmp_spice = vals[1::2]

    # paired L1 prediction on the SAME draws
    score_l1 = (queries * gmult).sum(axis=1)

    mm = ~adv
    res = {
        "k": K_L3, "sigma_g": SIGMA_G_L3, "n_mc": N_MC_L3,
        "m_margin": m_margin, "theta": K_L3 - m_margin,
        "R0": R0, "V_REF": V_REF, "RF": RF, "A0": A0,
        "runtime_s": dt,
        "sigma_tot_analytic": st,
        "match": {
            "spice_mean": float(score_spice[mm].mean()),
            "spice_std": float(score_spice[mm].std(ddof=1)),
            "l1_mean": float(score_l1[mm].mean()),
            "l1_std": float(score_l1[mm].std(ddof=1)),
        },
        "adv": {
            "spice_mean": float(score_spice[adv].mean()),
            "spice_std": float(score_spice[adv].std(ddof=1)),
            "l1_mean": float(score_l1[adv].mean()),
            "l1_std": float(score_l1[adv].std(ddof=1)),
        },
        "paired_max_abs_dev": float(np.abs(score_spice - score_l1).max()),
        "paired_rel_rms": float(np.linalg.norm(score_spice - score_l1)
                                / np.linalg.norm(score_l1)),
        "fa_count_spice": int(cmp_spice[adv].sum()),
        "fa_n": int(adv.sum()),
        "fa_rate_spice": float(cmp_spice[adv].mean()),
        "fa_count_l1": int((score_l1[adv] > K_L3 - m_margin).sum()),
        "fr_count_spice": int((1 - cmp_spice[mm]).sum()),
        "fr_rate_spice": float(1 - cmp_spice[mm].mean()),
        "gain_error_predicted": (1.0 + K_L3 * RF / R0) / A0,
    }
    res["sigma_ratio_match"] = res["match"]["spice_std"] / res["match"]["l1_std"]
    res["sigma_ratio_l1_vs_analytic"] = res["match"]["l1_std"] / st
    res["sigma_ratio_spice_vs_analytic"] = res["match"]["spice_std"] / st
    res["gain_error_frac"] = 1.0 / A0

    # analytic reference at this margin
    fr_an, fa_an = analytic_rates(K_L3, m_margin, st)
    res["fa_rate_analytic"] = fa_an
    res["fr_rate_analytic"] = fr_an
    # and the pre-registered eps ~ 1e-3 operating point (not countable at n=200)
    m_1e3 = 2.0 - z_for_eps(1e-3) * st
    res["m_for_fa_1e-3"] = float(m_1e3)
    res["fa_1e-3_check"] = float(analytic_rates(K_L3, m_1e3, st)[1])
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--skip-l3", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    WORKDIR.mkdir(exist_ok=True, parents=True)
    t_all = time.time()
    out = {"meta": {"seed": SEED, "kT_J": KT, "T_K": T_AMB, "C_ML_F": C_ML,
                    "V_FS_V": V_FS, "procs": args.procs}}

    n_is = 100_000 if args.quick else 400_000
    n_plain = 200_000 if args.quick else 1_000_000

    ctx = mp.get_context("fork")
    with ctx.Pool(args.procs) as pool:
        t0 = time.time()
        out["l1"] = run_l1(pool, n_plain=n_plain, n_is=n_is)
        print(f"[L1] {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        out["l1_yield"] = run_l1_yield(pool, n_rows=50_000 if args.quick else 200_000)
        print(f"[L1 yield] {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        Ns = (1000,) if args.quick else (1000, 2000, 4000, 8000)
        # N = 8000 needs ~256 MB of J per worker -> its own small pool
        with ctx.Pool(8) as bigpool:
            out["l2_ags"] = run_l2_ags(pool, Ns=Ns,
                                       reps=8 if args.quick else 24,
                                       big_pool=bigpool)
        print(f"[L2 AGS] {time.time()-t0:.1f}s", flush=True)
        for s in out["l2_ags"]["summary"]:
            print(f"   N={s['N']}  alpha_c={s['alpha_c']}  "
                  f"dev={s['rel_dev_from_AGS']}", flush=True)
        print(f"   extrap {out['l2_ags']['finite_size_extrapolation']}", flush=True)

        t0 = time.time()
        out["l2_thermal"] = run_l2_thermal(pool, N=800 if args.quick else 1500)
        print(f"[L2 thermal] {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        out["l2_mismatch"] = run_l2_mismatch(pool, N=800 if args.quick else 1500,
                                             reps=4 if args.quick else 8)
        print(f"[L2 mismatch] {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        out["l2_retention"] = run_l2_retention(
            pool, N=600 if args.quick else 1000, reps=6 if args.quick else 12)
        print(f"[L2 retention] {time.time()-t0:.1f}s", flush=True)
        for f in out["l2_retention"]["arrhenius_fits"]:
            print(f"   alpha={f['alpha']}  DeltaE={f['barrier_DeltaE']:.3f}  "
                  f"r2={f['r2']:.4f}", flush=True)

        t0 = time.time()
        out["l2_alpha_renorm"] = run_l2_alpha_renorm(
            pool, N=1000 if args.quick else 4000, reps=8 if args.quick else 16)
        print(f"[L2 alpha-renorm] {time.time()-t0:.1f}s", flush=True)
        for r in out["l2_alpha_renorm"]["rows"]:
            print(f"   sigma_G={r['sigma_g']:.2f}  alpha_c={r['alpha_c']}  "
                  f"pred={r['alpha_c_predicted']}  dev={r['rel_dev']}", flush=True)

    if not args.skip_l3:
        if not shutil.which("ngspice"):
            sys.exit("ngspice not found in PATH")
        t0 = time.time()
        out["l3"] = run_l3()
        print(f"[L3 ngspice] {time.time()-t0:.1f}s", flush=True)
        print(f"   sigma ratio spice/L1 = {out['l3']['sigma_ratio_match']:.4f}",
              flush=True)

    out["meta"]["total_runtime_s"] = time.time() - t_all
    path = WORKDIR / "symbol_margin_results.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}  ({out['meta']['total_runtime_s']:.1f}s total)")


if __name__ == "__main__":
    main()
