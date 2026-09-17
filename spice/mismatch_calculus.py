#!/usr/bin/env python3
"""
Mismatch calculus: a composable error-propagation algebra for the prime
compiler, plus its retrodiction against every number this repo has measured.

Part 1 (this file, section A) encodes the characterization tuples and the
composition rules R1-R13 of `docs/mismatch_calculus.md` as code.

Part 2 (section B) applies them to reproduce, *without running ngspice*,
the measured numbers of

  Table 6  (open-loop translinear RMSNorm, 3 corners)        rmsnorm_agc_sim.py
  Table 7  (AGC decomposition, detector-only and VGA-only)   agc_mismatch_sweep.py
  Result 5 (softmax open-loop vs sum-feedback AGC)           softmax_agc_sim.py
  exp_mamba_vcrc.md   K2 (triode / pair under mismatch, + refit)
  exp_symbol_margin.md K3 (feasibility of a k=256 dense code)
  exp_bounded_recursion.md A4 (race logit-error budget)
  exp_mismatch_absorption.md K2/K4 (training floor, perturbative cost)

Everything here is numpy only.  Where the algebra predicts a *distribution*
rather than a number, the distribution is evaluated by cheap Monte Carlo over
the same mismatch model the SPICE harness uses -- the point is that no
circuit is simulated: the junction law has been replaced by its sensitivity.

Usage:  OPENBLAS_NUM_THREADS=1 python3 mismatch_calculus.py [--json out.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SECTION A -- the algebra
# ---------------------------------------------------------------------------

VT_NG = 0.025864890          # ngspice SPICE3-legacy kT/q (spice/README.md)
ISAT = 1e-16                 # diode saturation current of the RMSNorm bench
IUNIT = 1e-6                 # 1 uA = one normalized unit
LAM0 = float(np.log(IUNIT / ISAT))     # 23.0259 -- the log lever at 1 uA

SQRT_2_PI = float(np.sqrt(2.0 / np.pi))
Z50 = 0.6744897501960817     # median of |N(0,1)|


# --- R1  log-domain lever -------------------------------------------------
def log_lever(current, isat=ISAT):
    """lambda = ln(I/Is): the factor by which a junction multiplies relative
    ideality mismatch.  ~23 at uA with Is=1e-16, ~19.1 for the race cell."""
    return np.log(np.asarray(current) / isat)


def junction_sigma(lam, s_is, s_n):
    """R1.  One junction, traversed once (either direction), contributes
        d ln I = +/- ( nu*lambda - iota ),   nu ~ N(0,s_n), iota ~ N(0,s_is)
    so its sigma on the relative current error is hypot(s_is, s_n*lambda)."""
    return np.hypot(s_is, np.asarray(lam) * s_n)


# --- R3  per-channel vs shared path ---------------------------------------
def compose_quadrature(coeffs, sigmas, extra_var=0.0):
    """R3.  Independent mismatch sources compose in quadrature with the
    sensitivity coefficients of the signal-flow graph; no cross terms."""
    c = np.asarray(coeffs, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    return float(np.sqrt(np.sum((c * s) ** 2) + extra_var))


# --- R2  feedback (Bode) suppression --------------------------------------
def feedback_demote(sigma_pc, loop_gain, upstream=False):
    """R2.  A stage at or downstream of the loop input has its per-channel
    exposure demoted to common mode and suppressed by 1/(1+L); a stage
    upstream of the loop input keeps full per-channel exposure."""
    if upstream:
        return sigma_pc, 0.0
    return 0.0, sigma_pc / (1.0 + loop_gain)


# --- R3b  the two canonical N-dependent factors ---------------------------
def energy_weights(x):
    """w_i = x_i^2 / sum_j x_j^2  -- the weight with which channel i enters
    any shared (summed) path.  Appears in every AGC/softmax formula."""
    x = np.asarray(x, dtype=float)
    return x ** 2 / np.sum(x ** 2)


def per_channel_residual_sigma(w, sigma):
    """R3.  sigma_VGA on a per-channel gain, normalised by a shared loop,
    leaves RMS_i residual  sigma * sqrt(1 - sum_i w_i^2)."""
    return sigma * np.sqrt(1.0 - np.sum(np.asarray(w) ** 2))


# --- R4/R5  threshold decisions -------------------------------------------
from math import erf, sqrt as _sqrt


def q_gauss(z):
    return 0.5 * (1.0 - erf(z / _sqrt(2.0)))


def z_of_eps(eps):
    """Gaussian quantile z_eps with Q(z) = eps (bisection, exact enough)."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q_gauss(mid) > eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def margin_required(sigma_tot, eps):
    """R4.  m* = z_eps * sigma_tot, in score units (one bit flip = 2)."""
    return z_of_eps(eps) * sigma_tot


def sigma_max_feasible(k, eps, d_min=1):
    """R4.  sigma_tot <= d_min/z_eps  =>  sigma_G <= d_min/(z_eps*sqrt(k))."""
    return d_min / (z_of_eps(eps) * np.sqrt(k))


def dlneps_dg(k, sigma_tot, eps):
    """R5.  d ln(eps)/dg = lambda(z) * k/sigma_tot, lambda = phi/Q."""
    z = z_of_eps(eps)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return (phi / q_gauss(z)) * k / sigma_tot


# --- R9  static reparametrization -----------------------------------------
def two_terminal_delta_perturb(delta, delta_max, eps_k, dvt, vov_max=1.0):
    """R9a.  A *two-terminal* voltage-controlled resistor carries zero current
    at zero differential voltage whatever its threshold, so a V_t offset is a
    conductance error, never an offset:
        Delta_eff = (1+eps_K) * ( Delta - dVt * Delta_max / V_ov,max )
    clamped at zero (cutoff), which is the one non-reparametrizable part."""
    floor = dvt * delta_max / vov_max
    out = (1.0 + eps_k) * (delta - floor)
    return np.clip(out, 0.0, None)


def three_terminal_offset(v_os, swing, vt=VT_NG):
    """R9b.  A *three-terminal* transconductor pair has an input-referred
    offset: its leak relaxes the state to -V_os, and its input pair injects
    2*VT*tanh(V_os/2VT) of phantom drive.  Both measured in full scales of
    the state swing."""
    return abs(v_os) / swing, 2 * vt * np.tanh(v_os / (2 * vt)) / swing


# ---------------------------------------------------------------------------
# SECTION B -- retrodiction
# ---------------------------------------------------------------------------

RESULTS = {}
ROWS = []      # (item, quantity, measured, predicted, ratio)


def row(item, what, meas, pred):
    r = float(pred) / float(meas) if meas else float("nan")
    ROWS.append((item, what, float(meas), float(pred), r))
    return r


# --- helper: replay the exact RNG streams of the SPICE harnesses -----------

def rms_x_draw():
    """rmsnorm_agc_sim.py: rng = default_rng(2026); x = randn(8)*1.5."""
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(8) * 1.5
    return np.clip(np.abs(x), 1e-3, None)


def agc_cell_draws():
    """agc_mismatch_sweep.py consumes one rng in a fixed order.  Replay it to
    recover the *exact* x vector each cell used (each run_cell draws its own
    x, so Sum w^2 differs per cell -- that is the dominant scatter term)."""
    rng = np.random.default_rng(2026)
    out = {}

    def cell(N, mc, s_is, s_n, s_mir, s_vga, key):
        x = rng.standard_normal(N) * 1.5
        x = np.where(np.abs(x) < 0.05,
                     0.05 * np.sign(x) + (x == 0) * 0.05, x)
        eps_all = []
        for _ in range(mc):
            rng.lognormal(0, s_is, 2 * N + 1)
            rng.normal(1.0, s_n, 2 * N + 1)
            rng.normal(1.0, s_mir)
            eps_all.append(rng.normal(0.0, s_vga, N))
        out[key] = dict(x=x, eps=np.array(eps_all))

    for lbl, si, sn, sm in (("detA_bjt", .01, .0005, .005),
                            ("detA_mos", .03, .003, .01),
                            ("detA_wst", .05, .005, .02)):
        cell(8, 100, si, sn, sm, 0.0, lbl)
    for lbl, sv in (("vgaB_0p5", .005), ("vgaB_1", .01), ("vgaB_2", .02)):
        cell(8, 100, 0.0, 0.0, 0.0, sv, lbl)
    return out


# ===========================================================================
# ITEM 1 -- Table 6: open-loop translinear RMSNorm at three corners
# ===========================================================================
#
# Signal-flow graph of build_translinear_netlist(), written out in the log
# domain.  b_j = nu_j*lambda_j - iota_j is the error of junction j in units of
# n0*VT; forward (I->V) enters with +b, backward (V->I) with -b.
#
#   eps_out,i = (1-w_i) b_x,i - sum_{j!=i} w_j b_x,j        [R1 x R3]
#             - b_o,i + 0.5 sum_j w_j b_sq,j
#             + b_ref - 0.5 b_ms - b_inv + b_invl - 0.5 g_mir
#
# Every term is a junction traversal; the lever lambda_j is the log-argument
# at that device.  R3 then adds them in quadrature.

CORNERS_T6 = [("BJT-grade", 0.01, 0.0005, 0.005, 2.2),
              ("MOS typical", 0.03, 0.003, 0.01, 11.4),
              ("MOS worst", 0.05, 0.005, 0.02, 18.7)]


def item1_table6():
    xa = rms_x_draw()
    N = len(xa)
    ms = float(np.mean(xa ** 2))
    inv = 1.0 / np.sqrt(ms)
    ref = xa * inv
    w = energy_weights(xa)

    lam_x = LAM0 + np.log(xa)
    lam_sq = LAM0 + np.log(xa ** 2)
    lam_ref = LAM0
    lam_ms = LAM0 + np.log(ms)
    lam_inv = LAM0 + np.log(inv)
    lam_o = LAM0 + np.log(ref)

    out = {}
    for label, s_is, s_n, s_mir, meas in CORNERS_T6:
        sj = lambda lam: junction_sigma(lam, s_is, s_n)
        var_i = np.zeros(N)
        for i in range(N):
            c_x = -w.copy()
            c_x[i] += 1.0
            var = np.sum((c_x * sj(lam_x)) ** 2)            # b_x,j
            var += np.sum((0.5 * w * sj(lam_sq)) ** 2)      # b_sq,j
            var += (sj(lam_o[i])) ** 2                      # b_o,i
            var += (sj(lam_ref)) ** 2                       # b_ref
            var += (0.5 * sj(lam_ms)) ** 2                  # b_ms
            var += 2.0 * (sj(lam_inv)) ** 2                 # b_inv, b_invl
            var += (0.5 * s_mir) ** 2                       # mirror
            var_i[i] = var
        # metric of the harness: mean_i |out_i - ref_i| with RMS(ref)=1
        pred = float(np.mean(ref * SQRT_2_PI * np.sqrt(var_i)))
        out[label] = dict(measured_pct=meas, predicted_pct=pred * 100)
        row("1 Table 6", f"{label} mean err", meas, pred * 100)
    RESULTS["item1_table6"] = out
    return out


# ===========================================================================
# ITEM 2 -- Table 7: AGC decomposition
# ===========================================================================
#
# (a) VGA-only.  With the loop closed, y = G x(1+eps), G = 1/RMS(x(1+eps)).
#     decompose_error() divides out the best-fit scalar alpha, so
#         resid_i = y0_i (eps_i - sum_j w_j eps_j)
#         per-channel = sqrt( sum_i w_i (eps_i - eps_bar_w)^2 )   [R3]
#     and the *common-mode* residue is second order,
#         alpha - 1 = -0.5 sum_i w_i eps_i^2 + 0.5 (sum_i w_i eps_i)^2,
#     which sits under the bisection floor of settle_gain (16 halvings of
#     [1e-3,1e3] => quantization q = ln(1e6)/2^16 = 2.108e-4 in ln G).
#
# (b) Detector-only.  The detector is a *shared scalar* path (R3), so
#         d ln G = -0.5 [ g_mir - b_ref + 2 sum_i w_i b_y,i - sum_i w_i b_sq,i ]
#     and the per-channel residual is identically zero (R2/R3).

BISECT_Q = float(np.log(1e6) / 2 ** 16)


def item2_table7():
    cells = agc_cell_draws()
    out = {}

    # --- (b) detector-only ------------------------------------------------
    for key, label, s_is, s_n, s_mir, meas_cm in (
            ("detA_bjt", "det BJT", .01, .0005, .005, 1.0),
            ("detA_mos", "det MOS typ", .03, .003, .01, 3.0),
            ("detA_wst", "det MOS worst", .05, .005, .02, 8.1)):
        x = cells[key]["x"]
        w = energy_weights(x)
        ref = np.abs(x) / np.sqrt(np.mean(x ** 2))       # |y| at equilibrium
        lam_y = LAM0 + np.log(np.maximum(ref, 1e-4))
        lam_sq = LAM0 + np.log(np.maximum(ref, 1e-4) ** 2)
        sj = lambda lam: junction_sigma(lam, s_is, s_n)
        var = 0.25 * (s_mir ** 2 + sj(LAM0) ** 2
                      + np.sum((2 * w * sj(lam_y)) ** 2)
                      + np.sum((w * sj(lam_sq)) ** 2))
        var += BISECT_Q ** 2 / 12.0
        pred = Z50 * np.sqrt(var) * 100
        out[label] = dict(measured_cm_pct=meas_cm, predicted_cm_pct=pred,
                          measured_pc_pct=0.0, predicted_pc_pct=0.0,
                          sum_w2=float(np.sum(w ** 2)))
        row("2 Table 7", f"{label} common-mode", meas_cm, pred)

    # --- (a) VGA-only -----------------------------------------------------
    for key, label, sv, meas_pc, meas_cm in (
            ("vgaB_0p5", "VGA 0.5%", .005, 0.37, 0.006),
            ("vgaB_1", "VGA 1%", .01, 0.69, 0.006),
            ("vgaB_2", "VGA 2%", .02, 1.66, 0.015)):
        x = cells[key]["x"]
        eps = cells[key]["eps"]                 # exact eps draws of the run
        w = energy_weights(x)
        ebar = eps @ w
        pc = np.sqrt(((eps - ebar[:, None]) ** 2) @ w)
        # alpha from the closed-form loop equilibrium + bisection quantization
        alpha = (1.0 + ebar) / np.sqrt(1.0 + 2 * ebar + (eps ** 2) @ w)
        qz = np.random.default_rng(7).uniform(-BISECT_Q / 2, BISECT_Q / 2,
                                              len(alpha))
        cm = np.abs(alpha * np.exp(qz) - 1.0)
        out[label] = dict(measured_pc_pct=meas_pc,
                          predicted_pc_pct=float(np.median(pc)) * 100,
                          measured_cm_pct=meas_cm,
                          predicted_cm_pct=float(np.median(cm)) * 100,
                          closed_form_pc_pct=per_channel_residual_sigma(w, sv)
                          * 100,
                          sum_w2=float(np.sum(w ** 2)))
        row("2 Table 7", f"{label} per-channel", meas_pc,
            float(np.median(pc)) * 100)
        row("2 Table 7", f"{label} common-mode", meas_cm,
            float(np.median(cm)) * 100)
    RESULTS["item2_table7"] = out
    return out


# ===========================================================================
# ITEM 3 -- Result 5: softmax, open-loop vs sum-feedback AGC
# ===========================================================================
#
# Both share the exponential stage (R2: it sits *upstream* of the loop input
# and keeps full per-channel exposure).  The open-loop path adds three more
# junction traversals per channel (log, out) plus two shared (sum, ref) that
# are NOT removed, because the open-loop softmax never renormalises:
#
#   open: Delta_i = (e_i - sum_j p_j e_j) + b_l,i - b_o,i + b_ref - b_sum
#   AGC : delta_i = (e_i + eps_i) - sum_j p_j (e_j + eps_j)
#
#   E[L1] = sum_i p_i E|.| = sqrt(2/pi) * sum_i p_i * sd_i        [R3]

CORNERS_SM = [("BJT-grade", .01, .0005, .005, 3.1),
              ("MOS typical", .03, .003, .01, 3.3),
              ("MOS worst", .05, .005, .02, 3.4)]
SM_MEAS_ABS = {"MOS typical": (13.6, 4.1)}
BIAS_SM = 23.0


def item3_softmax():
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(8) * 2.0
    p = np.exp(x) / np.exp(x).sum()
    N = len(x)
    sw = float(np.sum(p ** 2))
    S = 1.0 - 2 * p + sw                     # (1-p_i)^2 + sum_{j!=i} p_j^2

    lam_e = BIAS_SM + x                      # exp stage, driven at (BIAS+x)*VT
    i_i = ISAT * np.exp(lam_e)
    lam_l = np.log(i_i / ISAT)
    lam_sum = np.log(i_i.sum() / ISAT)
    lam_ref = np.log(1e-6 / ISAT)
    lam_o = np.log(1e-6 * p / ISAT)

    out = {}
    for label, s_is, s_n, s_vga, meas_ratio in CORNERS_SM:
        sj = lambda lam: junction_sigma(lam, s_is, s_n)
        var_open = (sj(lam_e) ** 2) * S + sj(lam_l) ** 2 + sj(lam_o) ** 2 \
            + sj(lam_ref) ** 2 + sj(lam_sum) ** 2
        var_agc = (sj(lam_e) ** 2 + s_vga ** 2) * S
        l1_open = SQRT_2_PI * float(np.sum(p * np.sqrt(var_open)))
        l1_agc = SQRT_2_PI * float(np.sum(p * np.sqrt(var_agc)))
        ratio = l1_open / l1_agc
        out[label] = dict(measured_ratio=meas_ratio, predicted_ratio=ratio,
                          predicted_L1_open_pct=l1_open * 100,
                          predicted_L1_agc_pct=l1_agc * 100)
        row("3 Result 5", f"{label} AGC/open ratio", meas_ratio, ratio)
        if label in SM_MEAS_ABS:
            mo, ma = SM_MEAS_ABS[label]
            out[label]["measured_L1_open_pct"] = mo
            out[label]["measured_L1_agc_pct"] = ma
            row("3 Result 5", f"{label} L1 open-loop", mo, l1_open * 100)
            row("3 Result 5", f"{label} L1 AGC", ma, l1_agc * 100)
    RESULTS["item3_softmax"] = out
    return out


# ===========================================================================
# ITEM 4 -- Mamba VCRC under 3 % K' / 10 mV V_t
# ===========================================================================
# R9 applied to the two device classes, evaluated on the *same* Delta/x
# sequences and the same mismatch RNG stream as mamba_vcrc_sim.py, but with
# the circuit replaced by its characterization tuple.  No ngspice.

C_STATE, T_TOK, VOV_MAX, SWING, X_SWING = 1e-12, 10e-9, 1.0, 4e-3, 4e-3
A_CH = (-1.0, -4.0, -16.0)
D_RANGES = ((1e-3, 1e-1), (1e-3, 1e0), (1e-3, 1e1))
MAMBA_MEAS = {  # (variant, dmin, dmax, A) -> (p50 %, refit p50 %)
    ("triode", 1e-3, 1e-1, -1.0): (3.34, 1.04),
    ("triode", 1e-3, 1e-1, -4.0): (5.10, 2.88),
    ("triode", 1e-3, 1e-1, -16.0): (7.43, 8.35),
    ("triode", 1e-3, 1e0, -1.0): (6.03, 5.69),
    ("triode", 1e-3, 1e0, -4.0): (8.99, 9.86),
    ("triode", 1e-3, 1e0, -16.0): (13.73, 14.72),
    ("triode", 1e-3, 1e1, -1.0): (12.78, 14.76),
    ("triode", 1e-3, 1e1, -4.0): (16.70, 18.73),
    ("triode", 1e-3, 1e1, -16.0): (18.79, 21.18),
    ("pair", 1e-3, 1e-1, -1.0): (2122.0, 14.5),
    ("pair", 1e-3, 1e-1, -4.0): (945.0, 4.8),
    ("pair", 1e-3, 1e-1, -16.0): (723.0, 13.1),
    ("pair", 1e-3, 1e0, -1.0): (710.0, 8.6),
    ("pair", 1e-3, 1e0, -4.0): (860.0, 32.8),
    ("pair", 1e-3, 1e0, -16.0): (1022.0, 43.6),
    ("pair", 1e-3, 1e1, -1.0): (1102.0, 42.5),
    ("pair", 1e-3, 1e1, -4.0): (996.0, 61.7),
    ("pair", 1e-3, 1e1, -16.0): (959.0, 102.9),
}


def _softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _make_delta(x, marker, dmin, dmax):
    z = -3.0 + 3.0 * np.abs(x) + 9.0 * marker
    sp = _softplus(z)
    lo, hi = float(sp.min()), float(sp.max())
    u = (sp - lo) / (hi - lo) if hi > lo else np.zeros_like(sp)
    return dmin + (dmax - dmin) * u


def _kappa(A, delta):
    u = np.abs(A) * delta
    return np.where(u < 1e-9, 1.0 + u / 2.0, u / (1.0 - np.exp(-u)))


def _recurrence(abar, bbar, x):
    h = np.zeros(len(x))
    s = 0.0
    for t in range(len(x)):
        s = abar[t] * s + bbar[t] * x[t]
        h[t] = s
    return h


def _mc_gauss_seq(n_tok=128):
    rng = np.random.default_rng(2027)      # seed + 1 in mamba_vcrc_sim.py
    x = rng.standard_normal(n_tok)
    return x / np.max(np.abs(x)), np.zeros(n_tok)


def item4_mamba(mc_runs=100):
    x, m = _mc_gauss_seq()
    cells = []
    for (dmin, dmax) in D_RANGES:
        delta = _make_delta(x, m, dmin, dmax)
        for A in A_CH:
            kap = _kappa(A, delta)
            ref = _recurrence(np.exp(delta * A), delta * 1.0, x)
            peak = max(np.max(np.abs(ref)), 1e-12)
            cells.append(dict(A=A, dmin=dmin, dmax=dmax, delta=delta,
                              kappa=kap, ref=ref, s_h=SWING / peak,
                              s_x=X_SWING / np.max(np.abs(x))))
    nc = len(cells)
    rng = np.random.default_rng(2126)      # seed + 100
    draws = [dict(kp_l=rng.normal(1, .03, nc), kp_i=rng.normal(1, .03, nc),
                  vt_l=rng.normal(0, .010, nc), vt_i=rng.normal(0, .010, nc))
             for _ in range(mc_runs)]

    out, tri, pai = {}, [], []
    for k, cl in enumerate(cells):
        A, d, kap, ref = cl["A"], cl["delta"], cl["kappa"], cl["ref"]
        dmax = cl["dmax"]
        g_leak = abs(A) * C_STATE * d / T_TOK
        g_in = C_STATE * d * kap / T_TOK * cl["s_h"] / cl["s_x"]
        vov_l = g_leak / np.max(g_leak) * VOV_MAX
        vov_i = g_in / np.max(g_in) * VOV_MAX

        # ---- triode (R9a: two-terminal, offset -> conductance floor) -----
        errs_t = []
        for mis in draws:
            gl = (1 + (mis["kp_l"][k] - 1)) * np.clip(vov_l - mis["vt_l"][k],
                                                      0, None)
            gi = (1 + (mis["kp_i"][k] - 1)) * np.clip(vov_i - mis["vt_i"][k],
                                                      0, None)
            d_leak = d * gl / np.maximum(vov_l, 1e-30)
            dk_in = d * kap * gi / np.maximum(vov_i, 1e-30)
            u = np.abs(A) * d_leak
            abar = np.exp(-u)
            bbar = np.where(u > 1e-12, dk_in * (1 - abar) / np.maximum(u, 1e-30),
                            dk_in)
            h = _recurrence(abar, bbar, x)
            errs_t.append(np.linalg.norm(h - ref) / np.linalg.norm(ref))
        p50_t = float(np.percentile(errs_t, 50)) * 100
        meas_t, refit_t = MAMBA_MEAS[("triode", cl["dmin"], dmax, A)]
        tri.append((meas_t, p50_t))
        out[f"triode|{cl['dmin']}-{dmax}|{A}"] = dict(
            measured_p50_pct=meas_t, predicted_p50_pct=p50_t)

        # ---- pair (R9b: three-terminal, offset -> moved tanh operating pt) -
        H = _pair_trajectories(cl, x, draws, k)          # (n_draws, n_tok)
        errs_p = np.linalg.norm(H - ref[None], axis=1) / np.linalg.norm(ref)
        errs_pf = [_refit(H[j], ref, x, d, kap, A)
                   for j in range(min(10, len(draws)))]
        p50_p = float(np.percentile(errs_p, 50)) * 100
        p50_pf = float(np.percentile(errs_pf, 50)) * 100
        meas_p, refit_p = MAMBA_MEAS[("pair", cl["dmin"], dmax, A)]
        pai.append((meas_p, p50_p))
        out[f"pair|{cl['dmin']}-{dmax}|{A}"] = dict(
            measured_p50_pct=meas_p, predicted_p50_pct=p50_p,
            measured_refit_pct=refit_p, predicted_refit_pct=p50_pf,
            narrow=bool(dmax <= 1.0))

    # summary rows: geometric-mean of the cell family (the doc quotes ranges)
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    row("4 Mamba", "triode p50 (geo-mean of 9 cells)",
        gm([a for a, _ in tri]), gm([b for _, b in tri]))
    row("4 Mamba", "triode p50 (min cell)", min(a for a, _ in tri),
        min(b for _, b in tri))
    row("4 Mamba", "triode p50 (max cell)", max(a for a, _ in tri),
        max(b for _, b in tri))
    row("4 Mamba", "pair p50 (geo-mean of 9 cells)",
        gm([a for a, _ in pai]), gm([b for _, b in pai]))
    refits_m = [v["measured_refit_pct"] for kk, v in out.items()
                if kk.startswith("pair") and v.get("narrow")]
    refits_p = [v["predicted_refit_pct"] for kk, v in out.items()
                if kk.startswith("pair") and v.get("narrow")]
    row("4 Mamba", "pair refit residual (geo-mean, narrow ranges)",
        gm(refits_m), gm(refits_p))
    RESULTS["item4_mamba"] = out
    return out


def _pair_trajectories(cl, x, draws, k, n_sub=4):
    """R9b as an ODE, vectorised over mismatch draws.

    The differential pair's leak is  I = I_tail*tanh((v + V_os)/(2 V_T)),
    so with  a = (s_h*h + V_os,l)/(2 V_T):
        da/dt = q - p*tanh(a),
        p = g_leak*kp_l/C,
        q = g_in*kp_i*2V_T*tanh((s_x*x + V_os,i)/(2V_T)) / (2 V_T C)
    An exponential (linearly-implicit) integrator is used because p*T_tok
    reaches 160 at A=-16, Delta_max=10 -- the recurrence is stiff and an
    explicit scheme would be an artefact, not a result.

    Two mismatch mechanisms fall out of this single equation and are the
    content of R9b: (i) the leak relaxes the state to -V_os, and (ii) the
    operating point of BOTH tanh's moves by V_os/(2V_T), which for
    V_os = 10 mV against a 4 mV swing is 2.5 full scales -- so the
    compression is no longer a constant gain but state-dependent.
    """
    A, d, kap = cl["A"], cl["delta"], cl["kappa"]
    s_h, s_x = cl["s_h"], cl["s_x"]
    g_leak = abs(A) * C_STATE * d / T_TOK
    g_in = C_STATE * d * kap / T_TOK * s_h / s_x
    kp_l = np.array([m["kp_l"][k] for m in draws])
    kp_i = np.array([m["kp_i"][k] for m in draws])
    vos_l = np.array([m["vt_l"][k] for m in draws])
    vos_i = np.array([m["vt_i"][k] for m in draws])
    tv = 2.0 * VT_NG
    dt = T_TOK / n_sub

    a = vos_l / tv                       # h = 0 at t = 0
    H = np.zeros((len(draws), len(x)))
    for t in range(len(x)):
        p = g_leak[t] * kp_l / C_STATE
        i_in = g_in[t] * kp_i * tv * np.tanh((s_x * x[t] + vos_i) / tv)
        q = i_in / (tv * C_STATE)
        for _ in range(n_sub):
            th = np.tanh(a)
            r = q - p * th
            lam = p * (1.0 - th * th)
            step = np.where(lam * dt > 1e-9,
                            r / np.maximum(lam, 1e-300)
                            * (1.0 - np.exp(-lam * dt)),
                            r * dt)
            a = a + step
        H[:, t] = (tv * a - vos_l) / s_h
    return H


def _refit(h_circ, ref, x, delta, kappa, A, n_par=5):
    """R9: how much of the mismatch is a *static reparametrization*?

    Identical family, bounds and normalisation to
    `mamba_vcrc_sim.refit_residual` (n_par = 5, i.e. q pinned to zero):
        abar_t = exp(-|A|(a*Delta_t + p*Delta_med))
        bbar_t = b*(Delta_t*kappa_t + q*Delta_med)
        h_t    = abar_t h_{t-1} + bbar_t (x_t + x0) + (1-abar_t) c
    The residual that survives the fit is the part the device does that no
    re-training of a Mamba block could represent.
    """
    from scipy.optimize import least_squares

    scale = float(np.sqrt(np.mean(ref ** 2))) + 1e-30
    dmed = float(np.median(delta))

    def model(par):
        a, bb, p, q, c, x0 = par
        abar = np.exp(-abs(A) * np.maximum(a * delta + p * dmed, 0.0))
        bbar = bb * (delta * kappa + q * dmed)
        hh = np.zeros(len(x))
        s = 0.0
        for t in range(len(x)):
            s = abar[t] * s + bbar[t] * (x[t] + x0) + (1 - abar[t]) * c
            hh[t] = s
        return hh

    p0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    lo = np.array([0.02, 0.02, -0.95, -0.95, -1e3, -1e3])
    hi = np.array([50.0, 50.0, 50.0, 50.0, 1e3, 1e3])
    if n_par == 5:
        lo[3], hi[3] = -1e-12, 1e-12
    base = float(np.sqrt(np.mean((h_circ - ref) ** 2)) / scale)
    try:
        sol = least_squares(lambda p: (model(p) - h_circ) / scale, p0,
                            bounds=(lo, hi), method="trf", x_scale="jac",
                            max_nfev=400)
        return min(float(np.sqrt(np.mean(sol.fun ** 2))), base)
    except Exception:
        return base


# ===========================================================================
# ITEM 5 -- symbol margin (R4)
# ===========================================================================

def item5_symbol():
    k, s_g, eps = 256, 0.03, 1e-9
    sig_tot = s_g * np.sqrt(k)
    m_star = margin_required(sig_tot, eps)
    s_max = sigma_max_feasible(k, eps)
    k_max = int(np.floor((1.0 / (z_of_eps(eps) * 0.03)) ** 2))
    out = dict(sigma_tot=float(sig_tot), z_eps=z_of_eps(eps),
               measured_margin=2.879, predicted_margin=float(m_star),
               measured_sigmaG_max_pct=1.04,
               predicted_sigmaG_max_pct=float(s_max) * 100,
               measured_k_max_at_3pct=30, predicted_k_max_at_3pct=k_max,
               measured_dlneps_dg=3283.0,
               predicted_dlneps_dg=float(dlneps_dg(k, sig_tot, eps)))
    row("5 Symbol", "m*(k=256, sG=3%, 1e-9) [units of 2.0]", 2.879, m_star)
    row("5 Symbol", "sigma_G,max at k=256 [%]", 1.04, s_max * 100)
    row("5 Symbol", "k_max at sigma_G=3%", 30, k_max)
    row("5 Symbol", "d ln eps/dg at k=256", 3283.0,
        dlneps_dg(k, sig_tot, eps))
    RESULTS["item5_symbol"] = out
    return out


# ===========================================================================
# ITEM 6 -- race ranking under mismatch (R1)
# ===========================================================================

def item6_race():
    i_max, is_race = 2e-6, 1e-14
    lam = float(np.log(i_max / is_race))          # 19.114
    e_n = 0.003 * lam
    e_is = float(np.log1p(0.03))
    tot = float(np.hypot(e_n, e_is))
    out = dict(lever=lam, measured_n_nat=0.0573, predicted_n_nat=e_n,
               measured_is_nat=0.0300, predicted_is_nat=e_is,
               measured_total_nat=0.0647, predicted_total_nat=tot,
               measured_pairwise_nat=0.0915,
               predicted_pairwise_nat=float(tot * np.sqrt(2)))
    row("6 Race", "0.3% n -> nat", 0.0573, e_n)
    row("6 Race", "3% Is -> nat", 0.0300, e_is)
    row("6 Race", "total per element [nat]", 0.0647, tot)
    RESULTS["item6_race"] = out
    return out


# ===========================================================================
# ITEM 7 -- training: the least-squares floor and the perturbative cost
# ===========================================================================
#
# R10.  With the loop closed, z = gamma (x) y and the metric divides out a
# per-sample scalar, so what training can reach is the *best constant*
# u = gamma*(1+eps); the floor is the deviation of the LS optimum of the
# LOSS (which does not divide out the scalar) from that constant.
# R11.  A perturbative (SPSA/ESC) gradient costs ~N x the evaluations of an
# exact one, because its variance per coordinate grows with N.

TRAIN_MEAS = {(8, 0.01): 0.155, (8, 0.02): 0.140,
              (32, 0.01): 0.057, (32, 0.02): 0.057}
COST_MEAS = {(8, 0.01): 12.5, (8, 0.02): 12.5, (32, 0.01): 25.0,
             (32, 0.02): 37.5}


def _ls_floor(N, sigma, draws=20, n_train=2000, n_test=500, M=4, seed=99):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        eps = rng.normal(0, sigma, N)
        W = rng.standard_normal((M, N)) / np.sqrt(N)
        G = W.T @ W
        xs = rng.standard_normal((n_train, N)) * 1.5
        xs = np.where(np.abs(xs) < 0.05, 0.05 * np.sign(xs + 1e-300), xs)
        y0 = xs / np.sqrt(np.mean(xs ** 2, axis=1, keepdims=True))
        s = (np.sqrt(np.mean(xs ** 2, axis=1))
             / np.sqrt(np.mean((xs * (1 + eps)) ** 2, axis=1)))
        # A_ij = E[s^2 y0_i y0_j G_ij], b_i = E[s y0_i sum_j G_ij y0_j]
        A = ((s ** 2)[:, None, None] * y0[:, :, None] * y0[:, None, :]
             * G[None]).mean(0)
        b = ((s[:, None] * y0) * (y0 @ G)).mean(0)
        u = np.linalg.solve(A + 1e-12 * np.eye(N), b)
        xt = rng.standard_normal((n_test, N)) * 1.5
        xt = np.where(np.abs(xt) < 0.05, 0.05 * np.sign(xt + 1e-300), xt)
        w = xt ** 2 / np.sum(xt ** 2, axis=1, keepdims=True)
        ub = w @ u
        pc = np.sqrt(np.sum(w * (u[None] / ub[:, None] - 1) ** 2, axis=1))
        vals.append(np.median(pc) / sigma)
    return float(np.median(vals))


def item7_training():
    out = {}
    for (N, sv), meas in TRAIN_MEAS.items():
        pred = _ls_floor(N, sv)
        out[f"N={N},sigma={sv}"] = dict(measured_floor_sigma=meas,
                                        predicted_floor_sigma=pred)
        row("7 Training", f"LS floor N={N} s={sv} [sigma]", meas, pred)
    # untrained / train-in-sim baseline: R3 closed form
    for N, meas in ((8, 0.79), (32, 0.95)):
        # R3: E[sum_i w_i^2] = 2(N-1)/(N(N+2)) + 1/N  (w_i ~ Beta(1/2,(N-1)/2))
        pred = float(np.sqrt(1.0 - (2 * (N - 1) / (N * (N + 2)) + 1.0 / N)))
        out[f"untrained N={N}"] = dict(measured_sigma=meas,
                                       predicted_sigma=pred)
        row("7 Training", f"untrained residual N={N} [sigma]", meas, pred)
    # perturbative cost: R11 predicts D/B ∝ N (SPSA variance per coordinate
    # is ‖g‖² instead of g_i²).  The absolute prefactor is set by the
    # optimiser schedule and is NOT predicted; the scaling factor is.
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    m8 = gm([COST_MEAS[(8, .01)], COST_MEAS[(8, .02)]])
    m32 = gm([COST_MEAS[(32, .01)], COST_MEAS[(32, .02)]])
    out["perturbative cost scaling 8->32"] = dict(
        measured_D_over_B_N8=m8, measured_D_over_B_N32=m32,
        measured_scaling=m32 / m8, predicted_scaling=32.0 / 8.0)
    row("7 Training", "perturbative cost scaling N=8 -> 32", m32 / m8, 4.0)
    RESULTS["item7_training"] = out
    return out


# ===========================================================================
# PART 3 -- prospective prediction (GELU and LayerNorm), no SPICE
# ===========================================================================
#
# Shared corner (the paper's MOS-typical):
#   junctions   sigma_Is = 3 %, sigma_n = 0.3 %
#   mirrors     sigma_mir = 1 %       per-channel gain sigma_VGA = 1 %
#   sigmoid pair extra:  sigma_Vt = 10 mV (input-referred), sigma_K' = 3 %
#
# GELU = P11 . P6 :  g(x) = x * sigma(1.702 x)
#   the multiplier is a translinear loop  Dx . Ds / Dref -> Do  (4 junctions,
#   3 of them per-channel), the sigmoid is a subthreshold/bipolar pair whose
#   whole mismatch is an input-referred offset on its own argument.
#
#   NEW RULE R12 (sigmoid slope lever): an input offset d on a saturating
#   transfer function f enters the output *relatively* as d*f'(u)/f(u);
#   for the logistic that is d*(1 - sigma(u)) -- large exactly where the
#   output is small, so it is a per-channel error that the normalisation
#   of a downstream layer cannot remove.

MOS_TYP = dict(s_is=0.03, s_n=0.003, s_mir=0.01, s_vga=0.01,
               s_vt=0.010, s_kp=0.03)
GELU_N = 8
GELU_ITAIL = 4e-6
GELU_IREF = 4e-6


def gelu_inputs(N=GELU_N, seed=2026):
    x = np.random.default_rng(seed).standard_normal(N)
    x = np.sign(x) * np.clip(np.abs(x), 0.1, 2.5)
    return x


def gelu_map(x, d, corner=MOS_TYP):
    """The calculus' transfer map for one mismatch draw `d` (a dict of
    per-device deviations).  Exactly the composition R1 x R3 x R12."""
    u = 1.702 * x
    # sigmoid pair: IS mismatch and V_t offset are both input-referred
    delta = (d["vos"] / VT_NG + (d["iota_a"] - d["iota_b"])
             + (d["nu_a"] - d["nu_b"]) * np.abs(u))
    sig = 1.0 / (1.0 + np.exp(-(u + delta)))
    i_s = GELU_ITAIL * (1.0 + d["g_tail"]) * sig
    i_x = np.abs(x) * IUNIT * (1.0 + d["eps_vga"])
    # translinear loop: b_j = nu_j*lambda_j - iota_j, forward +b, backward -b
    lam_x = np.log(i_x / ISAT)
    lam_s = np.log(np.maximum(i_s, 1e-15) / ISAT)
    lam_r = np.log(GELU_IREF / ISAT)
    b_x = d["nu_x"] * lam_x - d["iota_x"]
    b_s = d["nu_s"] * lam_s - d["iota_s"]
    b_r = d["nu_r"] * lam_r - d["iota_r"]
    lam_o = lam_x + lam_s - lam_r
    b_o = d["nu_o"] * lam_o - d["iota_o"]
    gain = np.exp(b_x + b_s - b_r - b_o) * (1.0 + d["g_out"])
    return np.sign(x) * (i_x / IUNIT) * (i_s / GELU_IREF) * gain


def gelu_draw(rng, N, c=MOS_TYP):
    f = lambda s: rng.normal(0.0, s, N)
    return dict(vos=f(c["s_vt"]), g_tail=f(c["s_mir"]),
                eps_vga=f(c["s_vga"]), g_out=f(c["s_mir"]),
                iota_a=f(c["s_kp"]), iota_b=f(c["s_kp"]),
                nu_a=f(c["s_n"]), nu_b=f(c["s_n"]),
                iota_x=f(c["s_is"]), nu_x=f(c["s_n"]),
                iota_s=f(c["s_is"]), nu_s=f(c["s_n"]),
                iota_o=f(c["s_is"]), nu_o=f(c["s_n"]),
                iota_r=np.full(N, rng.normal(0.0, c["s_is"])),
                nu_r=np.full(N, rng.normal(0.0, c["s_n"])))


def decompose(y_hat, y_ref):
    a = float(np.dot(y_hat, y_ref) / np.dot(y_ref, y_ref))
    r = y_hat / a - y_ref
    return abs(a - 1.0), float(np.sqrt(np.mean(r ** 2))
                               / np.sqrt(np.mean(y_ref ** 2)))


def predict_gelu(n_draws=2000, seed=31337):
    x = gelu_inputs()
    ref = x / (1.0 + np.exp(-1.702 * x))
    rng = np.random.default_rng(seed)
    cm, pc = [], []
    for _ in range(n_draws):
        y = gelu_map(x, gelu_draw(rng, len(x)))
        a, p = decompose(y, ref)
        cm.append(a)
        pc.append(p)
    return dict(pc_p50=float(np.percentile(pc, 50)) * 100,
                pc_p95=float(np.percentile(pc, 95)) * 100,
                cm_p50=float(np.percentile(cm, 50)) * 100,
                cm_p95=float(np.percentile(cm, 95)) * 100)


LN_N = 8


def ln_inputs(N=LN_N, seed=4242):
    x = np.random.default_rng(seed).standard_normal(N) * 1.5
    return np.where(np.abs(x) < 0.05, 0.05 * np.sign(x + 1e-300), x)


def ln_map(x, d, c=MOS_TYP):
    """Mean subtraction by mirrors (P1.P2, shared + per-channel gain errors)
    followed by the validated AGC loop.  The loop equilibrium is solved in
    closed form, which is what the junction detector bisects to."""
    N = len(x)
    m = np.mean(x) * (1.0 + d["g_mir_mean"])
    cc = x - m * (1.0 + d["g_copy"])
    ce = cc * (1.0 + d["eps_vga"])
    # junction RMS detector, R1 x R3: a shared scalar path
    absy = np.abs(ce)
    w = absy ** 2 / np.sum(absy ** 2)
    lam_y = np.log(np.maximum(absy, 1e-4) * IUNIT / ISAT)
    lam_sq = 2 * lam_y - np.log(IUNIT / ISAT)
    b_y = d["nu_y"] * lam_y - d["iota_y"]
    b_sq = d["nu_sq"] * lam_sq - d["iota_sq"]
    b_r = d["nu_r"] * np.log(IUNIT / ISAT) - d["iota_r"]
    dln = np.sum(w * (2 * b_y - b_r - b_sq)) + d["g_mir_det"]
    G = 1.0 / (np.sqrt(np.mean(ce ** 2)) * np.exp(0.5 * dln))
    return G * ce


def ln_draw(rng, N, c=MOS_TYP):
    f = lambda s: rng.normal(0.0, s, N)
    return dict(g_mir_mean=rng.normal(0.0, c["s_mir"]),
                g_copy=f(c["s_mir"]), eps_vga=f(c["s_vga"]),
                iota_y=f(c["s_is"]), nu_y=f(c["s_n"]),
                iota_sq=f(c["s_is"]), nu_sq=f(c["s_n"]),
                iota_r=rng.normal(0.0, c["s_is"]),
                nu_r=rng.normal(0.0, c["s_n"]),
                g_mir_det=rng.normal(0.0, c["s_mir"]))


def predict_layernorm(n_draws=2000, seed=31338):
    x = ln_inputs()
    c0 = x - np.mean(x)
    ref = c0 / np.sqrt(np.mean(c0 ** 2))
    rng = np.random.default_rng(seed)
    cm, pc = [], []
    for _ in range(n_draws):
        y = ln_map(x, ln_draw(rng, len(x)))
        a, p = decompose(y, ref)
        cm.append(a)
        pc.append(p)
    w = c0 ** 2 / np.sum(c0 ** 2)
    closed = np.sqrt(MOS_TYP["s_vga"] ** 2 * (1 - np.sum(w ** 2))
                     + 2 * np.mean(x) ** 2 * MOS_TYP["s_mir"] ** 2
                     / np.mean(c0 ** 2))
    return dict(pc_p50=float(np.percentile(pc, 50)) * 100,
                pc_p95=float(np.percentile(pc, 95)) * 100,
                cm_p50=float(np.percentile(cm, 50)) * 100,
                cm_p95=float(np.percentile(cm, 95)) * 100,
                pc_closed_form=float(closed) * 100)


def do_predict():
    g, l = predict_gelu(), predict_layernorm()
    print("=" * 96)
    print("  PROSPECTIVE PREDICTION (calculus only, MOS-typical corner, N=8)")
    print("=" * 96)
    for name, r in (("GELU  = P11.P6", g), ("LayerNorm = P1.P2 + AGC", l)):
        print(f"  {name:<26} per-channel p50 {r['pc_p50']:7.3f} %   "
              f"p95 {r['pc_p95']:7.3f} %")
        print(f"  {'':<26} common-mode p50 {r['cm_p50']:7.3f} %   "
              f"p95 {r['cm_p95']:7.3f} %")
    print(f"  LayerNorm closed form (R3):   "
          f"{l['pc_closed_form']:.3f} % per-channel")
    RESULTS["prediction"] = dict(gelu=g, layernorm=l)
    return g, l


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).parent / "out"
                                          / "mismatch_calculus.json"))
    ap.add_argument("--mc-runs", type=int, default=100)
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    if args.predict_only:
        do_predict()
        return

    item1_table6()
    item2_table7()
    item3_softmax()
    item4_mamba(args.mc_runs)
    item5_symbol()
    item6_race()
    item7_training()
    do_predict()

    print("=" * 96)
    print("  MISMATCH CALCULUS -- RETRODICTION (no ngspice was run)")
    print("=" * 96)
    print(f"  {'item':<12} {'quantity':<44} {'measured':>10} "
          f"{'predicted':>11} {'ratio':>7}")
    worst = []
    for item, what, meas, pred, r in ROWS:
        flag = "" if 0.5 <= r <= 2.0 else "   <-- MISS (>2x)"
        print(f"  {item:<12} {what:<44} {meas:>10.4g} {pred:>11.4g} "
              f"{r:>7.2f}{flag}")
        worst.append((item, what, r))
    miss = [(i, w, r) for i, w, r in worst if not (0.5 <= r <= 2.0)]
    print("-" * 96)
    print(f"  rows: {len(ROWS)}   outside 2x: {len(miss)}")
    for i, w, r in miss:
        print(f"    MISS  {i} {w}: ratio {r:.2f}")

    RESULTS["_rows"] = [dict(item=i, quantity=w, measured=m, predicted=p,
                             ratio=r) for i, w, m, p, r in ROWS]
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(RESULTS, indent=1))
    print(f"  raw: {args.json}")


if __name__ == "__main__":
    main()
