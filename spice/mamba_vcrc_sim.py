#!/usr/bin/env python3
"""
SPICE test of kill-gate 1 (docs/missing_primes_mapping.md sec. 8):

    Mamba's selective SSM is NOT dynamic topology (X2) but a data-dependent
    gain (P11) acting on an integrator (P8).

Circuit under test: a voltage-controlled-resistor / capacitor cell ("VCRC").
Per state channel n a capacitor C carries the state h; a voltage-controlled
leak conductance g_leak(x_t) sets the forget rate and a voltage-controlled
input transconductance g_in(x_t) sets the write gain.  Both conductances are
real MOS devices (level 1) in triode -- variant A -- or a junction-exact
differential pair with a current-mode tail -- variant B.

Level of fidelity (same convention as rmsnorm_agc_sim.py):
    device-exact, wiring-ideal.  The transconductor devices are real SPICE
    devices with real I-V laws and Monte-Carlo mismatch; current copying
    (mirrors, common-mode bookkeeping) is done with ideal F-sources.

Everything that is digital in Mamba (the softplus for Delta, the projections
for B and C) stays digital here and enters the circuit as PWL gate voltages /
PWL tail currents.  That boundary is the D/A boundary and is stated in the
report.

Usage:
    python3 mamba_vcrc_sim.py [--tokens 256] [--mc-runs 100] [--seed 2026]
    python3 mamba_vcrc_sim.py --quick        # small smoke run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Constants / design parameters
# --------------------------------------------------------------------------

# ngspice's SPICE3-legacy thermal voltage (see spice/README.md gotcha).
VT_NG = 0.025864890

C_STATE = 1e-12          # 1 pF integration capacitor per half-node
T_TOK = 10e-9            # one token per 10 ns
VCM = 0.9                # common-mode of the differential state pair
VTO_NOM = 0.7            # nominal NMOS threshold
KP_NOM = 100e-6          # nominal K' = mu*Cox [A/V^2]
VOV_MAX = 1.0            # max gate overdrive used by the DAC map
SWING = 4e-3             # target peak differential state swing [V]
X_SWING = 4e-3           # peak differential input voltage [V]

# S4D-real init: three state channels with fixed negative A
A_CHANNELS = (-1.0, -4.0, -16.0)

# Delta dynamic ranges to sweep (units of 1/T_tok)
DELTA_RANGES = ((1e-3, 1e-1), (1e-3, 1e0), (1e-3, 1e1))

WORKDIR = Path(__file__).parent / "out"


# --------------------------------------------------------------------------
# Reference model (numpy)
# --------------------------------------------------------------------------

def softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def mamba_recurrence(A, delta, B, x, mode="euler"):
    """Discrete selective-SSM recurrence, one scalar state channel.

    mode='euler' : Mamba Alg. 2 as implemented -- Abar=exp(dA), Bbar=d*B
    mode='zoh'   : exact zero-order hold -- Bbar=(exp(dA)-1)/A * B
    """
    dA = delta * A
    abar = np.exp(dA)
    if mode == "euler":
        bbar = delta * B
    elif mode == "zoh":
        bbar = np.where(np.abs(dA) < 1e-12, delta, (abar - 1.0) / A) * B
    else:
        raise ValueError(mode)
    h = np.zeros(len(x) + 1)
    for t in range(len(x)):
        h[t + 1] = abar[t] * h[t] + bbar[t] * x[t]
    return h[1:]


def euler_match_kappa(A, delta):
    """g_in correction that turns the (exact) RC response into Mamba's Euler B.

    Circuit over one token:  h+ = e^{dA} h + (1-e^{dA}) * (g_in/g_leak) * x
    Mamba Euler wants the input term to be  delta*B*x, so
        g_in/g_leak = delta*B / (1-e^{dA}),  and with g_leak = |A| C delta/T
        g_in = C*delta*B/T * kappa,   kappa = |A|*delta/(1-e^{-|A|delta}).
    """
    u = np.abs(A) * delta
    return np.where(u < 1e-9, 1.0 + u / 2.0, u / (1.0 - np.exp(-u)))


# --------------------------------------------------------------------------
# Input sequences
# --------------------------------------------------------------------------

def make_sequences(n_tok, rng):
    """Return {name: (x, marker)} with x the SSM input, marker the Delta cue."""
    seqs = {}

    # (a) random Gaussian token stream
    x = rng.standard_normal(n_tok)
    seqs["gauss"] = (x / np.max(np.abs(x)), np.zeros(n_tok))

    # (b) selective-copy toy: sparse spikes, a long hold window, then a flush
    x = np.zeros(n_tok)
    m = np.zeros(n_tok)
    n_sp = max(3, n_tok // 42)
    spike_pos = np.sort(rng.choice(np.arange(4, n_tok // 4), size=n_sp,
                                   replace=False))
    x[spike_pos] = rng.choice([-1.0, 1.0], size=n_sp)
    flush0 = int(0.68 * n_tok)
    m[flush0:flush0 + 5] = 1.0
    seqs["copy"] = (x, m)
    seqs["_copy_meta"] = dict(spikes=spike_pos.tolist(), flush=flush0)

    # (c) step + reset
    x = np.zeros(n_tok)
    m = np.zeros(n_tok)
    a, b, c = int(0.25 * n_tok), int(0.62 * n_tok), int(0.66 * n_tok)
    x[a:b] = 1.0
    m[b:c] = 1.0          # reset burst: Delta large, input zero
    x[c:] = 0.5
    seqs["step"] = (x, m)
    return seqs


def make_delta(x, marker, dmin, dmax):
    """Delta_t = softplus(w.x + b), affinely rescaled to span [dmin, dmax].

    The affine rescale stands for the learned scale of the Delta projection.
    It is normalised on the *realised* sequence so that min_t Delta_t = dmin
    and max_t Delta_t = dmax exactly -- otherwise the swept "dynamic range"
    would not be the dynamic range the circuit actually has to cover.
    """
    z = -3.0 + 3.0 * np.abs(x) + 9.0 * marker
    sp = softplus(z)
    lo, hi = float(np.min(sp)), float(np.max(sp))
    u = (sp - lo) / (hi - lo) if hi > lo else np.zeros_like(sp)
    return dmin + (dmax - dmin) * u


# --------------------------------------------------------------------------
# PWL helpers
# --------------------------------------------------------------------------

def pwl_string(values, t_tok=T_TOK, ramp=1e-12):
    """Piecewise-constant per token, switching 1 ps after each boundary.

    A breakpoint sits exactly at k*t_tok, so ngspice puts a solution point
    there and the token-boundary sample is not an interpolation artefact.
    """
    pts = [f"0 {values[0]:.10e}"]
    for k in range(1, len(values)):
        tb = k * t_tok
        pts.append(f"{tb:.10e} {values[k - 1]:.10e}")
        pts.append(f"{tb + ramp:.10e} {values[k]:.10e}")
    pts.append(f"{len(values) * t_tok:.10e} {values[-1]:.10e}")
    return "PWL(" + " ".join(pts) + ")"


def run_ngspice(netlist, tag, timeout=900):
    WORKDIR.mkdir(exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=timeout, cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-3000:])
        sys.stderr.write(res.stderr[-3000:])
        raise RuntimeError(f"ngspice failed for {tag}")
    return res.stdout


def parse_wrdata(tag, n_vec):
    data = np.loadtxt(WORKDIR / f"{tag}.txt")
    t = data[:, 0]
    vals = data[:, 1::2]
    assert vals.shape[1] == n_vec, (vals.shape, n_vec)
    return t, vals


# --------------------------------------------------------------------------
# Cell design: map (Delta_t, A, B_t) onto conductances / tail currents
# --------------------------------------------------------------------------

class Cell:
    """One (Delta-range, channel) VCRC cell."""

    def __init__(self, idx, A, dmin, dmax, delta, B, x, mapping,
                 swing=SWING, x_swing=X_SWING):
        self.idx = idx
        self.A = A
        self.dmin, self.dmax = dmin, dmax
        self.delta = delta
        self.B = B
        self.mapping = mapping
        self.swing, self.x_swing = swing, x_swing

        self.kappa = (euler_match_kappa(A, delta) if mapping == "euler"
                      else np.ones_like(delta))
        self.ref = mamba_recurrence(A, delta, B, x,
                                    "euler" if mapping == "euler" else "zoh")
        peak = max(np.max(np.abs(self.ref)), 1e-12)
        self.s_h = swing / peak                      # volts per unit of h
        self.s_x = x_swing / max(np.max(np.abs(x)), 1e-12)

        # continuous-time targets (single-ended equivalent, see the .md)
        self.g_leak = abs(A) * C_STATE * delta / T_TOK
        self.g_in = (C_STATE * delta * B * self.kappa / T_TOK
                     * self.s_h / self.s_x)

    # ---- variant A: MOS triode voltage-controlled resistors ----
    def triode_design(self):
        # the two devices of the differential cell each carry g/2
        bl = np.max(self.g_leak) / 2.0 / VOV_MAX
        bi = np.max(self.g_in) / 2.0 / VOV_MAX
        vov_l = (self.g_leak / 2.0) / bl
        vov_i = (self.g_in / 2.0) / bi
        return bl, bi, vov_l, vov_i

    # ---- variant B: junction-exact differential pair, current tail ----
    def pair_design(self):
        i_tail_l = 2.0 * VT_NG * C_STATE * abs(self.A) * self.delta / T_TOK
        i_tail_i = (2.0 * VT_NG * C_STATE * self.delta * self.B * self.kappa
                    / T_TOK * self.s_h / self.s_x)
        return i_tail_l, i_tail_i


def build_cells(x, marker, mapping, B_input_dependent=False,
                ranges=DELTA_RANGES, swing=SWING, x_swing=X_SWING):
    cells = []
    for (dmin, dmax) in ranges:
        delta = make_delta(x, marker, dmin, dmax)
        if B_input_dependent:
            B = 0.5 + 0.5 * np.tanh(2.0 * x)
        else:
            B = np.ones_like(x)
        for A in A_CHANNELS:
            cells.append(Cell(len(cells), A, dmin, dmax, delta, B, x, mapping,
                              swing=swing, x_swing=x_swing))
    return cells


# --------------------------------------------------------------------------
# Netlists
# --------------------------------------------------------------------------

HDR_OPTS = (".options reltol=1e-6 abstol=1e-16 vntol=1e-10 chgtol=1e-17 "
            "gmin=1e-14 method=gear maxord=2")


def _wl_from_beta(beta):
    """Pick W,L (in um) realising beta = KP*(W/L) with KP = KP_NOM."""
    ratio = beta / KP_NOM
    if ratio >= 1.0:
        return ratio, 1.0
    return 1.0, 1.0 / ratio


def build_triode_netlist(cells, x, tag, mis=None):
    """mis: dict with per-device 'kp' multipliers and 'vt' offsets, or None."""
    n = len(cells)
    L = [f"* Mamba VCRC, MOS-triode variant ({tag})", HDR_OPTS]
    L.append(f"Vxp xp 0 {pwl_string(VCM + cells[0].s_x * x / 2.0)}")
    L.append(f"Vxn xn 0 {pwl_string(VCM - cells[0].s_x * x / 2.0)}")

    for k, cl in enumerate(cells):
        bl, bi, vov_l, vov_i = cl.triode_design()
        kp_l = KP_NOM * (mis["kp_l"][k] if mis else 1.0)
        kp_i = KP_NOM * (mis["kp_i"][k] if mis else 1.0)
        vt_l = VTO_NOM + (mis["vt_l"][k] if mis else 0.0)
        vt_i = VTO_NOM + (mis["vt_i"][k] if mis else 0.0)
        wl, ll = _wl_from_beta(bl)
        wi, li = _wl_from_beta(bi)
        L.append(f".model ML{k} NMOS(LEVEL=1 VTO={vt_l:.6f} KP={kp_l:.6e} "
                 "LAMBDA=0 GAMMA=0 PHI=0.6 CGSO=0 CGDO=0 CBD=0 CBS=0 "
                 "CJ=0 CJSW=0 IS=1e-18)")
        L.append(f".model MI{k} NMOS(LEVEL=1 VTO={vt_i:.6f} KP={kp_i:.6e} "
                 "LAMBDA=0 GAMMA=0 PHI=0.6 CGSO=0 CGDO=0 CBD=0 CBS=0 "
                 "CJ=0 CJSW=0 IS=1e-18)")
        # gate voltages: the D/A boundary (softplus computed in numpy)
        L.append(f"Vgl{k} gl{k} 0 {pwl_string(VTO_NOM + VCM + vov_l)}")
        L.append(f"Vgi{k} gi{k} 0 {pwl_string(VTO_NOM + VCM + vov_i)}")
        # state pair
        L.append(f"Chp{k} hp{k} 0 {C_STATE:.6e}")
        L.append(f"Chn{k} hn{k} 0 {C_STATE:.6e}")
        L.append(f"Ml{k} hp{k} gl{k} hn{k} 0 ML{k} W={wl:.6f}u L={ll:.6f}u")
        # input transconductor + ideal differential copy into the state pair
        L.append(f"Mi{k} xp gi{k} xs{k} 0 MI{k} W={wi:.6f}u L={li:.6f}u")
        L.append(f"Vs{k} xs{k} xn DC 0")
        L.append(f"Fp{k} 0 hp{k} Vs{k} 1.0")
        L.append(f"Fn{k} hn{k} 0 Vs{k} 1.0")
        L.append(f"Ed{k} d{k} 0 VALUE={{V(hp{k})-V(hn{k})}}")

    L.append(".ic " + " ".join(f"V(hp{k})={VCM} V(hn{k})={VCM}"
                               for k in range(n)))
    vecs = " ".join(f"v(d{k})" for k in range(n))
    tstop = len(x) * T_TOK
    tmax = T_TOK / 50.0
    L += [".control", "set interp",
          f"tran {T_TOK:.6e} {tstop:.6e} 0 {tmax:.6e} uic",
          f"wrdata {tag}.txt {vecs}", "quit", ".endc", ".end"]
    return "\n".join(L)


def build_pair_netlist(cells, x, tag, mis=None):
    """Variant B: junction-exact differential pairs with current-mode tails."""
    n = len(cells)
    L = [f"* Mamba VCRC, differential-pair variant ({tag})", HDR_OPTS]
    L.append("Vcc vcc 0 DC 2.0")
    L.append("Vee vee 0 DC -2.0")
    L.append(f"Vxp xp 0 {pwl_string(VCM + cells[0].s_x * x / 2.0)}")
    L.append(f"Vxn xn 0 {pwl_string(VCM - cells[0].s_x * x / 2.0)}")

    for k, cl in enumerate(cells):
        itl, iti = cl.pair_design()
        gl = mis["kp_l"][k] if mis else 1.0      # tail-mirror gain error
        gi = mis["kp_i"][k] if mis else 1.0
        vos_l = mis["vt_l"][k] if mis else 0.0   # input-referred offset
        vos_i = mis["vt_i"][k] if mis else 0.0
        for nm, vos in ((f"QL{k}A", +vos_l / 2), (f"QL{k}B", -vos_l / 2),
                        (f"QI{k}A", +vos_i / 2), (f"QI{k}B", -vos_i / 2)):
            iss = 1e-16 * np.exp(vos / VT_NG)
            L.append(f".model {nm} NPN(IS={iss:.8e} BF=1e6 VAF=1e12 RB=0 "
                     "RE=0 RC=0 CJE=0 CJC=0 TF=0 TR=0)")
        # leak pair
        L.append(f"Ql{k}a cl{k}a hp{k} el{k} QL{k}A")
        L.append(f"Ql{k}b cl{k}b hn{k} el{k} QL{k}B")
        L.append(f"Il{k} el{k} vee {pwl_string(itl * gl)}")
        L.append(f"Vcl{k}a vcc cl{k}a DC 0")
        L.append(f"Vcl{k}b vcc cl{k}b DC 0")
        # input pair
        L.append(f"Qi{k}a ci{k}a xp ei{k} QI{k}A")
        L.append(f"Qi{k}b ci{k}b xn ei{k} QI{k}B")
        L.append(f"Ii{k} ei{k} vee {pwl_string(iti * gi)}")
        L.append(f"Vci{k}a vcc ci{k}a DC 0")
        L.append(f"Vci{k}b vcc ci{k}b DC 0")
        # state pair + ideal differential current bookkeeping
        L.append(f"Chp{k} hp{k} 0 {C_STATE:.6e}")
        L.append(f"Chn{k} hn{k} 0 {C_STATE:.6e}")
        L.append(f"Flp{k}a hp{k} 0 Vcl{k}a 0.5")
        L.append(f"Flp{k}b 0 hp{k} Vcl{k}b 0.5")
        L.append(f"Fln{k}a hn{k} 0 Vcl{k}b 0.5")
        L.append(f"Fln{k}b 0 hn{k} Vcl{k}a 0.5")
        L.append(f"Fip{k}a 0 hp{k} Vci{k}a 0.5")
        L.append(f"Fip{k}b hp{k} 0 Vci{k}b 0.5")
        L.append(f"Fin{k}a 0 hn{k} Vci{k}b 0.5")
        L.append(f"Fin{k}b hn{k} 0 Vci{k}a 0.5")
        L.append(f"Ed{k} d{k} 0 VALUE={{V(hp{k})-V(hn{k})}}")

    L.append(".ic " + " ".join(f"V(hp{k})={VCM} V(hn{k})={VCM}"
                               for k in range(n)))
    vecs = " ".join(f"v(d{k})" for k in range(n))
    tstop = len(x) * T_TOK
    tmax = T_TOK / 50.0
    L += [".control", "set interp",
          f"tran {T_TOK:.6e} {tstop:.6e} 0 {tmax:.6e} uic",
          f"wrdata {tag}.txt {vecs}", "quit", ".endc", ".end"]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def rel_rms(h_circ, h_ref):
    den = np.sqrt(np.mean(h_ref ** 2))
    if den <= 0:
        return np.nan
    return float(np.sqrt(np.mean((h_circ - h_ref) ** 2)) / den)


def eff_bits(rel):
    if not np.isfinite(rel) or rel <= 0:
        return float("inf")
    return float(-np.log2(rel))


def refit_residual(h_circ, cell, x, n_par=5):
    """Is the mismatch a *static reparametrisation* rather than noise?

    Physical model of a mismatched cell (both variants):
        g_leak = (1+eps_l) * (g_leak_nom + off_l)
        g_in   = (1+eps_i) * (g_in_nom   + off_i)
        plus, for the differential pair, an input-referred offset that makes
        the leak element relax to a constant state c instead of 0.
    In state-space terms:
        abar_t = exp(-|A|(a*Delta_t + p))
        bbar_t = b*(Delta_t*kappa_t + q) * B_t
        h_t    = abar_t h_{t-1} + bbar_t (x_t + x0) + (1-abar_t)*c
    with 6 constants per channel.  Where each lives in a real Mamba block:
        a   -> the learned A_n                              (in the family)
        p   -> a Delta floor / dt_min                       (in the family)
        b   -> the learned scale of the B projection        (in the family)
        c   -> a constant state offset; since the SSM is linear this is an
               exact shift h -> h + c and is absorbed by an output bias
        x0  -> the bias of the depthwise Conv1d in front of the SSM
        q   -> needs the B projection to carry a Delta-dependent term
               (the only parameter outside the standard family)
    n_par=5 drops q.  Returns (residual/||h_ref||, fitted parameters).
    """
    from scipy.optimize import least_squares

    d, B, kap, A = cell.delta, cell.B, cell.kappa, cell.A
    scale = np.sqrt(np.mean(cell.ref ** 2)) + 1e-30
    dmed = float(np.median(d))

    def model(par):
        a, bb, p, q, c, x0 = par
        abar = np.exp(-abs(A) * np.maximum(a * d + p * dmed, 0.0))
        bbar = bb * (d * kap + q * dmed) * B
        h = np.zeros(len(x) + 1)
        for t in range(len(x)):
            h[t + 1] = abar[t] * h[t] + bbar[t] * (x[t] + x0) + (1 - abar[t]) * c
        return h[1:]

    def resid(par):
        return (model(par) - h_circ) / scale

    p0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    lo = np.array([0.02, 0.02, -0.95, -0.95, -1e3, -1e3])
    hi = np.array([50.0, 50.0, 50.0, 50.0, 1e3, 1e3])
    if n_par == 5:
        lo[3], hi[3] = -1e-12, 1e-12
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                            x_scale="jac", max_nfev=900)
        r = float(np.sqrt(np.mean(sol.fun ** 2)))
        return min(r, rel_rms(h_circ, cell.ref)), sol.x
    except Exception:
        return float("nan"), np.full(6, np.nan)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def run_ideal(builder, cells, x, tag):
    net = builder(cells, x, tag)
    t0 = time.time()
    run_ngspice(net, tag)
    dt = time.time() - t0
    _, vals = parse_wrdata(tag, len(cells))
    out = []
    for k, cl in enumerate(cells):
        h = vals[:, k] / cl.s_h
        out.append(h)
    return np.array(out), dt


def mc_draws(n_cells, n_runs, rng, sig_kp=0.03, sig_vt=0.010):
    for _ in range(n_runs):
        yield dict(kp_l=rng.normal(1.0, sig_kp, n_cells),
                   kp_i=rng.normal(1.0, sig_kp, n_cells),
                   vt_l=rng.normal(0.0, sig_vt, n_cells),
                   vt_i=rng.normal(0.0, sig_vt, n_cells))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--mc-runs", type=int, default=100)
    ap.add_argument("--mc-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default="mamba_vcrc_results.json")
    args = ap.parse_args()
    if args.quick:
        args.tokens, args.mc_runs, args.mc_tokens = 48, 6, 32

    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")
    WORKDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    W = 78
    print("=" * W)
    print("  Kill-gate 1: Mamba selective SSM as P11 gain on a P8 integrator")
    print(f"  tokens={args.tokens}  T_tok={T_TOK*1e9:.0f} ns  C={C_STATE*1e12:.0f} pF"
          f"  MC={args.mc_runs}x{args.mc_tokens} tok")
    print("=" * W)

    seqs = make_sequences(args.tokens, rng)
    results = {"config": dict(tokens=args.tokens, mc_runs=args.mc_runs,
                              mc_tokens=args.mc_tokens, T_tok=T_TOK,
                              C=C_STATE, swing=SWING, vov_max=VOV_MAX,
                              seed=args.seed)}

    # ---------------------------------------------------------------- 0
    # Pure-numerics: how far apart are the Euler and ZOH references?
    print("\n[0] discretisation gap (numpy only): Mamba-Euler vs exact ZOH")
    gap = {}
    xg, mg = seqs["gauss"]
    for (dmin, dmax) in DELTA_RANGES:
        d = make_delta(xg, mg, dmin, dmax)
        for A in A_CHANNELS:
            he = mamba_recurrence(A, d, np.ones_like(xg), xg, "euler")
            hz = mamba_recurrence(A, d, np.ones_like(xg), xg, "zoh")
            r = rel_rms(hz, he)
            gap[f"{dmin}-{dmax}|A={A}"] = r
            print(f"    range {dmin:g}-{dmax:<5g} A={A:6.1f}: "
                  f"rel.RMS(ZOH vs Euler) = {r*100:8.3f} %")
    results["discretisation_gap"] = gap

    # ---------------------------------------------------------------- 1
    # Ideal devices, both variants, both gate mappings, all three inputs
    print("\n[1] ideal devices: rel. RMS error vs the discrete recurrence")
    ideal = {}
    for variant, builder in (("triode", build_triode_netlist),
                             ("pair", build_pair_netlist)):
        for mapping, refmode in (("euler", "euler"), ("zoh", "zoh")):
            for sname in ("gauss", "copy", "step"):
                x, m = seqs[sname]
                cells = build_cells(x, m, mapping)
                tag = f"mvcrc_{variant}_{mapping}_{sname}"
                hs, dt = run_ideal(builder, cells, x, tag)
                for k, cl in enumerate(cells):
                    r = rel_rms(hs[k], cl.ref)
                    ideal[f"{variant}|{mapping}|{sname}|{cl.dmin}-{cl.dmax}|{cl.A}"] = r
                if sname == "gauss":
                    print(f"    {variant:6s} map={mapping:5s} ({dt:.1f}s)")
                    for k, cl in enumerate(cells):
                        r = ideal[f"{variant}|{mapping}|gauss|{cl.dmin}-{cl.dmax}|{cl.A}"]
                        print(f"        D in [{cl.dmin:g},{cl.dmax:<5g}] A={cl.A:6.1f}"
                              f" : {r*100:9.4f} %   {eff_bits(r):5.1f} bit")
    results["ideal"] = ideal

    # store trajectories for the figure (input b, mapping=euler)
    figdata = {}
    for variant, builder in (("triode", build_triode_netlist),
                             ("pair", build_pair_netlist)):
        x, m = seqs["copy"]
        cells = build_cells(x, m, "euler")
        hs, _ = run_ideal(builder, cells, x, f"mvcrc_fig_{variant}")
        figdata[variant] = hs
        figdata[f"{variant}_ref"] = np.array([c.ref for c in cells])
        # one representative mismatched draw, for the figure
        mis = next(mc_draws(len(cells), 1, np.random.default_rng(7)))
        run_ngspice(builder(cells, x, f"mvcrc_figmis_{variant}", mis=mis),
                    f"mvcrc_figmis_{variant}")
        _, vv = parse_wrdata(f"mvcrc_figmis_{variant}", len(cells))
        figdata[f"{variant}_mis"] = np.array(
            [vv[:, k] / c.s_h for k, c in enumerate(cells)])
    results["_cells_meta"] = [
        dict(dmin=c.dmin, dmax=c.dmax, A=c.A) for c in
        build_cells(*seqs["copy"], "euler")]

    # ---------------------------------------------------------------- 2
    # K3: selective-copy behaviour
    print("\n[2] K3 selective-copy toy (Delta range 1e-3..1, mapping=euler)")
    x, m = seqs["copy"]
    meta = seqs["_copy_meta"]
    cells = build_cells(x, m, "euler")
    k3 = {}
    for variant in ("triode", "pair"):
        hs = figdata[variant]
        for k, cl in enumerate(cells):
            if (cl.dmin, cl.dmax) != (1e-3, 1e0):
                continue
            last_spike = meta["spikes"][-1]
            fl = meta["flush"]
            wamp = cl.ref[last_spike]
            ent = dict(
                write_ref=float(cl.ref[last_spike]),
                write_cir=float(hs[k][last_spike]),
                hold_ref=float(cl.ref[fl - 1] / wamp),
                hold_cir=float(hs[k][fl - 1] / wamp),
                erase_ref=float(abs(cl.ref[fl + 7]) / max(abs(cl.ref[fl - 1]), 1e-30)),
                erase_cir=float(abs(hs[k][fl + 7]) / max(abs(cl.ref[fl - 1]), 1e-30)),
            )
            k3[f"{variant}|A={cl.A}"] = ent
            print(f"    {variant:6s} A={cl.A:6.1f}: retention ref="
                  f"{ent['hold_ref']:+.3f} cir={ent['hold_cir']:+.3f} | "
                  f"post-flush ref={ent['erase_ref']:.2e} "
                  f"cir={ent['erase_cir']:.2e}")
    results["k3"] = k3

    # ---------------------------------------------------------------- 3
    # K2: Monte-Carlo mismatch
    print(f"\n[3] K2 Monte-Carlo mismatch ({args.mc_runs} draws, "
          "3% K'/tail, 10 mV Vt/Vos)")
    mc_seqs = make_sequences(args.mc_tokens, np.random.default_rng(args.seed + 1))
    xm, mm = mc_seqs["gauss"]
    mc = {}
    for variant, builder in (("triode", build_triode_netlist),
                             ("pair", build_pair_netlist)):
        cells = build_cells(xm, mm, "euler")
        # ideal baseline at the MC token count
        hs0, _ = run_ideal(builder, cells, xm, f"mc_{variant}_ideal")
        raw = [[] for _ in cells]
        fit = [[] for _ in cells]
        t0 = time.time()
        for j, mis in enumerate(mc_draws(len(cells), args.mc_runs,
                                         np.random.default_rng(args.seed + 100))):
            tag = f"mc_{variant}_{j}"
            net = builder(cells, xm, tag, mis=mis)
            run_ngspice(net, tag)
            _, vals = parse_wrdata(tag, len(cells))
            for k, cl in enumerate(cells):
                h = vals[:, k] / cl.s_h
                raw[k].append(rel_rms(h, cl.ref))
                if j < min(20, args.mc_runs):
                    fit[k].append(refit_residual(h, cl, xm)[0])
        dt = time.time() - t0

        print(f"    {variant} MC done in {dt:.0f}s")
        for k, cl in enumerate(cells):
            r = np.array(raw[k], dtype=float)
            f = np.array(fit[k], dtype=float)
            ent = dict(p50=float(np.nanpercentile(r, 50)),
                       p95=float(np.nanpercentile(r, 95)),
                       ideal=float(rel_rms(hs0[k], cl.ref)),
                       refit_p50=float(np.nanpercentile(f, 50)) if len(f) else None)
            mc[f"{variant}|{cl.dmin}-{cl.dmax}|{cl.A}"] = ent
            print(f"        {variant:6s} D in [{cl.dmin:g},{cl.dmax:<5g}] "
                  f"A={cl.A:6.1f}: ideal={ent['ideal']*100:8.4f}%  "
                  f"p50={ent['p50']*100:9.3f}%  p95={ent['p95']*100:9.3f}%  "
                  f"refit p50={ent['refit_p50']*100:8.3f}%")
    results["mc"] = mc

    # ---------------------------------------------------------------- 4
    # triode feasibility diagnostic
    print("\n[4] triode-region feasibility (analytic)")
    feas = {}
    x, m = seqs["gauss"]
    for mapping in ("euler", "zoh"):
        for cl in build_cells(x, m, mapping):
            _, _, vov_l, vov_i = cl.triode_design()
            need = cl.swing / 2.0
            ok = float(np.mean((vov_l >= need) & (vov_i >= cl.x_swing / 2.0)))
            # P(device pushed into cutoff by a 10 mV Vt offset), token-averaged
            from math import erf
            phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
            cut = float(np.mean([1.0 - phi(v / 0.010) for v in vov_l])
                        + np.mean([1.0 - phi(v / 0.010) for v in vov_i])) / 2
            feas[f"{mapping}|{cl.dmin}-{cl.dmax}|{cl.A}"] = dict(
                vov_l_min=float(np.min(vov_l)), vov_i_min=float(np.min(vov_i)),
                frac_in_triode=ok, p_cutoff=cut)
            if mapping == "euler":
                print(f"    D in [{cl.dmin:g},{cl.dmax:<5g}] A={cl.A:6.1f}: "
                      f"Vov_leak_min={np.min(vov_l)*1e3:9.4f} mV  "
                      f"Vov_in_min={np.min(vov_i)*1e3:9.4f} mV  "
                      f"tokens in triode={ok*100:5.1f}%  "
                      f"P(cutoff|sig_Vt=10mV)={cut*100:4.1f}%")
    results["triode_feasibility"] = feas

    # ---------------------------------------------------------------- 5
    # Mechanism separation: re-run the triode variant with the signal swing
    # shrunk until every device stays in triode for every token.
    print("\n[5] triode variant on the selective-copy input, swing shrunk "
          "until no device ever leaves triode")
    sl = {}
    x, m = seqs["copy"]
    for (dmin, dmax) in DELTA_RANGES:
        probe = build_cells(x, m, "euler", ranges=[(dmin, dmax)])
        vmin = min(min(np.min(c.triode_design()[2]),
                       np.min(c.triode_design()[3])) for c in probe)
        sw = min(SWING, 1.6 * vmin)
        cells = build_cells(x, m, "euler", ranges=[(dmin, dmax)],
                            swing=sw, x_swing=sw)
        tag = f"mvcrc_swlim_{dmax:g}"
        hs, _ = run_ideal(build_triode_netlist, cells, x, tag)
        for k, cl in enumerate(cells):
            r = rel_rms(hs[k], cl.ref)
            base = ideal[f"triode|euler|copy|{cl.dmin}-{cl.dmax}|{cl.A}"]
            sl[f"{cl.dmin}-{cl.dmax}|{cl.A}"] = dict(swing_V=sw, rel=r,
                                                     rel_full_swing=base)
            print(f"    D in [{dmin:g},{dmax:<5g}] A={cl.A:6.1f}: "
                  f"swing {SWING*1e3:.1f} mV -> {r*0+sw*1e3:10.6f} mV | "
                  f"rel.RMS {base*100:8.4f}% -> {r*100:8.4f}%  "
                  f"({eff_bits(r):4.1f} bit)")
    results["swing_limited"] = sl

    # ---------------------------------------------------------------- 6
    # How much offset can each variant take?  Same MC at sigma = 1 mV.
    print("\n[6] mismatch sensitivity: same MC with sigma(Vt/Vos) = 1 mV")
    sens = {}
    n6 = max(10, args.mc_runs // 2)
    for variant, builder in (("triode", build_triode_netlist),
                             ("pair", build_pair_netlist)):
        cells = build_cells(xm, mm, "euler")
        raw = [[] for _ in cells]
        for j, mis in enumerate(mc_draws(len(cells), n6,
                                         np.random.default_rng(args.seed + 900),
                                         sig_kp=0.03, sig_vt=0.001)):
            tag = f"s6_{variant}_{j}"
            run_ngspice(builder(cells, xm, tag, mis=mis), tag)
            _, vals = parse_wrdata(tag, len(cells))
            for k, cl in enumerate(cells):
                raw[k].append(rel_rms(vals[:, k] / cl.s_h, cl.ref))
        for k, cl in enumerate(cells):
            r = np.array(raw[k], dtype=float)
            sens[f"{variant}|{cl.dmin}-{cl.dmax}|{cl.A}"] = dict(
                p50=float(np.nanpercentile(r, 50)),
                p95=float(np.nanpercentile(r, 95)))
            print(f"    {variant:6s} D in [{cl.dmin:g},{cl.dmax:<5g}] "
                  f"A={cl.A:6.1f}: p50={np.nanpercentile(r,50)*100:9.3f}%  "
                  f"p95={np.nanpercentile(r,95)*100:9.3f}%")
    results["mc_sigma1mV"] = sens

    (WORKDIR / args.json).write_text(json.dumps(results, indent=1))
    print(f"\n    results -> {WORKDIR / args.json}")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = results["_cells_meta"]
        sel = [k for k, c in enumerate(cm) if c["dmin"] == 1e-3 and c["dmax"] == 1.0]
        fig, axes = plt.subplots(len(sel), 2, figsize=(12.5, 2.4 * len(sel)),
                                 sharex=True)
        meta = seqs["_copy_meta"]
        for row, k in enumerate(sel):
            axl, axr = axes[row]
            ref = figdata["triode_ref"][k]
            axl.plot(ref, "k-", lw=2.0, label="discrete Mamba (numpy)")
            axl.plot(figdata["triode"][k], "C0--", lw=1.1, label="VCRC triode")
            axl.plot(figdata["pair"][k], "C3:", lw=1.4, label="VCRC diff-pair")
            axl.plot(figdata["triode_mis"][k], "C2-", lw=0.9, alpha=0.8,
                     label="triode, one mismatch draw")
            for s in meta["spikes"]:
                axl.axvline(s, color="0.85", lw=0.7)
            axl.axvspan(meta["flush"], meta["flush"] + 5, color="C1", alpha=0.18)
            axl.set_ylabel(f"h   (A = {cm[k]['A']:g})")
            axl.legend(fontsize=7, ncol=2, loc="best")
            sc = np.sqrt(np.mean(ref ** 2))
            for nm, st, c in (("triode", "C0-", None), ("pair", "C3-", None)):
                axr.semilogy(np.abs(figdata[nm][k] - ref) / sc + 1e-16, st,
                             lw=1.0, label=f"{nm} (ideal devices)")
            axr.semilogy(np.abs(figdata["triode_mis"][k] - ref) / sc + 1e-16,
                         "C2-", lw=0.9, label="triode (mismatch draw)")
            axr.axhline(0.01, color="k", ls="--", lw=0.8)
            axr.set_ylim(1e-7, 10)
            axr.set_ylabel("|h_circ - h_ref| / rms(h_ref)")
            axr.legend(fontsize=7, loc="best")
        axes[-1][0].set_xlabel("token index")
        axes[-1][1].set_xlabel("token index")
        axes[0][0].set_title("Selective-copy toy, Delta in [1e-3, 1]  "
                             "(grey = spikes, orange = flush burst)")
        axes[0][1].set_title("normalised error (dashed line = K1 threshold, 1 %)")
        fig.tight_layout()
        fig.savefig(WORKDIR / "mamba_vcrc.png", dpi=130)
        print(f"    figure  -> {WORKDIR / 'mamba_vcrc.png'}")
    except ImportError:
        print("    (matplotlib absent, no figure)")

    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
