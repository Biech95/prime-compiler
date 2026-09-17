#!/usr/bin/env python3
"""X3 bounded-recursion experiment: latency/energy of the B(.) reclassifications.

Companion to `docs/exp_bounded_recursion.md` (pre-registration in its section 0,
written before this script was run).

Two algorithms from `docs/missing_primes_mapping.md` section 6 are costed at the
level of the paper's own transition model (section 4.4 of prime_compiler_v2.tex:
ADC ~100 fJ / ~1 ns, DAC ~50 fJ / ~0.5 ns; AGC settle 4-6 ns) plus one
circuit-level ngspice check each:

  Part A  P5* Beam search  -> B(B, L)
          circuit check: race over N = B*V exponential elements; time between
          the 1st and the B-th arrival vs the logit gap; Monte-Carlo mismatch
          (3 % Is, 0.3 % n) on top-B set correctness.

  Part B  L5 MCTS          -> B(M nodes, d)
          circuit check: d-level sequential race chain (b = 8 children per
          level, each level gated on the previous level's WTA detection,
          comparator delay modelled as a stated first-order lag).

Fidelity level (spice/README.md): junction-exact, wiring-ideal.  Every current
comes from a real diode I-V law with per-device Is/n mismatch; current mirrors,
the WTA threshold and the gating are ideal behavioural elements with stated
delays.

Gotchas honoured (spice/README.md):
  * every junction here is driven by an externally computed voltage, so VT MUST
    be ngspice's SPICE3-legacy 0.025864890 V, not the CODATA value;
  * junction currents are kept at uA scale (Imax = 2 uA, tail elements >= ~1 nA)
    so that the default GMIN = 1e-12 S leakage stays out of the signal path;
  * multiprocessing is used from a real .py file with OPENBLAS_NUM_THREADS=1
    (reference_python_mp_gotcha / house rule), max 8 workers.

Usage:
    python3 bounded_recursion_latency.py --part a
    python3 bounded_recursion_latency.py --part b
    python3 bounded_recursion_latency.py --part all --mc-runs 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# ---------------------------------------------------------------------------
# Constants.  Sourced constants are marked [S]; design constants declared in the
# pre-registration are marked [D].
# ---------------------------------------------------------------------------
VT = 0.025864890          # [S] spice/README.md -- ngspice kT/q at 27 C
IS_NOM = 1e-14            # [D] diode saturation current
N_NOM = 1.0               # [D] nominal ideality factor
C_INT = 10e-15            # [D] race integration capacitor, 10 fF
V_TH = 0.2                # [D] race threshold, 200 mV
I_MAX = 2e-6              # [D] current of the top-logit element, 2 uA
V_DD = 0.8                # [D] supply for energy accounting
T_CMP = 200e-12           # [D] WTA / comparator detection delay
R_FILT = 1000.0           # [D] comparator lag network
C_FILT = T_CMP / math.log(2.0) / R_FILT

T1_NS = C_INT * V_TH / I_MAX * 1e9        # fastest arrival, ns  (= 1.0)
V_OFF = N_NOM * VT * math.log(I_MAX / IS_NOM)   # bias so that z = 0 -> I_MAX

# paper's transition costs [S] prime_compiler_v2.tex section 4.4
E_ADC, T_ADC = 100e-15, 1e-9
E_DAC, T_DAC = 50e-15, 0.5e-9
T_AGC_COLD, T_AGC_STEP = 4.4e-9, 5.3e-9          # [S] section 5.3 Result 3
T_SOFTMAX_COLD, T_SOFTMAX_SWITCH = 3.2e-9, 2.1e-9  # [S] section 5.3 Result 5

MISMATCH_IS = 0.03        # [D/task] 3 % sigma on Is
MISMATCH_N = 0.003        # [D/task] 0.3 % sigma on n

NGSPICE = shutil.which("ngspice") or "ngspice"
MEAS_RE = re.compile(r"^\s*([a-z_]\w*)\s*=\s*([-+0-9.eE]+)")


# ---------------------------------------------------------------------------
# ngspice driver
# ---------------------------------------------------------------------------
def run_ngspice(netlist: str, tag: str, keep: bool = False) -> dict:
    """Run one netlist in batch mode; return the .measure results as a dict."""
    d = Path(tempfile.mkdtemp(prefix=f"brl_{tag}_", dir=str(OUT / "tmp")))
    cir = d / "n.cir"
    cir.write_text(netlist)
    try:
        r = subprocess.run([NGSPICE, "-b", str(cir)], capture_output=True,
                           text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {}
    res = {}
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        m = MEAS_RE.match(line)
        if m and "failed" not in line.lower():
            try:
                res[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return res


OPTIONS = (".options gmin=1e-12 reltol=1e-5 abstol=1e-15 vntol=1e-9 "
           "chgtol=1e-17 temp=27")


# ---------------------------------------------------------------------------
# Part A -- beam-search race
# ---------------------------------------------------------------------------
def race_netlist(z, is_dev, n_dev, tstop_ns, tstep_ps=20.0, measure_idx=None):
    """N-element exponential race.  z = logits (max should be 0)."""
    N = len(z)
    L = ["* bounded-recursion Part A: race over N exponential elements", OPTIONS]
    for i in range(N):
        v = N_NOM * VT * z[i] + V_OFF          # ideal drive (wiring-ideal)
        L += [f".model DN{i} D(IS={is_dev[i]:.9e} N={n_dev[i]:.9f})",
              f"V{i} a{i} 0 {v:.9f}",
              f"D{i} a{i} m{i} DN{i}",
              f"Vs{i} m{i} 0 0",
              f"B{i} 0 c{i} I=I(Vs{i})",
              f"C{i} c{i} 0 {C_INT:.6e}"]
    L.append(f".tran {tstep_ps:g}p {tstop_ns:g}n uic")
    idx = range(N) if measure_idx is None else measure_idx
    for i in idx:
        L.append(f".measure tran t{i} WHEN v(c{i})={V_TH} RISE=1")
    L.append(".end")
    return "\n".join(L) + "\n"


def logits_topgap(N, B, g1B, rng):
    """Top-B spread over g1B nats, the rest safely below."""
    z = np.empty(N)
    z[:B] = -np.linspace(0.0, g1B, B)
    tail = g1B + 0.5 + rng.exponential(1.0, N - B)
    z[B:] = -np.sort(tail)
    return z


def logits_selgap(N, B, g_sel, rng):
    """Top-B tight cluster; the (B+1)-th exactly g_sel below the B-th."""
    z = np.empty(N)
    z[:B] = -np.sort(rng.uniform(0.0, 0.05, B))
    z[B] = z[B - 1] - g_sel
    if N > B + 1:
        z[B + 1:] = z[B] - np.sort(rng.exponential(0.4, N - B - 1))
    return z


def _mc_one(args):
    N, B, g_sel, seed, tstop = args
    rng = np.random.default_rng(seed)
    z = logits_selgap(N, B, g_sel, rng)
    is_dev = IS_NOM * np.clip(1.0 + MISMATCH_IS * rng.standard_normal(N), 0.2, 5)
    n_dev = N_NOM * (1.0 + MISMATCH_N * rng.standard_normal(N))
    res = run_ngspice(race_netlist(z, is_dev, n_dev, tstop), "mc")
    if len(res) < B + 1:
        return None
    times = {int(k[1:]): v for k, v in res.items() if k.startswith("t")}
    order = sorted(times, key=lambda i: times[i])[:B]
    ok = set(order) == set(range(B))          # truth: indices 0..B-1
    # also record the realised effective-logit error of every measured element
    return ok


def part_a(mc_runs, workers, seed=2026):
    rng = np.random.default_rng(seed)
    out = {"design": {"C_int_F": C_INT, "V_th_V": V_TH, "I_max_A": I_MAX,
                      "t1_ns": T1_NS, "V_T": VT, "t_cmp_s": T_CMP}}

    # --- A1: deterministic gap sweep, ideal devices -------------------------
    B = 4
    gaps = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    print("\n[A1] race: 1st vs B-th arrival (B=4), ideal devices")
    print(f"{'N':>6} {'g_1B':>6} {'t_1/ns':>9} {'t_B/ns':>10} "
          f"{'dt/ns':>10} {'t_B/t_1':>9} {'pred':>9} {'err%':>7}")
    for N in (64, 256, 1024):
        for g in gaps:
            z = logits_topgap(N, B, g, rng)
            ones = np.full(N, IS_NOM)
            nn = np.full(N, N_NOM)
            tstop = max(8.0, 3.0 * T1_NS * math.exp(g))
            res = run_ngspice(race_netlist(z, ones, nn, tstop), "a1")
            t = np.array([res.get(f"t{i}", np.inf) for i in range(N)])
            srt = np.sort(t)
            t1, tB = srt[0] * 1e9, srt[B - 1] * 1e9
            pred = T1_NS * math.exp(g)
            rows.append(dict(N=N, g=g, t1_ns=t1, tB_ns=tB, dt_ns=tB - t1,
                             ratio=tB / t1, pred_ratio=math.exp(g),
                             pred_tB_ns=pred,
                             err_pct=100 * (tB - pred) / pred))
            print(f"{N:>6} {g:>6.2f} {t1:>9.4f} {tB:>10.4f} {tB-t1:>10.4f} "
                  f"{tB/t1:>9.4f} {math.exp(g):>9.4f} "
                  f"{100*(tB-pred)/pred:>7.3f}")
    out["A1_gap_sweep"] = rows

    # --- A2: energy law check ----------------------------------------------
    # E_race = V_dd * t_(B) * sum_i I_i ;  I_i = C*V_th/t_i  (exact for a ramp)
    print("\n[A2] race energy vs N and Z_norm (from the A1 transients)")
    erows = []
    for N in (64, 256, 1024):
        for g in (0.25, 1.0, 2.0):
            z = logits_topgap(N, B, g, rng)
            ones = np.full(N, IS_NOM)
            nn = np.full(N, N_NOM)
            tstop = max(8.0, 3.0 * T1_NS * math.exp(g))
            res = run_ngspice(race_netlist(z, ones, nn, tstop), "a2")
            t = np.array([res.get(f"t{i}", np.inf) for i in range(N)])
            I = C_INT * V_TH / np.where(np.isfinite(t), t, np.inf)
            # elements that never fire inside tstop: use the analytic current
            miss = ~np.isfinite(t)
            I[miss] = I_MAX * np.exp(z[miss])
            tB = np.sort(t)[B - 1]
            Z = float(np.sum(np.exp(z - z.max())))
            E = V_DD * tB * float(I.sum())
            E_pred = V_DD * C_INT * V_TH * Z * math.exp(g)
            erows.append(dict(N=N, g=g, Z_norm=Z, E_fJ=E * 1e15,
                              E_pred_fJ=E_pred * 1e15,
                              E_adc_all_fJ=N * E_ADC * 1e15,
                              E_adc_topB_fJ=B * E_ADC * 1e15))
            print(f"  N={N:>5} g={g:<5} Z={Z:>8.3f}  E_race={E*1e15:>9.3f} fJ "
                  f"(pred {E_pred*1e15:>9.3f})  ADC-all={N*E_ADC*1e15:>8.1f} fJ")
    out["A2_energy"] = erows

    # --- A3: Monte-Carlo mismatch on the top-B set --------------------------
    print(f"\n[A3] MC mismatch ({MISMATCH_IS:.0%} Is, {MISMATCH_N:.1%} n), "
          f"{mc_runs} draws; P(top-{B} set correct) vs selection gap")
    gsels = [0.1, 0.25, 0.5, 1.0, 2.0]
    mrows = []
    print(f"{'N':>6} " + " ".join(f"{g:>8.2f}" for g in gsels))
    for N in (64, 256, 1024):
        line = []
        for g in gsels:
            tstop = max(10.0, 3.0 * T1_NS * math.exp(g + 0.3))
            jobs = [(N, B, g, seed * 100003 + N * 7919 + int(g * 1000) * 97 + k,
                     tstop) for k in range(mc_runs)]
            with Pool(workers) as p:
                got = p.map(_mc_one, jobs, chunksize=4)
            ok = [g_ for g_ in got if g_ is not None]
            frac = float(np.mean(ok)) if ok else float("nan")
            mrows.append(dict(N=N, g_sel=g, p_correct=frac, n_valid=len(ok)))
            line.append(f"{frac:>8.3f}")
        print(f"{N:>6} " + " ".join(line))
    out["A3_mc"] = mrows

    # --- A4: effective-logit error budget (analytic, SPICE-anchored) --------
    # ln(I_i/I_nom) = ln(1+dIs) + (z_i + ln(I_max/Is))*(1/n_i - 1)
    sig_is = MISMATCH_IS
    sig_n_term = abs(math.log(I_MAX / IS_NOM)) * MISMATCH_N
    sig_tot = math.hypot(sig_is, sig_n_term)
    out["A4_error_budget"] = dict(sigma_from_Is_nats=sig_is,
                                  sigma_from_n_nats=sig_n_term,
                                  ln_Imax_over_Is=math.log(I_MAX / IS_NOM),
                                  sigma_total_nats=sig_tot,
                                  sigma_pairwise_nats=sig_tot * math.sqrt(2))
    print(f"\n[A4] effective-logit error budget: sigma(Is) = {sig_is:.4f} nat, "
          f"sigma(n) = {sig_n_term:.4f} nat (amplified by "
          f"ln(Imax/Is) = {math.log(I_MAX/IS_NOM):.2f}), "
          f"sigma_tot = {sig_tot:.4f} nat, pairwise {sig_tot*1.41421:.4f} nat")

    # --- A5: realistic-tail extrapolation to the real beam-search N ---------
    # numpy only (N = 131072 in SPICE is out of budget); uses the SPICE-verified
    # law t_i = C*V_th/I_i with the same mismatch model.
    print("\n[A5] realistic tail, N = B*V = 4*32768 = 131072 (numpy, "
          "SPICE-verified arrival law)")
    rrows = []
    rng5 = np.random.default_rng(seed + 1)
    Nreal = 4 * 32768
    for g in gsels:
        ok = 0
        trials = 400
        for _ in range(trials):
            z = np.empty(Nreal)
            z[:B] = -np.sort(rng5.uniform(0.0, 0.05, B))
            z[B] = z[B - 1] - g
            z[B + 1:] = z[B] - np.sort(rng5.exponential(0.4, Nreal - B - 1))
            eff = (z + np.log(1 + MISMATCH_IS * rng5.standard_normal(Nreal))
                   + (z + math.log(I_MAX / IS_NOM))
                   * (1.0 / (1 + MISMATCH_N * rng5.standard_normal(Nreal)) - 1))
            ok += set(np.argsort(-eff)[:B]) == set(range(B))
        rrows.append(dict(g_sel=g, p_correct=ok / trials, N=Nreal))
        print(f"  g_sel={g:<5} P(top-{B} set correct) = {ok/trials:.3f}")
    out["A5_large_N"] = rrows
    return out


# ---------------------------------------------------------------------------
# Part B -- MCTS selection chain
# ---------------------------------------------------------------------------
def chain_netlist(zs, is_dev, n_dev, d, b, tstop_ns, tstep_ps=5.0):
    L = ["* bounded-recursion Part B: d-level sequential race chain", OPTIONS,
         "Ven0 en0 0 DC 1"]
    for k in range(d):
        for j in range(b):
            v = N_NOM * VT * zs[k, j] + V_OFF
            L += [f".model D{k}_{j} D(IS={is_dev[k,j]:.9e} N={n_dev[k,j]:.9f})",
                  f"V{k}_{j} a{k}_{j} 0 {v:.9f}",
                  f"D{k}_{j} a{k}_{j} m{k}_{j} D{k}_{j}",
                  f"Vs{k}_{j} m{k}_{j} 0 0",
                  f"B{k}_{j} 0 c{k}_{j} I=I(Vs{k}_{j})*u(v(en{k})-0.5)",
                  f"C{k}_{j} c{k}_{j} 0 {C_INT:.6e}"]
        e = [f"v(c{k}_{j})" for j in range(b)]
        while len(e) > 1:
            e = [f"max({e[i]},{e[i+1]})" if i + 1 < len(e) else e[i]
                 for i in range(0, len(e), 2)]
        L += [f"Bw{k} w{k} 0 V=u({e[0]}-{V_TH})",
              f"Rf{k} w{k} en{k+1} {R_FILT}",
              f"Cf{k} en{k+1} 0 {C_FILT:.9e}"]
    L.append(f".tran {tstep_ps:g}p {tstop_ns:g}n uic")
    for k in range(d):
        L.append(f".measure tran tl{k} WHEN v(en{k+1})=0.5 RISE=1")
    L.append(".end")
    return "\n".join(L) + "\n"


def _chain_one(args):
    d, b, seed, mismatch, tstop = args
    rng = np.random.default_rng(seed)
    zs = np.zeros((d, b))
    for k in range(d):
        zs[k, 0] = 0.0
        zs[k, 1:] = -rng.uniform(0.2, 2.0, b - 1)   # UCB spread among siblings
    if mismatch:
        isd = IS_NOM * np.clip(1 + MISMATCH_IS * rng.standard_normal((d, b)),
                               0.2, 5)
        nd = N_NOM * (1 + MISMATCH_N * rng.standard_normal((d, b)))
    else:
        isd = np.full((d, b), IS_NOM)
        nd = np.full((d, b), N_NOM)
    res = run_ngspice(chain_netlist(zs, isd, nd, d, b, tstop), "b")
    key = f"tl{d-1}"
    return res.get(key, float("nan"))


def part_b(mc_runs, workers, seed=2026):
    out = {}
    b = 8
    print("\n[B1] d-level sequential race chain (b = 8), nominal devices")
    print(f"{'d':>4} {'t_sel/ns':>10} {'per level/ns':>13} "
          f"{'model d*(t1+t_cmp)':>20}")
    rows = []
    for d in (8, 16, 32):
        tstop = 2.0 * d * (T1_NS + T_CMP * 1e9) + 10
        t = _chain_one((d, b, seed, False, tstop))
        model = d * (T1_NS + T_CMP * 1e9)
        rows.append(dict(d=d, t_sel_ns=t * 1e9, per_level_ns=t * 1e9 / d,
                         model_ns=model, err_pct=100 * (t * 1e9 - model) / model))
        print(f"{d:>4} {t*1e9:>10.4f} {t*1e9/d:>13.4f} {model:>20.4f}")
    out["B1_nominal"] = rows

    print(f"\n[B2] same chain under mismatch ({MISMATCH_IS:.0%} Is, "
          f"{MISMATCH_N:.1%} n), {mc_runs} draws")
    print(f"{'d':>4} {'mean/ns':>10} {'sd/ns':>9} {'sd/mean':>9} "
          f"{'p5/ns':>9} {'p95/ns':>9}")
    mrows = []
    for d in (8, 16, 32):
        tstop = 3.0 * d * (T1_NS + T_CMP * 1e9) + 20
        jobs = [(d, b, seed * 7919 + d * 104729 + k, True, tstop)
                for k in range(mc_runs)]
        with Pool(workers) as p:
            got = np.array(p.map(_chain_one, jobs, chunksize=2), dtype=float)
        got = got[np.isfinite(got)] * 1e9
        mrows.append(dict(d=d, n=len(got), mean_ns=float(got.mean()),
                          sd_ns=float(got.std(ddof=1)),
                          cv=float(got.std(ddof=1) / got.mean()),
                          p5_ns=float(np.percentile(got, 5)),
                          p95_ns=float(np.percentile(got, 95))))
        print(f"{d:>4} {got.mean():>10.4f} {got.std(ddof=1):>9.4f} "
              f"{got.std(ddof=1)/got.mean():>9.5f} "
              f"{np.percentile(got,5):>9.4f} {np.percentile(got,95):>9.4f}")
    out["B2_mismatch"] = mrows
    return out


# ---------------------------------------------------------------------------
# Cost models (analytic, on top of the SPICE-measured primitives)
# ---------------------------------------------------------------------------
def cost_models(a_res, b_res):
    out = {}
    B, V, L = 4, 32768, 64
    N = B * V

    # --- beam search --------------------------------------------------------
    # measured B-th arrival ratio at a given top gap comes from A1; use the
    # verified law t_(B) = t_1 * exp(g_1B).
    rows = []
    for g1B in (0.25, 0.5, 1.0, 2.0):
        t_race = T1_NS * 1e-9 * math.exp(g1B)
        for slot, t_copy in (("switched-cap slot", 1e-9),
                             ("memristive slot 10 ns", 10e-9),
                             ("memristive slot 100 ns", 100e-9),
                             ("memristive slot 1 us", 1e-6)):
            for ro, t_ro in (("B serial ADC", B * T_ADC),
                             ("B parallel TDC", T_ADC)):
                t_step = (T_DAC + T_SOFTMAX_COLD + t_race + T_CMP + t_ro
                          + t_copy)
                rows.append(dict(g1B=g1B, slot=slot, readout=ro,
                                 t_race_s=t_race, t_step_s=t_step,
                                 t_seq_L_s=t_step * L))
    out["beam_latency"] = rows

    Z_typ = 50.0   # peaked LM distribution; swept below
    ebeam = []
    for Z in (10.0, 50.0, 200.0, 1000.0):
        for g1B in (0.25, 1.0):
            E_race = V_DD * C_INT * V_TH * Z * math.exp(g1B)
            ebeam.append(dict(Z=Z, g1B=g1B, E_race_fJ=E_race * 1e15,
                              E_adc_all_nJ=N * E_ADC * 1e9,
                              E_adc_topB_fJ=B * E_ADC * 1e15,
                              E_dac_all_nJ=N * E_DAC * 1e9,
                              saving_vs_adc_all=N * E_ADC / (E_race + B * E_ADC)))
    out["beam_energy"] = ebeam
    out["beam_N"] = N

    # --- MCTS ---------------------------------------------------------------
    per_level = b_res["B1_nominal"][-1]["per_level_ns"] * 1e-9 if b_res else \
        (T1_NS * 1e-9 + T_CMP)
    mrows = []
    K = 800
    for d in (8, 16, 32):
        t_sel = d * per_level
        t_alloc = T1_NS * 1e-9 + T_CMP           # one race over free flags
        for t_wr in (10e-9, 30e-9, 100e-9, 300e-9, 1e-6):
            for bk, t_bk in (("sequential d x P1.P8",
                              d * (T1_NS * 1e-9 + T_CMP)),
                             ("parallel one-hot", T1_NS * 1e-9 + T_CMP)):
                t_sim = t_sel + t_alloc + t_wr + t_bk
                mrows.append(dict(d=d, t_wr_s=t_wr, backup=bk,
                                  t_sel_s=t_sel, t_bk_s=t_bk, t_sim_s=t_sim,
                                  write_frac=t_wr / t_sim,
                                  t_move_s=K * t_sim))
    out["mcts"] = mrows
    out["mcts_per_level_s"] = per_level
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    ap.add_argument("--mc-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    (OUT / "tmp").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a_res = b_res = None
    if args.part in ("a", "all"):
        a_res = part_a(args.mc_runs, args.workers, args.seed)
    if args.part in ("b", "all"):
        b_res = part_b(args.mc_runs, args.workers, args.seed)

    cm = cost_models(a_res, b_res) if b_res else None
    blob = dict(part_a=a_res, part_b=b_res, cost_models=cm,
                elapsed_s=time.time() - t0,
                constants=dict(VT=VT, C_INT=C_INT, V_TH=V_TH, I_MAX=I_MAX,
                               T_CMP=T_CMP, E_ADC=E_ADC, E_DAC=E_DAC,
                               T_ADC=T_ADC, T_DAC=T_DAC, V_DD=V_DD))
    p = OUT / "bounded_recursion.json"
    p.write_text(json.dumps(blob, indent=1, default=float))
    print(f"\nwrote {p}   ({time.time()-t0:.1f} s)")

    if cm:
        print("\n[C1] beam search per-step latency, B=4 V=32768 L=64 "
              f"(N = {cm['beam_N']})")
        seen = set()
        for r in cm["beam_latency"]:
            if r["readout"] != "B serial ADC" or r["g1B"] != 1.0:
                continue
            print(f"   g_1B=1.0  {r['slot']:<24} t_step = "
                  f"{r['t_step_s']*1e9:>9.2f} ns   L=64 total "
                  f"{r['t_seq_L_s']*1e6:>8.3f} us")
        print("\n[C2] MCTS per-simulation latency (b=8), K=800 per move")
        for r in cm["mcts"]:
            if r["d"] != 32:
                continue
            print(f"   d=32 t_wr={r['t_wr_s']*1e9:>7.0f} ns  {r['backup']:<22}"
                  f" t_sim = {r['t_sim_s']*1e9:>9.1f} ns  "
                  f"write {r['write_frac']*100:>5.1f} %  "
                  f"move {r['t_move_s']*1e6:>8.1f} us")


if __name__ == "__main__":
    main()
