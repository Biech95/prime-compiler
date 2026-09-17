#!/usr/bin/env python3
"""The three argument-only entries of `docs/missing_primes_mapping.md` section 6,
turned into numbers.

Companion to `docs/exp_argument_entries.md` (pre-registration in its section 0,
written before this script existed).

  E1  D8  CTC loss            -> one L=4 trellis section, log domain (Loeliger
                                 et al. 2001 sum-product) vs charge domain
                                 (switched-capacitor charge redistribution);
                                 per-section error in ngspice, accumulation over
                                 T in {16,64,256} in numpy.
  E2  A20 Speculative decoding-> k-bit coded token comparison (symbol-margin
                                 model R4/R5), match line in ngspice, latency and
                                 energy from the paper's transition constants.
  E3  L3  Experience replay   -> 256-way one-hot gated read in ngspice (leakage
                                 floor from 255 off-gates), decoder tree depth
                                 from the P1 fan-in bound, retention/storage
                                 class from the gain-cell vs memristor numbers.

Fidelity level (spice/README.md): junction-exact, wiring-ideal.

Gotchas honoured (spice/README.md, house rules):
  * VT = 0.025864890 V (ngspice SPICE3-legacy kT/q at 27 C) wherever a junction
    is driven by an externally computed voltage;
  * junction currents kept at uA scale so GMIN = 1e-12 S stays out of the path;
  * multiprocessing only from this real .py file, OPENBLAS_NUM_THREADS=1,
    <= 4 workers (CPU budget).

Usage:
    python3 argument_entries_sim.py --part all --mc-runs 200 --workers 4
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
WORK = OUT / "argent"

# ---------------------------------------------------------------------------
# Constants.  [S] = sourced (see docs/exp_argument_entries.md section 0.1),
# [D] = declared design constant.
# ---------------------------------------------------------------------------
VT = 0.025864890          # [S] spice/README.md
IS_NOM = 1e-14            # [D]
N_NOM = 1.0               # [D]
I_U = 2e-6                # [D] unit current, 2 uA

SIG_IS = 0.03             # [S] MOS-typical corner
SIG_N = 0.003             # [S]
SIG_VT = 0.010            # [S] 10 mV
SIG_G_GATE = 0.01         # [S/task] 1 % gate (tail) mismatch for E3

V_FS = 0.5                # [D] charge-domain full scale
R_ON = 1e3                # [D] switch on-resistance
KB = 1.380649e-23
T_AMB = 300.0
KT = KB * T_AMB

# transition costs [S] prime_compiler_v2.tex 4.4
E_ADC, T_ADC = 100e-15, 1e-9
E_DAC, T_DAC = 50e-15, 0.5e-9
T_CMP = 200e-12           # [D] exp_bounded_recursion.md 0.2
T_RACE = 1.2e-9           # [S] exp_bounded_recursion.md 5.1 (measured)
C_ML, V_ML = 10e-15, 1.0  # [S] exp_symbol_margin.md 8
E_LINE = C_ML * V_ML ** 2  # 10 fJ
R0_ML, RF_ML, A0_ML = 1e5, 1e5, 1e4   # [S] exp_symbol_margin.md 7
V_REF_ML = 0.1            # [S] ibid (1 uA per line)

TAU_GAINCELL = 128.6e-6   # [S] exp_kv_leak_eviction.md / kv_gain_cell.py
TAU_LEROUX = 5e-3         # [S] Leroux et al. 2025

NGSPICE = shutil.which("ngspice") or "ngspice"
MEAS_RE = re.compile(r"^\s*([a-z_]\w*)\s*=\s*([-+0-9.eE]+)")

OPTS = (".options gmin=1e-12 reltol=1e-9 abstol=1e-16 vntol=1e-10 "
        "chgtol=1e-18 temp=27")


def run_ngspice(netlist: str, tag: str, wrdata: str | None = None,
                timeout: int = 1800):
    """Run one netlist in batch mode.  Returns (.measure dict, wrdata array)."""
    d = Path(tempfile.mkdtemp(prefix=f"ae_{tag}_", dir=str(WORK)))
    cir = d / "n.cir"
    cir.write_text(netlist)
    try:
        r = subprocess.run([NGSPICE, "-b", str(cir)], capture_output=True,
                           text=True, timeout=timeout, cwd=d)
    except subprocess.TimeoutExpired:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError(f"ngspice timeout on {tag}")
    res = {}
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        m = MEAS_RE.match(line)
        if m and "failed" not in line.lower():
            try:
                res[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    arr = None
    if wrdata is not None:
        p = d / wrdata
        if not p.exists():
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
            shutil.rmtree(d, ignore_errors=True)
            raise RuntimeError(f"ngspice produced no {wrdata} for {tag}")
        arr = np.loadtxt(p)
    shutil.rmtree(d, ignore_errors=True)
    return res, arr


# ===========================================================================
# E1 -- CTC trellis section
# ===========================================================================
L_STATES = 4


def draw_section(rng):
    """One trellis section's data: alpha_in (L,), p (L,L) with p[s,sp]."""
    a = rng.uniform(0.15, 1.0, L_STATES)
    a = a / a.max()
    p = rng.uniform(0.15, 1.0, (L_STATES, L_STATES))
    p = p / p.max()
    return a, p


def draw_log_devices(rng, n):
    """Per-junction Is and n draws."""
    return (IS_NOM * (1.0 + SIG_IS * rng.standard_normal(n)),
            N_NOM * (1.0 + SIG_N * rng.standard_normal(n)))


def log_section_numpy(a, p, dev):
    """Closed-form map of the log-domain trellis section (the calculus's own
    'evaluate the map by cheap MC' mode, mismatch_calculus.md section 7).

    dev = dict with is_a,n_a (L,), is_p,n_p (L,L), is_o,n_o (L,L), is_r,n_r ().
    Returns the L output currents.
    """
    Ia = I_U * a
    Ip = I_U * p
    Va = dev["n_a"] * VT * np.log(Ia / dev["is_a"])                  # (L,)
    Vp = dev["n_p"] * VT * np.log(Ip / dev["is_p"])                  # (L,L)
    Vr = dev["n_r"] * VT * math.log(I_U / dev["is_r"])
    Vloop = Va[None, :] + Vp - Vr                                    # (L,L)
    Iout = dev["is_o"] * np.exp(Vloop / (dev["n_o"] * VT))
    return Iout.sum(axis=1)


def log_section_netlist(blocks):
    """blocks: list of (a, p, dev).  One .op netlist, wrdata of L sums per block."""
    Lst = [".title E1 log-domain trellis section (Loeliger sum-product)", OPTS]
    cols = []
    for r, (a, p, dev) in enumerate(blocks):
        # reference diode
        Lst += [f".model DR{r} D(IS={dev['is_r']:.9e} N={dev['n_r']:.9f})",
                f"IR{r} 0 nr{r} DC {I_U:.9e}",
                f"DR{r} nr{r} 0 DR{r}"]
        for s in range(L_STATES):
            Lst += [f".model DA{r}_{s} D(IS={dev['is_a'][s]:.9e} "
                    f"N={dev['n_a'][s]:.9f})",
                    f"IA{r}_{s} 0 na{r}_{s} DC {I_U*a[s]:.9e}",
                    f"DA{r}_{s} na{r}_{s} 0 DA{r}_{s}"]
        for s in range(L_STATES):
            for sp in range(L_STATES):
                Lst += [f".model DP{r}_{s}_{sp} D(IS={dev['is_p'][s,sp]:.9e} "
                        f"N={dev['n_p'][s,sp]:.9f})",
                        f"IP{r}_{s}_{sp} 0 np{r}_{s}_{sp} DC "
                        f"{I_U*p[s,sp]:.9e}",
                        f"DP{r}_{s}_{sp} np{r}_{s}_{sp} 0 DP{r}_{s}_{sp}",
                        # translinear loop closed by ideal voltage summation
                        f"B{r}_{s}_{sp} nv{r}_{s}_{sp} 0 V = "
                        f"v(na{r}_{sp})+v(np{r}_{s}_{sp})-v(nr{r})",
                        f".model DO{r}_{s}_{sp} D(IS={dev['is_o'][s,sp]:.9e} "
                        f"N={dev['n_o'][s,sp]:.9f})",
                        f"DO{r}_{s}_{sp} nv{r}_{s}_{sp} nm{r}_{s}_{sp} "
                        f"DO{r}_{s}_{sp}",
                        f"VS{r}_{s}_{sp} nm{r}_{s}_{sp} 0 0"]
            expr = "+".join(f"I(VS{r}_{s}_{sp})" for sp in range(L_STATES))
            Lst.append(f"BS{r}_{s} nsum{r}_{s} 0 V = ({expr})*1e6")  # in uA
            Lst.append(f"RS{r}_{s} nsum{r}_{s} 0 1e12")
            cols.append(f"v(nsum{r}_{s})")
    Lst += [".control", "op", "wrdata e1log.txt " + " ".join(cols),
            ".endc", ".end"]
    return "\n".join(Lst) + "\n", len(cols)


def _e1log_worker(job):
    seed, nblk = job
    rng = np.random.default_rng(seed)
    blocks, ideal = [], []
    for _ in range(nblk):
        a, p = draw_section(rng)
        dev = dict(
            is_a=IS_NOM * (1 + SIG_IS * rng.standard_normal(L_STATES)),
            n_a=N_NOM * (1 + SIG_N * rng.standard_normal(L_STATES)),
            is_p=IS_NOM * (1 + SIG_IS * rng.standard_normal((L_STATES,) * 2)),
            n_p=N_NOM * (1 + SIG_N * rng.standard_normal((L_STATES,) * 2)),
            is_o=IS_NOM * (1 + SIG_IS * rng.standard_normal((L_STATES,) * 2)),
            n_o=N_NOM * (1 + SIG_N * rng.standard_normal((L_STATES,) * 2)),
            is_r=IS_NOM * (1 + SIG_IS * rng.standard_normal()),
            n_r=N_NOM * (1 + SIG_N * rng.standard_normal()),
        )
        blocks.append((a, p, dev))
        ideal.append(I_U * (p * a[None, :]).sum(axis=1))
    net, ncol = log_section_netlist(blocks)
    _, arr = run_ngspice(net, "e1log", wrdata="e1log.txt")
    row = arr if arr.ndim == 1 else arr[0]
    vals = row[1::2] * 1e-6          # back to amps
    assert vals.size == ncol, (vals.size, ncol)
    spice = vals.reshape(nblk, L_STATES)
    npmod = np.array([log_section_numpy(a, p, d) for a, p, d in blocks])
    return spice, npmod, np.array(ideal)


def charge_section_numpy(a, p, dC, Cu, Ctot):
    """Charge-redistribution MAC: V_s = sum_sp C_ss'(1+d) V_sp / Ctot(1+dbar)."""
    Vin = V_FS * a
    Cw = p * Cu * (1.0 + dC["w"])                  # (L,L)
    Cd = np.maximum(Ctot - (p * Cu).sum(axis=1), 0.05 * Cu) * (1.0 + dC["d"])
    num = (Cw * Vin[None, :]).sum(axis=1)
    den = Cw.sum(axis=1) + Cd
    return num / den


def charge_section_netlist(blocks, Cu, Ctot, tstop_ns=2.0):
    Lst = [".title E1 charge-domain trellis section (SC redistribution)", OPTS]
    for r, (a, p, dC) in enumerate(blocks):
        Vin = V_FS * a
        for s in range(L_STATES):
            for sp in range(L_STATES):
                cval = p[s, sp] * Cu * (1.0 + dC["w"][s, sp])
                Lst += [f"C{r}_{s}_{sp} t{r}_{s}_{sp} 0 {cval:.9e} "
                        f"IC={Vin[sp]:.9f}",
                        f"R{r}_{s}_{sp} t{r}_{s}_{sp} q{r}_{s} {R_ON:.4e}"]
            cd = max(Ctot - (p[s] * Cu).sum(), 0.05 * Cu) * (1.0 + dC["d"][s])
            Lst += [f"CD{r}_{s} q{r}_{s} 0 {cd:.9e} IC=0",
                    f"RL{r}_{s} q{r}_{s} 0 1e13"]
    Lst.append(f".tran 1p {tstop_ns:g}n uic")
    for r in range(len(blocks)):
        for s in range(L_STATES):
            Lst.append(f".measure tran v{r}_{s} FIND v(q{r}_{s}) "
                       f"AT={tstop_ns:g}n")
    Lst.append(".end")
    return "\n".join(Lst) + "\n"


def _e1chg_worker(job):
    seed, nblk, sigC, Cu = job
    rng = np.random.default_rng(seed)
    Ctot = 1.35 * Cu * L_STATES / 2.0   # generous, never clipped
    blocks, ideal, npmod = [], [], []
    for _ in range(nblk):
        a, p = draw_section(rng)
        dC = dict(w=sigC * rng.standard_normal((L_STATES,) * 2),
                  d=sigC * rng.standard_normal(L_STATES))
        blocks.append((a, p, dC))
        z = dict(w=np.zeros((L_STATES,) * 2), d=np.zeros(L_STATES))
        ideal.append(charge_section_numpy(a, p, z, Cu, Ctot))
        npmod.append(charge_section_numpy(a, p, dC, Cu, Ctot))
    net = charge_section_netlist(blocks, Cu, Ctot)
    res, _ = run_ngspice(net, "e1chg")
    spice = np.array([[res.get(f"v{r}_{s}", np.nan) for s in range(L_STATES)]
                      for r in range(nblk)])
    return spice, np.array(npmod), np.array(ideal)


def decompose(circ, ideal):
    """Return (e_lnZ, e_pc) per row.  e_lnZ = common-mode (log of the sum
    ratio); e_pc = RMS per-channel residual after removing the best scalar."""
    ratio = circ / ideal
    lnZ = np.log(circ.sum(axis=1) / ideal.sum(axis=1))
    # best scalar in the least-squares sense on the log ratios, energy weighted
    w = ideal / ideal.sum(axis=1, keepdims=True)
    lr = np.log(ratio)
    mu = (w * lr).sum(axis=1, keepdims=True)
    e_pc = np.sqrt((w * (lr - mu) ** 2).sum(axis=1))
    return lnZ, e_pc, np.sqrt((w * lr ** 2).sum(axis=1))


def e1_chain_log(n_dev, n_seq, T, rng, sigma_vos, reuse):
    """Forward recursion over T sections, log domain, closed-form map."""
    errs = np.empty((n_dev, n_seq))
    for d in range(n_dev):
        dev0 = dict(
            is_a=IS_NOM * (1 + SIG_IS * rng.standard_normal(L_STATES)),
            n_a=N_NOM * (1 + SIG_N * rng.standard_normal(L_STATES)),
            is_p=IS_NOM * (1 + SIG_IS * rng.standard_normal((L_STATES,) * 2)),
            n_p=N_NOM * (1 + SIG_N * rng.standard_normal((L_STATES,) * 2)),
            is_o=IS_NOM * (1 + SIG_IS * rng.standard_normal((L_STATES,) * 2)),
            n_o=N_NOM * (1 + SIG_N * rng.standard_normal((L_STATES,) * 2)),
            is_r=IS_NOM * (1 + SIG_IS * rng.standard_normal()),
            n_r=N_NOM * (1 + SIG_N * rng.standard_normal()))
        for q in range(n_seq):
            a_c, a_i = None, None
            lnP_c = lnP_i = 0.0
            for t in range(T):
                if a_c is None:
                    a_c, _ = draw_section(rng)
                    a_i = a_c.copy()
                    _, p = draw_section(rng)
                else:
                    _, p = draw_section(rng)
                dev = dev0 if reuse else dict(
                    is_a=IS_NOM * (1 + SIG_IS * rng.standard_normal(L_STATES)),
                    n_a=N_NOM * (1 + SIG_N * rng.standard_normal(L_STATES)),
                    is_p=IS_NOM * (1 + SIG_IS
                                   * rng.standard_normal((L_STATES,) * 2)),
                    n_p=N_NOM * (1 + SIG_N
                                 * rng.standard_normal((L_STATES,) * 2)),
                    is_o=IS_NOM * (1 + SIG_IS
                                   * rng.standard_normal((L_STATES,) * 2)),
                    n_o=N_NOM * (1 + SIG_N
                                 * rng.standard_normal((L_STATES,) * 2)),
                    is_r=IS_NOM * (1 + SIG_IS * rng.standard_normal()),
                    n_r=N_NOM * (1 + SIG_N * rng.standard_normal()))
                Ic = log_section_numpy(a_c, p, dev)
                Ii = I_U * (p * a_i[None, :]).sum(axis=1)
                if sigma_vos > 0:
                    Ic = Ic * (1 + (sigma_vos / V_FS)
                               * rng.standard_normal(L_STATES))
                Zc, Zi = Ic.sum(), Ii.sum()
                lnP_c += math.log(Zc / I_U)
                lnP_i += math.log(Zi / I_U)
                a_c = Ic / Ic.max()
                a_i = Ii / Ii.max()
            errs[d, q] = lnP_c - lnP_i
    return errs


def e1_chain_charge(n_dev, n_seq, T, rng, sigC, Cu, sigma_vos, reuse):
    Ctot = 1.35 * Cu * L_STATES / 2.0
    kTC = math.sqrt(KT / Cu)
    errs = np.empty((n_dev, n_seq))
    for d in range(n_dev):
        d0 = dict(w=sigC * rng.standard_normal((L_STATES,) * 2),
                  d=sigC * rng.standard_normal(L_STATES))
        for q in range(n_seq):
            a_c = a_i = None
            lnP_c = lnP_i = 0.0
            for t in range(T):
                if a_c is None:
                    a_c, _ = draw_section(rng)
                    a_i = a_c.copy()
                _, p = draw_section(rng)
                dC = d0 if reuse else dict(
                    w=sigC * rng.standard_normal((L_STATES,) * 2),
                    d=sigC * rng.standard_normal(L_STATES))
                Vc = charge_section_numpy(a_c, p, dC, Cu, Ctot)
                Vc = Vc + kTC * rng.standard_normal(L_STATES)
                if sigma_vos > 0:
                    Vc = Vc + sigma_vos * rng.standard_normal(L_STATES)
                z = dict(w=np.zeros((L_STATES,) * 2), d=np.zeros(L_STATES))
                Vi = charge_section_numpy(a_i, p, z, Cu, Ctot)
                Vc = np.maximum(Vc, 1e-9)
                lnP_c += math.log(Vc.sum() / V_FS)
                lnP_i += math.log(Vi.sum() / V_FS)
                a_c = Vc / Vc.max()
                a_i = Vi / Vi.max()
            errs[d, q] = lnP_c - lnP_i
    return errs


def part_e1(mc_runs, workers, seed):
    print("\n" + "=" * 72)
    print("E1  D8 CTC loss -- one L=4 trellis section, two domains")
    print("=" * 72)
    out = {}

    # ---- E1a: log-domain section in ngspice --------------------------------
    nblk = 10
    njob = max(1, mc_runs // nblk)
    jobs = [(seed * 7919 + 101 * i, nblk) for i in range(njob)]
    t0 = time.time()
    with Pool(workers) as pool:
        got = pool.map(_e1log_worker, jobs)
    spice = np.vstack([g[0] for g in got])
    npmod = np.vstack([g[1] for g in got])
    ideal = np.vstack([g[2] for g in got])
    lnZ_s, epc_s, etot_s = decompose(spice, ideal)
    lnZ_n, epc_n, etot_n = decompose(npmod, ideal)
    pair = np.abs(spice / npmod - 1.0)
    out["e1a_log"] = dict(
        n=int(spice.shape[0]),
        sigma_lnZ_spice=float(lnZ_s.std()), sigma_lnZ_numpy=float(lnZ_n.std()),
        rms_epc_spice=float(np.sqrt((epc_s ** 2).mean())),
        rms_epc_numpy=float(np.sqrt((epc_n ** 2).mean())),
        rms_etot_spice=float(np.sqrt((etot_s ** 2).mean())),
        mean_lnZ_spice=float(lnZ_s.mean()),
        paired_rms_dev=float(np.sqrt((pair ** 2).mean())),
        paired_max_dev=float(pair.max()), elapsed_s=time.time() - t0)
    e = out["e1a_log"]
    print(f"[E1a] log domain, {e['n']} draws, {e['elapsed_s']:.1f} s")
    print(f"   sigma(ln Z)        SPICE {e['sigma_lnZ_spice']*100:7.3f} %   "
          f"numpy {e['sigma_lnZ_numpy']*100:7.3f} %")
    print(f"   per-channel RMS    SPICE {e['rms_epc_spice']*100:7.3f} %   "
          f"numpy {e['rms_epc_numpy']*100:7.3f} %")
    print(f"   total RMS          SPICE {e['rms_etot_spice']*100:7.3f} %")
    print(f"   paired SPICE/numpy RMS dev {e['paired_rms_dev']:.3e}  "
          f"max {e['paired_max_dev']:.3e}")

    # ---- E1b: charge-domain section in ngspice -----------------------------
    out["e1b_charge"] = []
    for sigC in (0.001, 0.01):
        for Cu in (100e-15, 10e-15):
            nblk = 10
            njob = max(1, mc_runs // nblk)
            jobs = [(seed * 104729 + 37 * i + int(sigC * 1e5) + int(Cu * 1e16),
                     nblk, sigC, Cu) for i in range(njob)]
            t0 = time.time()
            with Pool(workers) as pool:
                got = pool.map(_e1chg_worker, jobs)
            sp = np.vstack([g[0] for g in got])
            nm = np.vstack([g[1] for g in got])
            idl = np.vstack([g[2] for g in got])
            ok = np.isfinite(sp).all(axis=1)
            sp, nm, idl = sp[ok], nm[ok], idl[ok]
            lnZ_s, epc_s, etot_s = decompose(sp, idl)
            lnZ_n, _, _ = decompose(nm, idl)
            pair = np.abs(sp / nm - 1.0)
            row = dict(sigma_C=sigC, C_u_F=Cu, n=int(sp.shape[0]),
                       sigma_lnZ_spice=float(lnZ_s.std()),
                       sigma_lnZ_numpy=float(lnZ_n.std()),
                       rms_epc_spice=float(np.sqrt((epc_s ** 2).mean())),
                       rms_etot_spice=float(np.sqrt((etot_s ** 2).mean())),
                       kTC_rel=float(math.sqrt(KT / Cu) / V_FS),
                       paired_rms_dev=float(np.sqrt((pair ** 2).mean())),
                       elapsed_s=time.time() - t0)
            out["e1b_charge"].append(row)
            print(f"[E1b] charge sigma_C={sigC*100:.1f}% Cu={Cu*1e15:.0f}fF "
                  f"n={row['n']}: sigma(lnZ) SPICE "
                  f"{row['sigma_lnZ_spice']*100:.4f} % / numpy "
                  f"{row['sigma_lnZ_numpy']*100:.4f} %, per-ch "
                  f"{row['rms_epc_spice']*100:.4f} %, kT/C "
                  f"{row['kTC_rel']*100:.4f} %, paired dev "
                  f"{row['paired_rms_dev']:.2e}  ({row['elapsed_s']:.1f} s)")

    # ---- E1c: accumulation over T ------------------------------------------
    print("\n[E1c] accumulation over T (numpy, closed-form map validated "
          "against SPICE above)")
    rows = []
    n_dev, n_seq = 40, 6
    for T in (16, 64, 256):
        for vos in (0.0, 1e-3, 10e-3):
            for reuse in (False, True):
                rng = np.random.default_rng(seed + T * 13 + int(vos * 1e5) * 7
                                            + int(reuse))
                e = e1_chain_log(n_dev, n_seq, T, rng, vos, reuse)
                rows.append(dict(domain="log", T=T, sigma_vos=vos,
                                 model="A-reuse" if reuse else "A-indep",
                                 bias_nat=float(e.mean()),
                                 sigma_all_nat=float(e.std()),
                                 sigma_within_dev_nat=float(
                                     e.std(axis=1).mean()),
                                 rms_nat=float(np.sqrt((e ** 2).mean()))))
    for T in (16, 64, 256):
        for sigC in (0.001, 0.01):
            for vos in (0.0, 1e-3, 10e-3):
                for reuse in (False, True):
                    rng = np.random.default_rng(seed + T * 29
                                                + int(sigC * 1e5) * 11
                                                + int(vos * 1e5) * 3
                                                + int(reuse))
                    e = e1_chain_charge(n_dev, n_seq, T, rng, sigC, 100e-15,
                                        vos, reuse)
                    rows.append(dict(domain=f"charge sigC={sigC*100:.1f}%",
                                     T=T, sigma_vos=vos,
                                     model="A-reuse" if reuse else "A-indep",
                                     bias_nat=float(e.mean()),
                                     sigma_all_nat=float(e.std()),
                                     sigma_within_dev_nat=float(
                                         e.std(axis=1).mean()),
                                     rms_nat=float(np.sqrt((e ** 2).mean()))))
    out["e1c_accum"] = rows
    print(f"{'domain':<22}{'T':>5}{'Vos/mV':>8}{'model':>10}"
          f"{'bias':>10}{'sd(all)':>10}{'sd(seq|dev)':>13}{'rms':>10}")
    for r in rows:
        print(f"{r['domain']:<22}{r['T']:>5}{r['sigma_vos']*1e3:>8.0f}"
              f"{r['model']:>10}{r['bias_nat']:>10.4f}"
              f"{r['sigma_all_nat']:>10.4f}{r['sigma_within_dev_nat']:>13.4f}"
              f"{r['rms_nat']:>10.4f}")
    return out


# ===========================================================================
# E2 -- speculative decoding token comparison
# ===========================================================================
from statistics import NormalDist
_ND = NormalDist()
Qtail = lambda z: _ND.cdf(-z)
Zeps = lambda e: -_ND.inv_cdf(e)


def e2_match_netlist(gmult, queries, theta):
    """aCAM-style match line, k lines, finite-gain transimpedance (the
    exp_symbol_margin.md section 7 protocol at k = 18)."""
    k = gmult.shape[1]
    Lst = [".title E2 token match line", OPTS]
    scale = V_REF_ML * RF_ML / R0_ML
    cols = []
    for r in range(gmult.shape[0]):
        for i in range(k):
            Lst += [f"V{r}_{i} in{r}_{i} 0 DC "
                    f"{V_REF_ML*queries[r,i]:.10e}",
                    f"R{r}_{i} in{r}_{i} vg{r} {R0_ML/gmult[r,i]:.10e}"]
        Lst += [f"E{r} o{r} 0 0 vg{r} {A0_ML:.6e}",
                f"Rf{r} vg{r} o{r} {RF_ML:.10e}",
                f"Bs{r} sc{r} 0 V = -V(o{r})/{scale:.10e}",
                f"Rl{r} sc{r} 0 1e12",
                f"Bc{r} cm{r} 0 V = (V(sc{r}) > {theta:.10e}) ? 1 : 0",
                f"Rc{r} cm{r} 0 1e12"]
        cols += [f"v(sc{r})", f"v(cm{r})"]
    Lst += [".control", "op", "wrdata e2.txt " + " ".join(cols),
            ".endc", ".end"]
    return "\n".join(Lst) + "\n"


def e2_settle_netlist(k):
    """Passive KCL match line: k source conductances charging C_ML.
    Measures the 0.1 %-settling time of the match-line node."""
    vfin = V_REF_ML * k / (k + 1.0)      # k source lines against one R0 load
    Lst = [".title E2 match-line settling", OPTS]
    for i in range(k):
        Lst += [f"V{i} in{i} 0 PULSE(0 {V_REF_ML:.6f} 0 1p 1p 10n 20n)",
                f"R{i} in{i} ml {R0_ML:.6e}"]
    Lst += [f"RL ml 0 {R0_ML:.6e}", f"CML ml 0 {C_ML:.6e}",
            ".tran 0.2p 6n uic",
            f".measure tran vfin FIND v(ml) AT=5.5n",
            f".measure tran t999 WHEN v(ml)={0.999*vfin:.10e} RISE=1",
            f".measure tran t99 WHEN v(ml)={0.99*vfin:.10e} RISE=1",
            ".end"]
    return "\n".join(Lst) + "\n"


def part_e2(mc_runs, seed):
    global T_ML_MEASURED
    print("\n" + "=" * 72)
    print("E2  A20 Speculative decoding -- coded token comparison")
    print("=" * 72)
    out = {}

    # ---- E2a: feasibility / margin table (R4, union-bounded) ---------------
    rows = []
    for V in (32768, 262144):
        k = math.ceil(math.log2(V))
        for sG in (0.03, 0.01):
            st_mis = sG * math.sqrt(k)
            for eps in (1e-3, 1e-6, 1e-9):
                z1, zu = Zeps(eps), Zeps(eps / k)
                lam = _ND.pdf(zu) / Qtail(zu)
                gmax = math.log(2) * st_mis / (k * lam)
                rows.append(dict(
                    V=V, k=k, sigma_G=sG, eps=eps, sigma_tot=st_mis,
                    z_plain=z1, z_union=zu,
                    m_star_plain=z1 * st_mis, m_star_union=zu * st_mis,
                    feasible=bool(zu * st_mis < 2.0),
                    sigma_G_max=1.0 / (zu * math.sqrt(k)),
                    gain_gmax=gmax, gain_bits=math.log2(1 / gmax),
                    thermal_room=float(
                        math.sqrt(max((1 / zu) ** 2 - st_mis ** 2, 0.0))),
                    sigma_ml_noise_max=float(
                        math.sqrt(max((1 / zu) ** 2 - st_mis ** 2, 0.0)) / k)))
    out["e2a_margin"] = rows
    print(f"{'V':>8}{'k':>4}{'sG':>6}{'eps':>8}{'s_tot':>9}{'z_u':>7}"
          f"{'m*':>8}{'of 2.0':>9}{'sG_max':>9}{'gain b':>8}{'ML noise':>10}")
    for r in rows:
        print(f"{r['V']:>8}{r['k']:>4}{r['sigma_G']*100:>5.0f}%"
              f"{r['eps']:>8.0e}{r['sigma_tot']:>9.4f}{r['z_union']:>7.3f}"
              f"{r['m_star_union']:>8.4f}{r['m_star_union']/2*100:>8.1f}%"
              f"{r['sigma_G_max']*100:>8.2f}%{r['gain_bits']:>8.1f}"
              f"{r['sigma_ml_noise_max']*100:>9.3f}%")

    # ---- E2b: ngspice match line at k = 18 ---------------------------------
    k = 18
    rng = np.random.default_rng(seed)
    sG = 0.03
    s_ln = math.sqrt(math.log1p(sG ** 2))
    nu = rng.normal(0.0, s_ln, size=(mc_runs, k))
    g = np.exp(nu - 0.5 * s_ln ** 2)
    g = np.vstack([g, g])
    q = np.ones((2 * mc_runs, k))
    q[mc_runs:, 0] = -1.0                      # Hamming-1 adversary
    st = sG * math.sqrt(k)
    m = float(2.0 - Zeps(0.1) * st)            # ~10 % false accepts, countable
    t0 = time.time()
    _, arr = run_ngspice(e2_match_netlist(g, q, k - m), "e2", wrdata="e2.txt")
    row = arr if arr.ndim == 1 else arr[0]
    vals = row[1::2]
    sc = vals[0::2]
    cm = vals[1::2]
    sc_np = (q * g).sum(axis=1)
    out["e2b_spice"] = dict(
        k=k, sigma_G=sG, n_draws=mc_runs, margin=m,
        match_mean_spice=float(sc[:mc_runs].mean()),
        match_mean_numpy=float(sc_np[:mc_runs].mean()),
        match_sd_spice=float(sc[:mc_runs].std()),
        match_sd_numpy=float(sc_np[:mc_runs].std()),
        adv_mean_spice=float(sc[mc_runs:].mean()),
        adv_sd_spice=float(sc[mc_runs:].std()),
        adv_sd_numpy=float(sc_np[mc_runs:].std()),
        sd_ratio=float(sc[:mc_runs].std() / sc_np[:mc_runs].std()),
        gain_err=float(sc[:mc_runs].mean() / sc_np[:mc_runs].mean() - 1),
        false_accept_spice=int(cm[mc_runs:].sum()),
        false_accept_numpy=int((sc_np[mc_runs:] > k - m).sum()),
        paired_rms=float(np.sqrt(((sc - sc_np) ** 2).mean())),
        elapsed_s=time.time() - t0)
    e = out["e2b_spice"]
    print(f"\n[E2b] ngspice match line k=18, sigma_G=3 %, {mc_runs} draws "
          f"({e['elapsed_s']:.1f} s)")
    print(f"   score sd  SPICE {e['match_sd_spice']:.6f}  numpy "
          f"{e['match_sd_numpy']:.6f}  ratio {e['sd_ratio']:.5f}")
    print(f"   gain error {e['gain_err']*100:.4f} %   "
          f"(finite-A0 prediction {(1+k*RF_ML/R0_ML)/A0_ML*100:.4f} %)")
    print(f"   false accepts SPICE {e['false_accept_spice']} / numpy "
          f"{e['false_accept_numpy']} of {mc_runs} at m={m:.3f}")

    # ---- E2c: match-line settling ------------------------------------------
    res, _ = run_ngspice(e2_settle_netlist(k), "e2s")
    t999 = res.get("t999", float("nan"))
    t99 = res.get("t99", float("nan"))
    out["e2c_settle"] = dict(k=k, C_ML_F=C_ML, R0=R0_ML,
                             t_settle_99_s=t99, t_settle_999_s=t999,
                             tau_pred_s=(R0_ML / (k + 1)) * C_ML)
    print(f"[E2c] match-line settle: 99 % {t99*1e12:.1f} ps, "
          f"99.9 % {t999*1e12:.1f} ps  "
          f"(RC prediction tau = {(R0_ML/(k+1))*C_ML*1e12:.1f} ps)")

    # ---- E2d: latency and energy per verified token ------------------------
    t_ML = t999 if np.isfinite(t999) else 4e-10
    T_ML_MEASURED = t_ML
    lat, ene = [], []
    for slot, t_shift in (("switched-capacitor", 1e-9), ("memristive 10 ns",
                                                         10e-9),
                          ("memristive 100 ns", 100e-9),
                          ("memristive 1 us", 1e-6)):
        t_step = t_ML + T_CMP + T_RACE + t_shift + T_ADC
        for a in (0.6, 0.8, 0.9):
            gmm = 5
            acc = (1 - a ** (gmm + 1)) / (1 - a)
            lat.append(dict(slot=slot, t_ML_s=t_ML, t_step_s=t_step,
                            gamma=gmm, accept_rate=a, exp_accepted=acc,
                            t_per_verified_token_s=t_step / acc))
    for a in (0.6, 0.8, 0.9):
        gmm = 5
        acc = (1 - a ** (gmm + 1)) / (1 - a)
        e_step = gmm * E_LINE + E_ADC + gmm * E_LINE  # compare + slot shift
        ene.append(dict(gamma=gmm, accept_rate=a, exp_accepted=acc,
                        E_step_J=e_step, E_per_verified_token_J=e_step / acc,
                        E_digital_xor_J=k * 0.1e-15,
                        ratio_vs_digital=(e_step / acc) / (k * 0.1e-15)))
    out["e2d_cost"] = dict(latency=lat, energy=ene)
    print("\n[E2d] verify-step latency / energy (gamma = 5)")
    for r in lat:
        if r["accept_rate"] != 0.8:
            continue
        print(f"   {r['slot']:<22} t_step = {r['t_step_s']*1e9:8.3f} ns   "
              f"per verified token {r['t_per_verified_token_s']*1e9:8.3f} ns")
    for r in ene:
        print(f"   a={r['accept_rate']}: E/verified token = "
              f"{r['E_per_verified_token_J']*1e15:7.2f} fJ   vs digital "
              f"18-bit XOR {r['E_digital_xor_J']*1e15:.2f} fJ  "
              f"({r['ratio_vs_digital']:.1f}x)")

    # ---- E2e: rollback register retention ----------------------------------
    rb = []
    for a in (0.6, 0.8, 0.9):
        t_step = t_ML + T_CMP + T_RACE + 1e-9 + T_ADC
        t_hold = 5 * t_step
        for V, kk in ((32768, 15), (262144, 18)):
            mstar = Zeps(1e-6 / kk) * 0.03 * math.sqrt(kk)
            tol = (2.0 - mstar) / (2.0 * kk)      # relative droop budget
            tau_req = t_hold / tol
            rb.append(dict(V=V, k=kk, accept_rate=a, t_hold_s=t_hold,
                           droop_tol=tol, tau_required_s=tau_req,
                           tau_gaincell_s=TAU_GAINCELL,
                           margin=TAU_GAINCELL / tau_req))
    out["e2e_rollback"] = rb
    r = rb[3]
    print(f"[E2e] rollback register: gamma=5 slots, t_hold = "
          f"{r['t_hold_s']*1e9:.2f} ns, droop budget {r['droop_tol']*100:.2f} %"
          f" -> tau >= {r['tau_required_s']*1e9:.0f} ns; measured gain cell "
          f"{TAU_GAINCELL*1e6:.1f} us = {r['margin']:.0f}x margin")
    return out


# ===========================================================================
# E3 -- experience replay: 256-way one-hot gated read
# ===========================================================================
FANIN = 256
T_ML_MEASURED = None


def e3_read_netlist(blocks, dv):
    """blocks: list of (x (256,), gmis (256,), vos (256,), is_a, is_b, n_a, n_b,
    j_sel).  Current-steering pair per slot; selected gate at +dv, others -dv."""
    Lst = [".title E3 256-way one-hot gated read", OPTS]
    cols = []
    vref = 0.9
    for r, b in enumerate(blocks):
        x, gm, vos, isa, isb, na, nb, jsel = b
        for j in range(FANIN):
            sel = (j == jsel)
            vg = vref + (dv if sel else -dv) + vos[j]
            Ij = I_U * x[j] * (1.0 + gm[j])
            Lst += [f".model DA{r}_{j} D(IS={isa[j]:.9e} N={na[j]:.9f})",
                    f".model DB{r}_{j} D(IS={isb[j]:.9e} N={nb[j]:.9f})",
                    f"VA{r}_{j} na{r}_{j} 0 DC {vg:.9f}",
                    f"VB{r}_{j} nb{r}_{j} 0 DC {vref:.9f}",
                    f"VSA{r}_{j} na{r}_{j} nad{r}_{j} 0",
                    f"VSB{r}_{j} nb{r}_{j} nbd{r}_{j} 0",
                    f"DA{r}_{j} nad{r}_{j} nt{r}_{j} DA{r}_{j}",
                    f"DB{r}_{j} nbd{r}_{j} nt{r}_{j} DB{r}_{j}",
                    f"IT{r}_{j} nt{r}_{j} 0 DC {Ij:.9e}"]
        tot = "+".join(f"I(VSA{r}_{j})" for j in range(FANIN))
        off = "+".join(f"I(VSA{r}_{j})" for j in range(FANIN) if j != jsel)
        Lst += [f"BT{r} ntot{r} 0 V = ({tot})*1e6",
                f"RT{r} ntot{r} 0 1e12",
                f"BO{r} noff{r} 0 V = ({off})*1e6",
                f"RO{r} noff{r} 0 1e12",
                f"BSL{r} nsl{r} 0 V = I(VSA{r}_{jsel})*1e6",
                f"RSL{r} nsl{r} 0 1e12"]
        cols += [f"v(ntot{r})", f"v(noff{r})", f"v(nsl{r})"]
    Lst += [".control", "op", "wrdata e3.txt " + " ".join(cols),
            ".endc", ".end"]
    return "\n".join(Lst) + "\n"


def _e3_worker(job):
    seed, nblk, dv, mismatch, xrange_ = job
    rng = np.random.default_rng(seed)
    blocks, truth = [], []
    for _ in range(nblk):
        x = rng.uniform(1.0 / xrange_, 1.0, FANIN)
        if mismatch:
            gm = SIG_G_GATE * rng.standard_normal(FANIN)
            vos = SIG_VT * rng.standard_normal(FANIN)
            isa = IS_NOM * (1 + SIG_IS * rng.standard_normal(FANIN))
            isb = IS_NOM * (1 + SIG_IS * rng.standard_normal(FANIN))
            na = N_NOM * (1 + SIG_N * rng.standard_normal(FANIN))
            nb = N_NOM * (1 + SIG_N * rng.standard_normal(FANIN))
        else:
            gm = np.zeros(FANIN); vos = np.zeros(FANIN)
            isa = np.full(FANIN, IS_NOM); isb = np.full(FANIN, IS_NOM)
            na = np.full(FANIN, N_NOM); nb = np.full(FANIN, N_NOM)
        jsel = int(rng.integers(FANIN))
        blocks.append((x, gm, vos, isa, isb, na, nb, jsel))
        truth.append(I_U * x[jsel])
    net = e3_read_netlist(blocks, dv)
    _, arr = run_ngspice(net, "e3", wrdata="e3.txt")
    row = arr if arr.ndim == 1 else arr[0]
    v = row[1::2] * 1e-6
    v = v.reshape(nblk, 3)
    return v, np.array(truth)


def part_e3(mc_runs, workers, seed):
    print("\n" + "=" * 72)
    print("E3  L3 Experience replay -- 256-way one-hot gated read")
    print("=" * 72)
    out = {}

    # ---- E3a: deterministic dV sweep ---------------------------------------
    rows = []
    print("[E3a] ideal devices, equal amplitudes (x = 1), dV sweep")
    for dv in (0.10, 0.15, 0.20, 0.26, 0.30, 0.40):
        v, truth = _e3_worker((seed, 1, dv, False, 1.0000001))
        tot, off, sl = v[0]
        rows.append(dict(dV=dv, I_tot=float(tot), I_off=float(off),
                         I_sel=float(sl), I_true=float(truth[0]),
                         crosstalk=float(off / truth[0]),
                         read_err=float(tot / truth[0] - 1),
                         pred_crosstalk=255 * math.exp(-dv / (N_NOM * VT))))
        r = rows[-1]
        print(f"   dV={dv*1e3:5.0f} mV  crosstalk = {r['crosstalk']*100:9.4f} %"
              f"  (law {r['pred_crosstalk']*100:9.4f} %)  read err "
              f"{r['read_err']*100:9.4f} %")
    out["e3a_sweep"] = rows

    # ---- E3b: Monte-Carlo at the repo design point -------------------------
    out["e3b_mc"] = []
    for dv in (0.30, 0.26, 0.20):
        for xr in (1.0000001, 10.0):
            nblk = 5
            njob = max(1, mc_runs // nblk)
            jobs = [(seed * 31337 + 71 * i + int(dv * 1e4) + int(xr),
                     nblk, dv, True, xr) for i in range(njob)]
            t0 = time.time()
            with Pool(workers) as pool:
                got = pool.map(_e3_worker, jobs)
            v = np.vstack([g[0] for g in got])
            tr = np.concatenate([g[1] for g in got])
            ct = v[:, 1] / tr
            re_ = v[:, 0] / tr - 1
            sel_err = v[:, 2] / tr - 1
            row = dict(dV=dv, x_range=xr, n=int(v.shape[0]),
                       crosstalk_mean=float(ct.mean()),
                       crosstalk_p95=float(np.percentile(ct, 95)),
                       read_err_mean=float(re_.mean()),
                       read_err_sd=float(re_.std()),
                       sel_err_sd=float(sel_err.std()),
                       elapsed_s=time.time() - t0)
            out["e3b_mc"].append(row)
            print(f"[E3b] dV={dv*1e3:.0f} mV, x-range {xr:.0f}:1, n={row['n']}"
                  f"  crosstalk {row['crosstalk_mean']*100:.4f} % (p95 "
                  f"{row['crosstalk_p95']*100:.4f} %)  read err "
                  f"{row['read_err_mean']*100:+.4f} % +- "
                  f"{row['read_err_sd']*100:.4f} %  "
                  f"({row['elapsed_s']:.1f} s)")

    # ---- E3c: decoder tree, latency ----------------------------------------
    t_ML = T_ML_MEASURED if T_ML_MEASURED else 3.7e-10
    tree = []
    for B in (10 ** 4, 10 ** 6):
        depth = math.ceil(math.log(B) / math.log(FANIN))
        tree.append(dict(B=B, addr_bits=math.ceil(math.log2(B)), depth=depth,
                         capacity=FANIN ** depth,
                         gates=B, predecode_lines=2 * math.ceil(math.log2(B)),
                         t_read_s=depth * (t_ML + T_CMP),
                         crosstalk_levels=depth))
    out["e3c_tree"] = tree
    print("\n[E3c] decoder tree from the P1 fan-in <= 256 bound")
    for r in tree:
        print(f"   B={r['B']:>8}: {r['addr_bits']} addr bits, depth "
              f"{r['depth']} (256^{r['depth']} = {r['capacity']}), read "
              f"{r['t_read_s']*1e9:.3f} ns")

    # ---- E3d: retention and storage class ----------------------------------
    ret = []
    for tau, nm in ((TAU_GAINCELL, "gain cell (repo, 128.6 us)"),
                    (TAU_LEROUX, "gain cell (Leroux, 5 ms)")):
        for n in (6, 8):
            t_ref = tau * (-math.log(1 - 2 ** -(n + 1)))
            p_cell = E_LINE / t_ref
            for E_wr in (0.1e-12, 1e-12, 10e-12):
                ret.append(dict(cell=nm, tau_s=tau, bits=n, t_ref_s=t_ref,
                                refresh_power_per_cell_W=p_cell,
                                E_wr_J=E_wr,
                                crossover_horizon_s=E_wr * t_ref / E_LINE))
    out["e3d_retention"] = ret
    print("\n[E3d] retention: refresh vs non-volatile write")
    for r in ret:
        if r["E_wr_J"] != 1e-12:
            continue
        print(f"   {r['cell']:<28} n={r['bits']}b  t_ref = "
              f"{r['t_ref_s']*1e6:8.3f} us  {r['refresh_power_per_cell_W']*1e9:7.3f}"
              f" nW/cell  crossover horizon {r['crossover_horizon_s']*1e3:7.3f} ms")
    bud = []
    for B in (10 ** 4, 10 ** 6):
        for dcell in (64, 128):
            for tau, nm in ((TAU_GAINCELL, "gain cell 128.6 us"),
                            (TAU_LEROUX, "gain cell 5 ms")):
                t_ref = tau * (-math.log(1 - 2 ** -7))
                P = B * dcell * E_LINE / t_ref
                bud.append(dict(B=B, cells_per_slot=dcell, cell=nm,
                                refresh_power_W=P))
        for E_wr in (1e-12,):
            for rate in (1e3,):
                for dcell in (64, 128):
                    bud.append(dict(B=B, cells_per_slot=dcell,
                                    cell=f"memristive {E_wr*1e12:.0f} pJ/cell",
                                    refresh_power_W=rate * dcell * E_wr))
    out["e3d_budget"] = bud
    print("   buffer-level power (n = 6 b, 128 cells/slot, 1e3 writes/s):")
    for r in bud:
        if r["cells_per_slot"] != 128:
            continue
        print(f"     B={r['B']:>8}  {r['cell']:<28} "
              f"{r['refresh_power_W']:.4g} W")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["e1", "e2", "e3", "all"], default="all")
    ap.add_argument("--mc-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    blob = {}
    if args.part in ("e2", "all"):
        blob["E2"] = part_e2(args.mc_runs, args.seed)
    if args.part in ("e3", "all"):
        if "E2" in blob:
            pass
        blob["E3"] = part_e3(args.mc_runs, args.workers, args.seed)
    if args.part in ("e1", "all"):
        blob["E1"] = part_e1(args.mc_runs, args.workers, args.seed)
    blob["elapsed_s"] = time.time() - t0
    blob["constants"] = dict(VT=VT, IS_NOM=IS_NOM, I_U=I_U, SIG_IS=SIG_IS,
                             SIG_N=SIG_N, SIG_VT=SIG_VT,
                             SIG_G_GATE=SIG_G_GATE, V_FS=V_FS, R_ON=R_ON,
                             E_ADC=E_ADC, E_DAC=E_DAC, T_CMP=T_CMP,
                             T_RACE=T_RACE, C_ML=C_ML, E_LINE=E_LINE,
                             TAU_GAINCELL=TAU_GAINCELL, TAU_LEROUX=TAU_LEROUX)
    p = OUT / "argument_entries.json"
    p.write_text(json.dumps(blob, indent=1, default=float))
    print(f"\nwrote {p}   ({blob['elapsed_s']:.1f} s)")


if __name__ == "__main__":
    main()
