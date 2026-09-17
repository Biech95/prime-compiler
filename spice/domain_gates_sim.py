#!/usr/bin/env python3
"""Kill-gates D1 and D2 of `docs/domain_axis.md` section 5.

Companion to `docs/exp_domain_gates.md` (its section 1 carries the
pre-registration, copied verbatim from `docs/domain_axis.md` section 5 before
this script was ever run).

D1  exp in the log domain vs exp in the time domain under identical mismatch
    arm (a)  open-loop translinear softmax  -- the netlist of
             `spice/softmax_agc_sim.py` is imported unmodified, so this arm is
             bit-identical to the cell that produced the paper's Result 5.
    arm (b)  the same logits as a time-to-threshold race: I_i = I_max e^{z_i}
             charges C to V_th, the softmax vector is read as the normalised
             inverse arrival times.  SAME junction draws as arm (a) (the first
             N of each draw's exponential stage), plus a comparator offset
             sigma_Vos = 10 mV.
    plus     a V_th sweep that locates the crossing of the comparator lever
             sigma_Vos/V_th with the (V_th-independent) log lever.

D2  P11 (a*b) in three signal domains at one corner (3 % K', 10 mV V_t,
    sigma_C in {0.1 %, 1 %}):
      (i)   subthreshold Gilbert cell (junction-exact BJT quad)
      (ii)  triode-region VCR multiplier (level-1 MOS, V_ov = 300 mV)
      (iii) charge-domain switched-capacitor multiply, 4 mismatched unit caps,
            one operand a capacitor code / charge packet, the other a voltage
      (iv)  third arm, pre-registered AGAINST the hypothesis: a four-quadrant
            analog x analog charge multiplier whose second operand must pass a
            transconductor (triode V-to-I, V_ov = 300 mV).

Fidelity level (spice/README.md convention): device-exact, wiring-ideal.
Every junction / channel is a real ngspice device with real I-V law and
Monte-Carlo mismatch; current copying, buffering and the comparator are ideal
behavioural elements, the comparator carrying an explicit offset source.

Gotchas honoured (spice/README.md):
  * VT = 0.025864890 V (ngspice SPICE3-legacy kT/q at 27 C) wherever a junction
    is driven by an externally computed voltage;
  * junction currents at uA scale; GMIN lowered to 1e-15 S for the race arm,
    whose tail elements run into the sub-nA decade;
  * multiprocessing only from a real .py file with OPENBLAS_NUM_THREADS=1,
    at most 6 workers (another SPICE job shares the machine).

Usage:
    python3 domain_gates_sim.py                 # full run, ~15 s on 6 workers
    python3 domain_gates_sim.py --quick         # smoke run
    python3 domain_gates_sim.py --gate d1
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
sys.path.insert(0, str(HERE))

# arm (a) is the published cell, imported rather than re-typed
from softmax_agc_sim import (build_softmax_netlist, VT as VT_SM,  # noqa: E402
                             ITARGET as ITARGET_SM, CORNERS as SM_CORNERS)

# ---------------------------------------------------------------------------
# Constants.  [S] = sourced, [D] = design constant declared in the
# pre-registration / docs/domain_axis.md section 0.1.
# ---------------------------------------------------------------------------
VT = 0.025864890          # [S] spice/README.md -- ngspice kT/q at 27 C
KB = 1.380649e-23         # [S] CODATA
TEMP_K = 300.0            # [D]

# --- D1 race cell (bounded_recursion_latency.py design point) --------------
IS_NOM = 1e-14            # [D] race diode saturation current
N_NOM = 1.0               # [D] nominal ideality factor
C_INT = 10e-15            # [D] race integration capacitor, 10 fF
V_TH_BASE = 0.200         # [D] race comparator threshold, 200 mV (task)
V_TH_PREREG = 0.500       # [S] threshold used in domain_axis section 5 D1
I_MAX = 2e-6              # [D] current of the top-logit element, 2 uA
SIG_VOS = 0.010           # [D] comparator input offset, 10 mV, 1 sigma
V_OFF = N_NOM * VT * math.log(I_MAX / IS_NOM)   # bias so z = 0 -> I_MAX
LN_BIAS = math.log(I_MAX / IS_NOM)              # = 19.11, the log lever

SIG_IS = 0.03             # [D] 3 % on saturation current
SIG_N = 0.003             # [D] 0.3 % on ideality factor

VTH_SWEEP = [0.050, 0.100, 0.155, 0.250, 0.400, 0.800]   # pre-registered
VTH_EXTRA = [0.200, 0.500]                               # reported alongside

# --- D2 shared corner ------------------------------------------------------
SIG_KP = 0.03             # [D] 3 % on K' (and on the BJT tail mirror)
SIG_VT_MOS = 0.010        # [D] 10 mV, differential-pair-referred (repo
                          #     convention: mamba_vcrc_sim.mc_draws applies
                          #     +vos/2 / -vos/2 to the two devices of a pair)

# (i) subthreshold Gilbert
I_EE = 10e-6              # [D] tail current, uA scale
IS_BJT = 1e-16            # [D]
FS_PAIR = VT              # [D] full scale of both operands = n V_T = 25.86 mV

# (ii) triode VCR
KP_NOM = 100e-6           # [D] K' = mu Cox
VTO_NOM = 0.7             # [D]
V_OV0 = 0.300             # [D] design point of domain_axis section 0.1
VCM_T = 0.9               # [D] common mode of the differential x nodes
FS_TRI_Y = V_OV0          # full scale of the gate operand
FS_TRI_X = 0.100          # [D] V_ds swing (cancels in the ratio)

# (iii)/(iv) charge domain
C_UNIT = 250e-15          # [D] unit cap; 4 units = 1 pF (section 0.1)
C_LOAD = 1e-12            # [D] summing-node cap
V_FS_Q = 1.0              # [D] V_FS = 1 V (section 0.1)
GBUF = 0.2                # [D] ideal interstage scaling, third arm only

GRID_NX, GRID_NY = 7, 5   # operand grid, |x|,|y| <= FS/2

NGSPICE = shutil.which("ngspice") or "ngspice"
MEAS_RE = re.compile(r"^\s*([a-z_]\w*)\s*=\s*([-+0-9.eE]+)")
PRINT_RE = re.compile(r"^\s*([a-z_(][\w()]*)\s*=\s*([-+0-9.eE]+)\s*$")

OPTS_DC = (".options gmin=1e-15 reltol=1e-6 abstol=1e-18 vntol=1e-9 "
           "temp=27")
OPTS_MOS = (".options gmin=1e-12 reltol=1e-6 abstol=1e-16 vntol=1e-10 "
            "temp=27")
OPTS_TRAN = (".options reltol=1e-6 abstol=1e-16 vntol=1e-10 chgtol=1e-18 "
             "gmin=1e-12 method=gear maxord=2 temp=27")


# ---------------------------------------------------------------------------
# ngspice driver
# ---------------------------------------------------------------------------
def _run(netlist: str, files=(), timeout=600):
    """Run one netlist in a private temp dir.  Returns (measures, {file:text})."""
    d = Path(tempfile.mkdtemp(prefix="dg_", dir=str(OUT / "tmp")))
    (d / "n.cir").write_text(netlist)
    try:
        r = subprocess.run([NGSPICE, "-b", "n.cir"], capture_output=True,
                           text=True, timeout=timeout, cwd=str(d))
    except subprocess.TimeoutExpired:
        shutil.rmtree(d, ignore_errors=True)
        return {}, {}
    meas = {}
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        m = MEAS_RE.match(line)
        if m and "failed" not in line.lower():
            try:
                meas[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    texts = {}
    for f in files:
        p = d / f
        texts[f] = p.read_text() if p.exists() else ""
    shutil.rmtree(d, ignore_errors=True)
    return meas, texts


def parse_print(txt):
    out = {}
    for line in txt.splitlines():
        m = PRINT_RE.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


# ===========================================================================
# D1
# ===========================================================================
def race_op_netlist(z, is_dev, n_dev):
    """DC operating point of N race elements: the exponential current
    generators I_i = Is_i exp((n0 VT z_i + V_OFF)/(n_i VT))."""
    N = len(z)
    L = ["* D1 arm (b): race current generators, DC operating point", OPTS_DC]
    for i in range(N):
        v = N_NOM * VT * z[i] + V_OFF
        L += [f".model DN{i} D(IS={is_dev[i]:.9e} N={n_dev[i]:.9f})",
              f"V{i} a{i} 0 {v:.9f}",
              f"D{i} a{i} m{i} DN{i}",
              f"Vs{i} m{i} 0 0"]
    pr = " ".join(f"i(Vs{i})" for i in range(N))
    L += [".control", "op", f"print {pr} > race.txt", "quit", ".endc", ".end"]
    return "\n".join(L) + "\n"


def race_tran_netlist(z, is_dev, n_dev, vth, tstop_ns):
    """Full transient of the race: each element charges its own 10 fF cap and
    trips a comparator whose threshold carries that element's offset."""
    N = len(z)
    L = ["* D1 arm (b): race transient, per-element comparator offset", OPTS_DC]
    for i in range(N):
        v = N_NOM * VT * z[i] + V_OFF
        L += [f".model DN{i} D(IS={is_dev[i]:.9e} N={n_dev[i]:.9f})",
              f"V{i} a{i} 0 {v:.9f}",
              f"D{i} a{i} m{i} DN{i}",
              f"Vs{i} m{i} 0 0",
              f"B{i} 0 c{i} I=I(Vs{i})",
              f"C{i} c{i} 0 {C_INT:.6e}"]
    L.append(f".tran {tstop_ns/4000.0:.6e}n {tstop_ns:.6e}n uic")
    for i in range(N):
        L.append(f".measure tran t{i} WHEN v(c{i})={vth[i]:.9f} RISE=1")
    L.append(".end")
    return "\n".join(L) + "\n"


def _d1_one(args):
    """One paired Monte-Carlo draw: arm (a) netlist + arm (b) currents.

    The draws are generated in the parent, in the exact order that
    softmax_agc_sim.main() generates them, so that arm (a) reproduces the
    published MOS-typical open-loop number rather than merely matching it in
    distribution."""
    N, x, is_m, n_m, eps, vos = args

    net = build_softmax_netlist(x, is_m, n_m, eps, "a")
    _, tx = _run(net, files=("sm_a.txt",))
    d = parse_print(tx["sm_a.txt"])
    try:
        p_open = np.array([d[f"i(vo{i})"] for i in range(N)]) / ITARGET_SM
    except KeyError:
        p_open = None

    # arm (b): the SAME exponential stage draws (first N devices)
    z = x - x.max()
    _, tx = _run(race_op_netlist(z, IS_NOM * is_m[:N], N_NOM * n_m[:N]),
                 files=("race.txt",))
    d = parse_print(tx["race.txt"])
    try:
        cur = np.array([d[f"i(vs{i})"] for i in range(N)])
    except KeyError:
        cur = None
    return dict(p_open=p_open, cur=cur, vos=vos)


def d1(mc_runs, workers, quick=False):
    res = {"design": dict(C_int_F=C_INT, V_th_base_V=V_TH_BASE,
                          V_th_prereg_V=V_TH_PREREG, I_max_A=I_MAX,
                          sigma_Vos_V=SIG_VOS, ln_Imax_over_Is=LN_BIAS,
                          VT=VT, sigma_Is=SIG_IS, sigma_n=SIG_N,
                          mc_runs=mc_runs)}
    print("\n" + "=" * 78)
    print("  D1  exp in the log domain vs exp in the time domain")
    print("=" * 78)
    print(f"  log lever   sqrt(dIs^2 + (ln(Imax/Is) dn)^2) = "
          f"{math.hypot(SIG_IS, LN_BIAS*SIG_N)*100:.2f} %  "
          f"(ln(Imax/Is) = {LN_BIAS:.2f})")
    for v in (V_TH_BASE, V_TH_PREREG):
        print(f"  comparator lever sigma_Vos/V_th at {v*1e3:.0f} mV = "
              f"{SIG_VOS/v*100:.2f} %  -> combined "
              f"{math.hypot(math.hypot(SIG_IS, LN_BIAS*SIG_N), SIG_VOS/v)*100:.2f} %")

    for N in ([8] if quick else [8, 64]):
        # --- logits and mismatch draws.  Replay softmax_agc_sim.main()'s own
        #     RNG sequence (seed 2026, BJT-grade corner first) so that arm (a)
        #     reproduces the published Result-5 number, not merely its
        #     distribution.
        rng = np.random.default_rng(2026)
        x = rng.standard_normal(N) * 2.0
        nd = 3 * N + 2
        corner_draws = []
        for (_lab, s_is, s_n, s_vga) in SM_CORNERS:
            lst = []
            for _ in range(200):
                lst.append((rng.lognormal(0, s_is, nd),
                            rng.normal(1.0, s_n, nd),
                            rng.normal(0.0, s_vga, N)))
            corner_draws.append(lst)
        mos = corner_draws[1][:mc_runs]          # MOS-typical corner
        vrng = np.random.default_rng(20260916 + N)
        vos_all = vrng.normal(0.0, SIG_VOS, (len(mos), N))
        p_ref = np.exp(x) / np.exp(x).sum()
        z = x - x.max()

        # ideal-device reference currents (for the lever decomposition)
        _, tx = _run(race_op_netlist(z, np.full(N, IS_NOM), np.full(N, N_NOM)),
                     files=("race.txt",))
        d = parse_print(tx["race.txt"])
        cur_ideal = np.array([d[f"i(vs{i})"] for i in range(N)])
        p_ideal_race = cur_ideal / cur_ideal.sum()
        print(f"\n  N = {N}: ideal-device race L1 = "
              f"{np.abs(p_ideal_race - p_ref).sum()*100:.4f} %  "
              f"(arrival-law / GMIN floor)")

        jobs = [(N, x, mos[k][0], mos[k][1], mos[k][2], vos_all[k])
                for k in range(len(mos))]
        t0 = time.time()
        with Pool(workers) as pool:
            got = pool.map(_d1_one, jobs, chunksize=2)
        got = [g for g in got if g["cur"] is not None and g["p_open"] is not None]
        print(f"  {len(got)}/{mc_runs} valid draws  ({time.time()-t0:.0f} s)")

        cur = np.array([g["cur"] for g in got])
        vos = np.array([g["vos"] for g in got])
        popen = np.array([g["p_open"] for g in got])

        l1_a = np.abs(popen - p_ref).sum(axis=1)
        l1_a_norm = np.abs(popen / popen.sum(axis=1, keepdims=True)
                           - p_ref).sum(axis=1)

        def l1_b(vth, use_cur=None, use_vos=None):
            c = cur if use_cur is None else use_cur
            v = vos if use_vos is None else use_vos
            inv = c / (C_INT * (vth + v))
            p = inv / inv.sum(axis=1, keepdims=True)
            return np.abs(p - p_ref).sum(axis=1)

        zeros = np.zeros_like(vos)
        ideal_cur = np.tile(cur_ideal, (len(got), 1))

        rows = []
        for vth in sorted(set(VTH_SWEEP + VTH_EXTRA)):
            both = l1_b(vth)
            junc = l1_b(vth, use_vos=zeros)          # V_th-independent
            cmp_ = l1_b(vth, use_cur=ideal_cur)
            rows.append(dict(V_th_V=vth,
                             prereg=vth in VTH_SWEEP,
                             l1_both_p50=float(np.median(both)),
                             l1_both_p95=float(np.percentile(both, 95)),
                             l1_junction_only_p50=float(np.median(junc)),
                             l1_comparator_only_p50=float(np.median(cmp_)),
                             lever_cmp=SIG_VOS / vth))
        res[f"N{N}_vth_sweep"] = rows

        junc_l1 = rows[0]["l1_junction_only_p50"]
        # comparator-only L1 is proportional to 1/V_th; solve for the crossing
        cross = [r["V_th_V"] * r["l1_comparator_only_p50"] / junc_l1
                 for r in rows if r["prereg"]]
        v_star = float(np.median(cross))

        base = [r for r in rows if abs(r["V_th_V"] - V_TH_BASE) < 1e-12][0]
        pre = [r for r in rows if abs(r["V_th_V"] - V_TH_PREREG) < 1e-12][0]
        res[f"N{N}"] = dict(
            logits=x.tolist(),
            arm_a_l1_p50=float(np.median(l1_a)),
            arm_a_l1_p95=float(np.percentile(l1_a, 95)),
            arm_a_l1_renorm_p50=float(np.median(l1_a_norm)),
            arm_b_l1_p50_200mV=base["l1_both_p50"],
            arm_b_l1_p95_200mV=base["l1_both_p95"],
            arm_b_l1_p50_500mV=pre["l1_both_p50"],
            arm_b_l1_p95_500mV=pre["l1_both_p95"],
            ideal_race_l1=float(np.abs(p_ideal_race - p_ref).sum()),
            v_star_V=v_star,
            v_star_spread=[float(min(cross)), float(max(cross))],
            junction_only_l1_p50=junc_l1)

        print(f"  arm (a) open-loop translinear : p50 "
              f"{np.median(l1_a)*100:6.2f} %  p95 "
              f"{np.percentile(l1_a,95)*100:6.2f} %"
              f"   (renormalised p50 {np.median(l1_a_norm)*100:5.2f} %)")
        print(f"  arm (b) race @ V_th = 200 mV  : p50 "
              f"{base['l1_both_p50']*100:6.2f} %  p95 "
              f"{base['l1_both_p95']*100:6.2f} %")
        print(f"  arm (b) race @ V_th = 500 mV  : p50 "
              f"{pre['l1_both_p50']*100:6.2f} %  p95 "
              f"{pre['l1_both_p95']*100:6.2f} %")
        print(f"  kill threshold 0.75*(a) = "
              f"{0.75*np.median(l1_a)*100:.2f} %  ->  "
              f"{'PASS' if base['l1_both_p50'] < 0.75*np.median(l1_a) else 'KILL'}"
              f" at 200 mV, "
              f"{'PASS' if pre['l1_both_p50'] < 0.75*np.median(l1_a) else 'KILL'}"
              f" at 500 mV")
        print(f"\n  {'V_th/mV':>8} {'pre-reg':>8} {'junction-only':>14} "
              f"{'comparator-only':>16} {'both p50':>10} {'both p95':>10}")
        for r in rows:
            print(f"  {r['V_th_V']*1e3:>8.0f} {'yes' if r['prereg'] else '(extra)':>8} "
                  f"{r['l1_junction_only_p50']*100:>13.3f}% "
                  f"{r['l1_comparator_only_p50']*100:>15.3f}% "
                  f"{r['l1_both_p50']*100:>9.3f}% {r['l1_both_p95']*100:>9.3f}%")
        print(f"  crossing V_th* = {v_star*1e3:.1f} mV  in the median-L1 metric "
              f"(pre-registered 155 mV +/- 20 % = 124-186 mV)  -> "
              f"{'PASS' if 0.8*0.155 <= v_star <= 1.2*0.155 else 'KILL'}")

        # --- the same crossing in the metric the levers are defined in -------
        # Both levers are additive perturbations of the same log-domain
        # quantity: ln I_i (junction) and -ln(1 + Vos_i/V_th) (comparator).
        sig_log = float(np.std(np.log(cur / cur_ideal)))
        sig_vos_meas = float(np.std(vos))
        v_star_nat = float(sig_vos_meas / sig_log)   # sigma_Vos/V_th = sigma_log
        ln_eff = float(np.mean(z + LN_BIAS))
        res[f"N{N}_nat"] = dict(sigma_log_nats=sig_log,
                                sigma_log_predicted=math.hypot(
                                    SIG_IS, ln_eff * SIG_N),
                                mean_ln_I_over_Is=ln_eff,
                                sigma_Vos_measured=sig_vos_meas,
                                v_star_nat_V=v_star_nat)
        print(f"  effective-logit lever: measured sigma_log = "
              f"{sig_log*100:.3f} % (lever with <ln(I/Is)> = {ln_eff:.2f}: "
              f"{math.hypot(SIG_IS, ln_eff*SIG_N)*100:.3f} %)")
        print(f"  crossing V_th* = {v_star_nat*1e3:.1f} mV  in the "
              f"effective-logit (nat) metric  -> "
              f"{'PASS' if 0.8*0.155 <= v_star_nat <= 1.2*0.155 else 'KILL'}")

        # --- rank error vs logit gap, at the base threshold ------------------
        inv_t = cur / (C_INT * (V_TH_BASE + vos))
        order_meas = np.argsort(-inv_t, axis=1)
        bins = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9]
        tri = np.triu_indices(N, 1)
        gaps = np.abs(z[tri[0]] - z[tri[1]])
        inv_pairs = ((inv_t[:, tri[0]] - inv_t[:, tri[1]]) *
                     (z[tri[0]] - z[tri[1]])) < 0
        rrows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (gaps >= lo) & (gaps < hi)
            if not m.any():
                continue
            npairs = int(m.sum()) * len(got)
            ninv = int(inv_pairs[:, m].sum())
            rrows.append(dict(gap_lo=lo, gap_hi=None if hi > 1e8 else hi,
                              n_pairs=npairs, n_inversions=ninv,
                              rate=ninv / npairs))
        m = gaps >= 0.5
        res[f"N{N}_rank"] = dict(bins=rrows,
                                 n_pairs_ge_05=int(m.sum()) * len(got),
                                 n_inversions_ge_05=int(inv_pairs[:, m].sum()),
                                 full_order_exact=float(np.mean(
                                     [np.array_equal(o, np.argsort(-z))
                                      for o in order_meas])))
        print(f"\n  rank error vs true logit gap (V_th = 200 mV):")
        for r in rrows:
            hi = "inf" if r["gap_hi"] is None else f"{r['gap_hi']:g}"
            print(f"    gap [{r['gap_lo']:g}, {hi}) : "
                  f"{r['n_inversions']:>7d} inversions / {r['n_pairs']:>8d} "
                  f"pairs = {r['rate']*100:8.4f} %")
        print(f"    gaps >= 0.5 nat : "
              f"{res[f'N{N}_rank']['n_inversions_ge_05']} / "
              f"{res[f'N{N}_rank']['n_pairs_ge_05']} pairs inverted")

        # --- transient cross-check of the arrival law ------------------------
        nchk = 5 if quick else 20
        devs = []
        for k in range(nchk):
            g = got[k]
            vth = V_TH_BASE + g["vos"]
            t_pred = C_INT * vth / g["cur"]
            meas, _ = _run(race_tran_netlist(z, IS_NOM * np.ones(N),
                                             N_NOM * np.ones(N), vth,
                                             float(t_pred.max() * 1.5e9)),
                           timeout=900)
            # ideal devices here: predicted arrival from the ideal op currents
            t_pred = C_INT * vth / cur_ideal
            for i in range(N):
                if f"t{i}" in meas and math.isfinite(meas[f"t{i}"]):
                    devs.append(abs(meas[f"t{i}"] - t_pred[i]) / t_pred[i])
        res[f"N{N}_arrival_law"] = dict(
            n=len(devs), max_rel_dev=float(np.max(devs)) if devs else None,
            median_rel_dev=float(np.median(devs)) if devs else None)
        if devs:
            print(f"\n  transient cross-check of t_i = C(V_th+Vos_i)/I_i : "
                  f"{len(devs)} arrivals, median dev "
                  f"{np.median(devs)*100:.4f} %, max {np.max(devs)*100:.4f} %")
    return res


# ===========================================================================
# D2
# ===========================================================================
def op_grid(fs_x, fs_y, ny=GRID_NY):
    """|x| <= fs_x/2, |y| <= fs_y/2 on a (GRID_NX x ny) grid."""
    xs = np.linspace(-1, 1, GRID_NX) * fs_x / 2
    ys = np.linspace(-1, 1, ny) * fs_y / 2
    return [(float(a), float(b)) for b in ys for a in xs]


def metrics(out, ref, xs, fs_y, gain):
    """(A) repo convention rel.RMS = RMS(err)/RMS(ref)  -- the metric of
           mamba_vcrc_sim.rel_rms, which produced the 3.34 % triode anchor;
       (B) lever convention: the error referred to operand-y full scale, i.e.
           the quantity the sigma_rel formulas of domain_axis section 1.1
           actually compute;
       (C) as (A) but with one constant output offset per cell removed -- the
           'learnable bias' of domain_axis section 3.1 item 7."""
    err = out - ref
    rms_ref = np.sqrt(np.mean(ref ** 2))
    a = float(np.sqrt(np.mean(err ** 2)) / rms_ref)
    b = float(np.sqrt(np.mean(err ** 2)) /
              (abs(gain) * np.sqrt(np.mean(np.asarray(xs) ** 2)) * fs_y))
    c = float(np.sqrt(np.mean((err - err.mean()) ** 2)) / rms_ref)
    return a, b, c


def enob(sig):
    return float(-math.log2(2.0 * sig)) if sig > 0 else float("inf")


# --- (i) subthreshold Gilbert ---------------------------------------------
def gilbert_netlist(grid, dtail, dy, dx1, dx2, dis):
    L = ["* D2 (i): subthreshold Gilbert quad, junction-exact", OPTS_DC,
         "Vcc vcc 0 DC 3"]
    off = {"MYA": +dy / 2, "MYB": -dy / 2, "MXA": +dx1 / 2, "MXB": -dx1 / 2,
           "MXC": +dx2 / 2, "MXD": -dx2 / 2}
    for k, (nm, o) in enumerate(off.items()):
        iss = IS_BJT * (1.0 + dis[k]) * math.exp(o / VT)
        L.append(f".model {nm} NPN(IS={iss:.8e} BF=1e6 VAF=1e12 RB=0 RE=0 "
                 "RC=0 CJE=0 CJC=0 TF=0 TR=0)")
    for g, (x, y) in enumerate(grid):
        L += [f"Vyp{g} yp{g} 0 DC {1.5 + y/2:.9f}",
              f"Vyn{g} yn{g} 0 DC {1.5 - y/2:.9f}",
              f"Vxp{g} xp{g} 0 DC {2.0 + x/2:.9f}",
              f"Vxn{g} xn{g} 0 DC {2.0 - x/2:.9f}",
              f"I{g} e{g} 0 DC {I_EE*(1+dtail):.9e}",
              f"Qy1{g} f1{g} yp{g} e{g} MYA",
              f"Qy2{g} f2{g} yn{g} e{g} MYB",
              f"Qa{g} k1{g} xp{g} f1{g} MXA",
              f"Qb{g} k2{g} xn{g} f1{g} MXB",
              f"Qc{g} k1{g} xn{g} f2{g} MXC",
              f"Qd{g} k2{g} xp{g} f2{g} MXD",
              f"Vk1{g} vcc k1{g} DC 0",
              f"Vk2{g} vcc k2{g} DC 0"]
    pr = " ".join(f"i(Vk1{g}) i(Vk2{g})" for g in range(len(grid)))
    L += [".control", "op", f"print {pr} > g.txt", "quit", ".endc", ".end"]
    return "\n".join(L) + "\n"


def _gilbert_one(args):
    grid, seed = args
    rng = np.random.default_rng(seed)
    dtail = float(rng.normal(0, SIG_KP))
    dy, dx1, dx2 = rng.normal(0, SIG_VT_MOS, 3)
    dis = rng.normal(0, SIG_KP, 6)
    _, tx = _run(gilbert_netlist(grid, dtail, dy, dx1, dx2, dis),
                 files=("g.txt",))
    d = parse_print(tx["g.txt"])
    try:
        return np.array([d[f"i(vk1{g})"] - d[f"i(vk2{g})"]
                         for g in range(len(grid))])
    except KeyError:
        return None


# --- (ii) triode VCR multiplier -------------------------------------------
def triode_netlist(grid, dkp, dvt, vov0=V_OV0):
    L = ["* D2 (ii): triode-region VCR multiplier, level-1 MOS", OPTS_MOS]
    for nm, dk, dv in (("MA", +dkp / 2, +dvt / 2), ("MB", -dkp / 2, -dvt / 2)):
        L.append(f".model {nm} NMOS(LEVEL=1 VTO={VTO_NOM+dv:.9f} "
                 f"KP={KP_NOM*(1+dk):.9e} LAMBDA=0 GAMMA=0 PHI=0.6 "
                 "CGSO=0 CGDO=0 CBD=0 CBS=0 CJ=0 CJSW=0 IS=1e-18)")
    for g, (x, y) in enumerate(grid):
        L += [f"Vp{g} p{g} 0 DC {VCM_T + x/2:.9f}",
              f"Vn{g} n{g} 0 DC {VCM_T - x/2:.9f}",
              f"Vga{g} ga{g} 0 DC {VTO_NOM + VCM_T + vov0 + y/2:.9f}",
              f"Vgb{g} gb{g} 0 DC {VTO_NOM + VCM_T + vov0 - y/2:.9f}",
              f"MA{g} p{g} ga{g} sa{g} 0 MA W=1u L=1u",
              f"Vsa{g} sa{g} n{g} DC 0",
              f"MB{g} p{g} gb{g} sb{g} 0 MB W=1u L=1u",
              f"Vsb{g} sb{g} n{g} DC 0"]
    pr = " ".join(f"i(Vsa{g}) i(Vsb{g})" for g in range(len(grid)))
    L += [".control", "op", f"print {pr} > t.txt", "quit", ".endc", ".end"]
    return "\n".join(L) + "\n"


def _triode_one(args):
    grid, seed = args
    rng = np.random.default_rng(seed)
    dkp = float(rng.normal(0, SIG_KP))
    dvt = float(rng.normal(0, SIG_VT_MOS))
    _, tx = _run(triode_netlist(grid, dkp, dvt), files=("t.txt",))
    d = parse_print(tx["t.txt"])
    try:
        return np.array([d[f"i(vsa{g})"] - d[f"i(vsb{g})"]
                         for g in range(len(grid))])
    except KeyError:
        return None


# --- (iii) charge-domain SC multiply --------------------------------------
def charge_netlist(grid, dc, dcl, vn):
    """One operand is the capacitor code m (|m| of 4 unit caps, signed by the
    bottom-plate drive), the other the sampled voltage V_x.  The 4 unit caps
    and the summing cap are the SAME physical devices for every grid point."""
    L = ["* D2 (iii): charge-domain SC multiply, 4 mismatched unit caps",
         OPTS_TRAN,
         "Vctrl ctrl 0 PWL(0 1 20n 1 20.01n 0 40n 0)",
         ".model SWM SW(RON=100 ROFF=1e12 VT=0.5 VH=0)"]
    for g, (x, m) in enumerate(grid):
        sgn = 0.0 if m == 0 else math.copysign(1.0, m)
        for j in range(4):
            s = sgn if j < abs(m) else 0.0
            v = s * x
            # bottom plates move only AFTER the reset switch is fully open
            L += [f"Vb{g}_{j} b{g}_{j} 0 PWL(0 {v:.9e} 25n {v:.9e} "
                  f"25.01n 0 40n 0)",
                  f"C{g}_{j} b{g}_{j} t{g} {C_UNIT*(1+dc[j]):.9e}"]
        L += [f"CL{g} t{g} 0 {C_LOAD*(1+dcl):.9e}",
              f"S{g} t{g} r{g} ctrl 0 SWM",
              f"Vn{g} r{g} 0 DC {vn:.9e}",
              f"Rl{g} t{g} 0 1e12"]
    L.append(".tran 0.1n 40n 0 0.2n uic")
    for g in range(len(grid)):
        L.append(f".measure tran vo{g} FIND v(t{g}) AT=39n")
    L.append(".end")
    return "\n".join(L) + "\n"


def _charge_one(args):
    grid, seed, sig_c = args
    rng = np.random.default_rng(seed)
    dc = rng.normal(0, sig_c, 4)
    dcl = float(rng.normal(0, sig_c))
    ctot = 4 * C_UNIT + C_LOAD
    vn = float(rng.normal(0, math.sqrt(KB * TEMP_K / ctot)))
    meas, _ = _run(charge_netlist(grid, dc, dcl, vn))
    try:
        return np.array([meas[f"vo{g}"] for g in range(len(grid))])
    except KeyError:
        return None


# --- (iv) third arm: charge x transconductor ------------------------------
def thirdarm_netlist(grid, dc, dcl, vn, dkp, dvt):
    L = ["* D2 (iv): four-quadrant analog x analog charge multiplier -- the "
         "second operand passes a triode transconductor", OPTS_TRAN,
         "Vctrl ctrl 0 PWL(0 1 20n 1 20.01n 0 40n 0)",
         ".model SWM SW(RON=100 ROFF=1e12 VT=0.5 VH=0)"]
    for nm, dk, dv in (("MA", +dkp / 2, +dvt / 2), ("MB", -dkp / 2, -dvt / 2)):
        L.append(f".model {nm} NMOS(LEVEL=1 VTO={VTO_NOM+dv:.9f} "
                 f"KP={KP_NOM*(1+dk):.9e} LAMBDA=0 GAMMA=0 PHI=0.6 "
                 "CGSO=0 CGDO=0 CBD=0 CBS=0 CJ=0 CJSW=0 IS=1e-18)")
    for g, (x, y) in enumerate(grid):
        for j in range(4):
            L += [f"Vb{g}_{j} b{g}_{j} 0 PWL(0 {x:.9e} 25n {x:.9e} "
                  f"25.01n 0 40n 0)",
                  f"C{g}_{j} b{g}_{j} t{g} {C_UNIT*(1+dc[j]):.9e}"]
        L += [f"CL{g} t{g} 0 {C_LOAD*(1+dcl):.9e}",
              f"S{g} t{g} r{g} ctrl 0 SWM",
              f"Vn{g} r{g} 0 DC {vn:.9e}",
              f"Rl{g} t{g} 0 1e12",
              f"Ep{g} p{g} 0 VALUE={{{VCM_T}+{GBUF/2}*V(t{g})}}",
              f"En{g} n{g} 0 VALUE={{{VCM_T}-{GBUF/2}*V(t{g})}}",
              f"Vga{g} ga{g} 0 DC {VTO_NOM + VCM_T + V_OV0 + y/2:.9f}",
              f"Vgb{g} gb{g} 0 DC {VTO_NOM + VCM_T + V_OV0 - y/2:.9f}",
              f"MA{g} p{g} ga{g} sa{g} 0 MA W=1u L=1u",
              f"Vsa{g} sa{g} n{g} DC 0",
              f"MB{g} p{g} gb{g} sb{g} 0 MB W=1u L=1u",
              f"Vsb{g} sb{g} n{g} DC 0"]
    L.append(".tran 0.1n 40n 0 0.2n uic")
    for g in range(len(grid)):
        L.append(f".measure tran ia{g} FIND i(Vsa{g}) AT=39n")
        L.append(f".measure tran ib{g} FIND i(Vsb{g}) AT=39n")
    L.append(".end")
    return "\n".join(L) + "\n"


def _thirdarm_one(args):
    grid, seed, sig_c = args
    rng = np.random.default_rng(seed)
    dc = rng.normal(0, sig_c, 4)
    dcl = float(rng.normal(0, sig_c))
    ctot = 4 * C_UNIT + C_LOAD
    vn = float(rng.normal(0, math.sqrt(KB * TEMP_K / ctot)))
    dkp = float(rng.normal(0, SIG_KP))
    dvt = float(rng.normal(0, SIG_VT_MOS))
    meas, _ = _run(thirdarm_netlist(grid, dc, dcl, vn, dkp, dvt))
    try:
        return np.array([meas[f"ia{g}"] - meas[f"ib{g}"]
                         for g in range(len(grid))])
    except KeyError:
        return None


def run_arm(name, fn, grid, xs, ref, fs_y, gain, mc_runs, workers, seed,
            extra=None):
    jobs = [(grid, seed + k) if extra is None else (grid, seed + k, extra)
            for k in range(mc_runs)]
    t0 = time.time()
    with Pool(workers) as pool:
        got = pool.map(fn, jobs, chunksize=2)
    got = [g for g in got if g is not None]
    M = np.array([metrics(o, ref, xs, fs_y, gain) for o in got])
    A, B, C = M[:, 0], M[:, 1], M[:, 2]
    ent = dict(name=name, n=len(got),
               relRMS_p50=float(np.median(A)), relRMS_p95=float(np.percentile(A, 95)),
               relRMS_rms=float(np.sqrt(np.mean(A ** 2))),
               fsRMS_p50=float(np.median(B)), fsRMS_p95=float(np.percentile(B, 95)),
               fsRMS_rms=float(np.sqrt(np.mean(B ** 2))),
               offrem_p50=float(np.median(C)),
               enob_A=enob(float(np.median(A))), enob_B=enob(float(np.median(B))),
               enob_C=enob(float(np.median(C))),
               seconds=time.time() - t0)
    print(f"  {name:<44} n={ent['n']:>3}  relRMS p50 "
          f"{ent['relRMS_p50']*100:>9.4f} %  ({ent['enob_A']:>5.2f} b)   "
          f"FS-ref p50 {ent['fsRMS_p50']*100:>8.4f} % (rms "
          f"{ent['fsRMS_rms']*100:>8.4f} %)  "
          f"({ent['enob_B']:>5.2f} b)  offs.rem. "
          f"{ent['offrem_p50']*100:>8.4f} % ({ent['enob_C']:>5.2f} b) "
          f"[{ent['seconds']:.0f} s]")
    return ent


def d2(mc_runs, workers, quick=False):
    print("\n" + "=" * 78)
    print("  D2  P11 (a x b) in three signal domains, one corner")
    print("=" * 78)
    res = {"design": dict(sigma_Kprime=SIG_KP, sigma_Vt_pair=SIG_VT_MOS,
                          I_EE=I_EE, V_ov0=V_OV0, KP=KP_NOM, C_unit=C_UNIT,
                          C_load=C_LOAD, V_FS=V_FS_Q, grid=(GRID_NX, GRID_NY),
                          mc_runs=mc_runs)}
    arms = {}

    # ---- (i) subthreshold Gilbert
    grid = op_grid(FS_PAIR, FS_PAIR)
    xs = np.array([g[0] for g in grid]); ys = np.array([g[1] for g in grid])
    gain_i = I_EE / (4 * VT * VT)
    ref_i = gain_i * xs * ys
    _, tx = _run(gilbert_netlist(grid, 0.0, 0.0, 0.0, 0.0, np.zeros(6)),
                 files=("g.txt",))
    d = parse_print(tx["g.txt"])
    out0 = np.array([d[f"i(vk1{g})"] - d[f"i(vk2{g})"] for g in range(len(grid))])
    a0, b0, c0 = metrics(out0, ref_i, xs, FS_PAIR, gain_i)
    print(f"  ideal-device floor, Gilbert (tanh compression): relRMS "
          f"{a0*100:.3f} %, FS-ref {b0*100:.3f} %")
    res["ideal_floor_gilbert"] = dict(relRMS=a0, fsRMS=b0)
    arms["subthreshold_pair"] = run_arm(
        "(i)   subthreshold Gilbert / diff. pair", _gilbert_one, grid, xs,
        ref_i, FS_PAIR, gain_i, mc_runs, workers, 3_000_000)

    # ---- (ii) triode VCR
    grid = op_grid(FS_TRI_X, FS_TRI_Y)
    xs = np.array([g[0] for g in grid]); ys = np.array([g[1] for g in grid])
    gain_ii = KP_NOM
    ref_ii = gain_ii * xs * ys
    _, tx = _run(triode_netlist(grid, 0.0, 0.0), files=("t.txt",))
    d = parse_print(tx["t.txt"])
    out0 = np.array([d[f"i(vsa{g})"] - d[f"i(vsb{g})"] for g in range(len(grid))])
    a0, b0, c0 = metrics(out0, ref_ii, xs, FS_TRI_Y, gain_ii)
    print(f"  ideal-device floor, triode VCR                : relRMS "
          f"{a0*100:.4f} %, FS-ref {b0*100:.4f} %")
    res["ideal_floor_triode"] = dict(relRMS=a0, fsRMS=b0)
    arms["triode"] = run_arm(
        "(ii)  triode VCR multiplier (V_ov = 300 mV)", _triode_one, grid, xs,
        ref_ii, FS_TRI_Y, gain_ii, mc_runs, workers, 4_000_000)

    # ---- (iii) charge domain, both sigma_C
    gridq = [(x, m) for m in (-2, -1, 0, 1, 2)
             for x in np.linspace(-1, 1, GRID_NX) * V_FS_Q / 2]
    xs = np.array([g[0] for g in gridq])
    ys = np.array([g[1] / 4.0 for g in gridq])          # code -> [-1, 1]
    ctot = 4 * C_UNIT + C_LOAD
    gain_iii = -4 * C_UNIT / ctot
    ref_iii = gain_iii * xs * ys
    meas, _ = _run(charge_netlist(gridq, np.zeros(4), 0.0, 0.0))
    out0 = np.array([meas[f"vo{g}"] for g in range(len(gridq))])
    a0, b0, c0 = metrics(out0, ref_iii, xs, 1.0, gain_iii)
    print(f"  ideal-device floor, charge SC                 : relRMS "
          f"{a0*100:.5f} %, FS-ref {b0*100:.5f} %")
    res["ideal_floor_charge"] = dict(relRMS=a0, fsRMS=b0)
    for sc in (0.001, 0.01):
        arms[f"charge_sc{sc}"] = run_arm(
            f"(iii) charge SC multiply, sigma_C = {sc*100:g} %", _charge_one,
            gridq, xs, ref_iii, 1.0, gain_iii, mc_runs, workers,
            5_000_000 + int(sc * 1e6), extra=sc)

    # ---- (iv) third arm
    grid4 = op_grid(V_FS_Q, FS_TRI_Y)
    xs = np.array([g[0] for g in grid4]); ys = np.array([g[1] for g in grid4])
    gain_iv = KP_NOM * GBUF * (4 * C_UNIT / ctot) * (-1.0)
    ref_iv = gain_iv * xs * ys
    meas, _ = _run(thirdarm_netlist(grid4, np.zeros(4), 0.0, 0.0, 0.0, 0.0))
    out0 = np.array([meas[f"ia{g}"] - meas[f"ib{g}"] for g in range(len(grid4))])
    a0, b0, c0 = metrics(out0, ref_iv, xs, FS_TRI_Y, gain_iv)
    print(f"  ideal-device floor, third arm                 : relRMS "
          f"{a0*100:.4f} %, FS-ref {b0*100:.4f} %")
    res["ideal_floor_thirdarm"] = dict(relRMS=a0, fsRMS=b0)
    for sc in (0.001, 0.01):
        arms[f"thirdarm_sc{sc}"] = run_arm(
            f"(iv)  charge x transconductor, sigma_C = {sc*100:g} %",
            _thirdarm_one, grid4, xs, ref_iv, FS_TRI_Y, gain_iv, mc_runs,
            workers, 6_000_000 + int(sc * 1e6), extra=sc)

    res["arms"] = arms
    for label, scname in (("sigma_C = 0.1 %", "charge_sc0.001"),
                          ("sigma_C = 1 %", "charge_sc0.01")):
        vals = {"subthreshold pair": arms["subthreshold_pair"],
                "triode": arms["triode"], "charge": arms[scname]}
        for metric in ("relRMS_p50", "fsRMS_p50", "offrem_p50"):
            bits = {k: enob(v[metric]) for k, v in vals.items()}
            spread = max(bits.values()) - min(bits.values())
            res.setdefault("spread", {})[f"{label}|{metric}"] = dict(
                bits=bits, spread_bits=spread)
            print(f"  spread ({label}, {metric}): "
                  + "  ".join(f"{k} {b:.2f} b" for k, b in bits.items())
                  + f"  ->  {spread:.2f} bits  "
                  + ("PASS" if spread >= 3.0 else "KILL"))
    return res


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["d1", "d2", "all"], default="all")
    ap.add_argument("--mc-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="domain_gates.json")
    args = ap.parse_args()
    if args.quick:
        args.mc_runs = 8

    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")
    (OUT / "tmp").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    blob = {"constants": dict(VT=VT, C_INT=C_INT, I_MAX=I_MAX,
                              IS_NOM=IS_NOM, SIG_VOS=SIG_VOS,
                              ln_Imax_over_Is=LN_BIAS),
            "mc_runs": args.mc_runs}
    if args.gate in ("d1", "all"):
        blob["D1"] = d1(args.mc_runs, args.workers, args.quick)
    if args.gate in ("d2", "all"):
        blob["D2"] = d2(args.mc_runs, args.workers, args.quick)
    blob["elapsed_s"] = time.time() - t0
    p = OUT / args.json
    p.write_text(json.dumps(blob, indent=1, default=float))
    print(f"\nwrote {p}   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
