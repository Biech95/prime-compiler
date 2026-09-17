#!/usr/bin/env python3
"""Cross-bench reconciliation of the two AGC per-channel noise numbers.

Packet 1 (docs/exp_phase0_gaps.md, K-C)      : 0.682-0.687 % at I0 = 1 uA
Packet 2 (docs/exp_calculus_noise.md, K1)    : 1.998 %      at I0 = 1 uA

Both are "per-channel VGA shot noise at the loop bandwidth", they differ 2.9x.
This script puts BOTH conventions on ONE bench (packet 1's transient AGC
testbench, packet 2's input draw) and separates the factors:

  * bandwidth  -- envelope-settling f_loop (packet 1) vs closed-loop AC ENBW
                  (packet 2), measured on the SAME netlist;
  * junctions  -- 1 shot-noise junction per VGA (packet 1) vs 2 (packet 2);
  * I0         -- 1 uA in both;
  * metric     -- identical decomposition, RMS-over-time vs p50-over-draws.

It then fits the unified constant and reports the (I0, B) crossover against
the 0.69 % VGA-mismatch residual, plus a damping sweep (loop gain / integrator
zero) with its settling-time cost.

Usage:  python3 spice/noise_reconciliation.py [--seeds 4] [--tend 300]
Output: spice/out/noise_reconciliation.json  (+ stdout table)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

Q_E = 1.602176634e-19
KB = 1.380649e-23
WORKDIR = Path(__file__).parent / "out"
OPTS = ".options gmin=1e-14 reltol=1e-6 abstol=1e-16"
MISMATCH_REF = 0.0069          # VGA mismatch residual, sigma_VGA = 1 %

_UNIQ = [0]


def _tag(s):
    _UNIQ[0] += 1
    return f"nrec_{s}_{_UNIQ[0]}"


def _run(lines, tag, timeout=1800):
    WORKDIR.mkdir(parents=True, exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text("\n".join(lines))
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=timeout, cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-3000:])
        raise RuntimeError(f"ngspice failed ({tag})")
    cir.unlink(missing_ok=True)
    return res


# ---------------------------------------------------------------------------
# the shared testbench (packet 1's agc_body, with an optional integrator zero)
# ---------------------------------------------------------------------------

def agc_lines(x, gamma=1.0, tau_det_ns=0.5, c_pf=1.0, k_gain=2e-3,
              rz_ohm=0.0, na_ch=None, nt_ps=10.0, x2=None, t_step_ns=1e9):
    N = len(x)
    L = [f"* AGC RMSNorm N={N} (reconciliation bench)", OPTS]
    for i in range(N):
        if x2 is None:
            L.append(f"Vx{i} x{i} 0 DC {x[i]:.8f}")
        else:
            L.append(f"Vx{i} x{i} 0 PWL(0 {x[i]:.8f} {t_step_ns}n "
                     f"{x[i]:.8f} {t_step_ns+0.05}n {x2[i]:.8f})")
    for i in range(N):
        if na_ch is not None:
            L.append(f"Vn{i} nn{i} 0 DC 0 "
                     f"trnoise({na_ch[i]:.6e} {nt_ps}p 0 0)")
        else:
            L.append(f"Vn{i} nn{i} 0 DC 0")
        L.append(f"By{i} y{i} 0 V={{V(g)*V(x{i})+V(nn{i})}}")
    sq = "+".join(f"V(y{i})*V(y{i})" for i in range(N))
    L.append(f"Bmsq msqr 0 V={{({sq})/{N}}}")
    L.append(f"Rdet msqr msq {tau_det_ns*1e-9/1e-12:.4f}")
    L.append("Cdet msq 0 1p")
    L.append(f"Bint 0 g I={{{k_gain}*({gamma*gamma:.10f} - V(msq))}}")
    if rz_ohm > 0:                       # integrator zero at 1/(2 pi Rz Cg)
        L.append(f"Rz g gz {rz_ohm:.6e}")
        L.append(f"Cg gz 0 {c_pf:.6f}p")
    else:
        L.append(f"Cg g 0 {c_pf:.6f}p")
    L.append(".ic V(g)=0.1 V(msq)=0")
    return L


def tran(lines, vectors, tstep_ps, tend_ns, tag, seed=None, uic=True):
    out = f"{tag}.dat"
    ctrl = [".control"]
    if seed is not None:
        ctrl.append(f"set rndseed = {int(seed)}")
    ctrl += [f"tran {tstep_ps}p {tend_ns}n{' uic' if uic else ''}",
             f"wrdata {out} {' '.join(vectors)}", "quit", ".endc", ".end"]
    _run(lines + ctrl, tag)
    d = np.loadtxt(WORKDIR / out)
    (WORKDIR / out).unlink(missing_ok=True)
    return d[:, 0], d[:, 1::2]


def agc_ac_lines(x, which, tau_det_ns=0.5, c_pf=1.0, k_gain=2e-3,
                 rz_ohm=0.0):
    """Packet 2's AC probe, on the same loop: unit RELATIVE perturbations."""
    N = len(x)
    G0 = 1.0 / np.sqrt(np.mean(np.asarray(x) ** 2))
    L = [f"* AGC AC probe ({which})", OPTS,
         f"Ved ed 0 DC 0 AC {'1' if which == 'det' else '0'}",
         f"Ve0 e0 0 DC 0 AC {'1' if which == 'ch0' else '0'}"]
    for i in range(N):
        L.append(f"Vx{i} x{i} 0 DC {x[i]:.10e}")
    for i in range(N):
        pert = "*(1+V(e0))" if i == 0 else ""
        L.append(f"By{i} y{i} 0 V={{V(g)*V(x{i}){pert}}}")
    sq = "+".join(f"V(y{i})*V(y{i})" for i in range(N))
    L.append(f"Bmsq msqr 0 V={{(1+V(ed))*({sq})/{N}}}")
    L += [f"Rdet msqr msq {tau_det_ns*1e-9/1e-12:.6f}", "Cdet msq 0 1p",
          f"Bint 0 g I={{{k_gain}*(1.0 - V(msq))}}"]
    if rz_ohm > 0:
        L += [f"Rz g gz {rz_ohm:.6e}", f"Cg gz 0 {c_pf}p"]
    else:
        L.append(f"Cg g 0 {c_pf}p")
    L.append(f".nodeset v(g)={G0:.8f} v(msq)=1.0")
    return L


def agc_ac(x, which, **kw):
    tag = _tag(f"ac_{which}")
    L = agc_ac_lines(x, which, **kw)
    L += [".control", "op", f"print v(g) > {tag}_op.txt",
          "ac dec 40 1e5 1e11", f"wrdata {tag}.txt v(g) v(y0) v(y1)",
          "quit", ".endc", ".end"]
    _run(L, tag)
    txt = (WORKDIR / f"{tag}_op.txt").read_text()
    g0 = float([ln for ln in txt.splitlines() if "v(g)" in ln][-1].split()[-1])
    d = np.loadtxt(WORKDIR / f"{tag}.txt")
    for f_ in (f"{tag}.txt", f"{tag}_op.txt"):
        (WORKDIR / f_).unlink(missing_ok=True)
    f = d[:, 0]
    cx = lambda c: d[:, c] + 1j * d[:, c + 1]
    return f, cx(1), cx(4), cx(7), g0


def enbw(f, H):
    h0 = abs(H[0])
    return 0.0 if h0 == 0 else float(np.trapezoid(np.abs(H) ** 2, f) / h0 ** 2)


# ---------------------------------------------------------------------------
# band limiting: packet 1 uses a 1-pole IIR at f_loop; packet 2 uses a
# brick-wall integral to B (ngspice .noise integrates the spectrum to B).
# ---------------------------------------------------------------------------

def lowpass_1pole(sig, dt, fc):
    from scipy.signal import lfilter
    a = dt / (dt + 1.0 / (2 * math.pi * fc))
    return lfilter([a], [1.0, -(1.0 - a)], sig, axis=0)


def brickwall(sig, dt, B):
    n = sig.shape[0]
    F = np.fft.rfft(sig, axis=0)
    fr = np.fft.rfftfreq(n, dt)
    F[fr > B, :] = 0.0
    return np.fft.irfft(F, n=n, axis=0)


def decompose_rms(yb, y_ref):
    """Packet 1's metric: per-timestep alpha removal, RMS over time+channels."""
    al = (yb @ y_ref) / float(y_ref @ y_ref)
    res = yb / al[:, None] - y_ref[None, :]
    return float(np.sqrt(np.mean(res ** 2))), float(np.std(al))


def mc_decompose_p50(ref, sig, n=20000, seed=77):
    """Packet 2's metric: p50 of the same decomposition over draws."""
    rng = np.random.default_rng(seed)
    pc = []
    rr = float(np.dot(ref, ref))
    for _ in range(n):
        y = ref * (1 + rng.normal(0.0, sig))
        a = float(np.dot(y, ref) / rr)
        pc.append(float(np.sqrt(np.mean((y / a - ref) ** 2))))
    return float(np.percentile(pc, 50))


def na_from_psd(psd, nt_s):
    return math.sqrt(psd / (2.0 * nt_s))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tend", type=float, default=300.0)
    ap.add_argument("--tstep", type=float, default=5.0)
    args = ap.parse_args()
    out = {}
    W = 88
    print("=" * W)
    print("  NOISE RECONCILIATION  --  packet 1 (0.68 %) vs packet 2 (2.0 %)")
    print("=" * W)

    # packet 2's draw (seed 2026), so both conventions see identical channels
    rng = np.random.default_rng(2026)
    N = 8
    x = rng.standard_normal(N) * 1.5
    x = np.where(np.abs(x) < 0.05, 0.05 * np.sign(x) + (x == 0) * 0.05, x)
    y_ref = x / np.sqrt(np.mean(x ** 2))
    G0 = 1.0 / np.sqrt(np.mean(x ** 2))
    I0 = 1e-6
    print(f"\n[0] draw (packet 2, seed 2026): G0 = {G0:.6f}; "
          f"|y_i| = {np.round(np.abs(y_ref),3).tolist()}")
    out["x"] = x.tolist()
    out["I0"] = I0

    # ---------------------------------------------------------------- [1]
    print("\n[1] ONE loop, THREE bandwidth definitions "
          "(k=2e-3, C=1 pF, tau_det=0.5 ns)")
    base = agc_lines(x)
    t, v = tran(base, ["v(g)", "v(msq)"], args.tstep, 80, _tag("bw"))
    g = v[:, 0]
    m = (t > 8e-9) & (t < 30e-9)
    err = np.abs(g[m] - g[-1]) / g[-1]
    ok = err > 1e-9
    tau_eq = -1.0 / np.polyfit(t[m][ok], np.log(err[ok]), 1)[0]
    f_env = 1.0 / (2 * math.pi * tau_eq)
    off = np.abs(g - G0) / G0 > 0.01
    t_settle = t[int(np.max(np.nonzero(off))) + 1]
    print(f"    (a) envelope settling  : tau_eq = {tau_eq*1e9:.3f} ns "
          f"-> f_env = 1/(2 pi tau_eq) = {f_env/1e6:.1f} MHz   "
          f"[PACKET 1 used this]")

    f, Hg_d, Hy0_d, Hy1_d, g_op = agc_ac(x, "det")
    _, _, Hy0_c, Hy1_c, _ = agc_ac(x, "ch0")
    Hd = Hg_d / g_op
    Hc = Hy0_c / (g_op * x[0])
    i3 = int(np.argmax(np.abs(Hd) < abs(Hd[0]) / math.sqrt(2)))
    f3 = float(f[i3])
    B_loop = enbw(f, Hd)
    B_pc = enbw(f, Hc)
    pk = int(np.argmax(np.abs(Hd) ** 2 / abs(Hd[0]) ** 2))
    peak = float(np.abs(Hd[pk]) ** 2 / abs(Hd[0]) ** 2)
    print(f"    (b) closed-loop AC     : -3 dB = {f3/1e6:.1f} MHz, "
          f"ENBW(det->G) = {B_loop/1e6:.1f} MHz, peaking "
          f"{peak:.2f}x at {f[pk]/1e6:.0f} MHz   [PACKET 2 used this]")
    print(f"    (c) per-channel path   : ENBW(ch0->y0) = {B_pc/1e6:.0f} MHz "
          f"(|H|(0)={abs(Hc[0]):.6f}, |H|(inf)={abs(Hc[-1]):.6f}) "
          f"-- the loop does NOT band-limit it")
    print(f"    ratio (b)/(a) = {B_loop/f_env:.2f} in POWER "
          f"-> {math.sqrt(B_loop/f_env):.2f} in amplitude")
    print(f"    NB packet 1's 1-pole post-filter at f_env has "
          f"ENBW = (pi/2)*f_env = {math.pi/2*f_env/1e6:.1f} MHz, "
          f"so its EFFECTIVE band is {math.pi/2*f_env/1e6:.0f} MHz")

    # damping of the same loop, from the noiseless step
    bs = agc_lines(x, x2=(np.asarray(x) * 2.5).tolist(), t_step_ns=20.0)
    ts_, vs_ = tran(bs, ["v(g)"], 2.0, 60, _tag("ring"))
    gfin = float(vs_[-1, 0])
    seg = (ts_ > 20.2e-9) & (ts_ < 40e-9)
    d = vs_[seg, 0] - gfin
    tt = ts_[seg]
    cr = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    over = float(np.max(d[tt > tt[cr[0]]]) / gfin) if len(cr) else 0.0
    f_ring = (1.0 / (2 * float(np.median(np.diff(tt[cr])))) if len(cr) > 2
              else float("nan"))
    # analytic 2nd-order model of the same loop:
    #   dG/dt = (k/C)(gamma^2 - msq),  msq = LPF_tau(G^2 MS(x))
    #   => wn^2 = 2 k G0 MS /(C tau_det),  zeta = 1/(2 wn tau_det)
    MS = float(np.mean(np.asarray(x) ** 2))
    wn = math.sqrt(2 * 2e-3 * G0 * MS / (1e-12 * 0.5e-9))
    zeta = 1.0 / (2 * wn * 0.5e-9)
    print(f"    damping: first overshoot {over*100:+.1f} %, ringing "
          f"{f_ring/1e6:.0f} MHz")
    print(f"    analytic 2nd order: wn/2pi = {wn/2/math.pi/1e6:.1f} MHz, "
          f"zeta = {zeta:.4f}")
    print(f"      -> envelope pole  zeta*wn/2pi = "
          f"{zeta*wn/2/math.pi/1e6:7.1f} MHz   (measured {f_env/1e6:.1f})")
    print(f"      -> ENBW  wn/(8 zeta)          = "
          f"{wn/(8*zeta)/1e6:7.1f} MHz   (measured {B_loop/1e6:.1f})")
    print(f"      -> peaking 1/(4 z^2(1-z^2))   = "
          f"{1/(4*zeta**2*(1-zeta**2)):7.2f}x    (measured {peak:.2f})")
    print(f"      -> ENBW / f_env = pi/(4 zeta^2) = "
          f"{math.pi/(4*zeta**2):.2f}        (measured "
          f"{B_loop/f_env:.2f})")
    print(f"    ==> THE 3x IS THE DAMPING: sigma(P2 band)/sigma(P1 band) "
          f"= 1/(zeta*sqrt(2)) = {1/(zeta*math.sqrt(2)):.2f}")
    out["bandwidth"] = dict(tau_eq=tau_eq, f_env=f_env, f3db=f3,
                            enbw_loop=B_loop, enbw_pc=B_pc,
                            enbw_1pole_eff=math.pi / 2 * f_env,
                            peak=peak, f_peak=float(f[pk]),
                            overshoot=over, f_ring=f_ring,
                            zeta_analytic=zeta, wn=wn, MS=MS,
                            t_settle=float(t_settle))

    # ---------------------------------------------------------------- [2]
    print("\n[2] per-channel noise on ONE bench, both conventions "
          f"(I0 = 1 uA, {args.seeds} seeds x {args.tend:.0f} ns)")
    print("    junction count: packet 1 = 1 (VGA output only), "
          "packet 2 = 2 (Dlog + Dexp)")
    bands = [("1-pole @ f_env (P1)", f_env, "pole"),
             ("brick @ ENBW_loop (P2)", B_loop, "brick"),
             ("brick @ (pi/2)f_env", math.pi / 2 * f_env, "brick")]
    rows = {}
    tr_cache = {}
    for kj in (1, 2):
        psd = kj * 2 * Q_E * I0 * np.abs(y_ref) / I0 ** 2
        na = [na_from_psd(p, 10e-12) for p in psd]
        acc = {b[0]: [] for b in bands}
        jit = []
        for sd in range(args.seeds):
            body = agc_lines(x, na_ch=na)
            vecs = ["v(g)"] + [f"v(y{j})" for j in range(N)]
            tt2, vv = tran(body, vecs, args.tstep, args.tend,
                           _tag(f"n{kj}"), seed=3000 + sd)
            mm = tt2 > 60e-9
            dt = float(np.median(np.diff(tt2[mm])))
            gg, yy = vv[mm, 0], vv[mm, 1:]
            jit.append(float(np.std(gg) / np.mean(gg)))
            for lbl, B, kind in bands:
                yb = (lowpass_1pole(yy, dt, B) if kind == "pole"
                      else brickwall(yy, dt, B))
                acc[lbl].append(decompose_rms(yb, y_ref)[0])
        for lbl, B, kind in bands:
            rows[(kj, lbl)] = (float(np.mean(acc[lbl])), B)
        tr_cache[kj] = float(np.mean(jit))
    print(f"    {'band limit':<26} {'B_eff/MHz':>10} "
          f"{'1 junction':>12} {'2 junctions':>13}")
    for lbl, B, kind in bands:
        beff = (math.pi / 2 * B) if kind == "pole" else B
        print(f"    {lbl:<26} {beff/1e6:10.1f} "
              f"{rows[(1,lbl)][0]*100:11.4f}% {rows[(2,lbl)][0]*100:12.4f}%")
    out["transient"] = {f"{kj}j|{lbl}": rows[(kj, lbl)][0]
                        for kj in (1, 2) for lbl, _, _ in bands}

    p1 = rows[(1, "1-pole @ f_env (P1)")][0]
    p2 = rows[(2, "brick @ ENBW_loop (P2)")][0]
    print(f"\n    packet 1 cell  (1 junction, 1-pole @ {f_env/1e6:.0f} MHz) "
          f"= {p1*100:.4f} %   [published 0.682-0.687 %]")
    print(f"    packet 2 cell  (2 junctions, brick @ {B_loop/1e6:.0f} MHz) "
          f"= {p2*100:.4f} %   [published 1.998 %]")
    print(f"    -> reproduced ratio {p2/p1:.3f}x on ONE bench "
          f"(published ratio {1.998/0.684:.3f}x)")

    # factor split
    f_bw = math.sqrt(B_loop / (math.pi / 2 * f_env))
    f_kj = math.sqrt(2.0)
    print(f"\n    factor split:  bandwidth sqrt({B_loop/1e6:.0f}/"
          f"{math.pi/2*f_env/1e6:.0f}) = {f_bw:.3f}x  x  junctions "
          f"sqrt(2) = {f_kj:.3f}x  = {f_bw*f_kj:.3f}x")
    out["factors"] = dict(bandwidth=f_bw, junctions=f_kj,
                          product=f_bw * f_kj, measured=p2 / p1)

    # ---------------------------------------------------------------- [3]
    print("\n[3] metric cross-check (same sigma_i, both statistics)")
    for kj in (1, 2):
        for lbl, B, kind in bands:
            beff = (math.pi / 2 * B) if kind == "pole" else B
            sig = np.sqrt(kj * 2 * Q_E * beff / (I0 * np.abs(y_ref)))
            p50 = mc_decompose_p50(y_ref, sig)
            # res_i = y_i (eps_i - sum_j w_j eps_j); w_j = y_j^2 / sum y^2
            wv = y_ref ** 2 / np.sum(y_ref ** 2)
            rms = float(np.sqrt(
                np.mean(y_ref ** 2 * sig ** 2 * (1 - 2 * wv))
                + np.mean(y_ref ** 2) * np.sum(wv ** 2 * sig ** 2)))
            tr = rows[(kj, lbl)][0]
            print(f"    {kj}j {lbl:<26} analytic RMS {rms*100:7.4f}%  "
                  f"MC p50 {p50*100:7.4f}%  transient {tr*100:7.4f}%  "
                  f"(p50/RMS {p50/rms:.3f}, tran/RMS {tr/rms:.3f})")
            out.setdefault("metric", {})[f"{kj}j|{lbl}"] = dict(
                analytic_rms=rms, mc_p50=p50, transient=tr, B_eff=beff)

    # ---------------------------------------------------------------- [4]
    print("\n[4] unified constant:  sigma_pc = C * sqrt(k_j * 2q * B / I0)")
    cs = []
    for kj in (1, 2):
        for lbl, B, kind in bands:
            beff = (math.pi / 2 * B) if kind == "pole" else B
            base_ = math.sqrt(kj * 2 * Q_E * beff / I0)
            cs.append(rows[(kj, lbl)][0] / base_)
    C = float(np.mean(cs))
    wv = y_ref ** 2 / np.sum(y_ref ** 2)
    C_an = float(np.sqrt(np.mean(np.abs(y_ref) * (1 - 2 * wv))
                         + np.sum(wv ** 2 / np.abs(y_ref))))
    print(f"    C (fit, transient) = {C:.4f} +- {np.std(cs):.4f} over "
          f"{len(cs)} (k_j, B) cells  [spread "
          f"{np.min(cs):.4f}..{np.max(cs):.4f}]")
    print(f"    C (analytic, this draw) = sqrt( mean_i[|y_i|(1-2w_i)] + "
          f"sum_j w_j^2/|y_j| ) = {C_an:.4f}")
    print(f"    -> fit/analytic = {C/C_an:.3f}. C is a pure draw shape "
          f"factor (w_i = y_i^2/sum y^2); it is 1 for a flat draw.")
    out["C"] = C
    out["C_analytic"] = C_an
    out["C_spread"] = [float(np.min(cs)), float(np.max(cs))]

    # crossover
    print(f"\n    crossover against the {MISMATCH_REF*100:.2f} % mismatch "
          f"residual,  I0 = C^2 * k_j * 2q * B / sigma_mm^2 :")
    for kj in (1, 2):
        for Bm in (100e6, f_env, math.pi / 2 * f_env, B_loop, 1e9):
            i0c = C ** 2 * kj * 2 * Q_E * Bm / MISMATCH_REF ** 2
            print(f"      k_j = {kj}, B = {Bm/1e6:7.1f} MHz -> "
                  f"I0_cross = {i0c*1e6:7.3f} uA")
    for kj in (1, 2):
        for i0c in (1e-6, 10e-6):
            Bc = MISMATCH_REF ** 2 * i0c / (C ** 2 * kj * 2 * Q_E)
            print(f"      k_j = {kj}, I0 = {i0c*1e6:4.1f} uA -> "
                  f"B_cross = {Bc/1e6:8.1f} MHz")
    out["crossover"] = {
        f"kj{kj}|B{Bm/1e6:.0f}MHz":
            C ** 2 * kj * 2 * Q_E * Bm / MISMATCH_REF ** 2
        for kj in (1, 2)
        for Bm in (100e6, f_env, math.pi / 2 * f_env, B_loop, 1e9)}
    out["B_cross"] = {f"kj{kj}|I0{i*1e6:.0f}uA":
                      MISMATCH_REF ** 2 * i / (C ** 2 * kj * 2 * Q_E)
                      for kj in (1, 2) for i in (1e-6, 10e-6)}

    # ---------------------------------------------------------------- [5]
    print("\n[5] damping fix and its cost "
          "(lower loop gain k, or an integrator zero Rz = tau_det/C_g)")
    print("    small-signal probe: +10 % input step at t = 200 ns "
          "(the 2.5x step of packet 1 is SLEW limited and hides k)")
    # zeta*wn = 1/(2 tau_det) is INDEPENDENT of k, so the envelope rate is
    # fixed; zeta = 1 (critical) is reached at
    k_crit = (1.0 / (2 * 0.5e-9)) ** 2 * 1e-12 * 0.5e-9 / (2 * G0 * MS)
    print(f"    zeta*wn = 1/(2 tau_det) = {1/(2*0.5e-9)/1e9:.2f} Grad/s for "
          f"EVERY k -> the envelope rate does not depend on k;")
    print(f"    critical damping (zeta = 1) at k_crit = C_g/(8 G0 MS "
          f"tau_det) = {k_crit:.3e}  ({2e-3/k_crit:.1f}x below the "
          f"published k)")
    print(f"    {'variant':<26} {'zeta':>6} {'overshoot':>10} "
          f"{'t_set 1%':>10} {'ENBW_loop':>11} {'peaking':>9} "
          f"{'sig_pc(2j)':>11} {'sig*sqrt(tset)':>15}")
    damp = []
    T_STEP = 200.0
    for lbl, kg, rz in [("k=2e-3, Rz=0 (published)", 2e-3, 0.0),
                        ("k=1e-3, Rz=0", 1e-3, 0.0),
                        ("k=5e-4, Rz=0", 5e-4, 0.0),
                        ("k=2.5e-4, Rz=0", 2.5e-4, 0.0),
                        ("k=k_crit, Rz=0 (zeta=1)", k_crit, 0.0),
                        ("k=1e-4, Rz=0", 1e-4, 0.0),
                        ("k=5e-5, Rz=0", 5e-5, 0.0),
                        ("k=2e-3, Rz=250", 2e-3, 250.0),
                        ("k=2e-3, Rz=500 (=tau/C)", 2e-3, 500.0),
                        ("k=5e-4, Rz=500", 5e-4, 500.0),
                        ("k=k_crit, Rz=500", k_crit, 500.0)]:
        bs = agc_lines(x, k_gain=kg, rz_ohm=rz,
                       x2=(np.asarray(x) * 1.1).tolist(), t_step_ns=T_STEP)
        ts2, vs2 = tran(bs, ["v(g)"], 2.0, T_STEP + 250, _tag("dmp"))
        gf = float(vs2[-1, 0])
        g_pre = float(vs2[np.argmin(np.abs(ts2 - (T_STEP - 1) * 1e-9)), 0])
        step = abs(gf - g_pre)
        sg = ts2 > (T_STEP + 0.05) * 1e-9
        dd = vs2[sg, 0] - gf
        tt3 = ts2[sg]
        cr2 = np.nonzero(np.diff(np.sign(dd)) != 0)[0]
        ov = (float(np.max(dd[tt3 > tt3[cr2[0]]]) / step)
              if len(cr2) else 0.0)          # first overshoot, rel. to step
        offs = np.abs(dd) / step > 0.01
        tset = (tt3[int(np.max(np.nonzero(offs))) + 1] - T_STEP * 1e-9
                if offs.any() else 0.0)
        fq, Hgd, _, _, gop = agc_ac(x, "det", k_gain=kg, rz_ohm=rz)
        H = Hgd / gop
        Bl = enbw(fq, H)
        pkv = float(np.max(np.abs(H) ** 2) / abs(H[0]) ** 2)
        sg2 = C * math.sqrt(2 * 2 * Q_E * Bl / I0)
        fom = sg2 * math.sqrt(max(tset, 1e-12) * 1e9)
        wn_ = math.sqrt(2 * kg * G0 * MS / (1e-12 * 0.5e-9))
        zt = float("inf") if rz > 0 else 1.0 / (2 * wn_ * 0.5e-9)
        print(f"    {lbl:<26} {zt:6.2f} {ov*100:+9.1f}% {tset*1e9:9.2f}ns "
              f"{Bl/1e6:10.0f}M {pkv:9.2f} {sg2*100:10.4f}% "
              f"{fom*100:14.4f}")
        damp.append(dict(label=lbl, k=kg, rz=rz, zeta=zt, overshoot=ov,
                         t_settle=float(tset), enbw=Bl, peak=pkv,
                         sigma_pc=sg2, fom=fom))
    out["k_crit"] = k_crit
    out["damping"] = damp

    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "noise_reconciliation.json").write_text(json.dumps(out,
                                                                  indent=1))
    print("\n  raw -> spice/out/noise_reconciliation.json")
    print("=" * W)


if __name__ == "__main__":
    main()
