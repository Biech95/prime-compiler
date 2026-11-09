#!/usr/bin/env python3
"""
Step response of the published AGC-RMSNorm loop (k = 2e-3, C_g = 1 pF,
tau_det = 0.5 ns, N = 8, input draw seed 2026), noiseless.

  large : 2.5x input step at t = 20 ns (bench of spice/phase0_gaps_sim.py)
  small : +10 % input step at t = 200 ns (bench of spice/noise_reconciliation.py)
  settle: 2.5x input step at t = 40 ns (bench of spice/phase0_gaps_sim.py, G-D
          base case), 1 % settling to the run's own equilibrium, in three forms:
            free       the published model: the gain node V(g) is an ideal,
                       sign-free behavioural integrator; after a large step it
                       transiently goes negative, and the VGA uses it as is;
            vga-clamp  the node is left free (it may wind up below zero), but
                       the VGA uses max(V(g), 0): a magnitude VGA cannot
                       realise a negative gain;
            node-clamp the gain node itself is held at V(g) >= 0 by a 10 S
                       behavioural clamp current, Bcl 0 g I={10*max(-V(g),0)}
                       (an ideal limiter on the integration node; residual
                       negative excursion < 1e-3 V).

For each step the script stores the raw transient (t, G) as .npy and .csv
and reports: G0, G_final, step height, every extremum of G - G_final after
the step (time, value relative to the step height and to G_final), the
ringing frequency from the first extrema, the number of crossings of
G_final while the lobe amplitude exceeds 1e-3 of the step height, and the
raw number of sign changes in the 20 ns after the step.

Requires ngspice in PATH and the prime-compiler repository (spice/); the
repository is only imported, nothing in it is written.
Usage: python3 agc_step_response.py --repo <unzipped prime_compiler_v3.1_code> [--quick]
"""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
THR = 1e-3          # lobe-amplitude threshold, relative to the step height


def analyse(t, g, t_step_ns, label, window_ns=20.0):
    pre = t < (t_step_ns - 1.0) * 1e-9
    g0 = float(g[pre][-1])
    gf = float(g[-1])
    step = g0 - gf                       # > 0: the gain falls after an input increase
    after = t > (t_step_ns + 0.05) * 1e-9
    ta, d = t[after], g[after] - gf
    s = np.sign(d)
    cr = np.nonzero(np.diff(s) != 0)[0]  # index before each sign change
    # lobes between consecutive crossings; extremum of each lobe
    bounds = [0] + list(cr + 1) + [len(d)]
    ext = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        j = a + int(np.argmax(np.abs(d[a:b])))
        ext.append((float(ta[j]), float(d[j])))
    # ext[0] is the initial approach (d > 0 at the start); overshoots follow
    print(f"\n=== {label} ===")
    print(f"  G0 = {g0:.6f}   G_final = {gf:.6f}   step height G0-G_final = {step:.6f}"
          f"   (G0/G_final = {g0/gf:.4f})")
    print(f"  {'#':>3} {'t - t_step [ns]':>16} {'(G-Gf)/step':>12} {'(G-Gf)/Gf':>11}")
    for i, (tx, v) in enumerate(ext[1:13], 1):
        print(f"  {i:3d} {tx*1e9 - t_step_ns:16.4f} {v/step*100:+11.2f}% {v/gf*100:+10.2f}%")
    tx = np.array([e[0] for e in ext[1:]])
    f_ring = 1.0 / (2.0 * float(np.mean(np.diff(tx[:6])))) if len(tx) > 6 else float("nan")
    amp = np.array([abs(e[1]) for e in ext[1:]])
    n_thr = int(np.sum(amp > THR * step))          # overshoot lobes above threshold
    w = (ta > (t_step_ns + 0.2) * 1e-9) & (ta < (t_step_ns + window_ns) * 1e-9)
    n_raw = int(np.sum(np.diff(np.sign(d[w])) != 0))
    # legacy metric of the project scripts: max(d) after the first crossing
    legacy_gf = float(np.max(d[ta > ta[cr[0]]]) / gf) if len(cr) else 0.0
    legacy_st = float(np.max(d[ta > ta[cr[0]]]) / step) if len(cr) else 0.0
    lw_cr = np.nonzero(np.diff(np.sign(d[w])) != 0)[0]
    f_med = (1.0 / (2 * float(np.median(np.diff(ta[w][lw_cr]))))
             if len(lw_cr) > 2 else float("nan"))
    print(f"  ringing frequency from the first 6 extremum spacings: {f_ring/1e6:.0f} MHz")
    print(f"  overshoot lobes with |G-Gf| > {THR:g} x step: {n_thr} "
          f"(= crossings of G_final above that amplitude)")
    print(f"  raw sign changes of G-Gf in (t_step+0.2, t_step+{window_ns:g}) ns: {n_raw};"
          f" median-spacing frequency of those {f_med/1e6:.0f} MHz")
    small = [(e[0], e[1]) for e in ext[1:] if abs(e[1]) <= THR * step]
    if small:
        print(f"  smallest lobe amplitude |G-Gf|/step = {min(abs(e[1]) for e in small)/step:.2e};"
              f" lobes below threshold: {len(small)}")
    print(f"  legacy metric max(G-Gf after 1st crossing): {legacy_gf*100:+.2f}% of G_final,"
          f" {legacy_st*100:+.2f}% of step")
    return dict(g0=g0, gf=gf, step=step, ext=ext[1:], f_ring=f_ring, n_thr=n_thr,
                n_raw=n_raw, legacy_gf=legacy_gf, legacy_st=legacy_st)


def save(name, t, g):
    arr = np.column_stack([t, g])
    np.save(HERE / f"{name}.npy", arr)
    np.savetxt(HERE / f"{name}.csv", arr, delimiter=",", header="t_s,G", comments="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "prime-compiler"))
    ap.add_argument("--quick", action="store_true", help="short probe run only")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo) / "spice"))
    import phase0_gaps_sim as p0
    import noise_reconciliation as nr
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    p0.WORKDIR = work                    # keep all ngspice files out of the repository
    nr.WORKDIR = work

    rng = np.random.default_rng(2026)
    x = p0.draw_x(rng, 8)
    ms = float(np.mean(x ** 2))
    g0 = 1.0 / math.sqrt(ms)
    wn = math.sqrt(2 * 2e-3 * g0 * ms / (1e-12 * 0.5e-9))
    zeta = 1.0 / (2 * wn * 0.5e-9)
    print(f"input draw seed 2026, N = 8; analytic small-signal model: "
          f"f_n = {wn/2/math.pi/1e6:.1f} MHz, zeta = {zeta:.4f}, "
          f"ideal 2nd-order overshoot {100*math.exp(-math.pi*zeta/math.sqrt(1-zeta**2)):.1f} %, "
          f"f_d = {wn*math.sqrt(1-zeta**2)/2/math.pi/1e6:.0f} MHz")

    if a.quick:
        body = p0.agc_body(x, None, 1.0, 0.5, 1.0, 2e-3, 1e9)
        t, v = p0.run_tran(body, ["v(g)"], 10, 10, "probe")
        print(f"probe ok: {len(t)} points, G(10 ns) = {v[-1,0]:.6f}, exact {g0:.6f}")
        return

    # large signal: identical call to phase0_gaps_sim.py (cmd_gc, deterministic corroboration)
    bs = p0.agc_body(x, list(np.asarray(x) * 2.5), 1.0, 0.5, 1.0, 2e-3, 20.0)
    t, v = p0.run_tran(bs, ["v(g)"], 2, 60, "large")
    save("agc_large_step_2p5x", t, v[:, 0])
    analyse(t, v[:, 0], 20.0, "large signal: 2.5x input step at 20 ns (phase0 bench)")

    # small signal: identical call to noise_reconciliation.py (published loop row)
    T = 200.0
    bs = nr.agc_lines(x, k_gain=2e-3, rz_ohm=0.0,
                      x2=(np.asarray(x) * 1.1).tolist(), t_step_ns=T)
    t, v = nr.tran(bs, ["v(g)"], 2.0, T + 250, "small")
    save("agc_small_step_1p1x", t, v[:, 0])
    analyse(t, v[:, 0], T, "small signal: +10 % input step at 200 ns (reconciliation bench)")

    # settling: free vs. clamped gain (G-D bench)
    settle_variants(p0, x)


def settle_variants(p0, x):
    """2.5x step at 40 ns, G-D bench of phase0_gaps_sim.py (10 ps, 80 ns)."""
    x2 = list(np.asarray(x) * 2.5)
    print("\n=== settling after a 2.5x input step at 40 ns (G-D bench, 1 % band of own equilibrium) ===")
    print(f"  {'variant':<24}{'ts cold':>9}{'ts step':>9}{'min V(g)':>10}"
          f"{'G_final':>10}{'1st ext/step':>13}{'2nd ext/step':>13}{'crossings>1e-3':>15}")
    out = {}
    for cap, lab_c in ((0.0, ""), (0.02, ", C_g +2 %")):
        for mode in ("free", "vga-clamp", "node-clamp"):
            body = p0.agc_body(x, x2, 1.0, 0.5, 1.0, 2e-3, 40.0, cap_mm=cap)
            if mode == "vga-clamp":
                body = [l.replace("V={V(g)*V(x", "V={max(V(g),0)*V(x") for l in body]
                assert sum("max(V(g),0)" in l for l in body) == len(x)
            elif mode == "node-clamp":
                i = next(k for k, l in enumerate(body) if l.startswith(".ic"))
                body = body[:i] + ["Bcl 0 g I={10*max(-V(g),0)}"] + body[i:]
            t, v = p0.run_tran(body, ["v(g)"], 10, 80, "settle")
            g = v[:, 0]
            m1, m2 = t < 40e-9, t >= 40e-9
            ts1 = p0.settling_time(t[m1], g[m1], g[m1][-1])
            ts2 = p0.settling_time(t[m2] - 40e-9, g[m2], g[-1])
            gf, g0 = float(g[-1]), float(g[m1][-1])
            step = g0 - gf
            ta, d = t[m2], g[m2] - gf
            cr = np.nonzero(np.diff(np.sign(d)) != 0)[0]
            bounds = [0] + list(cr + 1) + [len(d)]
            ext = [float(d[a + int(np.argmax(np.abs(d[a:b])))]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
            e1 = ext[1] / step if len(ext) > 1 else float("nan")
            e2 = ext[2] / step if len(ext) > 2 else float("nan")
            ncr = sum(abs(e) > THR * step for e in ext[1:])
            lab = mode + lab_c
            print(f"  {lab:<24}{ts1*1e9:8.2f}n{ts2*1e9:8.2f}n{float(g.min()):10.4f}"
                  f"{gf:10.6f}{e1*100:+12.2f}%{e2*100:+12.2f}%{ncr:15d}")
            out[lab] = (ts1, ts2)
            save(f"agc_settle_{mode}{'_cap2' if cap else ''}", t, g)
    return out


if __name__ == "__main__":
    main()
