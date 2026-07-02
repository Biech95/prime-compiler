#!/usr/bin/env python3
"""
SPICE validation: Softmax via sum-feedback AGC vs. open-loop translinear.

Companion to rmsnorm_agc_sim.py — same methodology (junction-exact,
wiring-ideal), applied to softmax:

    softmax(x)_i = exp(x_i) / sum_j exp(x_j)

Both variants share the SAME per-channel exponential stage (P3, real
diode junctions with mismatch) — that exposure is irreducible. They
differ only in the normalization path:

  OPEN-LOOP (translinear): per-channel log diode + output exp diode
      + shared sum/ref diodes  ->  3 mismatched junctions per channel.
  AGC (settled loop):      VGA per channel (gain error eps_i), sum
      detector is KCL (exact), loop pins sum(y) = I_target. Gain is
      the settled-loop DC equivalent  G = target / sum(I_i(1+eps_i)).

Prediction (from the RMSNorm result): AGC removes the per-channel
normalization junctions; residual error = exp stage + VGA only.

Usage: python3 softmax_agc_sim.py [--mc-runs 200] [--n 8]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

VT = 0.025864890  # ngspice-46 effective kT/q at default temp (measured
                  # via diode probe; SPICE3-legacy physical constants)
ISAT = 1e-16
ITARGET = 1e-6   # loop target: total output current 1 uA
BIAS = 23.0      # common logit offset: puts exp-stage currents at uA scale
                 # (keeps GMIN leakage negligible; e^BIAS cancels in softmax)

WORKDIR = Path(__file__).parent / "out"


def run_ngspice(netlist: str, tag: str):
    WORKDIR.mkdir(exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)],
                         capture_output=True, text=True, timeout=300,
                         cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"ngspice failed for {tag}")


# ---------------------------------------------------------------------------
# DC netlist: shared exp stages -> both normalization paths
# ---------------------------------------------------------------------------

def build_softmax_netlist(x, is_m, n_m, eps_vga, tag):
    """x: logits. is_m/n_m: per-diode mismatch (26 devices). eps_vga: per
    channel VGA gain error. Diode order: exp[0..N-1], log[0..N-1],
    out[0..N-1], sum, ref."""
    N = len(x)
    L = [f"* softmax open-loop vs AGC N={N} ({tag})"]
    dm = []
    di = 0

    def diode(idx):
        m = f"DM{idx}"
        dm.append(f".model {m} D(IS={ISAT*is_m[idx]:.6e} N={n_m[idx]:.6f})")
        return m

    # --- shared exponential stage (P3): I_i = Is_i * exp((BIAS+x_i)*n0/n_i)
    for i in range(N):
        L.append(f"Vd{i} nd{i} 0 DC {(BIAS + x[i]) * VT:.9f}")
        L.append(f"De{i} nd{i} se{i} {diode(i)}")
        L.append(f"Vse{i} se{i} 0 DC 0")
    di = N

    # --- OPEN-LOOP path: log diode per channel, shared sum + ref, out diode
    for i in range(N):
        L.append(f"Fl{i} 0 nl{i} Vse{i} 1.0")
        L.append(f"Dl{i} nl{i} 0 {diode(di)}"); di += 1
    for i in range(N):
        L.append(f"Fs{i} 0 nsum Vse{i} 1.0")
    L.append(f"Dsum nsum 0 {diode(2*N)}")
    L.append(f"Iref 0 nref DC {ITARGET:.6e}")
    L.append(f"Dref nref 0 {diode(2*N+1)}")
    di = 2*N + 2
    for i in range(N):
        L.append(f"Eo{i} eo{i} 0 VALUE={{V(nl{i})+V(nref)-V(nsum)}}")
        L.append(f"Do{i} eo{i} no{i} {diode(di)}"); di += 1
        L.append(f"Vo{i} no{i} 0 DC 0")

    # --- AGC path (settled-loop DC equivalent): G pins sum(y)=ITARGET.
    # KCL sum detector is exact; only VGA gain errors eps_i are per-channel.
    den = "+".join(f"i(Vse{i})*{1+eps_vga[i]:.8f}" for i in range(N))
    L.append(f"Eg gctl 0 VALUE={{{ITARGET:.6e}/({den})}}")
    for i in range(N):
        L.append(f"Ba{i} 0 na{i} I={{V(gctl)*i(Vse{i})*{1+eps_vga[i]:.8f}}}")
        L.append(f"Va{i} na{i} 0 DC 0")

    L += dm
    prints = " ".join(f"i(Vo{i})" for i in range(N)) + " " + \
             " ".join(f"i(Va{i})" for i in range(N))
    L += [".control", "op", f"print {prints} > sm_{tag}.txt",
          "quit", ".endc", ".end"]
    return "\n".join(L)


def parse_output(tag, N):
    txt = (WORKDIR / f"sm_{tag}.txt").read_text()
    vo, va = {}, {}
    for line in txt.splitlines():
        parts = line.split("=")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        if key.startswith("i(vo"):
            vo[int(key[4:-1])] = float(parts[1])
        elif key.startswith("i(va"):
            va[int(key[4:-1])] = float(parts[1])
    open_p = np.array([vo[i] for i in range(N)]) / ITARGET
    agc_p = np.array([va[i] for i in range(N)]) / ITARGET
    return open_p, agc_p


CORNERS = [
    ("BJT-grade   (IS 1%, n 0.05%, VGA 0.5%)", 0.01, 0.0005, 0.005),
    ("MOS typical (IS 3%, n 0.3%,  VGA 1%)  ", 0.03, 0.003, 0.01),
    ("MOS worst   (IS 5%, n 0.5%,  VGA 2%)  ", 0.05, 0.005, 0.02),
]


# ---------------------------------------------------------------------------
# AGC transient: real capacitor, settling time
# ---------------------------------------------------------------------------

def build_agc_transient(x1, x2, t_step_ns, t_end_ns, tag):
    """Behavioral continuous-time sum-AGC on exp(x) inputs (normalized)."""
    N = len(x1)
    e1 = np.exp(x1); e2 = np.exp(x2)
    L = [f"* softmax AGC transient N={N}"]
    for i in range(N):
        L.append(f"Vx{i} x{i} 0 PWL(0 {e1[i]:.6f} {t_step_ns}n {e1[i]:.6f} "
                 f"{t_step_ns+0.05}n {e2[i]:.6f})")
    ssum = "+".join(f"V(x{i})" for i in range(N))
    L.append(f"Bsum msum 0 V={{V(g)*({ssum})}}")
    L.append("Bint 0 g I={4e-5*(1 - V(msum))}")
    L.append("Cg g 0 1p")
    L.append(".ic V(g)=0.05")
    for i in range(N):
        L.append(f"By{i} y{i} 0 V={{V(g)*V(x{i})}}")
    vecs = "v(g) v(msum) " + " ".join(f"v(y{i})" for i in range(N))
    L += [".control", f"tran 0.01n {t_end_ns}n uic",
          f"wrdata smagc_{tag}.txt {vecs}", "quit", ".endc", ".end"]
    return "\n".join(L)


def settling_time(t, s, target, tol=0.01):
    off = np.abs(s - target) / abs(target) > tol
    if not off.any():
        return t[0]
    idx = np.max(np.nonzero(off)) + 1
    return t[idx] if idx < len(t) else np.inf


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mc-runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")

    rng = np.random.default_rng(args.seed)
    N = args.n
    x = rng.standard_normal(N) * 2.0
    p_ref = np.exp(x) / np.exp(x).sum()
    n_dev = 3 * N + 2

    W = 74
    print("=" * W)
    print("  SPICE: Softmax — open-loop translinear vs sum-feedback AGC")
    print(f"  N={N} logits ~ N(0,2), MC={args.mc_runs}/corner, paired draws")
    print("=" * W)

    # ideal
    net = build_softmax_netlist(x, np.ones(n_dev), np.ones(n_dev),
                                np.zeros(N), "ideal")
    run_ngspice(net, "sm_ideal")
    po, pa = parse_output("ideal", N)
    print(f"\n  ideal devices: open-loop L1={np.abs(po-p_ref).sum():.2e}  "
          f"AGC L1={np.abs(pa-p_ref).sum():.2e}")

    # Monte Carlo, paired (same exp-stage + VGA draws feed both paths)
    print(f"\n  mismatch MC (L1 distance to exact softmax, "
          f"{args.mc_runs} paired runs):")
    print(f"  {'corner':<40} {'open p50/p95':>16} {'AGC p50/p95':>16} "
          f"{'ratio':>6}")
    results = {}
    for label, s_is, s_n, s_vga in CORNERS:
        l1_open, l1_agc = [], []
        for k in range(args.mc_runs):
            is_m = rng.lognormal(0, s_is, n_dev)
            n_m = rng.normal(1.0, s_n, n_dev)
            eps = rng.normal(0.0, s_vga, N)
            net = build_softmax_netlist(x, is_m, n_m, eps, f"mc{k}")
            run_ngspice(net, f"sm_mc{k}")
            po, pa = parse_output(f"mc{k}", N)
            l1_open.append(np.abs(po - p_ref).sum())
            l1_agc.append(np.abs(pa - p_ref).sum())
        lo, la = np.array(l1_open), np.array(l1_agc)
        results[label] = (lo, la)
        ratio = np.median(lo) / np.median(la)
        print(f"  {label:<40} "
              f"{np.median(lo)*100:6.2f}/{np.percentile(lo,95)*100:5.2f}% "
              f"{np.median(la)*100:6.2f}/{np.percentile(la,95)*100:5.2f}% "
              f"{ratio:5.1f}x")

    # transient settling
    x2 = rng.standard_normal(N) * 2.0
    net = build_agc_transient(x, x2, 40.0, 80.0, "main")
    run_ngspice(net, "smagc_main")
    data = np.loadtxt(WORKDIR / "smagc_main.txt")
    t = data[:, 0]
    msum = data[:, 3]  # columns: t,g,t,msum,...
    m1, m2 = t < 40e-9, t >= 40e-9
    ts1 = settling_time(t[m1], msum[m1], 1.0)
    ts2 = settling_time(t[m2] - 40e-9, msum[m2], 1.0)
    y_eq = data[m1][-1, 5::2]
    p1 = np.exp(x) / np.exp(x).sum()
    print(f"\n  AGC transient (1 pF, sum detector):")
    print(f"    cold-start settling to 1% band : {ts1*1e9:.2f} ns")
    print(f"    re-settling after logit switch : {ts2*1e9:.2f} ns")
    print(f"    equilibrium vs exact softmax   : max err "
          f"{np.abs(y_eq-p1).max():.2e}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(t * 1e9, msum, lw=1.5)
        ax[0].axhline(1.0, ls="--", c="gray", lw=0.8)
        ax[0].axvline(40, ls=":", c="red", lw=0.8)
        ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel("sum(y) / target")
        ax[0].set_title("Softmax-AGC settling (logit switch at 40 ns)")
        label = CORNERS[1][0]
        lo, la = results[label]
        bins = np.linspace(0, max(lo.max(), la.max()) * 100, 30)
        ax[1].hist(lo * 100, bins=bins, alpha=0.6, label="open-loop")
        ax[1].hist(la * 100, bins=bins, alpha=0.6, label="AGC")
        ax[1].set_xlabel("L1 error vs exact softmax [%]")
        ax[1].set_title("MC mismatch, MOS-typical corner")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(WORKDIR / "softmax_spice.png", dpi=130)
        print(f"\n    plot: {WORKDIR / 'softmax_spice.png'}")
    except ImportError:
        pass

    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
