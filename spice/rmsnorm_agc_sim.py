#!/usr/bin/env python3
"""
SPICE validation of "Computational Primes" §5: RMSNorm in analog.

Level: junction-exact, wiring-ideal.
  - Translinear pipeline (Impl. A): real diode I-V (exp junction law) does
    all log-domain arithmetic; voltage adders (E-sources) stand in for the
    ideal translinear loop wiring. Monte-Carlo mismatch on IS, ideality n,
    and mirror ratio gives the precision figure a chip designer needs.
  - AGC feedback (Impl. B): continuous-time transient with a real
    integration capacitor -> settling time in nanoseconds, equilibrium
    accuracy, and re-settling after an input step.

Requires: ngspice in PATH, numpy. matplotlib optional (plots).

Usage: python3 rmsnorm_agc_sim.py [--mc-runs 200] [--n 8]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

VT = 0.02585  # thermal voltage at 300K
ISAT = 1e-16  # diode saturation current
IUNIT = 1e-6  # 1 uA = unit current (normalized 1.0)

WORKDIR = Path(__file__).parent / "out"


def run_ngspice(netlist: str, tag: str) -> str:
    WORKDIR.mkdir(exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(
        ["ngspice", "-b", str(cir)],
        capture_output=True, text=True, timeout=300, cwd=WORKDIR,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"ngspice failed for {tag}")
    return res.stdout


# ---------------------------------------------------------------------------
# Implementation A: translinear pipeline, DC operating point
# ---------------------------------------------------------------------------
#
# Log-domain current-mode arithmetic with real diodes:
#   input diode:   V_i = n*VT*ln(I_i/IS)         (P4, junction law)
#   E-source:      linear combination of log-voltages (ideal TL wiring)
#   output diode:  I_out = IS*exp(V/(n*VT))       (P3)
#
# Pipeline (currents normalized to IUNIT):
#   sq_i  = x_i^2                per-channel squarer     (P11 via P3/P4)
#   sum   = KCL sum of sq_i      shared 0V sense source  (P1, exact)
#   ms    = sum / N              CCCS = current mirror   (P2, mismatch-prone)
#   inv   = 1/sqrt(ms)           log-domain *(-0.5)      (P12)
#   out_i = x_i * inv            per-channel multiply    (P11)

def build_translinear_netlist(x_abs, is_mult, n_mult, mirror_gain, tag):
    """x_abs: positive input currents (normalized). *_mult: mismatch factors."""
    N = len(x_abs)
    L = [f"* translinear RMSNorm N={N} ({tag})"]
    dm = []  # per-device diode models

    def diode(idx):
        m = f"DM{idx}"
        dm.append(f".model {m} D(IS={ISAT*is_mult[idx]:.6e} N={n_mult[idx]:.6f})")
        return m

    di = 0
    # reference log voltage: 1 unit current through a diode
    L.append(f"Iref 0 nref DC {IUNIT:.6e}")
    L.append(f"Dref nref 0 {diode(di)}"); di += 1

    for i, xa in enumerate(x_abs):
        # input log stage
        L.append(f"Ix{i} 0 nx{i} DC {xa*IUNIT:.6e}")
        L.append(f"Dx{i} nx{i} 0 {diode(di)}"); di += 1
        # squarer: V = 2*V(nx_i) - V(nref) -> I = x_i^2 (units of IUNIT)
        L.append(f"Esq{i} esq{i} 0 VALUE={{2*V(nx{i})-V(nref)}}")
        L.append(f"Dsq{i} esq{i} msum {diode(di)}"); di += 1

    # KCL sum at node msum, held at 0V by sense source (P1)
    L.append("Vsum msum 0 DC 0")

    # current mirror 1:N with gain error (P2): copy sum/N into ms diode
    L.append(f"Fms 0 nms Vsum {mirror_gain/N:.8f}")
    L.append(f"Dms nms 0 {diode(di)}"); di += 1

    # 1/sqrt (P12): V = 1.5*V(nref) - 0.5*V(nms) -> I = 1/sqrt(ms)
    L.append("Einv einv 0 VALUE={1.5*V(nref)-0.5*V(nms)}")
    L.append(f"Dinv einv ninv_s {diode(di)}"); di += 1
    L.append("Vinv ninv_s 0 DC 0")
    L.append("Finv 0 ninv Vinv 1.0")
    L.append(f"Dinvl ninv 0 {diode(di)}"); di += 1

    # per-channel normalize (P11): I_out_i = x_i * inv
    for i in range(N):
        L.append(f"Eo{i} eo{i} 0 VALUE={{V(nx{i})+V(ninv)-V(nref)}}")
        L.append(f"Do{i} eo{i} no{i} {diode(di)}"); di += 1
        L.append(f"Vo{i} no{i} 0 DC 0")

    L += dm
    prints = " ".join(f"i(Vo{i})" for i in range(N))
    L += [
        ".control", "op",
        f"print {prints} > tl_{tag}.txt",
        "quit", ".endc", ".end",
    ]
    return "\n".join(L)


def parse_tl_output(tag, N):
    txt = (WORKDIR / f"tl_{tag}.txt").read_text()
    vals = {}
    for line in txt.splitlines():
        parts = line.split("=")
        if len(parts) == 2 and parts[0].strip().startswith("i(vo"):
            idx = int(parts[0].strip()[4:-1])
            vals[idx] = float(parts[1])
    return np.array([vals[i] for i in range(N)]) / IUNIT


# mismatch corners: (label, sigma_IS, sigma_n, sigma_mirror)
# n-mismatch is the killer: it scales the full log magnitude ln(I/IS) ~ 23,
# so 0.1% n-spread -> ~2.3% current error. BJTs match n to <0.05%;
# subthreshold MOSFETs locally ~0.1-0.5%.
CORNERS = [
    ("BJT-grade   (IS 1%, n 0.05%, mir 0.5%)", 0.01, 0.0005, 0.005),
    ("MOS typical (IS 3%, n 0.3%,  mir 1%)  ", 0.03, 0.003, 0.01),
    ("MOS worst   (IS 5%, n 0.5%,  mir 2%)  ", 0.05, 0.005, 0.02),
]


def translinear_experiment(N, mc_runs, rng):
    x = rng.standard_normal(N) * 1.5
    x_abs = np.clip(np.abs(x), 1e-3, None)
    ref = x_abs / np.sqrt(np.mean(x_abs**2))

    results = {}
    # ideal devices
    net = build_translinear_netlist(
        x_abs, np.ones(64), np.ones(64), 1.0, "ideal")
    run_ngspice(net, "tl_ideal")
    out = parse_tl_output("ideal", N)
    results["ideal"] = np.abs(out - ref) / np.abs(ref)

    # mismatch Monte Carlo per corner
    # metric: absolute error against unit-RMS reference (output SNR scale);
    # relative-per-channel blows up on small channels and misleads.
    for label, s_is, s_n, s_mir in CORNERS:
        errs = []
        for k in range(mc_runs):
            is_m = rng.lognormal(0, s_is, 64)
            n_m = rng.normal(1.0, s_n, 64)
            mir = rng.normal(1.0, s_mir)
            net = build_translinear_netlist(x_abs, is_m, n_m, mir, f"mc{k}")
            run_ngspice(net, f"tl_mc{k}")
            out = parse_tl_output(f"mc{k}", N)
            errs.append(np.abs(out - ref))  # ref has RMS=1
        results[label] = np.array(errs)
    return x_abs, ref, results


# ---------------------------------------------------------------------------
# Implementation B: AGC feedback loop, transient
# ---------------------------------------------------------------------------
#
# Continuous-time AGC (currents/voltages normalized, time real):
#   y_i  = G * x_i                       VGA (4-quadrant Gilbert, behavioral)
#   msq  = mean(y_i^2) -> RC lowpass     RMS detector (tau_det)
#   C dVg/dt = k*(gamma^2 - msq)         real cap integrator (P8)
#   G    = V(g)                          linear gain control
# Equilibrium: mean(G^2 x^2) = gamma^2  ->  G = gamma/RMS(x)  == RMSNorm.

def build_agc_netlist(x1, x2, gamma, tau_det_ns, c_pf, k_gain, t_step_ns,
                      t_end_ns, tag):
    N = len(x1)
    L = [f"* AGC RMSNorm N={N} ({tag})"]
    for i in range(N):
        # input steps from x1 to x2 at t_step
        L.append(
            f"Vx{i} x{i} 0 PWL(0 {x1[i]:.6f} {t_step_ns}n {x1[i]:.6f} "
            f"{t_step_ns + 0.05}n {x2[i]:.6f})")
    sq = "+".join(f"V(x{i})*V(x{i})" for i in range(N))
    # VGA + squarer + mean: msq_raw = G^2 * mean(x^2)
    L.append(f"Bmsq msqr 0 V={{V(g)*V(g)*({sq})/{N}}}")
    # detector lowpass tau = R*C
    r_det = tau_det_ns * 1e-9 / (1e-12)  # with C=1pF
    L.append(f"Rdet msqr msq {r_det:.3f}")
    L.append("Cdet msq 0 1p")
    # integrator on gain node: C dVg/dt = k*(gamma^2 - msq)
    L.append(f"Bint 0 g I={{{k_gain}*({gamma*gamma} - V(msq))}}")
    L.append(f"Cg g 0 {c_pf}p")
    L.append(".ic V(g)=0.1 V(msq)=0")
    # outputs
    for i in range(N):
        L.append(f"By{i} y{i} 0 V={{V(g)*V(x{i})}}")
    vecs = "v(g) v(msq) " + " ".join(f"v(y{i})" for i in range(N))
    L += [
        ".control",
        f"tran 0.01n {t_end_ns}n uic",
        f"wrdata agc_{tag}.txt {vecs}",
        "quit", ".endc", ".end",
    ]
    return "\n".join(L)


def parse_agc_output(tag):
    data = np.loadtxt(WORKDIR / f"agc_{tag}.txt")
    # wrdata format: time v1 time v2 ... -> columns 0,1 then odd columns
    t = data[:, 0]
    vals = data[:, 1::2]
    return t, vals


def settling_time(t, g, target, tol=0.01):
    """last time |g-target|/target > tol, i.e. entry into the tol band."""
    off = np.abs(g - target) / abs(target) > tol
    if not off.any():
        return t[0]
    return t[np.max(np.nonzero(off)) + 1] if np.max(np.nonzero(off)) + 1 < len(t) else np.inf


def agc_experiment(N, rng):
    x1 = rng.standard_normal(N) * 1.5
    x2 = x1 * 2.5  # input step: RMS jumps 2.5x
    gamma = 1.0
    g_target1 = gamma / np.sqrt(np.mean(x1**2))
    g_target2 = gamma / np.sqrt(np.mean(x2**2))

    t_step, t_end = 40.0, 80.0
    net = build_agc_netlist(x1, x2, gamma, tau_det_ns=0.5, c_pf=1.0,
                            k_gain=2e-3, t_step_ns=t_step, t_end_ns=t_end,
                            tag="main")
    run_ngspice(net, "agc_main")
    t, vals = parse_agc_output("main")
    g = vals[:, 0]

    m1 = t < t_step * 1e-9
    m2 = t >= t_step * 1e-9
    ts1 = settling_time(t[m1], g[m1], g_target1)
    ts2 = settling_time(t[m2] - t_step * 1e-9, g[m2], g_target2)

    g_eq1 = g[m1][-1]
    g_eq2 = g[-1]
    err1 = abs(g_eq1 - g_target1) / g_target1
    err2 = abs(g_eq2 - g_target2) / g_target2

    # per-channel output vs exact RMSNorm at equilibrium
    y_eq = vals[m1][-1, 2:]
    ref = x1 / np.sqrt(np.mean(x1**2))
    ch_err = np.abs(y_eq - ref) / np.maximum(np.abs(ref), 1e-9)

    return dict(t=t, g=g, vals=vals, g_target1=g_target1,
                g_target2=g_target2, ts1=ts1, ts2=ts2, err1=err1,
                err2=err2, ch_err=ch_err, t_step_ns=t_step)


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
    W = 72
    print("=" * W)
    print("  SPICE validation: RMSNorm analog (Computational Primes §5)")
    print(f"  N={args.n} channels, MC runs={args.mc_runs}, ngspice="
          f"{shutil.which('ngspice')}")
    print("=" * W)

    # --- Implementation A ---
    print("\n[A] Translinear pipeline (junction-exact, DC)")
    x_abs, ref, res = translinear_experiment(args.n, args.mc_runs, rng)
    e0 = res["ideal"]
    print(f"    ideal devices : rel_err mean={e0.mean():.2e} "
          f"max={e0.max():.2e}")
    print(f"    mismatch MC (abs err vs unit-RMS output, "
          f"{args.mc_runs} runs each):")
    for label, *_ in CORNERS:
        mc = res[label]
        per_run = mc.mean(axis=1)
        p95 = np.percentile(per_run, 95)
        bits = -np.log2(p95) if p95 > 0 else 99
        print(f"      {label}: mean={per_run.mean()*100:.2f}%  "
              f"p95={p95*100:.2f}%  ~{bits:.1f} bit")

    # --- Implementation B ---
    print("\n[B] AGC feedback loop (transient, real capacitor)")
    agc = agc_experiment(args.n, rng)
    print(f"    settling from cold start : {agc['ts1']*1e9:.2f} ns "
          f"(to 1% band), equilibrium gain err={agc['err1']*100:.4f}%")
    print(f"    re-settling after 2.5x input step: {agc['ts2']*1e9:.2f} ns, "
          f"gain err={agc['err2']*100:.4f}%")
    print(f"    per-channel output vs exact RMSNorm: mean="
          f"{agc['ch_err'].mean()*100:.4f}%  max={agc['ch_err'].max()*100:.4f}%")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(agc["t"] * 1e9, agc["g"], lw=1.5)
        ax[0].axhline(agc["g_target1"], ls="--", c="gray", lw=0.8)
        ax[0].axhline(agc["g_target2"], ls="--", c="gray", lw=0.8)
        ax[0].axvline(agc["t_step_ns"], ls=":", c="red", lw=0.8)
        ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel("gain G")
        ax[0].set_title("AGC settling (input step at 40 ns)")
        for label, *_ in CORNERS:
            ax[1].hist(res[label].mean(axis=1) * 100, bins=30, alpha=0.6,
                       label=label.split("(")[0].strip())
        ax[1].set_xlabel("mean abs. error vs unit-RMS output [%]")
        ax[1].set_title("Translinear MC mismatch")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(WORKDIR / "rmsnorm_spice.png", dpi=130)
        print(f"\n    plot: {WORKDIR / 'rmsnorm_spice.png'}")
    except ImportError:
        pass

    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
