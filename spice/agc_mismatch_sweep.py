#!/usr/bin/env python3
"""
Closes the two open validation gaps of paper v2 (Section 5.3, "What this
validation does not cover"):

GAP 1 — AGC-path mismatch. v2 argues structurally that detector-side
mismatch shifts only the common-mode gain while the VGA array is the
only per-channel exposure. Here we TEST that: the RMS detector is built
from real mismatched junctions (log-domain squarers, mirrored sum), the
loop equilibrium G is found by bisection on the SPICE-measured detector
output (equivalent to the settled integrator validated in the transient
testbench), and the output error is decomposed into
    common-mode:  |G_settled - G_ideal| / G_ideal   (a gamma shift)
    per-channel:  residual distortion after removing the common factor.
Prediction: common-mode tracks detector mismatch, per-channel tracks
VGA mismatch only.

GAP 2 — N-scaling. Per-channel error sources scale per channel, shared
paths average: detector-induced gain error should shrink ~1/sqrt(N),
VGA-induced per-channel error should stay flat. Swept at N = 8/32/128.

Usage: python3 agc_mismatch_sweep.py [--mc-runs 100]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

VT = 0.025864890  # ngspice effective kT/q (diode-probed, SPICE3 constants)
ISAT = 1e-16
IUNIT = 1e-6
GAMMA = 1.0

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
# Junction-based RMS detector: y_i -> mean-square current
# ---------------------------------------------------------------------------
# Per channel: input current |y_i| -> log diode -> E(2*Vlog - Vref)
# -> squarer output diode -> KCL sum -> 1:N mirror (gain error).
# Diode order per netlist: log[0..N-1], sq[0..N-1], ms, ref.

def build_detector_netlist(y_abs, is_m, n_m, mirror_gain, tag):
    N = len(y_abs)
    L = [f"* junction RMS detector N={N} ({tag})"]
    dm = []
    di = 0

    def diode(idx):
        m = f"DM{idx}"
        dm.append(f".model {m} D(IS={ISAT*is_m[idx]:.6e} N={n_m[idx]:.6f})")
        return m

    L.append(f"Iref 0 nref DC {IUNIT:.6e}")
    L.append(f"Dref nref 0 {diode(2*N)}")
    for i in range(N):
        L.append(f"Iy{i} 0 ny{i} DC {max(y_abs[i],1e-4)*IUNIT:.6e}")
        L.append(f"Dy{i} ny{i} 0 {diode(di)}"); di += 1
        L.append(f"Esq{i} esq{i} 0 VALUE={{2*V(ny{i})-V(nref)}}")
        L.append(f"Dsq{i} esq{i} msum {diode(N+i)}")
    L.append("Vsum msum 0 DC 0")
    # 1:N mirror with gain error; output sensed at a 0V-held node
    L.append(f"Fms 0 nmss Vsum {mirror_gain/N:.8f}")
    L.append("Vmss nmss 0 DC 0")
    L += dm
    L += [".control", "op", f"print i(Vmss) > det_{tag}.txt",
          "quit", ".endc", ".end"]
    return "\n".join(L)


def detector_ms(x_eff, G, is_m, n_m, mirror_gain, tag):
    """SPICE-measured mean-square (normalized units) of y = G*x_eff."""
    net = build_detector_netlist(np.abs(G * x_eff), is_m, n_m,
                                 mirror_gain, tag)
    run_ngspice(net, f"det_{tag}")
    txt = (WORKDIR / f"det_{tag}.txt").read_text()
    for line in txt.splitlines():
        if "i(vmss)" in line:
            return float(line.split("=")[1]) / IUNIT
    raise RuntimeError("no detector output")


def settle_gain(x_eff, is_m, n_m, mirror_gain, tag, iters=16):
    """Bisection on G until detector output = GAMMA^2 (loop equilibrium)."""
    lo, hi = 1e-3, 1e3
    for k in range(iters):
        G = np.sqrt(lo * hi)
        ms = detector_ms(x_eff, G, is_m, n_m, mirror_gain, f"{tag}_{k}")
        if ms > GAMMA**2:
            hi = G
        else:
            lo = G
    return np.sqrt(lo * hi)


# ---------------------------------------------------------------------------

DET_CORNERS = [("det BJT   (IS 1%, n 0.05%, mir 0.5%)", 0.01, 0.0005, 0.005),
               ("det MOS   (IS 3%, n 0.3%,  mir 1%)  ", 0.03, 0.003, 0.01),
               ("det worst (IS 5%, n 0.5%,  mir 2%)  ", 0.05, 0.005, 0.02)]
VGA_LEVELS = [("VGA 0.5%", 0.005), ("VGA 1%  ", 0.01), ("VGA 2%  ", 0.02)]


def decompose_error(y_hat, y_ref):
    """Split error into common-mode scale and per-channel residual (RMS)."""
    alpha = float(np.dot(y_hat, y_ref) / np.dot(y_ref, y_ref))
    resid = y_hat / alpha - y_ref
    return abs(alpha - 1.0), float(np.sqrt(np.mean(resid**2)))


def run_cell(N, mc, s_is, s_n, s_mir, s_vga, rng, tag):
    cm_list, pc_list = [], []
    x = rng.standard_normal(N) * 1.5
    x = np.where(np.abs(x) < 0.05, 0.05 * np.sign(x) + (x == 0) * 0.05, x)
    y_ref = GAMMA * x / np.sqrt(np.mean(x**2))
    G_ideal = GAMMA / np.sqrt(np.mean(x**2))
    for k in range(mc):
        is_m = rng.lognormal(0, s_is, 2 * N + 1)
        n_m = rng.normal(1.0, s_n, 2 * N + 1)
        mir = rng.normal(1.0, s_mir)
        eps = rng.normal(0.0, s_vga, N)
        x_eff = x * (1 + eps)          # VGA array: the signal path
        G = settle_gain(x_eff, is_m, n_m, mir, f"{tag}{k}")
        y_hat = G * x_eff              # AGC output = VGA output, nothing else
        cm, pc = decompose_error(y_hat, y_ref)
        cm_list.append(cm)
        pc_list.append(pc)
    return (np.median(cm_list), np.percentile(cm_list, 95),
            np.median(pc_list), np.percentile(pc_list, 95), G_ideal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")

    rng = np.random.default_rng(args.seed)
    W = 76
    print("=" * W)
    print("  AGC-path mismatch sweep + N-scaling (closes v2 Section 5.3 gaps)")
    print(f"  junction RMS detector, loop closed by bisection, "
          f"MC={args.mc_runs}/cell")
    print("=" * W)

    N = 8
    print(f"\n[A] Detector-only mismatch (VGA ideal), N={N}")
    print(f"  {'cell':<40} {'common-mode p50/p95':>22} "
          f"{'per-channel p50/p95':>22}")
    for label, s_is, s_n, s_mir in DET_CORNERS:
        cm50, cm95, pc50, pc95, _ = run_cell(
            N, args.mc_runs, s_is, s_n, s_mir, 0.0, rng, "A")
        print(f"  {label:<40} {cm50*100:9.3f}/{cm95*100:6.3f}% "
              f"{pc50*100:12.4f}/{pc95*100:7.4f}%")

    print(f"\n[B] VGA-only mismatch (detector ideal), N={N}")
    print(f"  {'cell':<40} {'common-mode p50/p95':>22} "
          f"{'per-channel p50/p95':>22}")
    for label, s_vga in VGA_LEVELS:
        cm50, cm95, pc50, pc95, _ = run_cell(
            N, args.mc_runs, 0.0, 0.0, 0.0, s_vga, rng, "B")
        print(f"  {label:<40} {cm50*100:9.3f}/{cm95*100:6.3f}% "
              f"{pc50*100:12.4f}/{pc95*100:7.4f}%")

    print(f"\n[C] Both (detector MOS-typical + VGA 1%), N={N}")
    cm50, cm95, pc50, pc95, _ = run_cell(
        N, args.mc_runs, 0.03, 0.003, 0.01, 0.01, rng, "C")
    print(f"  combined                                 "
          f"{cm50*100:9.3f}/{cm95*100:6.3f}% "
          f"{pc50*100:12.4f}/{pc95*100:7.4f}%")

    print(f"\n[D] N-scaling (detector MOS-typical + VGA 1%)")
    print(f"  {'N':>5} {'common-mode p50':>16} {'per-channel p50':>16}")
    for n in (8, 32, 128):
        cm50, cm95, pc50, pc95, _ = run_cell(
            n, args.mc_runs, 0.03, 0.003, 0.01, 0.01, rng, f"D{n}")
        print(f"  {n:>5} {cm50*100:15.3f}% {pc50*100:15.4f}%")

    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
