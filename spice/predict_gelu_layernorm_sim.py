#!/usr/bin/env python3
"""
SPICE arbiter for the prospective prediction of `docs/exp_mismatch_calculus.md`.

Two circuits the mismatch calculus has never seen, both at the paper's
MOS-typical corner (IS 3 %, n 0.3 %, mirror 1 %, VGA 1 %; sigmoid pair
additionally V_t 10 mV, K' 3 %), N = 8 channels, 200 Monte-Carlo draws:

  [G] GELU  = P11 . P6 :  g(x) = x * sigma(1.702 x)
      translinear (Gilbert) multiplier -- four junctions in one loop:
      Dx . Ds / Dref -> Do -- driven by a bipolar/subthreshold differential
      pair whose collector current is I_tail * sigma((v_bp - v_bn)/V_T).

  [L] LayerNorm = P1 . P2 (mean subtraction by mirrors) + the validated
      AGC RMSNorm.  The junction RMS detector and the loop equilibrium are
      exactly `agc_mismatch_sweep.py`; the new part is the mean path in
      front of the loop input.

Fidelity: junction-exact, wiring-ideal (see spice/README.md).  All
exponential physics is real device I-V; current mirrors and the differential
copy are ideal sources carrying a *stated* gain error.

Predictions are read from `mismatch_calculus.py` (the algebra), never
recomputed here, so the comparison is genuinely out-of-sample.

Usage:
  OPENBLAS_NUM_THREADS=1 python3 predict_gelu_layernorm_sim.py \
      --mc-runs 200 --workers 12
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import mismatch_calculus as MC

VT = 0.025864890          # ngspice SPICE3-legacy kT/q  (spice/README.md)
ISAT = 1e-16
IUNIT = 1e-6
ITAIL = 4e-6              # sigmoid-pair tail current
IREF = 4e-6               # translinear reference (so I_out = |x| * sigma)
GAMMA = 1.0

WORKDIR = Path(__file__).parent / "out"
DIODE = ("BF=1e6 VAF=1e12 RB=0 RE=0 RC=0 CJE=0 CJC=0 TF=0 TR=0")
OPTS = ".options gmin=1e-14 reltol=1e-6 abstol=1e-16"


def run_ngspice(netlist, tag):
    WORKDIR.mkdir(exist_ok=True)
    cir = WORKDIR / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=300, cwd=WORKDIR)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise RuntimeError(f"ngspice failed for {tag}")


def parse_print(tag, prefix, n):
    txt = (WORKDIR / f"{tag}.txt").read_text()
    vals = {}
    for line in txt.splitlines():
        p = line.split("=")
        if len(p) != 2:
            continue
        key = p[0].strip()
        if key.startswith(prefix) and key.endswith(")"):
            try:
                idx = int(key[len(prefix):-1])
            except ValueError:
                continue
            vals[idx] = float(p[1])
    return np.array([vals[i] for i in range(n)])


def parse_scalar(tag, name):
    txt = (WORKDIR / f"{tag}.txt").read_text()
    for line in txt.splitlines():
        p = line.split("=")
        if len(p) == 2 and p[0].strip() == name:
            return float(p[1])
    raise RuntimeError(f"{name} not found in {tag}")


# ---------------------------------------------------------------------------
# [G]  GELU  =  Gilbert multiplier  x  differential-pair sigmoid
# ---------------------------------------------------------------------------

def build_gelu_netlist(x, d, tag):
    """d: dict of per-device deviations, same names as MC.gelu_draw()."""
    N = len(x)
    u = 1.702 * x
    L = [f"* GELU = P11.P6, N={N} ({tag})", OPTS,
         "Vcc vcc 0 DC 2.0", "Vee vee 0 DC -2.0",
         f"Iref 0 nref DC {IREF:.8e}",
         "Dref nref 0 DMR"]
    models = [f".model DMR D(IS={ISAT*np.exp(d['iota_r'][0]):.8e} "
              f"N={1.0+d['nu_r'][0]:.8f})"]
    for i in range(N):
        vd = u[i] * VT + d["vos"][i]          # differential base drive
        L += [f"Vbp{i} bp{i} 0 DC {vd/2:.10e}",
              f"Vbn{i} bn{i} 0 DC {-vd/2:.10e}",
              f"Q{i}a qa{i} bp{i} e{i} QA{i}",
              f"Q{i}b qb{i} bn{i} e{i} QB{i}",
              f"It{i} e{i} vee DC {ITAIL*(1+d['g_tail'][i]):.8e}",
              f"Vca{i} vcc qa{i} DC 0",
              f"Vcb{i} vcc qb{i} DC 0",
              # sigmoid current -> log stage of the translinear loop
              f"Fs{i} 0 ns{i} Vca{i} 1.0",
              f"Ds{i} ns{i} 0 DMS{i}",
              # |x| drive, carrying the per-channel VGA gain error
              f"Ix{i} 0 nx{i} DC {abs(x[i])*(1+d['eps_vga'][i])*IUNIT:.8e}",
              f"Dx{i} nx{i} 0 DMX{i}",
              # translinear loop closure: Vx + Vs - Vref -> Vout
              f"Eo{i} eo{i} 0 VALUE={{V(nx{i})+V(ns{i})-V(nref)}}",
              f"Do{i} eo{i} nd{i} DMO{i}",
              f"Vd{i} nd{i} 0 DC 0",
              # output mirror with its own gain error
              f"Fo{i} 0 no{i} Vd{i} {1+d['g_out'][i]:.8f}",
              f"Vo{i} no{i} 0 DC 0"]
        models += [
            f".model QA{i} NPN(IS={ISAT*np.exp(d['iota_a'][i]):.8e} "
            f"NF={1.0+d['nu_a'][i]:.8f} {DIODE})",
            f".model QB{i} NPN(IS={ISAT*np.exp(d['iota_b'][i]):.8e} "
            f"NF={1.0+d['nu_b'][i]:.8f} {DIODE})",
            f".model DMS{i} D(IS={ISAT*np.exp(d['iota_s'][i]):.8e} "
            f"N={1.0+d['nu_s'][i]:.8f})",
            f".model DMX{i} D(IS={ISAT*np.exp(d['iota_x'][i]):.8e} "
            f"N={1.0+d['nu_x'][i]:.8f})",
            f".model DMO{i} D(IS={ISAT*np.exp(d['iota_o'][i]):.8e} "
            f"N={1.0+d['nu_o'][i]:.8f})"]
    L += models
    pr = " ".join(f"i(Vo{i})" for i in range(N))
    L += [".control", "op", f"print {pr} > {tag}.txt", "quit", ".endc", ".end"]
    return "\n".join(L)


def gelu_one(job):
    j, x, d = job
    tag = f"gelu_{j}"
    run_ngspice(build_gelu_netlist(x, d, tag), tag)
    i_out = parse_print(tag, "i(vo", len(x)) / IUNIT
    for suf in (".cir", ".txt"):
        (WORKDIR / f"{tag}{suf}").unlink(missing_ok=True)
    return j, np.sign(x) * i_out


# ---------------------------------------------------------------------------
# [L]  LayerNorm = mean subtraction (mirrors) + AGC RMSNorm
# ---------------------------------------------------------------------------

def build_ln_netlist(x, d, G, tag):
    N = len(x)
    L = [f"* LayerNorm = P1.P2 mean + AGC RMSNorm, N={N} ({tag})", OPTS,
         f"Iref 0 nref DC {IUNIT:.8e}", "Dref nref 0 DMR"]
    models = [f".model DMR D(IS={ISAT*np.exp(d['iota_r']):.8e} "
              f"N={1.0+d['nu_r']:.8f})"]
    for i in range(N):
        L.append(f"Vx{i} nx{i} 0 DC {x[i]:.10e}")
    # shared 1:N mirror computing the mean (P1 . P2), one gain error
    msum = "+".join(f"V(nx{i})" for i in range(N))
    L.append(f"Emean nm 0 VALUE={{({msum})/{N}*{1+d['g_mir_mean']:.8f}}}")
    for i in range(N):
        # per-channel copy of the mean, each with its own mirror gain error
        L.append(f"Ec{i} nc{i} 0 VALUE={{V(nx{i})-V(nm)*"
                 f"{1+d['g_copy'][i]:.8f}}}")
        # VGA array (the only per-channel path inside the loop) + RMS detector
        L.append(f"By{i} 0 ny{i} I={{max(abs(V(nc{i}))*"
                 f"{G*(1+d['eps_vga'][i]):.10e},1e-4)*{IUNIT:.6e}}}")
        L.append(f"Dy{i} ny{i} 0 DMY{i}")
        L.append(f"Esq{i} esq{i} 0 VALUE={{2*V(ny{i})-V(nref)}}")
        L.append(f"Dsq{i} esq{i} msum DMSQ{i}")
        models += [
            f".model DMY{i} D(IS={ISAT*np.exp(d['iota_y'][i]):.8e} "
            f"N={1.0+d['nu_y'][i]:.8f})",
            f".model DMSQ{i} D(IS={ISAT*np.exp(d['iota_sq'][i]):.8e} "
            f"N={1.0+d['nu_sq'][i]:.8f})"]
    L += ["Vsum msum 0 DC 0",
          f"Fms 0 nmss Vsum {(1+d['g_mir_det'])/N:.10f}",
          "Vmss nmss 0 DC 0"]
    L += models
    L += [".control", "op", f"print i(Vmss) > {tag}.txt",
          "quit", ".endc", ".end"]
    return "\n".join(L)


def ln_one(job):
    j, x, d, iters = job
    lo, hi = 1e-3, 1e3
    for k in range(iters):
        G = float(np.sqrt(lo * hi))
        tag = f"lnp_{j}_{k}"
        run_ngspice(build_ln_netlist(x, d, G, tag), tag)
        ms = parse_scalar(tag, "i(vmss)") / IUNIT
        for suf in (".cir", ".txt"):
            (WORKDIR / f"{tag}{suf}").unlink(missing_ok=True)
        if ms > GAMMA ** 2:
            hi = G
        else:
            lo = G
    G = float(np.sqrt(lo * hi))
    c = x - np.mean(x) * (1 + d["g_mir_mean"]) * (1 + d["g_copy"])
    return j, G * c * (1 + d["eps_vga"])


# ---------------------------------------------------------------------------

def pct(v, q):
    return float(np.percentile(v, q)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-runs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bisect", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    if not shutil.which("ngspice"):
        sys.exit("ngspice not found in PATH")
    WORKDIR.mkdir(exist_ok=True)

    # ---- the predictions, taken verbatim from the calculus ----------------
    pred_g = MC.predict_gelu()
    pred_l = MC.predict_layernorm()

    W = 92
    print("=" * W)
    print("  SPICE ARBITER -- GELU (P11.P6) and full LayerNorm (P1.P2 + AGC)")
    print(f"  ngspice={shutil.which('ngspice')}  VT={VT}  N=8  "
          f"MC={args.mc_runs}  MOS-typical corner")
    print("=" * W)

    # ---- [G] GELU --------------------------------------------------------
    xg = MC.gelu_inputs()
    ref_g = xg / (1.0 + np.exp(-1.702 * xg))
    rng = np.random.default_rng(args.seed)
    draws_g = [MC.gelu_draw(rng, len(xg)) for _ in range(args.mc_runs)]

    # ideal-device control (the bench must reproduce GELU when devices match)
    zero = {k: np.zeros_like(v) for k, v in draws_g[0].items()}
    _, y0 = gelu_one((999999, xg, zero))
    err0 = float(np.max(np.abs(y0 - ref_g) / np.maximum(np.abs(ref_g), 1e-12)))
    print(f"\n[G] ideal-device control: max rel. deviation from "
          f"x*sigma(1.702x) = {err0:.3e}")

    jobs = [(j, xg, d) for j, d in enumerate(draws_g)]
    with mp.Pool(args.workers) as pool:
        res = pool.map(gelu_one, jobs, chunksize=4)
    cm_g, pc_g = [], []
    for _, y in sorted(res):
        a, p = MC.decompose(y, ref_g)
        cm_g.append(a)
        pc_g.append(p)
    gel = dict(pc_p50=pct(pc_g, 50), pc_p95=pct(pc_g, 95),
               cm_p50=pct(cm_g, 50), cm_p95=pct(cm_g, 95),
               ideal_max_rel=err0)

    # ---- [L] LayerNorm ---------------------------------------------------
    xl = MC.ln_inputs()
    c0 = xl - np.mean(xl)
    ref_l = c0 / np.sqrt(np.mean(c0 ** 2))
    rng = np.random.default_rng(args.seed + 1)
    draws_l = [MC.ln_draw(rng, len(xl)) for _ in range(args.mc_runs)]

    zero_l = {k: (np.zeros_like(v) if np.ndim(v) else 0.0)
              for k, v in draws_l[0].items()}
    _, yl0 = ln_one((999999, xl, zero_l, args.bisect))
    a0, p0 = MC.decompose(yl0, ref_l)
    print(f"[L] ideal-device control: common-mode {a0*100:.4f} %, "
          f"per-channel {p0*100:.4f} %  (bisection floor)")

    jobs = [(j, xl, d, args.bisect) for j, d in enumerate(draws_l)]
    with mp.Pool(args.workers) as pool:
        res = pool.map(ln_one, jobs, chunksize=2)
    cm_l, pc_l = [], []
    for _, y in sorted(res):
        a, p = MC.decompose(y, ref_l)
        cm_l.append(a)
        pc_l.append(p)
    lay = dict(pc_p50=pct(pc_l, 50), pc_p95=pct(pc_l, 95),
               cm_p50=pct(cm_l, 50), cm_p95=pct(cm_l, 95),
               ideal_cm=a0 * 100, ideal_pc=p0 * 100)

    # ---- verdicts --------------------------------------------------------
    print("\n" + "-" * W)
    print(f"  {'circuit':<12} {'quantity':<20} {'predicted':>10} "
          f"{'SPICE':>10} {'ratio':>7}  verdict")
    rows = []
    for name, pr, sp in (("GELU", pred_g, gel), ("LayerNorm", pred_l, lay)):
        for q, lbl in (("pc_p50", "per-channel p50"),
                       ("pc_p95", "per-channel p95"),
                       ("cm_p50", "common-mode p50"),
                       ("cm_p95", "common-mode p95")):
            r = pr[q] / sp[q] if sp[q] else float("nan")
            ok = 0.5 <= r <= 2.0
            v = "ok" if ok else "MISS"
            if q == "pc_p50":
                v = "HIT (kill-gate)" if ok else "FAIL (kill-gate)"
            print(f"  {name:<12} {lbl:<20} {pr[q]:>10.3f} {sp[q]:>10.3f} "
                  f"{r:>7.2f}  {v}")
            rows.append(dict(circuit=name, quantity=lbl, predicted=pr[q],
                             spice=sp[q], ratio=r, ok=bool(ok)))

    out = dict(prediction=dict(gelu=pred_g, layernorm=pred_l),
               spice=dict(gelu=gel, layernorm=lay), rows=rows,
               config=dict(mc_runs=args.mc_runs, bisect=args.bisect,
                           seed=args.seed, VT=VT, N=8))
    p = WORKDIR / "predict_gelu_layernorm.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n  raw: {p}")


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
