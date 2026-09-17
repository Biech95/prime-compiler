#!/usr/bin/env python3
"""
Kill-gate 3 of docs/missing_primes_mapping.md (§8), with the feedback-alignment
relaxation of §10 (X4 row).

QUESTION: does a gradient measured on the mismatched physical circuit absorb the
per-channel VGA mismatch that the AGC loop cannot remove (Table 7 of the paper:
per-channel residual ~0.7*sigma_VGA), and does it still hold when the backward
path is NOT the exact adjoint?

Pre-registration, model, metric and kill criteria: docs/exp_mismatch_absorption.md
(written before this script was run).

Conditions (identical mismatch draws / data / seeds):
  A  train-in-simulation, deploy-on-circuit   (ideal fwd, ideal bwd)
  B  physics-aware training (Wright 2022)     (mismatched fwd, ideal bwd)
  C  feedback alignment (Lillicrap 2016)      (mismatched fwd, random B + mismatched
                                               backward gain)
  D  perturbative / extremum seeking          (mismatched fwd, no backward)
  E  C and D with 0.5% multiplicative output noise

Forward surrogate reproduces the SPICE-validated structure exactly:
    y = G*x*(1+eps),  G = GAMMA / RMS(x*(1+eps))
with detector mismatch set to 0 (detector mismatch moves only the common mode;
see [A] of agc_mismatch_sweep.py). The trained gamma is verified on the REAL
junction-exact loop via agc_mismatch_sweep.settle_gain().

Usage:
    OPENBLAS_NUM_THREADS=1 python3 train_mismatch_absorption.py \
        --mc-runs 20 --spice-draws 5 --workers 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agc_mismatch_sweep as H  # noqa: E402  (settle_gain / detector_ms / decompose_error)

GAMMA = 1.0
M_OUT = 4
N_TRAIN = 2000
N_TEST = 500

# optimisation hyper-parameters (fixed before the production run)
# Plain SGD with a cosine-decayed learning rate for every condition -- the same
# optimiser everywhere, so the A/B/C/D comparison is not an optimiser comparison.
# (Adam was tried and is markedly WORSE for the perturbative estimator, whose
# per-coordinate variance the Adam normaliser amplifies; using it would have
# handicapped condition D.)
BATCH_GRAD = 64
LR_GRAD0, LR_GRAD1 = 2.0, 1e-3
BATCH_PERT = 16
LR_PERT0, LR_PERT1 = 0.5, 1e-3
SIGMA_P = 0.02           # Rademacher perturbation amplitude
NOISE_E = 0.005          # condition E multiplicative output noise
GCLIP = (1e-3, 20.0)     # gamma box; hitting it is recorded as divergence

# Budget ladders. Each budget is an independent run with its own cosine
# schedule, so "cost to reach a quality target" is not a schedule artefact.
LADDER_GRAD = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]     # x BATCH_GRAD
LADDER_PERT = [6250, 12500, 25000, 50000, 100000, 200000,
               400000, 800000]                                     # x 2 x BATCH_PERT
STEPS_GRAD = LADDER_GRAD[-1]
STEPS_PERT = LADDER_PERT[-1]


def cos_lr(lr0, lr1, t, steps):
    return lr1 + (lr0 - lr1) * 0.5 * (1.0 + np.cos(np.pi * t / steps))

CELLS = [(8, 0.01), (8, 0.02), (32, 0.01), (32, 0.02)]

DET_MOS = (0.03, 0.003, 0.01)   # MOS-typical detector corner (IS, n, mirror)


# ---------------------------------------------------------------------------
# surrogate forward model
# ---------------------------------------------------------------------------

def make_inputs(rng, N, n):
    """Same input distribution/clipping as agc_mismatch_sweep.run_cell."""
    X = rng.standard_normal((n, N)) * 1.5
    X = np.where(np.abs(X) < 0.05, 0.05 * np.sign(X) + (X == 0) * 0.05, X)
    return X


def norm_ideal(X):
    """y0 = x / RMS(x)  (ideal AGC, gamma applied separately)."""
    return X / np.sqrt(np.mean(X ** 2, axis=1, keepdims=True))


def norm_mis(X, eps):
    """y = G*x*(1+eps) with G = GAMMA/RMS(x*(1+eps)) -- the SPICE-validated loop."""
    Xe = X * (1.0 + eps)
    return GAMMA * Xe / np.sqrt(np.mean(Xe ** 2, axis=1, keepdims=True))


def decompose_batch(Z, Yref):
    """decompose_error() of the harness, vectorised over a batch of samples."""
    alpha = np.sum(Z * Yref, axis=1) / np.sum(Yref * Yref, axis=1)
    resid = Z / alpha[:, None] - Yref
    return np.abs(alpha - 1.0), np.sqrt(np.mean(resid ** 2, axis=1))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def evaluate(gamma, W, Yte_mis, Yref_te, T_te):
    """Per-channel residual / common mode / task MSE on the MISMATCHED forward."""
    Z = Yte_mis * gamma
    cm, pc = decompose_batch(Z, Yref_te)
    mse = float(np.mean((Z @ W.T - T_te) ** 2))
    return float(np.median(cm)), float(np.median(pc)), mse


def train_gradient(cond, gamma0, W0, Ytr_id, Ytr_mis, T_tr, W_learnable,
                   Bmat, delta, rng, noise=0.0, steps=STEPS_GRAD,
                   lr0=LR_GRAD0, lr1=LR_GRAD1, batch=BATCH_GRAD):
    """Conditions A / B / C / C2 -- plain SGD, cosine-decayed lr.

    cond "A"          ideal forward, exact ideal backward
    cond "B"          mismatched forward, exact backward of the IDEAL model
    cond "C" / "C2"   mismatched forward, backward through the fixed matrix
                      `Bmat` in place of W^T plus a mismatched backward gain
                      (1+delta); local activation is the PHYSICAL one.
    Returns (gamma, W, evals, diverged).
    """
    gamma = gamma0.copy()
    W = W0.copy()
    n = Ytr_id.shape[0]
    evals = 0
    hit_box = False
    for t in range(steps):
        lr = cos_lr(lr0, lr1, t, steps)
        idx = rng.integers(0, n, batch)
        y0 = Ytr_id[idx]
        ym = Ytr_mis[idx]
        tt = T_tr[idx]

        yf = y0 if cond == "A" else ym
        Z = yf * gamma
        if noise:
            Z = Z * (1.0 + rng.normal(0.0, noise, Z.shape))
        E = Z @ W.T - tt                  # (batch, M)
        evals += batch

        gE = (2.0 / M_OUT) * E
        if cond in ("A", "B"):
            back = (gE @ W) * y0          # exact adjoint of the ideal model
        else:
            back = (gE @ Bmat) * ym * (1.0 + delta)
        gnew = gamma - lr * back.mean(axis=0)
        if not np.all(np.isfinite(gnew)):
            gnew = np.nan_to_num(gnew, nan=GCLIP[1], posinf=GCLIP[1],
                                 neginf=GCLIP[0])
            hit_box = True
        gamma = np.clip(gnew, *GCLIP)
        if np.any(gamma <= GCLIP[0] * 1.001) or np.any(gamma >= GCLIP[1] * 0.999):
            hit_box = True

        if W_learnable:
            # the output-layer weight gradient is exact in every condition
            W = W - lr * (gE.T @ Z) / batch
            if not np.all(np.isfinite(W)):
                W = np.nan_to_num(W)
                hit_box = True
    return gamma, W, evals, hit_box


def train_perturbative(gamma0, W, Ytr_mis, T_tr, rng, noise=0.0,
                       steps=STEPS_PERT, lr0=LR_PERT0, lr1=LR_PERT1,
                       batch=BATCH_PERT, sigma_p=SIGMA_P):
    """Condition D: no backward at all. Two-sided Rademacher perturbation of
    gamma evaluated on the mismatched forward (Cauwenberghs 1992 / extremum
    seeking); the two probes share the minibatch (common random numbers)."""
    gamma = gamma0.copy()
    n = Ytr_mis.shape[0]
    N = gamma.shape[0]
    evals = 0
    hit_box = False
    for t in range(steps):
        lr = cos_lr(lr0, lr1, t, steps)
        idx = rng.integers(0, n, batch)
        ym = Ytr_mis[idx]
        tt = T_tr[idx]
        s = rng.integers(0, 2, N) * 2.0 - 1.0
        dp = sigma_p * s

        Zp = ym * (gamma + dp)
        Zm = ym * (gamma - dp)
        if noise:
            Zp = Zp * (1.0 + rng.normal(0.0, noise, Zp.shape))
            Zm = Zm * (1.0 + rng.normal(0.0, noise, Zm.shape))
        Lp = np.mean((Zp @ W.T - tt) ** 2)
        Lm = np.mean((Zm @ W.T - tt) ** 2)
        evals += 2 * batch

        gamma = np.clip(gamma - lr * (Lp - Lm) / (2.0 * sigma_p) * s, *GCLIP)
        if np.any(gamma <= GCLIP[0] * 1.001) or np.any(gamma >= GCLIP[1] * 0.999):
            hit_box = True
    return gamma, evals, hit_box


def ls_optimum(Ytr_mis, T_tr, W):
    """Exact minimiser of the training loss over gamma for fixed W.

    The loss is quadratic in gamma: o = W (gamma * y). With J_k = W * y_k
    (row-scaled), gamma* solves (sum_k J_k^T J_k) gamma = sum_k J_k^T t_k.
    This is the attainable floor of conditions B/C2/D on this training set and
    makes the forward-evaluation comparison independent of the lr schedule.
    """
    N = Ytr_mis.shape[1]
    A = np.zeros((N, N))
    b = np.zeros(N)
    for k in range(Ytr_mis.shape[0]):
        J = W * Ytr_mis[k]
        A += J.T @ J
        b += J.T @ T_tr[k]
    return np.linalg.solve(A, b)


def smallest_budget(ladder_results, target):
    """Smallest budget (in forward evaluations) whose FINAL residual <= target."""
    for evals, pc in ladder_results:
        if pc <= target:
            return evals
    return None


# ---------------------------------------------------------------------------
# one Monte-Carlo draw = one worker task
# ---------------------------------------------------------------------------

def run_draw(task):
    cell_idx, draw, N, sigma, master_seed = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(cell_idx, draw))
    rng = np.random.default_rng(ss)

    # --- shared across ALL conditions -------------------------------------
    eps = rng.normal(0.0, sigma, N)                    # VGA per-channel mismatch
    Xtr = make_inputs(rng, N, N_TRAIN)
    Xte = make_inputs(rng, N, N_TEST)
    W0 = rng.standard_normal((M_OUT, N)) / np.sqrt(N)
    Bfa = rng.standard_normal((M_OUT, N)) / np.sqrt(N)  # fixed random backward
    delta = rng.normal(0.0, 0.05, N)                    # mismatched backward gain
    # sign-concordant random backward (Liao et al. 2016 / Xiao et al. 2018
    # "sign symmetry"): random magnitudes, signs taken from the forward matrix
    Bsc = np.abs(rng.standard_normal((M_OUT, N))) / np.sqrt(N) * np.sign(W0)

    Ytr_id, Yte_id = norm_ideal(Xtr), norm_ideal(Xte)
    Ytr_mis, Yte_mis = norm_mis(Xtr, eps), norm_mis(Xte, eps)
    T_tr = Ytr_id @ W0.T
    T_te = Yte_id @ W0.T
    Yref_te = Yte_id
    g1 = np.ones(N)

    out = {"cell": cell_idx, "draw": draw, "N": N, "sigma": sigma}
    cm, pc, mse = evaluate(g1, W0, Yte_mis, Yref_te, T_te)
    out["untrained"] = {"cm": cm, "pc": pc, "mse": mse}
    # diagnostic: fraction of channels whose feedback-alignment loop gain
    # diag(B^T W) is positive, i.e. whose gamma update has the correct sign
    out["fa_diag_sign"] = float(np.mean(np.sum(Bfa * W0, axis=0) > 0))
    out["sc_diag_sign"] = float(np.mean(np.sum(Bsc * W0, axis=0) > 0))

    res = {}

    def grad_run(cond, wlearn, nz, Bmat, steps, seed):
        r = np.random.default_rng(seed)
        gam, W, ev, div = train_gradient(cond, g1, W0, Ytr_id, Ytr_mis, T_tr,
                                         wlearn, Bmat, delta, r, noise=nz,
                                         steps=steps)
        c, p, m = evaluate(gam, W, Yte_mis, Yref_te, T_te)
        return {"cm": c, "pc": p, "mse": m, "evals": ev,
                "diverged": bool(div or p > out["untrained"]["pc"]),
                "gamma": gam.tolist()}

    # --- ladder conditions (cost comparison): B, C, C2, D -----------------
    ladders = {}
    seeds = ss.spawn(24)
    si = 0
    for key, cond, Bmat in (("B", "B", W0), ("C", "C", Bfa), ("C2", "C2", Bsc)):
        lad = []
        for st in LADDER_GRAD:
            r = grad_run(cond, False, 0.0, Bmat, st, seeds[si])
            lad.append((r["evals"], r["pc"]))
            res[key] = r          # last one = largest budget
        ladders[key] = lad
        si += 1
    lad = []
    for st in LADDER_PERT:
        r = np.random.default_rng(seeds[si])
        gam, ev, div = train_perturbative(g1, W0, Ytr_mis, T_tr, r, steps=st)
        c, p, m = evaluate(gam, W0, Yte_mis, Yref_te, T_te)
        rd = {"cm": c, "pc": p, "mse": m, "evals": ev,
              "diverged": bool(div or p > out["untrained"]["pc"]),
              "gamma": gam.tolist()}
        lad.append((ev, p))
        res["D"] = rd
    ladders["D"] = lad
    si += 1

    # --- single-budget conditions -----------------------------------------
    single = [("A", "A", False, 0.0, W0), ("E_C", "C", False, NOISE_E, Bfa),
              ("E_C2", "C2", False, NOISE_E, Bsc),
              ("A_wlearn", "A", True, 0.0, W0), ("B_wlearn", "B", True, 0.0, W0),
              ("C_wlearn", "C", True, 0.0, Bfa),
              ("C2_wlearn", "C2", True, 0.0, Bsc)]
    for key, cond, wl, nz, Bmat in single:
        res[key] = grad_run(cond, wl, nz, Bmat, STEPS_GRAD, seeds[si])
        si += 1
    r = np.random.default_rng(seeds[si]); si += 1
    gam, ev, div = train_perturbative(g1, W0, Ytr_mis, T_tr, r, noise=NOISE_E,
                                      steps=STEPS_PERT)
    c, p, m = evaluate(gam, W0, Yte_mis, Yref_te, T_te)
    res["E_D"] = {"cm": c, "pc": p, "mse": m, "evals": ev,
                  "diverged": bool(div or p > out["untrained"]["pc"]),
                  "gamma": gam.tolist()}

    # --- analytic attainable floor (exact least squares on the training set)
    g_ls = ls_optimum(Ytr_mis, T_tr, W0)
    c, p, m = evaluate(g_ls, W0, Yte_mis, Yref_te, T_te)
    res["LS"] = {"cm": c, "pc": p, "mse": m, "evals": 0, "diverged": False,
                 "gamma": g_ls.tolist()}

    # --- cost: smallest budget reaching 1.2 x B's final residual (K4) ------
    tgt = 1.2 * res["B"]["pc"]
    res["target_pc"] = tgt
    res["cost"] = {k: smallest_budget(ladders[k], tgt) for k in ladders}
    out["ladders"] = ladders
    out["res"] = res
    return out


# ---------------------------------------------------------------------------
# SPICE verification on the real junction-exact loop
# ---------------------------------------------------------------------------

def spice_check(task):
    (cell_idx, draw, N, sigma, master_seed, gamma_tr, det_corner, n_inputs,
     tag) = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(cell_idx, draw))
    rng = np.random.default_rng(ss)
    eps = rng.normal(0.0, sigma, N)
    Xtr = make_inputs(rng, N, N_TRAIN)      # keep the stream aligned with run_draw
    Xte = make_inputs(rng, N, N_TEST)
    del Xtr

    gamma_tr = np.asarray(gamma_tr)
    if det_corner is None:
        is_m = np.ones(2 * N + 1)
        n_m = np.ones(2 * N + 1)
        mir = 1.0
    else:
        s_is, s_n, s_mir = det_corner
        drng = np.random.default_rng(
            np.random.SeedSequence(entropy=master_seed + 7,
                                   spawn_key=(cell_idx, draw)))
        is_m = drng.lognormal(0, s_is, 2 * N + 1)
        n_m = drng.normal(1.0, s_n, 2 * N + 1)
        mir = float(drng.normal(1.0, s_mir))

    rows = []
    for j in range(n_inputs):
        x = Xte[j]
        x_eff = x * (1.0 + eps)
        y_ref = GAMMA * x / np.sqrt(np.mean(x ** 2))
        G = H.settle_gain(x_eff, is_m, n_m, mir, f"{tag}_{j}")
        y_spice = G * x_eff
        G_sur = GAMMA / np.sqrt(np.mean(x_eff ** 2))
        y_sur = G_sur * x_eff
        r = {}
        for name, g in (("before", np.ones(N)), ("after", gamma_tr)):
            cm_s, pc_s = H.decompose_error(y_spice * g, y_ref)
            cm_u, pc_u = H.decompose_error(y_sur * g, y_ref)
            # scale-SENSITIVE error (no common-mode removal): this is the only
            # quantity in which the SPICE loop gain can differ from the surrogate
            tot_s = float(np.sqrt(np.mean((y_spice * g - y_ref) ** 2)))
            tot_u = float(np.sqrt(np.mean((y_sur * g - y_ref) ** 2)))
            r[name] = {"spice_cm": cm_s, "spice_pc": pc_s, "spice_tot": tot_s,
                       "sur_cm": cm_u, "sur_pc": pc_u, "sur_tot": tot_u}
        r["G_spice"] = G
        r["G_sur"] = G_sur
        r["G_rel_dev"] = abs(G - G_sur) / G_sur
        rows.append(r)
    return {"cell": cell_idx, "draw": draw, "det": "mos" if det_corner else "ideal",
            "rows": rows}


# ---------------------------------------------------------------------------

def pct(v):
    return 100.0 * v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--spice-draws", type=int, default=5)
    ap.add_argument("--spice-inputs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--out", type=str, default=str(HERE / "out" /
                                                   "mismatch_absorption.json"))
    args = ap.parse_args()

    t0 = time.time()
    tasks = [(ci, d, N, s, args.seed)
             for ci, (N, s) in enumerate(CELLS)
             for d in range(args.mc_runs)]
    print(f"[1/2] training: {len(tasks)} (cell, draw) tasks on {args.workers} workers")
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_draw, tasks, chunksize=1)
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- aggregate --------------------------------------------------------
    conds = ["LS", "A", "B", "C", "C2", "D", "E_C", "E_C2", "E_D",
             "A_wlearn", "B_wlearn", "C_wlearn", "C2_wlearn"]
    agg = {}
    for ci, (N, s) in enumerate(CELLS):
        rs = [r for r in results if r["cell"] == ci]
        cellkey = f"N{N}_s{s}"
        a = {"N": N, "sigma": s, "untrained": {
            "pc50": float(np.median([r["untrained"]["pc"] for r in rs])),
            "pc95": float(np.percentile([r["untrained"]["pc"] for r in rs], 95)),
            "cm50": float(np.median([r["untrained"]["cm"] for r in rs])),
            "mse": float(np.median([r["untrained"]["mse"] for r in rs]))}}
        for c in conds:
            pcs = [r["res"][c]["pc"] for r in rs]
            cms = [r["res"][c]["cm"] for r in rs]
            mses = [r["res"][c]["mse"] for r in rs]
            a[c] = {"pc50": float(np.median(pcs)),
                    "pc95": float(np.percentile(pcs, 95)),
                    "cm50": float(np.median(cms)),
                    "mse": float(np.median(mses)),
                    "evals": rs[0]["res"][c]["evals"],
                    "diverged": int(sum(r["res"][c]["diverged"] for r in rs))}
        a["fa_diag_sign"] = float(np.mean([r["fa_diag_sign"] for r in rs]))
        a["sc_diag_sign"] = float(np.mean([r["sc_diag_sign"] for r in rs]))
        cost = {}
        for k in ("B", "C", "C2", "D"):
            v = [r["res"]["cost"][k] for r in rs]
            cost[k] = float(np.median([x for x in v if x])) if any(v) else None
            cost[k + "_hits"] = int(sum(x is not None for x in v))
        rat = [r["res"]["cost"]["D"] / r["res"]["cost"]["B"] for r in rs
               if r["res"]["cost"]["D"] and r["res"]["cost"]["B"]]
        cost["D_over_B_p50"] = float(np.median(rat)) if rat else None
        cost["D_over_B_p95"] = float(np.percentile(rat, 95)) if rat else None
        cost["n"] = len(rs)
        a["cost"] = cost
        agg[cellkey] = a

    # ---- print tables -----------------------------------------------------
    print("\n" + "=" * 100)
    print("  per-channel residual p50/p95 [% of signal], common-mode p50 [%], "
          "task MSE (median)")
    print("=" * 100)
    for k, a in agg.items():
        print(f"\n  {k}  (sigma_VGA = {a['sigma']*100:.0f}%, N = {a['N']}), "
              f"0.7*sigma = {a['sigma']*70:.2f}%")
        print(f"    {'cond':<12} {'pc p50':>9} {'pc p95':>9} {'pc/sigma':>9} "
              f"{'cm p50':>9} {'task MSE':>11} {'fwd evals':>11} {'div':>5}")
        u = a["untrained"]
        print(f"    {'untrained':<12} {pct(u['pc50']):9.4f} {pct(u['pc95']):9.4f} "
              f"{u['pc50']/a['sigma']:9.3f} {pct(u['cm50']):9.4f} "
              f"{u['mse']:11.3e} {'-':>11} {'-':>5}")
        for c in conds:
            r = a[c]
            print(f"    {c:<12} {pct(r['pc50']):9.4f} {pct(r['pc95']):9.4f} "
                  f"{r['pc50']/a['sigma']:9.3f} {pct(r['cm50']):9.4f} "
                  f"{r['mse']:11.3e} {r['evals']:11d} {r['diverged']:5d}")
        e = a["cost"]
        print(f"    smallest budget [fwd evals] reaching 1.2x B_final:  "
              f"B={e['B']} ({e['B_hits']}/{e['n']})  D={e['D']} "
              f"({e['D_hits']}/{e['n']})  C={e['C']} ({e['C_hits']}/{e['n']})  "
              f"C2={e['C2']} ({e['C2_hits']}/{e['n']})")
        print(f"    D/B budget ratio p50={e['D_over_B_p50']} "
              f"p95={e['D_over_B_p95']}   "
              f"backward loop-gain sign>0: random B={a['fa_diag_sign']:.3f}, "
              f"sign-concordant B={a['sc_diag_sign']:.3f}")

    # ---- SPICE verification ----------------------------------------------
    print("\n[2/2] SPICE verification on the junction-exact loop")
    sp_tasks = []
    for ci, (N, s) in enumerate(CELLS):
        for d in range(args.spice_draws):
            r = next(x for x in results if x["cell"] == ci and x["draw"] == d)
            gam = r["res"]["B"]["gamma"]
            for det, dname in ((None, "id"), (DET_MOS, "mos")):
                sp_tasks.append((ci, d, N, s, args.seed, gam, det,
                                 args.spice_inputs, f"ma{ci}_{d}_{dname}"))
    t1 = time.time()
    with mp.Pool(min(args.workers, len(sp_tasks))) as pool:
        sp = pool.map(spice_check, sp_tasks, chunksize=1)
    print(f"      {len(sp_tasks)} SPICE cells in {time.time()-t1:.1f}s")

    print("\n" + "=" * 100)
    print("  SPICE (junction-exact loop) vs surrogate, trained gamma from "
          "condition B")
    print("=" * 100)
    print(f"  {'cell':<13} {'det':<6} {'pc SPICE bef':>13} {'pc SPICE aft':>13} "
          f"{'pc surr aft':>12} {'dev':>7} {'tot SPICE bef':>14} "
          f"{'tot SPICE aft':>14} {'G dev':>8}")
    spice_summary = {}
    max_dev = 0.0
    for ci, (N, s_v) in enumerate(CELLS):
        for dname in ("ideal", "mos"):
            rows = [rr for x in sp if x["cell"] == ci and x["det"] == dname
                    for rr in x["rows"]]
            if not rows:
                continue
            med = lambda f: float(np.median([f(r) for r in rows]))  # noqa: E731
            pb = med(lambda r: r["before"]["spice_pc"])
            pa = med(lambda r: r["after"]["spice_pc"])
            ub = med(lambda r: r["before"]["sur_pc"])
            ua = med(lambda r: r["after"]["sur_pc"])
            tb = med(lambda r: r["before"]["spice_tot"])
            ta = med(lambda r: r["after"]["spice_tot"])
            tub = med(lambda r: r["before"]["sur_tot"])
            tua = med(lambda r: r["after"]["sur_tot"])
            ca = med(lambda r: r["after"]["spice_cm"])
            gd = med(lambda r: r["G_rel_dev"])
            dev = abs(pa - ua) / max(ua, 1e-12)
            devb = abs(pb - ub) / max(ub, 1e-12)
            max_dev = max(max_dev, dev, devb)
            spice_summary[f"N{N}_s{s_v}_{dname}"] = {
                "pc_before_spice": pb, "pc_after_spice": pa,
                "pc_before_sur": ub, "pc_after_sur": ua,
                "tot_before_spice": tb, "tot_after_spice": ta,
                "tot_before_sur": tub, "tot_after_sur": tua,
                "cm_after_spice": ca, "G_rel_dev": gd,
                "rel_dev_after": dev, "rel_dev_before": devb,
                "n_rows": len(rows)}
            print(f"  N={N},s={s_v*100:.0f}%{'':<4} {dname:<6} "
                  f"{pct(pb):13.4f} {pct(pa):13.4f} {pct(ua):12.4f} "
                  f"{dev*100:6.2f}% {pct(tb):14.4f} {pct(ta):14.4f} "
                  f"{gd*100:7.4f}%")
    if max_dev > 0.20:
        print("\n  *** WARNING: SPICE and surrogate disagree by "
              f"{max_dev*100:.1f}% relative (> 20%) on the per-channel "
              "residual. ***")
    else:
        print(f"\n  SPICE/surrogate max relative deviation on per-channel "
              f"residual: {max_dev*100:.3f}% (< 20% threshold)")
    print("  NOTE: the per-channel residual is invariant under any scalar loop "
          "gain, so\n        SPICE vs surrogate can only differ there through "
          "the eps path; the\n        scale-sensitive check is the 'tot' "
          "columns (no common-mode removal).")

    payload = {"agg": agg, "spice": spice_summary, "max_spice_dev": max_dev,
               "args": vars(args),
               "hyper": {"steps_grad": STEPS_GRAD, "batch_grad": BATCH_GRAD,
                         "lr_grad": [LR_GRAD0, LR_GRAD1],
                         "steps_pert": STEPS_PERT,
                         "batch_pert": BATCH_PERT, "lr_pert": [LR_PERT0, LR_PERT1],
                         "sigma_p": SIGMA_P, "noise_E": NOISE_E,
                         "n_train": N_TRAIN, "n_test": N_TEST, "M": M_OUT}}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  wrote {args.out}   total wall clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
