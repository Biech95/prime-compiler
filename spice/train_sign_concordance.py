#!/usr/bin/env python3
"""
Pre-registered re-run of the exploratory condition C2 of
docs/exp_mismatch_absorption.md (sign concordance of the backward path), with
fresh seeds and a sharper condition set that locates the boundary of what the
backward path needs.

Pre-registration: docs/exp_sign_concordance.md section 1 (written before this
script was run).

Conditions (frozen W, identical mismatch draw / data / optimiser):
  S0      exact adjoint of the ideal model (= condition B, the anchor)
  S1      random magnitudes + signs of W, backward gain mismatch 5%  (= C2)
  S2      sign-concordant, log-uniform magnitudes over one decade (0.1-1x|W|),
          backward gain mismatch 20%
  S3(f)   S1 with a fraction f of the backward entries sign-flipped
          (nested flip sets), f in {0, .05, .10, .20, .35, .50}; S3(0) == S1
  S4      FULLY random backward path, but the trainable stage is a dense NxN
          matrix Gamma initialised to I (does Lillicrap-style FA come back once
          there is a matrix that can rotate?)
  S4ref   positive control: dense stage with the exact adjoint

The forward model, data, metric and optimiser are imported unchanged from
train_mismatch_absorption.py; only the backward path and the parametrisation of
the trainable stage differ.

Usage:
    OPENBLAS_NUM_THREADS=1 python3 train_sign_concordance.py \
        --mc-runs 20 --seed 3031 --workers 32
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
import train_mismatch_absorption as T  # noqa: E402  (forward model, optimiser, metric)

M_OUT = T.M_OUT
CELLS = T.CELLS
STEPS = T.STEPS_GRAD            # 32000
BATCH = T.BATCH_GRAD            # 64
LR0, LR1 = T.LR_GRAD0, T.LR_GRAD1   # 2.0 -> 1e-3
LR0_ALT, LR1_ALT = 0.2, 1e-4        # secondary schedule for the dense stage
GBOX = T.GCLIP                  # (1e-3, 20) for the diagonal gain
DBOX = (-20.0, 20.0)            # dense stage: off-diagonals may be negative

F_LIST = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def hidden_metric(Z, Yref):
    """Per-channel residual on the hidden signal (identical to the previous
    experiment): common mode removed by a per-sample least-squares scalar."""
    cm, pc = T.decompose_batch(Z, Yref)
    return float(np.median(cm)), float(np.median(pc))


def out_metric(O, Tt):
    """Output residual against the ideal network, common mode removed, and
    normalised by RMS(t) so it is dimensionless.  This is the identifiable
    observable -- a dense Gamma is not identifiable from the loss, the output
    error is."""
    a = np.sum(O * Tt, axis=1) / np.sum(Tt * Tt, axis=1)
    r = O / a[:, None] - Tt
    num = np.sqrt(np.mean(r ** 2, axis=1))
    den = np.sqrt(np.mean(Tt ** 2, axis=1))
    return float(np.median(num / den))


def eval_diag(gamma, W, Yte_mis, Yref_te, T_te):
    Z = Yte_mis * gamma
    cm, pc = hidden_metric(Z, Yref_te)
    O = Z @ W.T
    return {"cm": cm, "pc": pc, "pc_out": out_metric(O, T_te),
            "mse": float(np.mean((O - T_te) ** 2))}


def eval_dense(Gam, W, Yte_mis, Yref_te, T_te):
    Z = Yte_mis @ Gam.T
    cm, pc = hidden_metric(Z, Yref_te)
    O = Z @ W.T
    return {"cm": cm, "pc": pc, "pc_out": out_metric(O, T_te),
            "mse": float(np.mean((O - T_te) ** 2))}


# ---------------------------------------------------------------------------
# dense trainable stage (S4 / S4ref)
# ---------------------------------------------------------------------------

def train_dense(W, Ytr_mis, T_tr, Bmat, delta, rng, exact, steps=STEPS,
                lr0=LR0, lr1=LR1, batch=BATCH):
    """z = Gamma y with Gamma an NxN trainable matrix initialised to I; W frozen.

    exact=True   backward through W^T (positive control)
    exact=False  backward through the fixed random Bmat plus the per-channel
                 backward gain (1+delta) -- feedback alignment with a matrix
                 that can rotate.
    """
    N = Ytr_mis.shape[1]
    Gam = np.eye(N)
    n = Ytr_mis.shape[0]
    evals = 0
    hit_box = False
    for t in range(steps):
        lr = T.cos_lr(lr0, lr1, t, steps)
        idx = rng.integers(0, n, batch)
        ym = Ytr_mis[idx]
        tt = T_tr[idx]
        Z = ym @ Gam.T
        E = Z @ W.T - tt
        evals += batch
        gE = (2.0 / M_OUT) * E
        if exact:
            dz = gE @ W
        else:
            dz = (gE @ Bmat) * (1.0 + delta)
        Gnew = Gam - lr * (dz.T @ ym) / batch
        if not np.all(np.isfinite(Gnew)):
            Gnew = np.nan_to_num(Gnew, nan=DBOX[1], posinf=DBOX[1],
                                 neginf=DBOX[0])
            hit_box = True
        Gam = np.clip(Gnew, *DBOX)
        if np.any(np.abs(Gam) >= DBOX[1] * 0.999):
            hit_box = True
    return Gam, evals, hit_box


# ---------------------------------------------------------------------------
# one Monte-Carlo draw
# ---------------------------------------------------------------------------

def run_draw(task):
    cell_idx, draw, N, sigma, master_seed = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(cell_idx, draw))
    rng = np.random.default_rng(ss)

    # --- shared across ALL conditions (same construction order as the
    #     previous experiment, so the draw is structurally identical) --------
    eps = rng.normal(0.0, sigma, N)
    Xtr = T.make_inputs(rng, N, T.N_TRAIN)
    Xte = T.make_inputs(rng, N, T.N_TEST)
    W0 = rng.standard_normal((M_OUT, N)) / np.sqrt(N)
    Bfa = rng.standard_normal((M_OUT, N)) / np.sqrt(N)     # fully random backward
    delta = rng.normal(0.0, 0.05, N)                        # 5 % backward gain
    Bsc = np.abs(rng.standard_normal((M_OUT, N))) / np.sqrt(N) * np.sign(W0)

    # S2: sign-concordant, magnitudes log-uniform over one decade (0.1-1x|W|),
    #     backward gain mismatch 20 %
    U = rng.uniform(-1.0, 0.0, (M_OUT, N))
    Bdec = np.sign(W0) * np.abs(W0) * (10.0 ** U)
    delta20 = rng.normal(0.0, 0.20, N)

    # S3: one nested permutation of the M*N backward entries
    perm = rng.permutation(M_OUT * N)

    Ytr_id, Yte_id = T.norm_ideal(Xtr), T.norm_ideal(Xte)
    Ytr_mis, Yte_mis = T.norm_mis(Xtr, eps), T.norm_mis(Xte, eps)
    T_tr = Ytr_id @ W0.T
    T_te = Yte_id @ W0.T
    Yref_te = Yte_id
    g1 = np.ones(N)

    out = {"cell": cell_idx, "draw": draw, "N": N, "sigma": sigma}
    out["untrained"] = eval_diag(g1, W0, Yte_mis, Yref_te, T_te)

    # diagnostics -----------------------------------------------------------
    out["diag_sign_random"] = float(np.mean(np.sum(Bfa * W0, axis=0) > 0))
    out["diag_sign_sc"] = float(np.mean(np.sum(Bsc * W0, axis=0) > 0))
    out["diag_sign_dec"] = float(np.mean(np.sum(Bdec * W0, axis=0) > 0))
    ev = np.linalg.eigvals(W0 @ Bfa.T)          # governs the dense-stage dynamics
    out["WBt_all_pos"] = bool(np.all(ev.real > 0))
    out["WBt_min_re"] = float(np.min(ev.real))
    ev2 = np.linalg.eigvals(W0 @ W0.T)
    out["WWt_min_re"] = float(np.min(ev2.real))

    res = {}
    seeds = ss.spawn(32)
    si = 0

    def diag_run(cond, Bmat, dl):
        nonlocal si
        r = np.random.default_rng(seeds[si]); si += 1
        gam, _, ev_n, box = T.train_gradient(cond, g1, W0, Ytr_id, Ytr_mis, T_tr,
                                             False, Bmat, dl, r, noise=0.0,
                                             steps=STEPS)
        m = eval_diag(gam, W0, Yte_mis, Yref_te, T_te)
        m["evals"] = ev_n
        m["diverged"] = bool(box or m["pc"] > out["untrained"]["pc"])
        m["gamma"] = gam.tolist()
        return m

    def dense_run(exact, Bmat, dl, lr0, lr1):
        nonlocal si
        r = np.random.default_rng(seeds[si]); si += 1
        Gam, ev_n, box = train_dense(W0, Ytr_mis, T_tr, Bmat, dl, r, exact,
                                     steps=STEPS, lr0=lr0, lr1=lr1)
        m = eval_dense(Gam, W0, Yte_mis, Yref_te, T_te)
        m["evals"] = ev_n
        m["diverged"] = bool(box or m["pc_out"] > out["untrained"]["pc_out"])
        return m

    # --- S0 anchor, S1, S2 -------------------------------------------------
    res["S0"] = diag_run("B", W0, delta)
    res["S1"] = diag_run("C2", Bsc, delta)
    res["S2"] = diag_run("C2", Bdec, delta20)

    # --- S3 sign-flip sweep (nested flip sets; f=0 is S1 by construction) ---
    for f in F_LIST:
        key = f"S3_{int(round(f*100)):02d}"
        if f == 0.0:
            res[key] = res["S1"]
            res[key + "_nflip"] = 0
            continue
        k = int(round(f * M_OUT * N))
        flip = np.ones(M_OUT * N)
        flip[perm[:k]] = -1.0
        Bf = Bsc * flip.reshape(M_OUT, N)
        res[key] = diag_run("C2", Bf, delta)
        res[key + "_nflip"] = k
        res[key + "_diag_sign"] = float(np.mean(np.sum(Bf * W0, axis=0) > 0))

    # --- S4 dense stage ----------------------------------------------------
    res["S4"] = dense_run(False, Bfa, delta, LR0, LR1)
    res["S4_alt"] = dense_run(False, Bfa, delta, LR0_ALT, LR1_ALT)
    res["S4ref"] = dense_run(True, W0, delta, LR0, LR1)
    res["S4ref_alt"] = dense_run(True, W0, delta, LR0_ALT, LR1_ALT)

    # --- analytic floor (diagonal stage, exact least squares) --------------
    g_ls = T.ls_optimum(Ytr_mis, T_tr, W0)
    m = eval_diag(g_ls, W0, Yte_mis, Yref_te, T_te)
    m["evals"] = 0
    m["diverged"] = False
    res["LS"] = m

    # --- per-draw convergence flags relative to the S0 anchor --------------
    p0 = res["S0"]["pc"]
    o0 = res["S0"]["pc_out"]
    conv = {}
    for k in ["S1", "S2"] + [f"S3_{int(round(f*100)):02d}" for f in F_LIST]:
        conv[k] = bool((not res[k]["diverged"]) and res[k]["pc"] <= 2.0 * p0)
    for k in ("S4", "S4_alt", "S4ref", "S4ref_alt"):
        conv[k] = bool((not res[k]["diverged"]) and res[k]["pc_out"] <= 2.0 * o0)
    conv["S4_best"] = bool(conv["S4"] or conv["S4_alt"])
    conv["S4ref_best"] = bool(conv["S4ref"] or conv["S4ref_alt"])
    res["converged"] = conv

    for k in list(res):
        if isinstance(res[k], dict) and "gamma" in res[k]:
            del res[k]["gamma"]
    out["res"] = res
    return out


# ---------------------------------------------------------------------------

def run_draw_flipk(task):
    """POST-HOC ADDENDUM (not pre-registered, run after the main grid and
    labelled as exploratory in the write-up): the S3 grid's smallest point,
    f = 5 %, already fails K3, so this probes the absolute floor -- exactly
    k = 1 and k = 2 flipped entries of the backward matrix.  Setup is byte-for-
    byte the setup of run_draw() so the mismatch draw, W, Bsc and the flip
    permutation are the same."""
    cell_idx, draw, N, sigma, master_seed, klist = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(cell_idx, draw))
    rng = np.random.default_rng(ss)
    eps = rng.normal(0.0, sigma, N)
    Xtr = T.make_inputs(rng, N, T.N_TRAIN)
    Xte = T.make_inputs(rng, N, T.N_TEST)
    W0 = rng.standard_normal((M_OUT, N)) / np.sqrt(N)
    _Bfa = rng.standard_normal((M_OUT, N)) / np.sqrt(N)
    delta = rng.normal(0.0, 0.05, N)
    Bsc = np.abs(rng.standard_normal((M_OUT, N))) / np.sqrt(N) * np.sign(W0)
    _U = rng.uniform(-1.0, 0.0, (M_OUT, N))
    _d20 = rng.normal(0.0, 0.20, N)
    perm = rng.permutation(M_OUT * N)

    Ytr_id, Yte_id = T.norm_ideal(Xtr), T.norm_ideal(Xte)
    Ytr_mis, Yte_mis = T.norm_mis(Xtr, eps), T.norm_mis(Xte, eps)
    T_tr, T_te = Ytr_id @ W0.T, Yte_id @ W0.T
    g1 = np.ones(N)
    unt = eval_diag(g1, W0, Yte_mis, Yte_id, T_te)

    seeds = ss.spawn(40)[32:]
    r0 = np.random.default_rng(seeds[0])
    gam, _, _, box = T.train_gradient("B", g1, W0, Ytr_id, Ytr_mis, T_tr, False,
                                      W0, delta, r0, steps=STEPS)
    s0 = eval_diag(gam, W0, Yte_mis, Yte_id, T_te)

    out = {"cell": cell_idx, "draw": draw, "S0_pc": s0["pc"], "res": {}}
    for j, k in enumerate(klist):
        flip = np.ones(M_OUT * N)
        flip[perm[:k]] = -1.0
        Bf = Bsc * flip.reshape(M_OUT, N)
        r = np.random.default_rng(seeds[1 + j])
        gam, _, _, box = T.train_gradient("C2", g1, W0, Ytr_id, Ytr_mis, T_tr,
                                          False, Bf, delta, r, steps=STEPS)
        m = eval_diag(gam, W0, Yte_mis, Yte_id, T_te)
        m["diverged"] = bool(box or m["pc"] > unt["pc"])
        m["all_signs_ok"] = bool(np.all(np.sum(Bf * W0, axis=0) > 0))
        m["converged"] = bool((not m["diverged"]) and m["pc"] <= 2.0 * s0["pc"])
        out["res"][f"k{k}"] = m
    return out


CONDS = (["LS", "S0", "S1", "S2"]
         + [f"S3_{int(round(f*100)):02d}" for f in F_LIST]
         + ["S4", "S4_alt", "S4ref", "S4ref_alt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3031)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--out", type=str,
                    default=str(HERE / "out" / "sign_concordance.json"))
    ap.add_argument("--addendum", action="store_true",
                    help="POST-HOC (not pre-registered): k=1 and k=2 flipped "
                         "backward entries instead of the pre-registered grid")
    args = ap.parse_args()

    t0 = time.time()
    if args.addendum:
        tasks = [(ci, d, N, s, args.seed, [1, 2])
                 for ci, (N, s) in enumerate(CELLS)
                 for d in range(args.mc_runs)]
        with mp.Pool(args.workers) as pool:
            rs_all = pool.map(run_draw_flipk, tasks, chunksize=1)
        add = {}
        print("\n  POST-HOC ADDENDUM (exploratory): absolute floor on sign errors")
        for ci, (N, s) in enumerate(CELLS):
            rs = [r for r in rs_all if r["cell"] == ci]
            row = {}
            for k in (1, 2):
                key = f"k{k}"
                pcs = [r["res"][key]["pc"] for r in rs]
                row[key] = {"pc50": float(np.median(pcs)),
                            "pc95": float(np.percentile(pcs, 95)),
                            "converged": int(sum(r["res"][key]["converged"]
                                                 for r in rs)),
                            "diverged": int(sum(r["res"][key]["diverged"]
                                                for r in rs)),
                            "all_signs_ok": int(sum(r["res"][key]["all_signs_ok"]
                                                    for r in rs))}
                print(f"   N={N}, sigma={s*100:.0f}%  {key} flipped of "
                      f"{M_OUT*N} entries: pc p50={100*row[key]['pc50']:.4f}% "
                      f"conv={row[key]['converged']}/{len(rs)} "
                      f"div={row[key]['diverged']}/{len(rs)} "
                      f"all-channel-signs-ok={row[key]['all_signs_ok']}/{len(rs)}")
            add[f"N{N}_s{s}"] = row
        p = Path(args.out).with_name("sign_concordance_addendum.json")
        p.write_text(json.dumps(add, indent=1))
        print(f"\n  wrote {p}   wall clock {time.time()-t0:.1f}s")
        return
    tasks = [(ci, d, N, s, args.seed)
             for ci, (N, s) in enumerate(CELLS)
             for d in range(args.mc_runs)]
    print(f"[1/1] {len(tasks)} (cell, draw) tasks on {args.workers} workers, "
          f"master seed {args.seed}")
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_draw, tasks, chunksize=1)
    print(f"      done in {time.time()-t0:.1f}s")

    agg = {}
    for ci, (N, s) in enumerate(CELLS):
        rs = [r for r in results if r["cell"] == ci]
        a = {"N": N, "sigma": s, "n": len(rs)}
        for fld in ("pc", "pc_out"):
            a[f"untrained_{fld}50"] = float(np.median(
                [r["untrained"][fld] for r in rs]))
        for c in CONDS:
            pcs = [r["res"][c]["pc"] for r in rs]
            pos = [r["res"][c]["pc_out"] for r in rs]
            a[c] = {"pc50": float(np.median(pcs)),
                    "pc95": float(np.percentile(pcs, 95)),
                    "pc_out50": float(np.median(pos)),
                    "pc_out95": float(np.percentile(pos, 95)),
                    "cm50": float(np.median([r["res"][c]["cm"] for r in rs])),
                    "mse": float(np.median([r["res"][c]["mse"] for r in rs])),
                    "evals": rs[0]["res"][c]["evals"],
                    "diverged": int(sum(r["res"][c]["diverged"] for r in rs)),
                    "converged": int(sum(r["res"]["converged"].get(c, False)
                                         for r in rs))}
        a["S4_best_conv"] = int(sum(r["res"]["converged"]["S4_best"] for r in rs))
        a["S4ref_best_conv"] = int(sum(r["res"]["converged"]["S4ref_best"]
                                       for r in rs))
        for d in ("diag_sign_random", "diag_sign_sc", "diag_sign_dec",
                  "WBt_min_re", "WWt_min_re"):
            a[d] = float(np.mean([r[d] for r in rs]))
        a["WBt_all_pos"] = int(sum(r["WBt_all_pos"] for r in rs))
        a["S3_diag_sign"] = {
            f"S3_{int(round(f*100)):02d}":
                (1.0 if f == 0.0 else
                 float(np.mean([r["res"][f"S3_{int(round(f*100)):02d}_diag_sign"]
                                for r in rs])))
            for f in F_LIST}
        # K1/K2 relative deviation from the anchor (p50)
        for k in ("S1", "S2"):
            a[k]["rel_dev_vs_S0"] = abs(a[k]["pc50"] - a["S0"]["pc50"]) \
                / max(a["S0"]["pc50"], 1e-15)
        # K3: largest f with >= 18/20 converged
        fmax = None
        for f in F_LIST:
            k = f"S3_{int(round(f*100)):02d}"
            if a[k]["converged"] >= 18:
                fmax = f
            else:
                break
        a["f_max"] = fmax
        agg[f"N{N}_s{s}"] = a

    # ---- print ------------------------------------------------------------
    p = lambda v: 100.0 * v  # noqa: E731
    for key, a in agg.items():
        print("\n" + "=" * 108)
        print(f"  {key}   sigma_VGA = {a['sigma']*100:.0f}%, N = {a['N']}, "
              f"{a['n']} draws   (untrained pc = {p(a['untrained_pc50']):.3f}%, "
              f"pc_out = {p(a['untrained_pc_out50']):.3f}%)")
        print(f"    {'cond':<11}{'pc p50':>9}{'pc p95':>9}{'pc/sig':>8}"
              f"{'pc_out p50':>12}{'pc_out p95':>12}{'task MSE':>11}"
              f"{'fwd ev':>10}{'div':>5}{'conv':>6}")
        for c in CONDS:
            r = a[c]
            print(f"    {c:<11}{p(r['pc50']):9.4f}{p(r['pc95']):9.4f}"
                  f"{r['pc50']/a['sigma']:8.3f}{p(r['pc_out50']):12.4f}"
                  f"{p(r['pc_out95']):12.4f}{r['mse']:11.3e}"
                  f"{r['evals']:10d}{r['diverged']:5d}{r['converged']:6d}")
        print(f"    K1 S1 rel.dev vs S0 = {a['S1']['rel_dev_vs_S0']*100:.2f}%   "
              f"K2 S2 rel.dev vs S0 = {a['S2']['rel_dev_vs_S0']*100:.2f}%   "
              f"K3 f_max = {a['f_max']}")
        print(f"    diag(B^T W)>0 fraction: random={a['diag_sign_random']:.3f}  "
              f"sign-conc={a['diag_sign_sc']:.3f}  decade={a['diag_sign_dec']:.3f}"
              f"   per-f: " + " ".join(
                  f"{k.split('_')[1]}%={v:.3f}" for k, v in
                  a["S3_diag_sign"].items()))
        print(f"    dense dynamics: draws with all Re(eig(W B^T))>0 = "
              f"{a['WBt_all_pos']}/{a['n']}, mean min Re = {a['WBt_min_re']:.4f}"
              f"   (exact adjoint: mean min Re(eig(W W^T)) = "
              f"{a['WWt_min_re']:.4f})")
        print(f"    K4 S4 converged (best of 2 lr) = {a['S4_best_conv']}/{a['n']}"
              f"   control S4ref = {a['S4ref_best_conv']}/{a['n']}")

    payload = {"agg": agg, "args": vars(args),
               "hyper": {"steps": STEPS, "batch": BATCH, "lr": [LR0, LR1],
                         "lr_alt": [LR0_ALT, LR1_ALT], "f_list": F_LIST,
                         "n_train": T.N_TRAIN, "n_test": T.N_TEST, "M": M_OUT},
               "per_draw": [{"cell": r["cell"], "draw": r["draw"],
                             "untrained": r["untrained"],
                             "WBt_min_re": r["WBt_min_re"],
                             "WBt_all_pos": r["WBt_all_pos"],
                             "res": r["res"]} for r in results]}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\n  wrote {args.out}   total wall clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
