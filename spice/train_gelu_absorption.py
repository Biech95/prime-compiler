#!/usr/bin/env python3
"""
Part A: can physics-aware training absorb the GELU (P11.P6) mismatch?
Part B: how does the perturbative (SPSA) training cost scale with N?

Pre-registration, model, metric and kill criteria: docs/exp_gelu_training.md
(written before this script was run).

Part A
------
  x -> analog GELU stage with per-channel mismatch (drawn EXACTLY as
  predict_gelu_layernorm_sim.py does, via mismatch_calculus.gelu_draw at the
  MOS-typical corner: Gilbert K' 3 %, sigmoid pair V_t 10 mV, Is 3 %, n 0.3 %)
  -> per-channel affine (gamma_i, beta_i) -> frozen W (N -> M) -> MSE vs the
  ideal network.

  Conditions: A train-in-sim, B physics-aware, C2 sign-concordant backward with
  20 % gain mismatch, D perturbative.  N in {8, 32}, 20 MC draws.
  The forward surrogate is mismatch_calculus.gelu_map itself (batched); it is
  validated paired against the real ngspice bench, and the trained (gamma, beta)
  are pushed back through that bench (mandatory verification).

Part B
------
  Re-uses spice/train_mismatch_absorption.py unchanged (RMSNorm/VGA stage,
  sigma_VGA = 1 %) and measures the budget-to-reach-1.2xB of condition D at
  N in {8, 32, 128, 512}, then fits the exponent of D/B vs N.

Usage:
    OPENBLAS_NUM_THREADS=1 python3 train_gelu_absorption.py --part a --workers 6
    OPENBLAS_NUM_THREADS=1 python3 train_gelu_absorption.py --part b --workers 6
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
import mismatch_calculus as MC              # noqa: E402
import predict_gelu_layernorm_sim as PG     # noqa: E402
import train_mismatch_absorption as TMA     # noqa: E402

OUT = HERE / "out"

# ---------------------------------------------------------------------------
# Part A constants (frozen before the run, see docs/exp_gelu_training.md §1.2/1.5)
# ---------------------------------------------------------------------------
M_OUT = 4
N_TRAIN = 2000
N_TEST = 500
STEPS_GRAD = 32000
BATCH_GRAD = 64
STEPS_PERT = 800000
BATCH_PERT = 16
SIGMA_P = 0.02
LR_GRAD_ARMS = (2.0, 0.5, 0.2)
LR_PERT_ARMS = (0.5, 0.1)
GBOX = (1e-3, 20.0)
BBOX = (-5.0, 5.0)
DELTA_BWD = 0.20                 # 20 % backward gain mismatch for C2
WD_BETA = 0.01                   # condition D2 (post-hoc): beta decay
CELLS_A = [8, 32]

MASKS = {
    "R3":        ("eps_vga", "g_out", "g_tail"),
    "R1_iota":   ("iota_x", "iota_s", "iota_o", "iota_r"),
    "R1_nu":     ("nu_x", "nu_s", "nu_o", "nu_r"),
    "R12_off":   ("vos", "iota_a", "iota_b"),
    "R12_slope": ("nu_a", "nu_b"),
}


# ---------------------------------------------------------------------------
# batched surrogate  (line-by-line the same algebra as mismatch_calculus.gelu_map)
# ---------------------------------------------------------------------------

def gelu_map_batch(X, d):
    """gelu_map() of the calculus, broadcast over a batch of input vectors.

    X: (n, N).  d: dict of (N,) per-device deviation arrays (gelu_draw)."""
    u = 1.702 * X
    delta = (d["vos"] / MC.VT_NG + (d["iota_a"] - d["iota_b"])
             + (d["nu_a"] - d["nu_b"]) * np.abs(u))
    sig = 1.0 / (1.0 + np.exp(-(u + delta)))
    i_s = MC.GELU_ITAIL * (1.0 + d["g_tail"]) * sig
    i_x = np.abs(X) * MC.IUNIT * (1.0 + d["eps_vga"])
    lam_x = np.log(i_x / MC.ISAT)
    lam_s = np.log(np.maximum(i_s, 1e-15) / MC.ISAT)
    lam_r = np.log(MC.GELU_IREF / MC.ISAT)
    b_x = d["nu_x"] * lam_x - d["iota_x"]
    b_s = d["nu_s"] * lam_s - d["iota_s"]
    b_r = d["nu_r"] * lam_r - d["iota_r"]
    lam_o = lam_x + lam_s - lam_r
    b_o = d["nu_o"] * lam_o - d["iota_o"]
    gain = np.exp(b_x + b_s - b_r - b_o) * (1.0 + d["g_out"])
    return np.sign(X) * (i_x / MC.IUNIT) * (i_s / MC.GELU_IREF) * gain


def gelu_ideal(X):
    return X / (1.0 + np.exp(-1.702 * X))


def gelu_x(rng, N, n):
    """Same input distribution and clipping as mismatch_calculus.gelu_inputs."""
    x = rng.standard_normal((n, N))
    return np.sign(x) * np.clip(np.abs(x), 0.1, 2.5)


def zero_draw(d):
    return {k: np.zeros_like(v) for k, v in d.items()}


def mask_draw(d, keys):
    z = zero_draw(d)
    for k in keys:
        z[k] = d[k].copy()
    return z


def decompose_batch(Z, Yref):
    """MC.decompose(), vectorised over a batch of samples (relative residual)."""
    alpha = np.sum(Z * Yref, axis=1) / np.sum(Yref * Yref, axis=1)
    resid = Z / alpha[:, None] - Yref
    rms_ref = np.sqrt(np.mean(Yref ** 2, axis=1))
    return np.abs(alpha - 1.0), np.sqrt(np.mean(resid ** 2, axis=1)) / rms_ref


def evaluate_a(gam, bet, W, Hte_mis, Href_te, T_te):
    Z = Hte_mis * gam + bet
    cm, pc = decompose_batch(Z, Href_te)
    mse = float(np.mean((Z @ W.T - T_te) ** 2))
    return float(np.median(cm)), float(np.median(pc)), mse


def train_loss(gam, bet, W, H, T):
    return float(np.mean((( H * gam + bet) @ W.T - T) ** 2))


def cos_lr(lr0, lr1, t, steps):
    return lr1 + (lr0 - lr1) * 0.5 * (1.0 + np.cos(np.pi * t / steps))


# ---------------------------------------------------------------------------
# Part A training
# ---------------------------------------------------------------------------

def train_gradient_a(cond, W, Htr_id, Htr_mis, T_tr, Bmat, delta, rng,
                     lr0, steps=STEPS_GRAD, batch=BATCH_GRAD):
    """A / B / C2 -- plain SGD, cosine-decayed lr, frozen W."""
    N = W.shape[1]
    gam = np.ones(N)
    bet = np.zeros(N)
    n = Htr_id.shape[0]
    lr1 = lr0 / 2000.0
    evals = 0
    hit = False
    for t in range(steps):
        lr = cos_lr(lr0, lr1, t, steps)
        idx = rng.integers(0, n, batch)
        h0 = Htr_id[idx]
        hm = Htr_mis[idx]
        tt = T_tr[idx]
        hf = h0 if cond == "A" else hm
        Z = hf * gam + bet
        E = Z @ W.T - tt
        evals += batch
        gE = (2.0 / M_OUT) * E
        if cond in ("A", "B"):
            dz = gE @ W                      # exact adjoint of the ideal model
            loc = h0
        else:
            dz = (gE @ Bmat) * (1.0 + delta)  # sign-concordant backward
            loc = hm
        gg = (dz * loc).mean(axis=0)
        gb = dz.mean(axis=0)
        gnew = gam - lr * gg
        bnew = bet - lr * gb
        if not (np.all(np.isfinite(gnew)) and np.all(np.isfinite(bnew))):
            gnew = np.nan_to_num(gnew, nan=1.0, posinf=GBOX[1], neginf=GBOX[0])
            bnew = np.nan_to_num(bnew, nan=0.0, posinf=BBOX[1], neginf=BBOX[0])
            hit = True
        gam = np.clip(gnew, *GBOX)
        bet = np.clip(bnew, *BBOX)
        if (np.any(gam <= GBOX[0] * 1.001) or np.any(gam >= GBOX[1] * 0.999)
                or np.any(np.abs(bet) >= BBOX[1] * 0.999)):
            hit = True
    return gam, bet, evals, hit


def train_perturbative_a(W, Htr_mis, T_tr, rng, lr0, steps=STEPS_PERT,
                         batch=BATCH_PERT, sigma_p=SIGMA_P, wd_beta=0.0):
    """Condition D: no backward path.  Two-sided Rademacher SPSA on (gamma, beta).

    `wd_beta` > 0 adds a decay term `wd_beta * mean(beta^2)` to the objective
    (condition D2, a declared post-hoc control: see docs/exp_gelu_training.md
    §2.0).  beta is only identifiable up to null(W), which has dimension N - M;
    a backward-free estimator random-walks in that null space unless something
    pins it, and an analog offset store leaks anyway."""
    N = W.shape[1]
    gam = np.ones(N)
    bet = np.zeros(N)
    n = Htr_mis.shape[0]
    lr1 = lr0 / 2000.0
    evals = 0
    hit = False
    for t in range(steps):
        lr = cos_lr(lr0, lr1, t, steps)
        idx = rng.integers(0, n, batch)
        hm = Htr_mis[idx]
        tt = T_tr[idx]
        sg = rng.integers(0, 2, N) * 2.0 - 1.0
        sb = rng.integers(0, 2, N) * 2.0 - 1.0
        dg = sigma_p * sg
        db = sigma_p * sb
        Zp = hm * (gam + dg) + (bet + db)
        Zm = hm * (gam - dg) + (bet - db)
        Lp = np.mean((Zp @ W.T - tt) ** 2)
        Lm = np.mean((Zm @ W.T - tt) ** 2)
        if wd_beta:
            Lp += wd_beta * np.mean((bet + db) ** 2)
            Lm += wd_beta * np.mean((bet - db) ** 2)
        evals += 2 * batch
        c = lr * (Lp - Lm) / (2.0 * sigma_p)
        gam = np.clip(gam - c * sg, *GBOX)
        bet = np.clip(bet - c * sb, *BBOX)
        if (np.any(gam <= GBOX[0] * 1.001) or np.any(gam >= GBOX[1] * 0.999)
                or np.any(np.abs(bet) >= BBOX[1] * 0.999)):
            hit = True
    return gam, bet, evals, hit


def ls_optimum_a(H, T, W):
    """Exact minimiser of the training loss over (gamma, beta), frozen W.

    o = W (h*gamma + beta) = [W diag(h) | W] [gamma; beta]  -- quadratic."""
    N = W.shape[1]
    Mt = W.T @ W                                   # (N, N)
    S2 = H.T @ H                                   # sum_k h_k h_k^T
    hs = H.sum(axis=0)
    n = H.shape[0]
    A = np.empty((2 * N, 2 * N))
    A[:N, :N] = Mt * S2
    A[:N, N:] = Mt * hs[:, None]
    A[N:, :N] = Mt * hs[None, :]
    A[N:, N:] = Mt * n
    WT = T @ W                                     # (n, N) = (W^T t_k)^T
    b = np.concatenate([np.sum(H * WT, axis=0), WT.sum(axis=0)])
    A = A + np.eye(2 * N) * (1e-12 * np.trace(A) / (2 * N))
    th = np.linalg.solve(A, b)
    return th[:N], th[N:]


def run_draw_a(task):
    N, draw, master_seed = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(N, draw))
    rng = np.random.default_rng(ss)

    d = MC.gelu_draw(rng, N)
    Xtr = gelu_x(rng, N, N_TRAIN)
    Xte = gelu_x(rng, N, N_TEST)
    W = rng.standard_normal((M_OUT, N)) / np.sqrt(N)
    Bsc = np.abs(rng.standard_normal((M_OUT, N))) / np.sqrt(N) * np.sign(W)
    delta = rng.normal(0.0, DELTA_BWD, N)

    Htr_id, Hte_id = gelu_ideal(Xtr), gelu_ideal(Xte)
    Htr_mis, Hte_mis = gelu_map_batch(Xtr, d), gelu_map_batch(Xte, d)
    T_tr, T_te = Htr_id @ W.T, Hte_id @ W.T

    out = {"N": N, "draw": draw}
    g1, b0 = np.ones(N), np.zeros(N)
    cm, pc, mse = evaluate_a(g1, b0, W, Hte_mis, Hte_id, T_te)
    out["untrained"] = {"cm": cm, "pc": pc, "mse": mse}
    out["sc_diag_sign"] = float(np.mean(np.sum(Bsc * W, axis=0) > 0))
    # cross-reference: the bench's own fixed input vector (the 8.5 % protocol)
    xb = MC.gelu_inputs(N=N, seed=2026 if N == 8 else 2027)
    out["untrained_benchx_pc"] = float(MC.decompose(gelu_map_batch(xb[None, :],
                                                                  d)[0],
                                                    gelu_ideal(xb))[1])
    # basis of null(W): beta is identifiable only modulo this subspace
    Vt = np.linalg.svd(W, full_matrices=True)[2]
    Pnull = Vt[M_OUT:]

    seeds = ss.spawn(64)
    si = 0
    res = {}

    def best_grad(cond, Bmat, Htr_fwd_id, Htr_fwd_mis, T, Hte_eval, Tev, tag):
        nonlocal si
        best = None
        for lr0 in LR_GRAD_ARMS:
            r = np.random.default_rng(seeds[si]); si += 1
            gam, bet, ev, hit = train_gradient_a(cond, W, Htr_fwd_id,
                                                 Htr_fwd_mis, T, Bmat, delta,
                                                 r, lr0)
            tl = train_loss(gam, bet, W, Htr_fwd_id if cond == "A" else Htr_fwd_mis, T)
            if not np.isfinite(tl):
                tl = np.inf
            if best is None or tl < best[0]:
                best = (tl, gam, bet, ev, hit, lr0)
        tl, gam, bet, ev, hit, lr0 = best
        c, p, m = evaluate_a(gam, bet, W, Hte_eval, Hte_id, Tev)
        return {"cm": c, "pc": p, "mse": m, "evals": ev * len(LR_GRAD_ARMS),
                "lr0": lr0, "train_loss": tl,
                "diverged": bool(hit or p > out["untrained"]["pc"]),
                "gamma": gam.tolist(), "beta": bet.tolist()}

    # A: train on the ideal forward, DEPLOY on the mismatched one
    res["A"] = best_grad("A", W, Htr_id, Htr_mis, T_tr, Hte_mis, T_te, "A")
    # B: physics-aware (mismatched forward, ideal backward)
    res["B"] = best_grad("B", W, Htr_id, Htr_mis, T_tr, Hte_mis, T_te, "B")
    # C2: sign-concordant random backward, 20 % backward gain mismatch
    res["C2"] = best_grad("C2", Bsc, Htr_id, Htr_mis, T_tr, Hte_mis, T_te, "C2")

    # D: perturbative (pre-registered) and D2 (post-hoc: + beta decay)
    for key, wd in (("D", 0.0), ("D2", WD_BETA)):
        best = None
        for lr0 in LR_PERT_ARMS:
            r = np.random.default_rng(seeds[si]); si += 1
            gam, bet, ev, hit = train_perturbative_a(W, Htr_mis, T_tr, r, lr0,
                                                     wd_beta=wd)
            tl = train_loss(gam, bet, W, Htr_mis, T_tr)
            if not np.isfinite(tl):
                tl = np.inf
            if best is None or tl < best[0]:
                best = (tl, gam, bet, ev, hit, lr0)
        tl, gam, bet, ev, hit, lr0 = best
        c, p, m = evaluate_a(gam, bet, W, Hte_mis, Hte_id, T_te)
        res[key] = {"cm": c, "pc": p, "mse": m,
                    "evals": ev * len(LR_PERT_ARMS),
                    "lr0": lr0, "train_loss": tl,
                    "diverged": bool(hit or p > out["untrained"]["pc"]),
                    "gamma": gam.tolist(), "beta": bet.tolist()}

    # LS: analytic floor
    gls, bls = ls_optimum_a(Htr_mis, T_tr, W)
    c, p, m = evaluate_a(gls, bls, W, Hte_mis, Hte_id, T_te)
    res["LS"] = {"cm": c, "pc": p, "mse": m, "evals": 0, "lr0": None,
                 "train_loss": train_loss(gls, bls, W, Htr_mis, T_tr),
                 "diverged": False, "gamma": gls.tolist(), "beta": bls.tolist()}

    # gamma-only control (no beta): does the offset need the additive term?
    gls_g, _ = ls_optimum_a(Htr_mis, T_tr, W)
    A_only = None
    # (gamma-only least squares: solve the NxN block alone)
    Mt = W.T @ W
    S2 = Htr_mis.T @ Htr_mis
    WT = T_tr @ W
    Ag = Mt * S2
    Ag = Ag + np.eye(N) * (1e-12 * np.trace(Ag) / N)
    gg = np.linalg.solve(Ag, np.sum(Htr_mis * WT, axis=0))
    c, p, m = evaluate_a(gg, np.zeros(N), W, Hte_mis, Hte_id, T_te)
    res["LS_gamma_only"] = {"cm": c, "pc": p, "mse": m, "evals": 0,
                            "lr0": None, "train_loss": float("nan"),
                            "diverged": False, "gamma": gg.tolist(),
                            "beta": np.zeros(N).tolist()}
    del gls_g, A_only

    # --- component attribution: LS floor per mechanism mask ----------------
    masks = {}
    for name, keys in MASKS.items():
        dm = mask_draw(d, keys)
        Htr_m = gelu_map_batch(Xtr, dm)
        Hte_m = gelu_map_batch(Xte, dm)
        _, pc_u = decompose_batch(Hte_m, Hte_id)
        gm, bm = ls_optimum_a(Htr_m, T_tr, W)
        Zm = Hte_m * gm + bm
        _, pc_t = decompose_batch(Zm, Hte_id)
        masks[name] = {"pc_untrained": float(np.median(pc_u)),
                       "pc_trained": float(np.median(pc_t))}
    out["masks"] = masks
    # beta null-space diagnostic: ||P_null beta|| / ||beta||
    for k, v in res.items():
        b = np.asarray(v["beta"])
        nb = float(np.linalg.norm(b))
        v["beta_norm"] = nb
        v["beta_null_frac"] = float(np.linalg.norm(Pnull @ b) / nb) if nb > 1e-12 else 0.0
    out["res"] = res
    return out


# ---------------------------------------------------------------------------
# Part A: SPICE validation and verification
# ---------------------------------------------------------------------------

def spice_gelu(x, d, tag):
    PG.WORKDIR.mkdir(exist_ok=True)
    PG.run_ngspice(PG.build_gelu_netlist(x, d, tag), tag)
    i_out = PG.parse_print(tag, "i(vo", len(x)) / PG.IUNIT
    for suf in (".cir", ".txt"):
        (PG.WORKDIR / f"{tag}{suf}").unlink(missing_ok=True)
    return np.sign(x) * i_out


def validate_task(task):
    """V2/V3: paired SPICE vs surrogate on the bench's own fixed input vector."""
    N, j, seed = task
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed,
                                                       spawn_key=(N, j)))
    x = MC.gelu_inputs(N=N, seed=2026 + (0 if N == 8 else 1))
    ref = gelu_ideal(x)
    d = MC.gelu_draw(rng, N)
    y_sp = spice_gelu(x, d, f"vg_{N}_{j}")
    y_su = MC.gelu_map(x, d)
    cm_s, pc_s = MC.decompose(y_sp, ref)
    cm_u, pc_u = MC.decompose(y_su, ref)
    return {"N": N, "j": j, "pc_spice": pc_s, "pc_sur": pc_u,
            "cm_spice": cm_s, "cm_sur": cm_u,
            "max_rel_chan": float(np.max(np.abs(y_sp - y_su)
                                         / np.maximum(np.abs(y_su), 1e-12)))}


def verify_task(task):
    """Mandatory: push the trained (gamma, beta) through the real ngspice bench."""
    N, draw, master_seed, gam, bet, n_inputs, tag = task
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(N, draw))
    rng = np.random.default_rng(ss)
    d = MC.gelu_draw(rng, N)
    _ = gelu_x(rng, N, N_TRAIN)          # keep the stream aligned with run_draw_a
    Xte = gelu_x(rng, N, N_TEST)
    gam = np.asarray(gam)
    bet = np.asarray(bet)
    rows = []
    for j in range(n_inputs):
        x = Xte[j]
        ref = gelu_ideal(x)
        y_sp = spice_gelu(x, d, f"{tag}_{j}")
        y_su = gelu_map_batch(x[None, :], d)[0]
        r = {}
        for name, (g, b) in (("before", (np.ones(N), np.zeros(N))),
                             ("after", (gam, bet))):
            cm_s, pc_s = MC.decompose(y_sp * g + b, ref)
            cm_u, pc_u = MC.decompose(y_su * g + b, ref)
            tot_s = float(np.sqrt(np.mean((y_sp * g + b - ref) ** 2))
                          / np.sqrt(np.mean(ref ** 2)))
            tot_u = float(np.sqrt(np.mean((y_su * g + b - ref) ** 2))
                          / np.sqrt(np.mean(ref ** 2)))
            r[name] = {"spice_cm": cm_s, "spice_pc": pc_s, "spice_tot": tot_s,
                       "sur_cm": cm_u, "sur_pc": pc_u, "sur_tot": tot_u}
        rows.append(r)
    return {"N": N, "draw": draw, "rows": rows}


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

# Ladders extended DOWNWARD as well as upward (declared deviation, §2.0): a
# pilot showed that at N = 8 both B and D already reach the target at TMA's
# lowest rung, which would left-censor the interpolated cost estimate.
LADDER_GRAD_EXT = [2, 4, 8, 15, 30, 60, 125] + list(TMA.LADDER_GRAD)
LADDER_PERT_EXT = ([25, 50, 100, 200, 400, 800, 1600, 3200]
                   + list(TMA.LADDER_PERT)
                   + [1600000, 3200000])
CELLS_B = [8, 32, 128, 512]
SIGMA_B = 0.01
# pre-registered arm first (the harness default), then a declared
# post-hoc smaller-step arm (see docs/exp_gelu_training.md §2.0)
LR_PERT_ARMS_B = (0.5, 0.1)


def interp_budget(ladder, target):
    """Log-log interpolated crossing budget; None if never reached."""
    prev = None
    for ev, pc in ladder:
        if pc <= target:
            if prev is None or prev[1] <= target:
                return float(ev), True          # left-censored at the first rung
            e0, p0 = prev
            f = (np.log(p0) - np.log(target)) / (np.log(p0) - np.log(pc))
            f = min(max(f, 0.0), 1.0)
            return float(np.exp(np.log(e0) + f * (np.log(ev) - np.log(e0)))), False
        prev = (ev, pc)
    return None, False


def run_draw_b(task):
    N, draw, master_seed = task
    t0 = time.time()
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(N, draw))
    rng = np.random.default_rng(ss)
    eps = rng.normal(0.0, SIGMA_B, N)
    Xtr = TMA.make_inputs(rng, N, TMA.N_TRAIN)
    Xte = TMA.make_inputs(rng, N, TMA.N_TEST)
    W0 = rng.standard_normal((TMA.M_OUT, N)) / np.sqrt(N)
    delta = rng.normal(0.0, 0.05, N)
    Ytr_id, Yte_id = TMA.norm_ideal(Xtr), TMA.norm_ideal(Xte)
    Ytr_mis, Yte_mis = TMA.norm_mis(Xtr, eps), TMA.norm_mis(Xte, eps)
    T_tr, T_te = Ytr_id @ W0.T, Yte_id @ W0.T
    g1 = np.ones(N)
    cm_u, pc_u, mse_u = TMA.evaluate(g1, W0, Yte_mis, Yte_id, T_te)

    seeds = ss.spawn(8)
    ladB = []
    rB = None
    r = np.random.default_rng(seeds[0])
    for st in LADDER_GRAD_EXT:
        rr = np.random.default_rng(r.integers(0, 2 ** 62))
        gam, W, ev, div = TMA.train_gradient("B", g1, W0, Ytr_id, Ytr_mis,
                                             T_tr, False, W0, delta, rr,
                                             steps=st)
        c, p, m = TMA.evaluate(gam, W, Yte_mis, Yte_id, T_te)
        ladB.append((ev, p))
        rB = {"cm": c, "pc": p, "mse": m, "evals": ev, "diverged": bool(div)}

    target = 1.2 * rB["pc"]
    lads = {}
    rDs = {}
    for ai, lr0 in enumerate(LR_PERT_ARMS_B):
        lad = []
        r = np.random.default_rng(seeds[1 + ai])
        for st in LADDER_PERT_EXT:
            rr = np.random.default_rng(r.integers(0, 2 ** 62))
            gam, ev, div = TMA.train_perturbative(g1, W0, Ytr_mis, T_tr, rr,
                                                  steps=st, lr0=lr0,
                                                  lr1=lr0 / 500.0)
            c, p, m = TMA.evaluate(gam, W0, Yte_mis, Yte_id, T_te)
            lad.append((ev, p))
            rDs[lr0] = {"cm": c, "pc": p, "mse": m, "evals": ev,
                        "diverged": bool(div)}
            if p <= target and len(lad) >= 2:
                break            # crossing bracketed; higher rungs add nothing
        lads[lr0] = lad
    ladD = lads[LR_PERT_ARMS_B[0]]          # pre-registered arm (harness default)
    rD = rDs[LR_PERT_ARMS_B[0]]

    g_ls = TMA.ls_optimum(Ytr_mis, T_tr, W0)
    c, p, m = TMA.evaluate(g_ls, W0, Yte_mis, Yte_id, T_te)
    rLS = {"cm": c, "pc": p, "mse": m}

    rung = {k: TMA.smallest_budget(l, target) for k, l in
            (("B", ladB), ("D", ladD))}
    ip = {}
    for k, l in (("B", ladB), ("D", ladD)):
        v, cens = interp_budget(l, target)
        ip[k] = v
        ip[k + "_censored"] = cens
    # post-hoc "best lr" arm: the cheapest crossing over the lr sweep
    best_r, best_i = None, None
    per_arm = {}
    for lr0, lad in lads.items():
        rv = TMA.smallest_budget(lad, target)
        iv, _ = interp_budget(lad, target)
        per_arm[str(lr0)] = {"rung": rv, "interp": iv}
        if rv is not None and (best_r is None or rv < best_r):
            best_r = rv
        if iv is not None and (best_i is None or iv < best_i):
            best_i = iv
    rung["Dbest"] = best_r
    ip["Dbest"] = best_i
    return {"N": N, "draw": draw, "untrained": {"cm": cm_u, "pc": pc_u,
                                                "mse": mse_u},
            "B": rB, "D": rD, "LS": rLS, "target": target,
            "ladderB": ladB, "ladders_D": {str(k): v for k, v in lads.items()},
            "ladderD": ladD, "rung": rung, "interp": ip, "per_arm": per_arm,
            "secs": time.time() - t0}


# ---------------------------------------------------------------------------

def p50(v):
    return float(np.percentile(v, 50))


def p95(v):
    return float(np.percentile(v, 95))


def main_a(args):
    t0 = time.time()
    print("=" * 96)
    print("  PART A -- physics-aware training of the GELU (P11.P6) stage")
    print("=" * 96)

    # ---- V1: batching equivalence ----------------------------------------
    rng = np.random.default_rng(11)
    d8 = MC.gelu_draw(rng, 8)
    Xc = gelu_x(rng, 8, 200)
    ref_rows = np.array([MC.gelu_map(Xc[k], d8) for k in range(200)])
    bat = gelu_map_batch(Xc, d8)
    v1 = float(np.max(np.abs(bat - ref_rows) / np.maximum(np.abs(ref_rows), 1e-30)))
    print(f"\n[V1] gelu_map_batch vs mismatch_calculus.gelu_map, 200 rows: "
          f"max rel dev = {v1:.3e}  ({'OK' if v1 < 1e-12 else 'FAIL'})")

    # ---- ideal-device control on the netlist ------------------------------
    xg = MC.gelu_inputs()
    y0 = spice_gelu(xg, zero_draw(MC.gelu_draw(np.random.default_rng(0), 8)),
                    "vg_ideal8")
    ctrl = float(np.max(np.abs(y0 - gelu_ideal(xg)) /
                        np.maximum(np.abs(gelu_ideal(xg)), 1e-12)))
    print(f"[V0] ngspice ideal-device control (N=8): max rel dev from "
          f"x*sigma(1.702x) = {ctrl:.3e}")

    # ---- V2/V3: paired SPICE vs surrogate --------------------------------
    vtasks = ([(8, j, args.seed + 900) for j in range(args.val_draws8)]
              + [(32, j, args.seed + 901) for j in range(args.val_draws32)])
    with mp.Pool(args.workers) as pool:
        vres = pool.map(validate_task, vtasks, chunksize=1)
    val = {}
    for N in (8, 32):
        rs = [r for r in vres if r["N"] == N]
        ps = np.array([r["pc_spice"] for r in rs])
        pu = np.array([r["pc_sur"] for r in rs])
        paired = np.abs(ps - pu) / pu
        val[f"N{N}"] = {"n": len(rs), "pc_p50_spice": p50(ps) * 100,
                        "pc_p50_sur": p50(pu) * 100,
                        "pc_p95_spice": p95(ps) * 100,
                        "pc_p95_sur": p95(pu) * 100,
                        "paired_rel_dev_p50": p50(paired) * 100,
                        "paired_rel_dev_max": float(np.max(paired)) * 100,
                        "max_rel_chan": float(np.max([r["max_rel_chan"]
                                                      for r in rs])) * 100,
                        "dev_p50": abs(p50(ps) - p50(pu)) / p50(pu) * 100}
        v = val[f"N{N}"]
        print(f"[V{2 if N == 8 else 3}] N={N}, {v['n']} paired draws:  "
              f"per-channel p50 SPICE {v['pc_p50_spice']:.3f} % vs surrogate "
              f"{v['pc_p50_sur']:.3f} %  (dev {v['dev_p50']:.2f} %);  "
              f"paired per-draw rel dev p50 {v['paired_rel_dev_p50']:.2f} %, "
              f"max {v['paired_rel_dev_max']:.2f} %")
    gate = (val["N8"]["dev_p50"] < 20.0
            and abs(val["N8"]["pc_p50_sur"] - 8.5) / 8.5 < 0.20)
    print(f"      surrogate gate (<20 % vs SPICE and vs the 8.5 % headline): "
          f"{'PASS' if gate else 'FAIL -> switch to SPICE-in-the-loop'}")

    # ---- training ---------------------------------------------------------
    tasks = [(N, d, args.seed) for N in CELLS_A for d in range(args.mc_runs)]
    print(f"\n[train] {len(tasks)} (N, draw) tasks on {args.workers} workers")
    t1 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(run_draw_a, tasks, chunksize=1)
    print(f"        done in {time.time()-t1:.1f}s")

    conds = ["LS", "LS_gamma_only", "A", "B", "C2", "D", "D2"]
    agg = {}
    for N in CELLS_A:
        rs = [r for r in results if r["N"] == N]
        a = {"N": N, "untrained": {
            "pc50": p50([r["untrained"]["pc"] for r in rs]),
            "pc95": p95([r["untrained"]["pc"] for r in rs]),
            "cm50": p50([r["untrained"]["cm"] for r in rs]),
            "mse": p50([r["untrained"]["mse"] for r in rs]),
            "pc50_benchx": p50([r["untrained_benchx_pc"] for r in rs])}}
        thr = 0.25 * a["untrained"]["pc50"]
        for c in conds:
            a[c] = {"pc50": p50([r["res"][c]["pc"] for r in rs]),
                    "pc95": p95([r["res"][c]["pc"] for r in rs]),
                    "cm50": p50([r["res"][c]["cm"] for r in rs]),
                    "mse": p50([r["res"][c]["mse"] for r in rs]),
                    "evals": rs[0]["res"][c]["evals"],
                    "diverged": int(sum(r["res"][c]["diverged"] for r in rs)),
                    "beta_null_frac": p50([r["res"][c]["beta_null_frac"]
                                           for r in rs]),
                    "frac_draws_below_thr": float(np.mean(
                        [r["res"][c]["pc"] < 0.25 * r["untrained"]["pc"]
                         for r in rs])),
                    "lr_hist": {}}
            for r in rs:
                k = str(r["res"][c]["lr0"])
                a[c]["lr_hist"][k] = a[c]["lr_hist"].get(k, 0) + 1
        a["ka1_threshold"] = thr
        a["masks"] = {m: {"pc_untrained": p50([r["masks"][m]["pc_untrained"]
                                               for r in rs]),
                          "pc_trained": p50([r["masks"][m]["pc_trained"]
                                             for r in rs])}
                      for m in MASKS}
        a["sc_diag_sign"] = float(np.mean([r["sc_diag_sign"] for r in rs]))
        agg[f"N{N}"] = a

    print("\n" + "=" * 96)
    print("  per-channel residual [% of signal], common mode [%], task MSE "
          "(medians over draws)")
    print("=" * 96)
    for N in CELLS_A:
        a = agg[f"N{N}"]
        u = a["untrained"]
        print(f"\n  N = {N}   untrained pc p50 = {u['pc50']*100:.3f} %  "
              f"(bench fixed-x protocol: {u['pc50_benchx']*100:.3f} %)   "
              f"KA1 threshold (0.25x) = {u['pc50']*25:.3f} %")
        print(f"    {'cond':<15} {'pc p50':>9} {'pc p95':>9} {'x untr':>8} "
              f"{'cm p50':>9} {'task MSE':>11} {'fwd evals':>11} {'div':>5} "
              f"{'<thr':>6} {'b_null':>7}")
        print(f"    {'untrained':<15} {u['pc50']*100:9.4f} {u['pc95']*100:9.4f} "
              f"{1.0:8.3f} {u['cm50']*100:9.4f} {u['mse']:11.3e} "
              f"{'-':>11} {'-':>5} {'-':>6} {'-':>7}")
        for c in conds:
            r = a[c]
            print(f"    {c:<15} {r['pc50']*100:9.4f} {r['pc95']*100:9.4f} "
                  f"{r['pc50']/u['pc50']:8.3f} {r['cm50']*100:9.4f} "
                  f"{r['mse']:11.3e} {r['evals']:11d} {r['diverged']:5d} "
                  f"{r['frac_draws_below_thr']:6.2f} {r['beta_null_frac']:7.3f}")
        print(f"    lr arms chosen: B {a['B']['lr_hist']}, C2 "
              f"{a['C2']['lr_hist']}, D {a['D']['lr_hist']}, "
              f"D2 {a['D2']['lr_hist']}")
        print(f"    C2/B = {a['C2']['pc50']/a['B']['pc50']:.3f}   "
              f"sign-concordant diag(B^T W)>0 fraction = {a['sc_diag_sign']:.3f}")
        print(f"    component attribution (LS floor per mechanism mask, "
              f"pc p50 %):")
        for m in MASKS:
            mm = a["masks"][m]
            print(f"      {m:<11} untrained {mm['pc_untrained']*100:7.3f} "
                  f"-> trained {mm['pc_trained']*100:7.3f}   "
                  f"absorbed {100*(1-mm['pc_trained']/max(mm['pc_untrained'],1e-15)):6.1f} %")

    # ---- mandatory SPICE verification -------------------------------------
    print("\n[verify] pushing the trained (gamma, beta) of condition B through "
          "the real ngspice bench")
    vt = []
    for N in CELLS_A:
        for d in range(args.spice_draws):
            r = next(x for x in results if x["N"] == N and x["draw"] == d)
            vt.append((N, d, args.seed, r["res"]["B"]["gamma"],
                       r["res"]["B"]["beta"], args.spice_inputs,
                       f"gv{N}_{d}"))
    t2 = time.time()
    with mp.Pool(min(args.workers, len(vt))) as pool:
        vr = pool.map(verify_task, vt, chunksize=1)
    print(f"         {len(vt)} cells x {args.spice_inputs} inputs in "
          f"{time.time()-t2:.1f}s")
    sp = {}
    maxdev = 0.0
    print(f"\n  {'cell':<8} {'pc SPICE bef':>13} {'pc surr bef':>12} "
          f"{'pc SPICE aft':>13} {'pc surr aft':>12} {'dev aft':>8} "
          f"{'tot SPICE bef':>14} {'tot SPICE aft':>14}")
    for N in CELLS_A:
        rows = [rr for x in vr if x["N"] == N for rr in x["rows"]]
        med = lambda f: float(np.median([f(r) for r in rows]))   # noqa: E731
        pb, pa = med(lambda r: r["before"]["spice_pc"]), med(lambda r: r["after"]["spice_pc"])
        ub, ua = med(lambda r: r["before"]["sur_pc"]), med(lambda r: r["after"]["sur_pc"])
        tb, ta = med(lambda r: r["before"]["spice_tot"]), med(lambda r: r["after"]["spice_tot"])
        dev_a = abs(pa - ua) / max(ua, 1e-12)
        dev_b = abs(pb - ub) / max(ub, 1e-12)
        maxdev = max(maxdev, dev_a, dev_b)
        sp[f"N{N}"] = {"pc_before_spice": pb, "pc_after_spice": pa,
                       "pc_before_sur": ub, "pc_after_sur": ua,
                       "tot_before_spice": tb, "tot_after_spice": ta,
                       "rel_dev_before": dev_b, "rel_dev_after": dev_a,
                       "n_rows": len(rows)}
        print(f"  N={N:<6} {pb*100:13.4f} {ub*100:12.4f} {pa*100:13.4f} "
              f"{ua*100:12.4f} {dev_a*100:7.2f}% {tb*100:14.4f} {ta*100:14.4f}")
    if maxdev > 0.20:
        print(f"\n  *** WARNING: SPICE and surrogate disagree by "
              f"{maxdev*100:.1f} % relative (> 20 %) on the per-channel "
              f"residual. ***")
    else:
        print(f"\n  SPICE/surrogate max relative deviation on the per-channel "
              f"residual: {maxdev*100:.3f} % (< 20 % threshold)")

    # ---- verdicts ---------------------------------------------------------
    print("\n" + "=" * 96)
    for N in CELLS_A:
        a = agg[f"N{N}"]
        thr = 0.25 * a["untrained"]["pc50"]
        ka0 = a["A"]["pc50"] < thr
        ka1 = not (a["B"]["pc50"] < thr)
        ka2 = a["C2"]["pc50"] > 2.0 * a["B"]["pc50"]
        print(f"  N={N}: KA0 {'TRIGGERED' if ka0 else 'not triggered'} "
              f"(A={a['A']['pc50']*100:.3f} % vs {thr*100:.3f} %) | "
              f"KA1 {'TRIGGERED' if ka1 else 'not triggered'} "
              f"(B={a['B']['pc50']*100:.3f} % vs {thr*100:.3f} %) | "
              f"KA2 {'TRIGGERED' if ka2 else 'not triggered'} "
              f"(C2/B={a['C2']['pc50']/a['B']['pc50']:.3f})")

    payload = {"part": "A", "validation": {"v1_max_rel": v1,
                                           "ideal_control": ctrl, "paired": val,
                                           "gate_pass": bool(gate)},
               "agg": agg, "spice": sp, "max_spice_dev": maxdev,
               "args": vars(args), "wall_s": time.time() - t0}
    OUT.mkdir(exist_ok=True)
    p = OUT / "gelu_absorption_A.json"
    p.write_text(json.dumps(payload, indent=1))
    print(f"\n  wrote {p}   wall clock {time.time()-t0:.1f}s")


def main_b(args):
    t0 = time.time()
    print("=" * 96)
    print("  PART B -- perturbative (SPSA) cost scaling on the RMSNorm/VGA "
          "stage, sigma_VGA = 1 %")
    print("=" * 96)
    tasks = [(N, d, args.seed_b) for N in CELLS_B for d in range(args.mc_b)]
    print(f"  {len(tasks)} (N, draw) tasks on {args.workers} workers "
          f"(ladders: B {LADDER_GRAD_EXT}, D {LADDER_PERT_EXT})")
    with mp.Pool(args.workers) as pool:
        res = pool.map(run_draw_b, tasks, chunksize=1)
    print(f"  training done in {time.time()-t0:.1f}s")

    agg = {}
    print(f"\n  {'N':>5} {'untr pc%':>9} {'B pc%':>8} {'D pc%':>8} {'LS pc%':>8} "
          f"{'cost B':>11} {'cost D':>12} {'D/B rung':>9} {'D/B interp':>11} "
          f"{'Dbest/B':>11} {'hits':>6} {'s/draw':>8}")
    for N in CELLS_B:
        rs = [r for r in res if r["N"] == N]
        rb = [r["rung"]["B"] for r in rs]
        rd = [r["rung"]["D"] for r in rs]
        ib = [r["interp"]["B"] for r in rs]
        idd = [r["interp"]["D"] for r in rs]
        rdb = [r["rung"]["Dbest"] for r in rs]
        idb = [r["interp"]["Dbest"] for r in rs]
        rat_r = [d / b for d, b in zip(rd, rb) if d and b]
        rat_i = [d / b for d, b in zip(idd, ib) if d and b]
        rat_rb = [d / b for d, b in zip(rdb, rb) if d and b]
        rat_ib = [d / b for d, b in zip(idb, ib) if d and b]
        a = {"N": N, "n": len(rs),
             "untr_pc50": p50([r["untrained"]["pc"] for r in rs]),
             "B_pc50": p50([r["B"]["pc"] for r in rs]),
             "D_pc50": p50([r["D"]["pc"] for r in rs]),
             "LS_pc50": p50([r["LS"]["pc"] for r in rs]),
             "cost_B_rung": p50([x for x in rb if x]) if any(rb) else None,
             "cost_D_rung": p50([x for x in rd if x]) if any(rd) else None,
             "cost_B_interp": p50([x for x in ib if x]) if any(ib) else None,
             "cost_D_interp": p50([x for x in idd if x]) if any(idd) else None,
             "D_over_B_rung_p50": p50(rat_r) if rat_r else None,
             "D_over_B_rung_p95": p95(rat_r) if rat_r else None,
             "D_over_B_interp_p50": p50(rat_i) if rat_i else None,
             "D_over_B_interp_p95": p95(rat_i) if rat_i else None,
             "Dbest_over_B_rung_p50": p50(rat_rb) if rat_rb else None,
             "Dbest_over_B_interp_p50": p50(rat_ib) if rat_ib else None,
             "hits_Dbest": int(sum(x is not None for x in rdb)),
             "lr_arm_costs": {k: p50([x for x in
                                      [r["per_arm"][k]["interp"] for r in rs]
                                      if x]) if any(r["per_arm"][k]["interp"]
                                                    for r in rs) else None
                              for k in rs[0]["per_arm"]},
             "hits_B": int(sum(x is not None for x in rb)),
             "hits_D": int(sum(x is not None for x in rd)),
             "censored_D": int(sum(r["interp"]["D_censored"] for r in rs)),
             "censored_B": int(sum(r["interp"]["B_censored"] for r in rs)),
             "secs_per_draw": p50([r["secs"] for r in rs])}
        agg[f"N{N}"] = a
        print(f"  {N:>5} {a['untr_pc50']*100:9.4f} {a['B_pc50']*100:8.4f} "
              f"{a['D_pc50']*100:8.4f} {a['LS_pc50']*100:8.4f} "
              f"{str(a['cost_B_rung']):>11} {str(a['cost_D_rung']):>12} "
              f"{a['D_over_B_rung_p50'] or float('nan'):9.2f} "
              f"{a['D_over_B_interp_p50'] or float('nan'):11.2f} "
              f"{a['Dbest_over_B_interp_p50'] or float('nan'):11.2f} "
              f"{a['hits_D']:>3}/{a['n']:<2} {a['secs_per_draw']:8.1f}")

    fits = {}
    for tag in ("rung", "interp", "Dbest_rung", "Dbest_interp"):
        Ns, Rs = [], []
        for N in CELLS_B:
            key = (f"D_over_B_{tag}_p50" if tag in ("rung", "interp")
                   else f"Dbest_over_B_{tag.split('_')[1]}_p50")
            v = agg[f"N{N}"][key]
            if v:
                Ns.append(N)
                Rs.append(v)
        if len(Ns) >= 2:
            A = np.vstack([np.ones(len(Ns)), np.log(Ns)]).T
            coef, *_ = np.linalg.lstsq(A, np.log(Rs), rcond=None)
            pred = A @ coef
            ssr = float(np.sum((np.log(Rs) - pred) ** 2))
            sst = float(np.sum((np.log(Rs) - np.mean(np.log(Rs))) ** 2))
            fits[tag] = {"a": float(coef[0]), "p": float(coef[1]),
                         "R2": 1 - ssr / sst if sst > 0 else float("nan"),
                         "N": Ns, "R": Rs}
            print(f"\n  fit ({tag}):  D/B = {np.exp(coef[0]):.3f} * N^"
                  f"{coef[1]:.3f}   R^2 = {fits[tag]['R2']:.4f}   "
                  f"KB {'TRIGGERED' if not (0.7 <= coef[1] <= 1.3) else 'not triggered'}")
    payload = {"part": "B", "agg": agg, "fits": fits, "args": vars(args),
               "ladder_pert": LADDER_PERT_EXT,
               "ladder_grad": LADDER_GRAD_EXT,
               "wall_s": time.time() - t0,
               "raw": [{k: v for k, v in r.items()
                        if k in ("N", "draw", "target", "rung", "interp",
                                 "ladderB", "ladderD", "secs")} for r in res]}
    OUT.mkdir(exist_ok=True)
    p = OUT / "gelu_absorption_B.json"
    p.write_text(json.dumps(payload, indent=1))
    print(f"\n  wrote {p}   wall clock {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], default="a")
    ap.add_argument("--mc-runs", type=int, default=20)
    ap.add_argument("--mc-b", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5171)
    ap.add_argument("--seed-b", type=int, default=4127)
    ap.add_argument("--val-draws8", type=int, default=30)
    ap.add_argument("--val-draws32", type=int, default=10)
    ap.add_argument("--spice-draws", type=int, default=5)
    ap.add_argument("--spice-inputs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    # Python 3.14 defaults to "forkserver" on Linux, which re-imports this
    # module in every worker; fork keeps the already-loaded state and the
    # deterministic SeedSequence streams.
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    if args.part == "a":
        main_a(args)
    else:
        main_b(args)


if __name__ == "__main__":
    main()
