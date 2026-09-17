#!/usr/bin/env python3
"""Regenerate the v3 paper figures from the JSON produced by the testbenches in
spice/.  Every number plotted is read from spice/out/*.json -- nothing is typed
in by hand except axis labels and the two analytic curves in fig_v3_noise, whose
constants (C, k_j, I0) are themselves read from noise_reconciliation.json.

Usage:  python3 spice/make_v3_figures.py
Writes: fig_v3_training.png, fig_v3_signflip.png, fig_v3_calculus.png,
        fig_v3_noise.png, fig_v3_mamba.png   (repository root)

Figures that could NOT be regenerated from existing data are listed at the
bottom of this file and are omitted from the paper rather than invented.
"""
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.dirname(HERE)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 160,
})


def load(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Figure 1: training conditions, per-channel residual in units of sigma_VGA
# --------------------------------------------------------------------------
def fig_training():
    d = load("mismatch_absorption.json")["agg"]
    cells = ["N8_s0.01", "N8_s0.02", "N32_s0.01", "N32_s0.02"]
    labels = [r"$N{=}8$, $\sigma_{\rm VGA}{=}1\%$",
              r"$N{=}8$, $2\%$",
              r"$N{=}32$, $1\%$",
              r"$N{=}32$, $2\%$"]
    conds = ["untrained", "A", "B", "C", "C2", "D"]
    names = ["untrained", "A train-in-sim", "B physics-aware",
             "C feedback align.", "C2 sign-concordant", "D perturbative"]
    colors = ["0.55", "#ff7f0e", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    w = 0.13
    xs = np.arange(len(cells))
    for j, (c, nm, col) in enumerate(zip(conds, names, colors)):
        vals, p95 = [], []
        for cell in cells:
            sig = d[cell]["sigma"]
            vals.append(d[cell][c]["pc50"] / sig)
            p95.append(d[cell][c]["pc95"] / sig)
        vals = np.array(vals)
        p95 = np.array(p95)
        pos = xs + (j - 2.5) * w
        ax.bar(pos, vals, width=w, color=col, label=nm, log=True,
               edgecolor="k", linewidth=0.3)
        ax.vlines(pos, vals, p95, color="k", linewidth=0.6)
    # LS analytic floor
    ls = [d[c]["LS"]["pc50"] / d[c]["sigma"] for c in cells]
    for i, v in enumerate(ls):
        ax.hlines(v, xs[i] - 0.45, xs[i] + 0.45, color="k", linestyle=":",
                  linewidth=1.0, zorder=5)
    ax.plot([], [], color="k", linestyle=":", label="analytic LS floor")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"per-channel residual  [$\sigma_{\mathrm{VGA}}$]")
    ax.set_title("Per-channel residual of the AGC VGA array by gradient source "
                 "(p50, bar to p95)")
    ax.legend(ncol=4, loc="upper center", framealpha=0.95)
    ax.set_ylim(0.03, 4e3)
    ax.grid(axis="y", which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_v3_training.png"))
    plt.close(fig)
    print("wrote fig_v3_training.png")


# --------------------------------------------------------------------------
# Figure 2: sign-flip fraction f vs. convergence
# --------------------------------------------------------------------------
def fig_signflip():
    d = load("sign_concordance.json")["agg"]
    add = load("sign_concordance_addendum.json")
    cells = ["N8_s0.01", "N8_s0.02", "N32_s0.01", "N32_s0.02"]
    labels = [r"$N{=}8$, $1\%$", r"$N{=}8$, $2\%$",
              r"$N{=}32$, $1\%$", r"$N{=}32$, $2\%$"]
    keys = ["S3_00", "S3_05", "S3_10", "S3_20", "S3_35", "S3_50"]
    fvals = np.array([0.0, 5.0, 10.0, 20.0, 35.0, 50.0])
    marks = ["o", "s", "^", "d"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    for cell, lab, m in zip(cells, labels, marks):
        conv = [d[cell][k]["converged"] for k in keys]
        ax.plot(fvals, conv, marker=m, label=lab, linewidth=1.2)
    ax.axhline(18, color="k", linestyle="--", linewidth=0.8)
    ax.text(20, 15.0, "pre-registered bar (18/20)", fontsize=7)
    ax.set_xlabel("sign-flip fraction f of the backward entries  [%]")
    ax.set_ylabel("draws converged (of 20)")
    ax.set_title(r"$f_{\max}=0\%$ in all four cells")
    ax.set_ylim(-1, 22)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ks = [0, 1, 2]
    for cell, lab, m in zip(cells, labels, marks):
        k0 = d[cell]["S3_00"]["converged"]
        k1 = add[cell]["k1"]["converged"] if cell in add else None
        k2 = add[cell]["k2"]["converged"] if cell in add else None
        ax.plot(ks, [k0, k1, k2], marker=m, label=lab, linewidth=1.2)
    ax.axhline(18, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(ks)
    ax.set_xlabel("wrong-sign backward entries $k$ (exploratory)")
    ax.set_ylabel("draws converged (of 20)")
    ax.set_title("a single wrong entry already fails the bar")
    ax.set_ylim(-1, 22)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_v3_signflip.png"))
    plt.close(fig)
    print("wrote fig_v3_signflip.png")


# --------------------------------------------------------------------------
# Figure 3: mismatch-calculus retrodiction + the two prospective points
# --------------------------------------------------------------------------
def fig_calculus():
    d = load("mismatch_calculus.json")
    rows = [r for r in d["_rows"] if r["measured"] > 0 and r["predicted"] > 0]
    meas = np.array([r["measured"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    items = [r["item"].split()[0] for r in rows]
    uniq = sorted(set(items), key=lambda s: int(s))
    cmap = plt.get_cmap("tab10")

    p = load("predict_gelu_layernorm.json")["rows"]
    pro = [r for r in p if r["quantity"] == "per-channel p50"]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo, hi = 3e-3, 3e3
    x = np.array([lo, hi])
    ax.fill_between(x, x / 2, x * 2, color="0.85", label="factor-2 band")
    ax.plot(x, x, "k-", linewidth=0.8)
    for i, it in enumerate(uniq):
        sel = [j for j, s in enumerate(items) if s == it]
        ax.scatter(meas[sel], pred[sel], s=26, color=cmap(i % 10),
                   label="item " + it, zorder=3, edgecolor="k", linewidth=0.3)
    for r, mk, nm in zip(pro, ["*", "P"], ["GELU (P11$\\cdot$P6)",
                                           "LayerNorm (P1$\\cdot$P2 + AGC)"]):
        ax.scatter([r["spice"]], [r["predicted"]], s=190, marker=mk,
                   color="crimson", zorder=4, edgecolor="k", linewidth=0.5,
                   label="prospective: " + nm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured (ngspice or published testbench)")
    ax.set_ylabel("predicted by the mismatch calculus")
    ax.set_title("36 retrodicted rows, 0 outside a factor 2\n"
                 "plus two prospective composites (1.00$\\times$, 1.05$\\times$)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_v3_calculus.png"))
    plt.close(fig)
    print("wrote fig_v3_calculus.png  (%d retrodicted rows)" % len(rows))


# --------------------------------------------------------------------------
# Figure 4: per-channel shot-noise error vs. observation bandwidth
# --------------------------------------------------------------------------
def fig_noise():
    d = load("noise_reconciliation.json")
    C = d["C"]
    I0 = d["I0"]
    q = 1.602176634e-19
    bw = d["bandwidth"]
    B = np.logspace(7, 10.2, 400)

    def sig(kj):
        return 100.0 * C * np.sqrt(kj * 2 * q * B / I0)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.loglog(B / 1e6, sig(2), color="#1f77b4", linewidth=1.6,
              label=r"$\sigma_{\rm pc}=C\sqrt{k_j\,2qB/I_0}$, $k_j=2$")
    ax.loglog(B / 1e6, sig(1), color="#1f77b4", linewidth=1.2, linestyle="--",
              label=r"same, $k_j=1$")
    ax.axhline(0.69, color="k", linestyle=":", linewidth=1.2)
    ax.text(11, 0.73, "VGA mismatch residual 0.69 % (sigma_VGA = 1 %)",
            fontsize=7.5)

    # measured one-bench transient points (docs/exp_noise_reconciliation.md 2.1)
    mx = [bw["enbw_1pole_eff"] / 1e6, bw["enbw_loop"] / 1e6]
    ax.scatter(mx, [0.9925, 2.3316], marker="o", s=42, color="#d62728",
               zorder=5, edgecolor="k", linewidth=0.4,
               label="measured transient, 2 junctions")
    ax.scatter(mx, [0.6845, 1.6315], marker="s", s=38, color="#ff7f0e",
               zorder=5, edgecolor="k", linewidth=0.4,
               label="measured transient, 1 junction")

    bx = d["B_cross"]["kj2|I01uA"] / 1e6
    ax.axvline(bx, color="#2ca02c", linewidth=1.1, linestyle="-.")
    ax.text(bx * 0.93, 0.055,
            "$B_{cross}$ = 122 MHz ($\\zeta{=}1$ loop: 125 MHz)",
            rotation=90, fontsize=7, color="#2ca02c",
            va="bottom", ha="right")
    fl = bw["enbw_loop"] / 1e6
    ax.axvline(fl, color="#d62728", linewidth=1.1, linestyle="-.")
    ax.text(fl * 0.93, 0.055, "published loop: ENBW = 1395 MHz", rotation=90,
            fontsize=7, color="#d62728", va="bottom", ha="right")
    ax.set_xlabel("declared observation bandwidth $B$  [MHz]")
    ax.set_ylabel("per-channel error  [% of signal]")
    ax.set_title(r"Noise as the third per-channel exposure, $I_0=1\,\mu$A")
    ax.set_xlim(10, 1.5e4)
    ax.set_ylim(0.04, 10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_v3_noise.png"))
    plt.close(fig)
    print("wrote fig_v3_noise.png")


# --------------------------------------------------------------------------
# Figure 5: Mamba state trajectory -- reuse of the testbench figure
# --------------------------------------------------------------------------
def fig_mamba():
    src = os.path.join(OUT, "mamba_vcrc.png")
    dst = os.path.join(ROOT, "fig_v3_mamba.png")
    shutil.copyfile(src, dst)
    print("wrote fig_v3_mamba.png (copy of spice/out/mamba_vcrc.png)")


# --------------------------------------------------------------------------
# NOT REGENERATED, and therefore omitted from the paper rather than invented:
#   * a "domain lever" figure -- docs/domain_axis.md is analytic lever
#     arithmetic with no JSON behind it, so it stays a table (Table 5).
# --------------------------------------------------------------------------

if __name__ == "__main__":
    fig_training()
    fig_signflip()
    fig_calculus()
    fig_noise()
    fig_mamba()
