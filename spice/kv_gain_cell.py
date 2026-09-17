#!/usr/bin/env python3
"""SPICE side of the X2 KV-leak experiment: one gain-cell-like storage node.

Purpose: show that the discrete retention law used in `kv_leak_eviction.py`

    q_i <- q_i * exp(-1/tau) + kappa * a_{t,i},   live <=> q_i > theta

is what an RC storage node with charge-pulse refresh actually does, i.e. that
the discrete model is physically realisable.  This demonstrates realisability
only; it says nothing about whether the eviction policy is any good (that is
what the Python experiment measures).

Design (1 token = 1 us):
    C      = 10 fF                         storage node
    R_leak = tau / C                       tau = 128 token-times = 128 us
                                           -> R_leak = 1.28e10 ohm
    theta  = 100 mV                        eviction threshold
    kappa  = 4 * theta = 400 mV            charge delivered per unit attention
    refresh pulse: width 50 ns, amplitude I = kappa * C * a / width

Two cells are driven with real GPT-2 attention columns from the same trace:
one heavy hitter (large, sustained a_{t,i}) and one one-off token (written
once, then essentially ignored).  The heavy hitter stays above theta by
reinforcement alone; the one-off decays through theta = it is evicted.

Gotchas honoured (see spice/README.md): no junction is driven with an
externally computed voltage here, so the ngspice-VT issue does not apply; the
leak path is a linear resistor, so ngspice's junction GMIN does not add a
parasitic leakage term; chgtol is tightened because the node charges are ~fC.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

C_STORE = 10e-15          # F
TAU_TOKENS = 128.0        # token-times
T_TOKEN = 1e-6            # s per token
THETA_V = 0.100           # V
KAPPA_V = 0.400           # V per unit attention weight
PULSE_W = 50e-9           # s
R_LEAK = TAU_TOKENS * T_TOKEN / C_STORE


def get_attention_columns(n_tok, seed=2026):
    """Return (heavy_idx, heavy_col, oneoff_idx, oneoff_col) from a real trace.

    col[t] = attention weight a_{t,i} that query t pays to key i, for t >= i.
    Falls back to a synthetic Zipf/locality trace if the model is unavailable.
    """
    try:
        import torch
        sys.path.insert(0, str(HERE))
        import kv_leak_eviction as K
        from transformers import GPT2LMHeadModel
        torch.set_grad_enabled(False)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        W = K.GPT2Weights(m, dev, torch.float32)
        ids = K.load_sequences(1, n_tok, dev)
        _, li = K.forward_full(W, ids, want_layer_inputs=True)
        layer = 5
        q, k, v = K.qkv_of(W, W.blocks[layer], li[layer], W.n_head, W.head_dim)
        A, _ = K.full_attention(q, k, v, W.head_dim)
        A = A[0, 3].float().cpu().numpy()        # layer 5, head 3
        src = f"GPT-2 small, layer {layer}, head 3, wikitext-2"
    except Exception as e:                        # pragma: no cover
        sys.stderr.write(f"[warn] no model ({e}); synthetic trace\n")
        rng = np.random.default_rng(seed)
        import torch
        A = K_synth(n_tok, rng)
        src = "synthetic"

    recv = A.sum(axis=0)                          # total attention received
    # heavy hitter: strongest receiver that is not the token-0 sink (a sink is
    # too easy a win; picking the runner-up makes the demo honest)
    order = np.argsort(-recv)
    heavy = int(order[0]) if order[0] != 0 else int(order[1])
    # one-off: an early-but-not-sink token with the least received attention.
    # It must be born early enough that tau*ln(kappa/theta) token-times of
    # decay still fit inside the run, or the eviction cannot be observed.
    lo, hi = n_tok // 10, n_tok // 5
    mid = np.arange(lo, hi)
    oneoff = int(mid[np.argmin(recv[mid])])
    sink = 0
    cells = [("sk", sink, A[:, sink].copy(), "attention sink (token 0)"),
             ("hh", heavy, A[:, heavy].copy(), "heavy hitter (strongest non-sink)"),
             ("oo", oneoff, A[:, oneoff].copy(), "one-off token")]
    return cells, src


def K_synth(T, rng):
    A = np.zeros((T, T))
    for t in range(T):
        row = np.zeros(t + 1)
        row[: min(4, t + 1)] += 0.5 / min(4, t + 1)
        d = np.arange(t, -1, -1.0)
        loc = np.exp(-d / 8.0)
        row += 0.4 * loc / loc.sum()
        row += 0.1 / (t + 1)
        A[t, : t + 1] = row / row.sum()
    return A


def pwl_source(name, node, born, col, n_tok):
    """PWL current source: write pulse at t=born, refresh pulses afterwards."""
    pts = []
    for t in range(born, n_tok):
        a = float(col[t])
        if t == born:
            a = 1.0 + a       # write (q0 = kappa) plus the step's own refresh
        if a <= 0:
            continue
        amp = KAPPA_V * C_STORE * a / PULSE_W
        t0 = t * T_TOKEN
        pts += [(t0, 0.0), (t0 + 1e-12, amp), (t0 + PULSE_W, amp), (t0 + PULSE_W + 1e-12, 0.0)]
    body = " ".join(f"{t:.12g} {i:.9g}" for t, i in pts)
    lines, cur = [f"{name} 0 {node} pwl("], ""
    for tok in body.split(" "):
        if len(cur) + len(tok) > 200:
            lines.append("+ " + cur)
            cur = ""
        cur += tok + " "
    lines.append("+ " + cur + ")")
    return "\n".join(lines)


def build_netlist(cells, n_tok):
    t_end = n_tok * T_TOKEN
    L = [
        "* X2 KV gain cell: leaky storage node with attention-proportional refresh",
        f"* C={C_STORE:g}F  R_leak={R_LEAK:g}ohm  tau={TAU_TOKENS:g} token-times"
        f"  theta={THETA_V:g}V  kappa={KAPPA_V:g}V",
        ".options reltol=1e-4 chgtol=1e-18 vntol=1e-9 method=gear",
    ]
    for node, born, col, _ in cells:
        L += [f"C{node} {node} 0 {C_STORE:g} ic=0",
              f"R{node} {node} 0 {R_LEAK:g}",
              pwl_source(f"I{node}", node, born, col, n_tok)]
    vecs = " ".join(f"v({node})" for node, *_ in cells)
    L += [".control", f"tran 10n {t_end*1e6:g}u uic",
          f"wrdata kv_gain_cell.txt {vecs}", ".endc", ".end", ""]
    return "\n".join(L)


def run_ngspice(netlist, tag):
    OUT.mkdir(exist_ok=True)
    cir = OUT / f"{tag}.cir"
    cir.write_text(netlist)
    res = subprocess.run(["ngspice", "-b", str(cir)], capture_output=True,
                         text=True, timeout=1800, cwd=OUT)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"ngspice failed for {tag}")
    return res.stdout


def fit_tau(t, v, t0, t1):
    """Least-squares exponential fit of v over [t0,t1] -> measured tau (s)."""
    m = (t >= t0) & (t <= t1) & (v > 1e-6)
    if m.sum() < 10:
        return float("nan")
    slope, _ = np.polyfit(t[m], np.log(v[m]), 1)
    return -1.0 / slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tok", type=int, default=400)
    args = ap.parse_args()
    n = args.n_tok

    cells, src = get_attention_columns(n)
    print(f"[trace] {src}, T={n} tokens")
    for node, born, col, label in cells:
        print(f"[trace] {label:34s} i={born:3d}  received attention sum={col.sum():7.2f}  "
              f"mean a_(t,i) after birth={col[born:].mean():.3e}")

    net = build_netlist(cells, n)
    tau_des = R_LEAK * C_STORE
    print(f"[spice] C={C_STORE:.1e} F, R_leak={R_LEAK:.3e} ohm, "
          f"designed tau = R*C = {tau_des*1e6:.1f} us = {TAU_TOKENS:.0f} token-times; "
          f"theta={THETA_V*1e3:.0f} mV, kappa={KAPPA_V*1e3:.0f} mV")
    run_ngspice(net, "kv_gain_cell")

    d = np.loadtxt(OUT / "kv_gain_cell.txt")
    t = d[:, 0]
    V = {node: d[:, 2 * j + 1] for j, (node, *_rest) in enumerate(cells)}

    def crossing(v, born):
        i0 = int(np.searchsorted(t, (born + 1) * T_TOKEN))
        for i in range(i0, len(v)):
            if v[i] < THETA_V and v[i - 1] >= THETA_V:
                return t[i]
        return None

    # --- effective tau, measured on the one-off cell's free decay ------------
    oo_node, oo_born, oo_col, _ = cells[-1]
    tau_meas = fit_tau(t, V[oo_node], (oo_born + 5) * T_TOKEN, t[-1])
    print(f"[tau]   designed {tau_des*1e6:.2f} us   measured {tau_meas*1e6:.2f} us   "
          f"deviation {100*(tau_meas-tau_des)/tau_des:+.2f} %")
    pred = tau_des * math.log(KAPPA_V / THETA_V)
    print(f"[check] analytic decay-to-theta for a never-refreshed cell: "
          f"tau*ln(kappa/theta) = {pred*1e6:.1f} us = {pred/T_TOKEN:.0f} token-times")

    # --- retention per cell --------------------------------------------------
    res = {}
    for node, born, col, label in cells:
        v = V[node]
        i0 = int(np.searchsorted(t, (born + 1) * T_TOKEN))
        c = crossing(v, born)
        life = (c / T_TOKEN - born) if c else (n - born)
        res[node] = dict(born=born, label=label, peak=float(v.max()),
                         final=float(v[-1]), vmin=float(v[i0:].min()),
                         evicted_at=(float(c / T_TOKEN) if c else None), lifetime=float(life))
        print(f"[cell]  {label:34s} peak={v.max()*1e3:7.1f} mV  final={v[-1]*1e3:6.1f} mV  "
              + (f"evicted at token {c/T_TOKEN:.0f} ({life:.0f} token-times after write)"
                 if c else f"never evicted (alive >= {life:.0f} token-times)"))

    # --- discrete-model agreement -------------------------------------------
    dec = math.exp(-1.0 / TAU_TOKENS)
    worst = 0.0
    for node, born, col, label in cells:
        q, qs = 0.0, []
        for tt in range(born, n):
            q = q * dec + KAPPA_V * ((1.0 + col[tt]) if tt == born else col[tt])
            qs.append(q)
        qs = np.array(qs)
        samp = np.array([np.interp((born + j) * T_TOKEN + PULSE_W * 1.5, t, V[node])
                         for j in range(len(qs))])
        rel = float(np.abs(samp - qs).max() / max(qs.max(), 1e-12))
        worst = max(worst, rel)
        print(f"[model] {label:34s} SPICE node vs discrete difference equation: "
              f"max deviation {rel*100:.3f} % of peak")

    np.savez(OUT / "kv_gain_cell.npz", t=t, tau_meas=tau_meas, tau_des=tau_des,
             **{f"v_{k}": v for k, v in V.items()})
    import json
    (OUT / "kv_gain_cell.json").write_text(json.dumps(
        dict(tau_designed_us=tau_des * 1e6, tau_measured_us=float(tau_meas) * 1e6,
             tau_dev_pct=float(100 * (tau_meas - tau_des) / tau_des),
             theta_mV=THETA_V * 1e3, kappa_mV=KAPPA_V * 1e3, C_fF=C_STORE * 1e15,
             R_leak_ohm=R_LEAK, n_tok=n, model_max_dev_pct=worst * 100,
             cells=res), indent=1))
    print(f"[spice] wrote {OUT/'kv_gain_cell.json'}")


if __name__ == "__main__":
    main()
