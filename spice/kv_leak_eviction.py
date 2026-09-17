#!/usr/bin/env python3
"""X2 experiment: content-adaptive KV-cache eviction by leak alone.

Tests the claim in docs/missing_primes_mapping.md §10 (X2 row): a leaky analog
storage cell (gain cell, retention tau) whose charge is refreshed by every
attention read in proportion to that token's attention weight reproduces
heavy-hitter (H2O-style) KV eviction *by physics*, with no argmin and no
scoring logic.

Dynamics, one step per token, per cached token i:

    q_i <- q_i * exp(-1/tau) + kappa * a_{t,i}          (tau in token-times)
    token i is live  <=>  q_i > theta   (theta == 1 WLOG)

Two stages:

  A (`--stage a`)  oracle-trace sweep. Real GPT-2 attention matrices and value
      vectors; the policy state is driven by the attention weights the cache
      actually applies (renormalised over the live set), but the underlying
      scores come from a full-attention forward pass. Gives m1 (Jaccard with
      H2O) and m2 (attention-output error) over the full tau/kappa sweep.

  B (`--stage b`)  closed-loop perplexity. The cache policy actually drives the
      model forward pass: hidden states, and hence all later keys/values, are
      those produced under eviction. Gives m3.

  spice (`--stage spice`)  delegates to kv_gain_cell.py.

Everything is token-indexed rather than slot-indexed: a token occupies at most
one slot, so a per-token charge vector q[0..T-1] with the constraint
|{i : q_i > theta}| <= K is an exact re-encoding of K physical slots and makes
the whole sweep vectorisable over (config, sequence, head).

Usage:
    python3 kv_leak_eviction.py --stage a
    python3 kv_leak_eviction.py --stage b
    python3 kv_leak_eviction.py --check          # gauge anchor vs HuggingFace
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
SEED = 2026
THETA = 1.0


# ---------------------------------------------------------------------------
# GPT-2 weights + hand-rolled forward (full control over the attention step)
# ---------------------------------------------------------------------------

def gelu_new(x):
    import torch
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class GPT2Weights:
    """Flat container of GPT-2 tensors, device-resident."""

    def __init__(self, model, device, dtype):
        import torch
        sd = model.state_dict()
        g = lambda k: sd[k].to(device=device, dtype=dtype)
        self.wte = g("transformer.wte.weight")
        self.wpe = g("transformer.wpe.weight")
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.n_embd = model.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.blocks = []
        for i in range(self.n_layer):
            p = f"transformer.h.{i}."
            self.blocks.append(dict(
                ln1_w=g(p + "ln_1.weight"), ln1_b=g(p + "ln_1.bias"),
                qkv_w=g(p + "attn.c_attn.weight"), qkv_b=g(p + "attn.c_attn.bias"),
                proj_w=g(p + "attn.c_proj.weight"), proj_b=g(p + "attn.c_proj.bias"),
                ln2_w=g(p + "ln_2.weight"), ln2_b=g(p + "ln_2.bias"),
                fc_w=g(p + "mlp.c_fc.weight"), fc_b=g(p + "mlp.c_fc.bias"),
                mp_w=g(p + "mlp.c_proj.weight"), mp_b=g(p + "mlp.c_proj.bias"),
            ))
        self.lnf_w = g("transformer.ln_f.weight")
        self.lnf_b = g("transformer.ln_f.bias")


def layer_norm(x, w, b, eps=1e-5):
    import torch
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps)


def embed(W, ids):
    import torch
    T = ids.shape[1]
    pos = torch.arange(T, device=ids.device)
    return W.wte[ids] + W.wpe[pos][None]


def qkv_of(W, blk, h, n_head, head_dim):
    """h: (B,T,C) -> q,k,v each (B,n_head,T,head_dim)."""
    import torch
    x = layer_norm(h, blk["ln1_w"], blk["ln1_b"])
    qkv = x @ blk["qkv_w"] + blk["qkv_b"]
    B, T, _ = x.shape
    q, k, v = qkv.split(n_head * head_dim, dim=2)
    shape = (B, T, n_head, head_dim)
    q = q.view(shape).transpose(1, 2)
    k = k.view(shape).transpose(1, 2)
    v = v.view(shape).transpose(1, 2)
    return q, k, v


def attn_out_to_residual(W, blk, ctx, h):
    """ctx: (B,n_head,T,head_dim) -> residual add."""
    B, H, T, D = ctx.shape
    merged = ctx.transpose(1, 2).reshape(B, T, H * D)
    return h + (merged @ blk["proj_w"] + blk["proj_b"])


def mlp_block(blk, h):
    x = layer_norm(h, blk["ln2_w"], blk["ln2_b"])
    x = gelu_new(x @ blk["fc_w"] + blk["fc_b"])
    return h + (x @ blk["mp_w"] + blk["mp_b"])


def full_attention(q, k, v, head_dim):
    import torch
    T = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    A = torch.softmax(scores, dim=-1)
    return A, A @ v


def forward_full(W, ids, want_layer_inputs=False):
    """Standard full-attention forward. Returns logits (+ per-layer inputs)."""
    h = embed(W, ids)
    layer_inputs = []
    for blk in W.blocks:
        if want_layer_inputs:
            layer_inputs.append(h)
        q, k, v = qkv_of(W, blk, h, W.n_head, W.head_dim)
        _, ctx = full_attention(q, k, v, W.head_dim)
        h = attn_out_to_residual(W, blk, ctx, h)
        h = mlp_block(blk, h)
    h = layer_norm(h, W.lnf_w, W.lnf_b)
    logits = h @ W.wte.T
    return (logits, layer_inputs) if want_layer_inputs else logits


# ---------------------------------------------------------------------------
# Policy simulators (token-indexed, vectorised over config x row)
# ---------------------------------------------------------------------------
#
# Every simulator consumes, per step t, a row of raw attention weights
# a[.., t, :] (already softmaxed over the causal prefix) and produces the
# renormalised cache weights w with support on the live set.  Masked softmax
# over a subset S is exactly a_i / sum_{j in S} a_j, so this is not an
# approximation of a masked forward pass -- it *is* one.


def _renorm(a_row, live):
    """a_row: (...,T); live: (...,T) bool -> normalised weights over live set."""
    import torch
    w = torch.where(live, a_row, torch.zeros_like(a_row))
    z = w.sum(-1, keepdim=True).clamp_min(1e-30)
    return w / z


def run_fifo(A, V, o_full, K, t0, n_sink=0):
    """Sliding window of the last K tokens (n_sink>0: StreamingLLM, i.e. the
    first n_sink tokens are pinned and the window is K-n_sink)."""
    import torch
    N, T, _ = A.shape
    err_sum, err_cnt = 0.0, 0
    live_hist = torch.zeros((N, T, T), dtype=torch.bool, device=A.device)
    idx = torch.arange(T, device=A.device)
    Kw = K - n_sink
    for t in range(T):
        live = (idx <= t) & (idx > t - Kw)
        if n_sink:
            live = live | (idx < n_sink)
        live = live[None].expand(N, T)
        live_hist[:, t] = live
        if t >= t0:
            w = _renorm(A[:, t], live)
            o = torch.einsum("nt,ntd->nd", w, V)
            d = torch.linalg.norm(o - o_full[:, t], dim=-1) / torch.linalg.norm(o_full[:, t], dim=-1).clamp_min(1e-30)
            err_sum += d.sum().item()
            err_cnt += d.numel()
    return err_sum / max(err_cnt, 1), live_hist


def run_h2o_sweep(A, V, o_full, K, t0, ratios=(0.25, 0.5, 0.75)):
    """Tune H2O's heavy/recent split so that K2 compares against the *best* H2O."""
    best = None
    table = {}
    for hr in ratios:
        m2, live = run_h2o(A, V, o_full, K, t0, heavy_ratio=hr)
        table[hr] = m2
        if best is None or m2 < best[0]:
            best = (m2, live, hr)
    return best[0], best[1], best[2], table


def run_h2o(A, V, o_full, K, t0, heavy_ratio=0.5):
    """H2O: recent window of K/2 + greedy heavy-hitter retention.

    Eviction is permanent (H2O's actual algorithm): when the cache is full the
    live, non-recent token with the smallest accumulated attention is dropped.
    """
    import torch
    N, T, _ = A.shape
    K_rec = max(1, int(round(K * (1.0 - heavy_ratio))))
    live = torch.zeros((N, T), dtype=torch.bool, device=A.device)
    acc = torch.zeros((N, T), device=A.device)
    live_hist = torch.zeros((N, T, T), dtype=torch.bool, device=A.device)
    idx = torch.arange(T, device=A.device)
    err_sum, err_cnt = 0.0, 0
    inf = torch.tensor(float("inf"), device=A.device)
    for t in range(T):
        live[:, t] = True
        n = live.sum(-1)
        full = n > K
        if full.any():
            recent = (idx <= t) & (idx > t - K_rec)
            elig = live & ~recent[None]
            score = torch.where(elig, acc, inf)
            victim = score.argmin(dim=-1)
            kill = full & elig.any(-1)
            if kill.any():
                rows = torch.nonzero(kill, as_tuple=True)[0]
                live[rows, victim[rows]] = False
        live_hist[:, t] = live
        w = _renorm(A[:, t], live)
        acc = acc + w
        if t >= t0:
            o = torch.einsum("nt,ntd->nd", w, V)
            d = torch.linalg.norm(o - o_full[:, t], dim=-1) / torch.linalg.norm(o_full[:, t], dim=-1).clamp_min(1e-30)
            err_sum += d.sum().item()
            err_cnt += d.numel()
    return err_sum / max(err_cnt, 1), live_hist


def run_leak(A, V, o_full, K, t0, taus, kappas, variants, ref_live=None):
    """Leaky-cell policy, swept over (tau, kappa, variant) in one batched loop.

    variants: list of 'argmin' (P5 race on the fullest cache) or 'drop'
              (no comparison at all -- new token is simply not cached).
    ref_live: (N,T,T) bool live masks of the reference policy (H2O) for m1.
    Returns (configs, m1, m2, mean_live_fraction).
    """
    import torch
    dev = A.device
    N, T, _ = A.shape
    configs = [(tau, kap, var) for tau in taus for kap in kappas for var in variants]
    C = len(configs)
    d_vec = torch.tensor([math.exp(-1.0 / c[0]) for c in configs], device=dev).view(C, 1, 1)
    k_vec = torch.tensor([float(c[1]) for c in configs], device=dev).view(C, 1, 1)
    is_argmin = torch.tensor([c[2] == "argmin" for c in configs], device=dev).view(C, 1)

    q = torch.zeros((C, N, T), device=dev)
    inf = torch.tensor(float("inf"), device=dev)
    err_sum = torch.zeros(C, device=dev)
    jac_sum = torch.zeros(C, device=dev)
    livefrac_sum = torch.zeros(C, device=dev)
    cnt = 0
    of = o_full  # (N,T,D)
    for t in range(T):
        q = q * d_vec
        live = q > THETA
        q = torch.where(live, q, torch.zeros_like(q))  # dead cells hold no charge
        n = live.sum(-1)                                # (C,N)
        full = n >= K
        # --- write token t -------------------------------------------------
        # argmin variant: if full, evict the minimum-charge cell (one P5 race)
        need_evict = full & is_argmin
        if need_evict.any():
            score = torch.where(live, q, inf)
            victim = score.argmin(dim=-1)               # (C,N)
            ci, ni = torch.nonzero(need_evict, as_tuple=True)
            live[ci, ni, victim[ci, ni]] = False
            q[ci, ni, victim[ci, ni]] = 0.0
        write = (~full) | is_argmin                     # drop variant skips when full
        q[:, :, t] = torch.where(write, k_vec.view(C, 1).expand(C, N), q[:, :, t])
        live[:, :, t] |= write
        # --- read ----------------------------------------------------------
        a_row = A[:, t].unsqueeze(0)                    # (1,N,T)
        w = _renorm(a_row.expand(C, N, T), live)
        # --- refresh -------------------------------------------------------
        q = q + k_vec * w
        if t >= t0:
            o = torch.einsum("cnt,ntd->cnd", w, V)
            err = torch.linalg.norm(o - of[:, t][None], dim=-1) / torch.linalg.norm(of[:, t], dim=-1).clamp_min(1e-30)[None]
            err_sum += err.sum(-1)
            livefrac_sum += live.sum(-1).float().sum(-1) / (K * N)
            if ref_live is not None:
                r = ref_live[:, t][None]
                inter = (live & r).sum(-1).float()
                union = (live | r).sum(-1).float().clamp_min(1.0)
                jac_sum += (inter / union).sum(-1)
            cnt += 1
    denom = max(cnt, 1) * N
    return configs, (jac_sum / denom).cpu().numpy(), (err_sum / denom).cpu().numpy(), (livefrac_sum / max(cnt, 1)).cpu().numpy()


# ---------------------------------------------------------------------------
# Closed-loop forward pass with an eviction policy (stage B / m3)
# ---------------------------------------------------------------------------

def forward_policy(W, ids, policy, K, heavy_ratio=0.5, tau=None, kappa=None,
                   variant="argmin", n_sink=0):
    """Full forward where every layer's attention runs under the cache policy.

    The policy state is fed by the *masked* attention the cache actually
    produced, and the evicted hidden states propagate, so this is the real
    closed-loop behaviour, not an oracle-trace replay.
    """
    import torch
    dev = ids.device
    B, T = ids.shape
    H, D = W.n_head, W.head_dim
    idx = torch.arange(T, device=dev)
    inf = torch.tensor(float("inf"), device=dev)
    h = embed(W, ids)
    for blk in W.blocks:
        q_, k_, v_ = qkv_of(W, blk, h, H, D)
        N = B * H
        qf = q_.reshape(N, T, D)
        kf = k_.reshape(N, T, D)
        vf = v_.reshape(N, T, D)
        ctx = torch.empty((N, T, D), device=dev, dtype=qf.dtype)
        if policy == "full":
            A, c = full_attention(q_, k_, v_, D)
            ctx = c.reshape(N, T, D)
        else:
            live = torch.zeros((N, T), dtype=torch.bool, device=dev)
            acc = torch.zeros((N, T), device=dev)
            qq = torch.zeros((N, T), device=dev)
            decay = math.exp(-1.0 / tau) if tau else 0.0
            K_rec = max(1, int(round(K * (1.0 - heavy_ratio))))
            for t in range(T):
                if policy == "fifo":
                    lv = (idx <= t) & (idx > t - (K - n_sink))
                    if n_sink:
                        lv = lv | (idx < n_sink)
                    live = lv[None].expand(N, T).clone()
                elif policy == "h2o":
                    live[:, t] = True
                    full_c = live.sum(-1) > K
                    if full_c.any():
                        recent = (idx <= t) & (idx > t - K_rec)
                        elig = live & ~recent[None]
                        victim = torch.where(elig, acc, inf).argmin(dim=-1)
                        kill = full_c & elig.any(-1)
                        rows = torch.nonzero(kill, as_tuple=True)[0]
                        if rows.numel():
                            live[rows, victim[rows]] = False
                elif policy == "leak":
                    qq = qq * decay
                    live = qq > THETA
                    qq = torch.where(live, qq, torch.zeros_like(qq))
                    full_c = live.sum(-1) >= K
                    if variant == "argmin" and full_c.any():
                        victim = torch.where(live, qq, inf).argmin(dim=-1)
                        rows = torch.nonzero(full_c, as_tuple=True)[0]
                        live[rows, victim[rows]] = False
                        qq[rows, victim[rows]] = 0.0
                    wr = torch.ones(N, dtype=torch.bool, device=dev) if variant == "argmin" else ~full_c
                    qq[:, t] = torch.where(wr, torch.full_like(qq[:, t], kappa), qq[:, t])
                    live[:, t] |= wr
                else:
                    raise ValueError(policy)
                s = (qf[:, t : t + 1] @ kf.transpose(-1, -2)).squeeze(1) / math.sqrt(D)
                s = s.masked_fill(~live | (idx > t)[None], float("-inf"))
                w = torch.softmax(s, dim=-1)
                if policy == "h2o":
                    acc = acc + w
                elif policy == "leak":
                    qq = qq + kappa * w
                ctx[:, t] = torch.einsum("nt,ntd->nd", w, vf)
        ctx = ctx.view(B, H, T, D)
        h = attn_out_to_residual(W, blk, ctx, h)
        h = mlp_block(blk, h)
    h = layer_norm(h, W.lnf_w, W.lnf_b)
    return h @ W.wte.T


def perplexity(logits, ids):
    import torch
    lp = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        ids[:, 1:].reshape(-1), reduction="mean")
    return float(torch.exp(lp))


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

def load_sequences(n_seq, T, device):
    """n_seq contiguous non-overlapping wikitext-2 chunks of T tokens."""
    import torch
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    text = None
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as e:
        sys.stderr.write(f"[warn] wikitext unavailable ({e}); falling back to repo text\n")
        src = sorted(Path(__file__).resolve().parents[1].glob("docs/*.md"))
        text = "\n\n".join(p.read_text() for p in src)
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_seq * T
    if ids.numel() < need:
        reps = need // ids.numel() + 1
        ids = ids.repeat(reps)
    ids = ids[:need].view(n_seq, T)
    return ids.to(device)


def synthetic_traces(n_seq, n_head, T, device, rng):
    """Zipf sinks + exponential locality + uniform noise, row-normalised."""
    import torch
    S, H = n_seq, n_head
    A = np.zeros((S, H, T, T), dtype=np.float32)
    n_sink = 4
    zipf = 1.0 / (np.arange(1, n_sink + 1) ** 1.2)
    zipf /= zipf.sum()
    for s in range(S):
        for hh in range(H):
            wsink, wloc, wnoise = rng.dirichlet([2.0, 2.0, 0.5])
            lam = rng.uniform(2.0, 16.0)
            for t in range(T):
                row = np.zeros(t + 1, dtype=np.float64)
                row[: min(n_sink, t + 1)] += wsink * zipf[: min(n_sink, t + 1)]
                d = np.arange(t, -1, -1, dtype=np.float64)
                loc = np.exp(-d / lam)
                row += wloc * loc / loc.sum()
                row += wnoise / (t + 1)
                A[s, hh, t, : t + 1] = row / row.sum()
    V = rng.standard_normal((S, H, T, 64)).astype(np.float32)
    return torch.from_numpy(A).to(device), torch.from_numpy(V).to(device)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def gauge_check(device):
    """Anchor: hand-rolled forward must match HuggingFace's own GPT-2."""
    import torch
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    W = GPT2Weights(m, device, torch.float32)
    ids = torch.randint(0, 50257, (2, 64), generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.no_grad():
        mine = forward_full(W, ids)
        theirs = m.to(device)(ids).logits
    err = (mine - theirs).abs().max().item() / theirs.abs().max().item()
    print(f"[gauge] max relative logit deviation hand-rolled vs HuggingFace: {err:.3e}")
    assert err < 1e-4, "hand-rolled GPT-2 forward does not reproduce HuggingFace"
    return err


def stage_a(args):
    import torch
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    rng = np.random.default_rng(SEED)
    T = args.T
    taus = [8, 32, 128, 512]
    kappas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    variants = ["argmin", "drop"]
    ratios = [0.1, 0.2, 0.4]

    if args.synthetic:
        source = "synthetic"
        A_all, V_all = synthetic_traces(args.n_seq, args.n_head_syn, T, dev, rng)
        layers = [0]
        get_layer = lambda l: (A_all, V_all)
    else:
        source = "gpt2/wikitext-2"
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
        W = GPT2Weights(m, dev, torch.float32)
        ids = load_sequences(args.n_seq, T, dev)
        _, layer_inputs = forward_full(W, ids, want_layer_inputs=True)
        layers = args.layers if args.layers else list(range(W.n_layer))

        def get_layer(l):
            q, k, v = qkv_of(W, W.blocks[l], layer_inputs[l], W.n_head, W.head_dim)
            A, _ = full_attention(q, k, v, W.head_dim)
            return A, v

    results = []
    t_start = time.time()
    for l in layers:
        A4, V4 = get_layer(l)
        S, H = A4.shape[0], A4.shape[1]
        N = S * H
        A = A4.reshape(N, T, T).contiguous()
        V = V4.reshape(N, T, V4.shape[-1]).contiguous()
        o_full = torch.einsum("nts,nsd->ntd", A, V)
        for r in ratios:
            K = int(round(r * T))
            t0 = K  # metrics only where the cache is actually constrained
            h2o_m2, h2o_live, h2o_hr, h2o_tab = run_h2o_sweep(A, V, o_full, K, t0)
            fifo_m2, _ = run_fifo(A, V, o_full, K, t0)
            stream_m2, _ = run_fifo(A, V, o_full, K, t0, n_sink=4)
            cfgs, m1, m2, lf = run_leak(A, V, o_full, K, t0, taus, kappas, variants, ref_live=h2o_live)
            results.append(dict(layer=int(l), ratio=r, K=K, h2o_m2=float(h2o_m2),
                                h2o_heavy_ratio=float(h2o_hr),
                                h2o_table={str(k): float(v) for k, v in h2o_tab.items()},
                                fifo_m2=float(fifo_m2), stream_m2=float(stream_m2),
                                leak=[dict(tau=c[0], kappa=c[1], variant=c[2],
                                           m1=float(m1[i]), m2=float(m2[i]),
                                           livefrac=float(lf[i]))
                                      for i, c in enumerate(cfgs)]))
            print(f"  layer {l:2d} K/T={r:.1f} K={K:3d}  H2O m2={h2o_m2:.4f}  FIFO m2={fifo_m2:.4f}"
                  f"  FIFO+sink m2={stream_m2:.4f}"
                  f"  best-leak m2={m2.min():.4f}  ({time.time()-t_start:.0f}s)", flush=True)
            del h2o_live
        del A, V, o_full, A4, V4
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    OUT.mkdir(exist_ok=True)
    payload = dict(source=source, n_seq=args.n_seq, T=T, layers=[int(x) for x in layers],
                   taus=taus, kappas=kappas, results=results)
    (OUT / "kv_leak_stage_a.json").write_text(json.dumps(payload))
    print(f"[stage A] wrote {OUT/'kv_leak_stage_a.json'} ({time.time()-t_start:.0f}s)")
    summarise_a(payload)


def summarise_a(payload):
    """Aggregate over layers: mean m1/m2 per (ratio, config)."""
    ratios = sorted({r["ratio"] for r in payload["results"]})
    print("\n=== Stage A summary (mean over layers, sequences, heads, steps t>=K) ===")
    best = {}
    for r in ratios:
        rows = [x for x in payload["results"] if x["ratio"] == r]
        h2o = float(np.mean([x["h2o_m2"] for x in rows]))
        fifo = float(np.mean([x["fifo_m2"] for x in rows]))
        stream = float(np.mean([x.get("stream_m2", float("nan")) for x in rows]))
        ncfg = len(rows[0]["leak"])
        agg = []
        for i in range(ncfg):
            c = rows[0]["leak"][i]
            agg.append(dict(tau=c["tau"], kappa=c["kappa"], variant=c["variant"],
                            m1=float(np.mean([x["leak"][i]["m1"] for x in rows])),
                            m2=float(np.mean([x["leak"][i]["m2"] for x in rows])),
                            livefrac=float(np.mean([x["leak"][i]["livefrac"] for x in rows]))))
        agg.sort(key=lambda d: d["m2"])
        best[r] = dict(h2o=h2o, fifo=fifo, stream=stream, table=agg)
        b = agg[0]
        print(f"K/T={r}: FIFO m2={fifo:.4f}  FIFO+sink m2={stream:.4f}  H2O m2={h2o:.4f}  "
              f"best leak m2={b['m2']:.4f} (tau={b['tau']}, kappa={b['kappa']}, {b['variant']}, "
              f"m1={b['m1']:.3f}, live={b['livefrac']:.2f})")
        ba = [d for d in agg if d["variant"] == "argmin"][0]
        bd = [d for d in agg if d["variant"] == "drop"][0]
        print(f"        best argmin m2={ba['m2']:.4f} (tau={ba['tau']},k={ba['kappa']})   "
              f"best drop m2={bd['m2']:.4f} (tau={bd['tau']},k={bd['kappa']})   "
              f"drop/argmin = {bd['m2']/ba['m2']:.3f}")
        # per-tau best (K3: does one tau serve every budget?)
        pertau = {}
        for tau in payload["taus"]:
            sub = [d for d in agg if d["tau"] == tau and d["variant"] == "argmin"]
            pertau[tau] = min(sub, key=lambda d: d["m2"]) if sub else None
        best[r]["per_tau_argmin"] = pertau
        best[r]["best_argmin"], best[r]["best_drop"] = ba, bd
        print("        per-tau best (argmin): " +
              "  ".join(f"tau={t}:{pertau[t]['m2']:.4f}(k={pertau[t]['kappa']})" for t in payload["taus"]))
    # K3: is there ONE (tau, kappa) that is near-best at every budget?
    print("\n--- K3: cross-budget transferability (argmin variant) ---")
    keys = [(d["tau"], d["kappa"]) for d in best[ratios[0]]["table"] if d["variant"] == "argmin"]
    lut = {r: {(d["tau"], d["kappa"]): d["m2"] for d in best[r]["table"] if d["variant"] == "argmin"}
           for r in ratios}
    bestper = {r: min(lut[r].values()) for r in ratios}
    scored = sorted(keys, key=lambda k: max(lut[r][k] / bestper[r] for r in ratios))
    for k in scored[:3]:
        exc = {r: lut[r][k] / bestper[r] - 1.0 for r in ratios}
        print(f"  tau={k[0]:3d} kappa={k[1]:<5g}  excess over per-budget best: " +
              "  ".join(f"K/T={r}: {100*exc[r]:+.1f}%" for r in ratios))
    # single tau, kappa free per budget
    print("  single tau, kappa retuned per budget:")
    for tau in payload["taus"]:
        exc = {r: min(v for (t_, k_), v in lut[r].items() if t_ == tau) / bestper[r] - 1.0 for r in ratios}
        print(f"    tau={tau:3d}: " + "  ".join(f"K/T={r}: {100*exc[r]:+.1f}%" for r in ratios))
    best["k3_single_config"] = [dict(tau=k[0], kappa=k[1],
                                     excess={str(r): lut[r][k] / bestper[r] - 1.0 for r in ratios})
                                for k in scored[:3]]
    (OUT / "kv_leak_stage_a_summary.json").write_text(json.dumps(best, indent=1))
    # emit the stage-B config file: best leak config per budget, plus baselines
    cfgb = {}
    uni = best["k3_single_config"][0]
    for r in ratios:
        ba, bd = best[r]["best_argmin"], best[r]["best_drop"]
        hr = float(np.mean([x["h2o_heavy_ratio"] for x in payload["results"] if x["ratio"] == r]))
        cfgb[str(r)] = {
            "leak_argmin": dict(policy="leak", tau=ba["tau"], kappa=ba["kappa"], variant="argmin"),
            "leak_drop": dict(policy="leak", tau=bd["tau"], kappa=bd["kappa"], variant="drop"),
            "leak_universal": dict(policy="leak", tau=uni["tau"], kappa=uni["kappa"], variant="argmin"),
            "h2o": dict(policy="h2o", heavy_ratio=0.5 if hr not in (0.25, 0.75) else hr),
            "fifo": dict(policy="fifo"),
            "fifo_sink4": dict(policy="fifo", n_sink=4),
        }
    (OUT / "kv_leak_best.json").write_text(json.dumps(cfgb, indent=1))
    return best


def stage_b(args):
    import torch
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    W = GPT2Weights(m, dev, torch.float32)
    T = args.T
    ids = load_sequences(args.n_seq_b, T, dev)
    cfg = json.loads((OUT / "kv_leak_best.json").read_text())
    out = {}
    t0 = time.time()
    lg = forward_full(W, ids)
    out["full"] = perplexity(lg, ids)
    print(f"  full cache: ppl={out['full']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for r_s, spec in cfg.items():
        r = float(r_s)
        K = int(round(r * T))
        for name, kw in spec.items():
            lg = forward_policy(W, ids, K=K, **kw)
            out[f"{name}@{r}"] = perplexity(lg, ids)
            print(f"  {name:14s} K/T={r}: ppl={out[f'{name}@{r}']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    (OUT / "kv_leak_stage_b.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--n-seq-b", type=int, default=50)
    ap.add_argument("--n-head-syn", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        torch.set_num_threads(min(32, os.cpu_count()))
    if args.check:
        gauge_check(torch.device(args.device))
        return
    if args.stage == "a":
        stage_a(args)
    else:
        stage_b(args)


if __name__ == "__main__":
    main()
