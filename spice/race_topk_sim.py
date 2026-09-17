#!/usr/bin/env python3
"""Top-k set/order accuracy of the stochastic exponential race (sMTJ physics)
versus the deterministic race, and the K-race cost of an O(k) primacy readout.

Experiment for docs/exp_race_topk.md.  Pre-registration lives in that file's §0 and
was written before this script was run.

Model (paper: multi_computation_v2.tex, Op. 1-3, Op. 7):
    stochastic:    T_i ~ Exp(r_i),  r_i = exp(beta * z_i)      -> Plackett-Luce order
    deterministic: t_i = exp(-beta * z_i)                      -> exact order

Usage:
    source <your-venv>/bin/activate   # ROCm-enabled PyTorch
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python spice/race_topk_sim.py --out results.json
"""

import argparse
import itertools
import json
import math
import time

import torch

# ----------------------------------------------------------------------------
# pre-registered grid (docs/exp_race_topk.md §0.3)
# ----------------------------------------------------------------------------
N_GRID = [32, 256, 1024, 32768]
ENSEMBLES = ["gauss1", "gauss2", "gauss4", "peaked", "hard"]
K_GRID = [1, 2, 5, 10, 30]
KMAX = 30
BETA = 1.0
K_VOTE_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
R_RESAMPLE = 128
SEED0 = 20260916

DTYPE = torch.float64


def n_configs(N):
    return 16 if N >= 32768 else 64


def n_trials(N):
    return 100_000


def chunk_size(N):
    return max(256, min(8192, int(400e6 / (8 * N))))


# ----------------------------------------------------------------------------
# logit ensembles (§0.3).  Indices are randomly permuted for peaked/hard so that
# all ensembles are index-exchangeable (required by the MI identity, §0.6).
# ----------------------------------------------------------------------------
def make_logits(ens, N, M, gen, device):
    if ens.startswith("gauss"):
        sigma = float(ens[5:])
        return torch.randn(M, N, generator=gen, device=device, dtype=DTYPE) * sigma
    if ens == "peaked":
        z = torch.randn(M, N, generator=gen, device=device, dtype=DTYPE) * 1.0
        m = 10
        head = 4.0 - torch.log(
            torch.arange(1, m + 1, device=device, dtype=DTYPE)
        )  # z_(j) = 4 - ln j
        z[:, :m] = head.unsqueeze(0)
    elif ens == "hard":
        z = torch.randn(M, N, generator=gen, device=device, dtype=DTYPE) * 1.0 - 4.0
        m = 30
        head = -0.1 * torch.arange(0, m, device=device, dtype=DTYPE)  # 0.1 nat spacing
        z[:, :m] = head.unsqueeze(0)
    else:
        raise ValueError(ens)
    # random index permutation per configuration
    perm = torch.argsort(torch.rand(M, N, generator=gen, device=device), dim=1)
    return torch.gather(z, 1, perm)


# ----------------------------------------------------------------------------
# metric helpers
# ----------------------------------------------------------------------------
def kendall_tau_from_ranks(ranks_k):
    """ranks_k: (T,k) int64, the true ranks of the k elements in firing order.
    Returns mean tau-b (== tau-a, all values distinct) over T."""
    T, k = ranks_k.shape
    if k < 2:
        return float("nan")
    a = ranks_k.unsqueeze(2)  # (T,k,1)
    b = ranks_k.unsqueeze(1)  # (T,1,k)
    lt = (a < b)
    iu = torch.triu(torch.ones(k, k, dtype=torch.bool, device=ranks_k.device), 1)
    conc = (lt & iu).sum(dim=(1, 2)).to(DTYPE)
    npairs = k * (k - 1) / 2
    tau = 2.0 * conc / npairs - 1.0
    return tau.mean().item()


def analytic_pl(r_sorted_desc, k):
    """Exact Plackett-Luce P(order) and P(set) for the top-k, from the full rate
    vector sorted descending (python floats, float64).  Anchor A3."""
    Z = float(sum(r_sorted_desc))
    top = [float(x) for x in r_sorted_desc[:k]]
    # P(order): the top-k in exactly the true order
    p_ord, rem = 1.0, Z
    for j in range(k):
        p_ord *= top[j] / rem
        rem -= top[j]
    # P(set): sum over all k! orderings of the same k elements
    p_set = 0.0
    for perm in itertools.permutations(range(k)):
        p, rem = 1.0, Z
        for j in perm:
            p *= top[j] / rem
            rem -= top[j]
        p_set += p
    return p_ord, p_set


# ----------------------------------------------------------------------------
# core: simulate one (N, ensemble) cell
# ----------------------------------------------------------------------------
def run_cell(N, ens, device, verbose=True):
    M, T, B = n_configs(N), n_trials(N), chunk_size(N)
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED0 + N * 131 + ENSEMBLES.index(ens))

    z = make_logits(ens, N, M, gen, device)  # (M,N)
    r = torch.exp(BETA * z)
    Zpart = r.sum(dim=1)  # (M,)
    softmax_max = (r.max(dim=1).values / Zpart)  # (M,) anchor A1
    # true descending order of z -> rank_of[element] = its true rank (0 = best)
    true_order = torch.argsort(z, dim=1, descending=True)  # (M,N)
    rank_of = torch.empty_like(true_order)
    rank_of.scatter_(
        1, true_order, torch.arange(N, device=device).unsqueeze(0).expand(M, N)
    )

    # ---- anchor A2: deterministic race ----------------------------------
    t_det = torch.exp(-BETA * z)
    det_order = torch.argsort(t_det, dim=1)  # ascending time
    det_ranks = torch.gather(rank_of, 1, det_order[:, :KMAX])  # (M,KMAX)
    tgt = torch.arange(KMAX, device=device).unsqueeze(0)
    det_exact = bool((det_ranks == tgt).all().item())

    # ---- stochastic races -----------------------------------------------
    all_ranks = torch.empty(M, T, KMAX, dtype=torch.int32, device=device)
    nll_sum = torch.zeros(M, KMAX, dtype=DTYPE, device=device)  # sum of -ln P(prefix_k)
    for m in range(M):
        rm = r[m]
        Zm = Zpart[m]
        done = 0
        while done < T:
            b = min(B, T - done)
            E = torch.empty(b, N, dtype=DTYPE, device=device).exponential_(
                generator=gen
            )
            Tt = E / rm.unsqueeze(0)
            idx = torch.topk(Tt, KMAX, dim=1, largest=False, sorted=True).indices
            all_ranks[m, done : done + b] = torch.gather(
                rank_of[m].unsqueeze(0).expand(b, N), 1, idx
            ).to(torch.int32)
            # Plackett-Luce log-likelihood of the observed prefix (§0.6)
            rsel = rm[idx]  # (b,KMAX)
            cum = torch.cumsum(rsel, dim=1)
            denom = Zm - (cum - rsel)
            logp = torch.log(rsel / denom)
            nll_sum[m] += (-torch.cumsum(logp, dim=1)).sum(dim=0)
            done += b
            del E, Tt
    cond_entropy_nats = nll_sum / T  # (M,KMAX) H(Pi_1:k | z) in nats, MC-unbiased

    # ---- per-trial metrics ------------------------------------------------
    res = {}
    ranks64 = all_ranks.to(torch.int64)
    for k in K_GRID:
        if k > N:
            continue
        rk = ranks64[:, :, :k]  # (M,T,k)
        inter = (rk < k).sum(dim=2).to(DTYPE)  # (M,T) intersection size
        set_ok = (rk.max(dim=2).values == k - 1)  # distinct ranks -> set match
        ord_ok = (rk == torch.arange(k, device=device).view(1, 1, k)).all(dim=2)
        jac = inter / (2 * k - inter)
        p_set_cfg = set_ok.to(DTYPE).mean(dim=1)  # (M,)
        tau = (
            float("nan")
            if k < 2
            else sum(kendall_tau_from_ranks(ranks64[m, :, :k]) for m in range(M)) / M
        )
        res[k] = {
            "P_set": p_set_cfg.mean().item(),
            "P_set_std_cfg": p_set_cfg.std().item(),
            "P_order": ord_ok.to(DTYPE).mean().item(),
            "Jaccard": jac.mean().item(),
            "tau": tau,
            "inter_mean": inter.mean().item(),
        }

    # ---- anchor A1 ---------------------------------------------------------
    p_top1_mc = res[1]["P_set"]
    p_top1_th = softmax_max.mean().item()
    a1_rel = abs(p_top1_mc - p_top1_th) / p_top1_th

    # ---- anchor A3 (config 0, k<=5) ---------------------------------------
    a3 = []
    r0 = torch.sort(r[0], descending=True).values.cpu().tolist()
    for k in [1, 2, 5]:
        if k > N:
            continue
        p_ord_th, p_set_th = analytic_pl(r0, k)
        rk = ranks64[0, :, :k]
        p_set_mc = (rk.max(dim=1).values == k - 1).to(DTYPE).mean().item()
        p_ord_mc = (
            (rk == torch.arange(k, device=device).view(1, k)).all(dim=1).to(DTYPE).mean().item()
        )
        se = math.sqrt(max(p_set_th * (1 - p_set_th), 1e-12) / T)
        a3.append(
            {
                "k": k,
                "P_set_analytic": p_set_th,
                "P_set_mc": p_set_mc,
                "n_sigma_set": abs(p_set_mc - p_set_th) / max(se, 1e-12),
                "P_order_analytic": p_ord_th,
                "P_order_mc": p_ord_mc,
            }
        )

    # ---- anchor A4: H(Pi_1|z) vs exact softmax entropy (mean over configs) --
    pm = (r / Zpart.unsqueeze(1)).clamp_min(1e-300)
    h_exact = float((-(pm * torch.log(pm)).sum(dim=1)).mean().item())
    h_mc = cond_entropy_nats[:, 0].mean().item()
    a4_rel = abs(h_mc - h_exact) / h_exact

    # ---- information per race (§0.6) --------------------------------------
    info = {}
    lgam = math.lgamma(N + 1)
    for k in K_GRID:
        if k > N:
            continue
        H_marg_bits = (lgam - math.lgamma(N - k + 1)) / math.log(2)
        H_cond_bits = cond_entropy_nats[:, k - 1].mean().item() / math.log(2)
        info[k] = {
            "H_marginal_bits": H_marg_bits,
            "H_cond_bits": H_cond_bits,
            "MI_bits": H_marg_bits - H_cond_bits,
            "log2_binom_bits": (lgam - math.lgamma(k + 1) - math.lgamma(N - k + 1))
            / math.log(2),
        }
    info["log2_N_fact_bits"] = lgam / math.log(2)
    info["log2_N_bits"] = math.log2(N)

    # ---- empirical plug-in MI cross-check, k<=5 (capped rank labels) ------
    mi_plugin = {}
    for k in [2, 5]:
        if k > N:
            continue
        rk = ranks64[:, :, :k].clone()
        rk[rk >= k] = k  # cap: all deeper ranks -> one symbol
        base = k + 1
        code = torch.zeros(M, all_ranks.shape[1], dtype=torch.int64, device=device)
        for j in range(k):
            code = code * base + rk[:, :, j]
        # H(observed pattern) plug-in, Miller-Madow corrected, averaged over configs
        hs = []
        for m in range(M):
            cnt = torch.bincount(code[m])
            cnt = cnt[cnt > 0].to(DTYPE)
            p = cnt / cnt.sum()
            h = float((-(p * torch.log2(p)).sum()).item())
            h += (cnt.numel() - 1) / (2 * cnt.sum().item() * math.log(2))  # Miller-Madow
            hs.append(h)
        H_pattern = sum(hs) / len(hs)
        # MI(true top-k order ; observed first-k order) via exchangeability:
        #   = log2(N!/(N-k)!) - H(relative pattern)   [capped -> lower bound on H]
        H_marg_bits = (lgam - math.lgamma(N - k + 1)) / math.log(2)
        mi_plugin[k] = {"H_pattern_bits": H_pattern, "MI_upper_bits": H_marg_bits - H_pattern}

    out = {
        "N": N,
        "ensemble": ens,
        "M": M,
        "T": T,
        "races": M * T,
        "metrics": res,
        "anchors": {
            "A1_P_top1_mc": p_top1_mc,
            "A1_P_top1_theory": p_top1_th,
            "A1_rel_err": a1_rel,
            "A2_det_exact": det_exact,
            "A3": a3,
            "A4_H_exact": h_exact,
            "A4_H_mc": h_mc,
            "A4_rel_err": a4_rel,
        },
        "info": info,
        "mi_plugin": mi_plugin,
    }

    # ---- majority vote over K races (§0.5) --------------------------------
    out["vote"] = vote_analysis(all_ranks, N, device, gen)
    out["vote_extrap"] = vote_extrapolation(all_ranks, N, device)

    if verbose:
        print(
            f"  N={N:>6} {ens:<7} P_set: "
            + " ".join(f"k{k}={res[k]['P_set']:.3f}" for k in res)
            + f"  A1rel={a1_rel:.2e} A2={det_exact}"
        )
    del all_ranks, ranks64, r, z
    torch.cuda.empty_cache()
    return out


def vote_extrapolation(all_ranks, N, device, n_rival=200):
    """Large-K extrapolation of the majority-vote accuracy.

    The binding constraint of the vote is the count of the weakest true member
    (true rank a = k-1) against the strongest outsider (true ranks b >= k).  With
    q_i = P(i in first-k of one race), the count difference over K races has mean
    K*(q_a - q_b) and variance K*Var(1{a in} - 1{b in}), so by the CLT plus a union
    bound over the outsiders

        P(vote wrong) ~<  sum_b  Phi( -(q_a-q_b) sqrt(K) / sqrt(v_b) ).

    Validated against the measured curve for K <= 1000 (see §2.4 of the doc).
    Returns predicted accuracy curves and the extrapolated K(0.9), K(0.99).
    """
    M, T, _ = all_ranks.shape
    out = {}
    nrm = torch.distributions.Normal(0.0, 1.0)
    for k in K_GRID:
        if k > N:
            continue
        a = k - 1
        S = torch.arange(a, min(N, k + n_rival), device=device)  # [a, k, k+1, ...]
        acc_curve = {}
        Kext = {}
        per_cfg = []
        for m in range(M):
            rk = all_ranks[m, :, :k].to(torch.int64)
            hits = torch.zeros(T, S.numel(), dtype=torch.bool, device=device)
            step = max(1, 2_000_000 // (k * S.numel()))
            for s0 in range(0, T, step):
                blk = rk[s0 : s0 + step]
                hits[s0 : s0 + step] = (blk.unsqueeze(2) == S.view(1, 1, -1)).any(dim=1)
            q = hits.to(DTYPE).mean(dim=0)  # (|S|,)
            qa, qb = q[0], q[1:]
            both = (hits[:, :1] & hits[:, 1:]).to(DTYPE).mean(dim=0)
            mu = qa - qb
            v = (qa + qb - 2 * both - mu**2).clamp_min(1e-12)
            se = math.sqrt(1.0 / T)
            ok = mu > 2 * se  # margin resolvable above MC noise
            per_cfg.append((mu, v, ok))
        for K in [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]:
            accs = []
            for mu, v, ok in per_cfg:
                if not bool(ok.all().item()):
                    accs.append(0.0)
                    continue
                z = -mu * math.sqrt(K) / torch.sqrt(v)
                perr = nrm.cdf(z).sum().item()
                accs.append(max(0.0, 1.0 - perr))
            acc_curve[K] = sum(accs) / len(accs)
        for p in (0.9, 0.99):
            lo, hi = 1, 10**9
            if acc_curve[10_000_000] < p:
                Kext[p] = None
            else:
                for _ in range(40):
                    mid = int(math.sqrt(lo * hi)) + 1
                    accs = []
                    for mu, v, ok in per_cfg:
                        if not bool(ok.all().item()):
                            accs.append(0.0)
                            continue
                        z = -mu * math.sqrt(mid) / torch.sqrt(v)
                        accs.append(max(0.0, 1.0 - nrm.cdf(z).sum().item()))
                    if sum(accs) / len(accs) >= p:
                        hi = mid
                    else:
                        lo = mid
                    if hi - lo <= max(1, lo // 50):
                        break
                Kext[p] = hi
        out[k] = {
            "acc_curve_pred": acc_curve,
            "K90_ext": Kext[0.9],
            "K99_ext": Kext[0.99],
            "n_rival": int(min(N, k + n_rival) - k),
        }
    return out


def vote_analysis(all_ranks, N, device, gen):
    """Majority vote: over K races keep the k most frequent first-k members.
    Random tie-break.  Returns K(0.9), K(0.99) per k."""
    M, T, _ = all_ranks.shape
    R = R_RESAMPLE
    Kmax = K_VOTE_GRID[-1]
    res = {}
    for k in K_GRID:
        if k > N:
            continue
        acc_by_K = {K: 0.0 for K in K_VOTE_GRID}
        for m in range(M):
            memb = all_ranks[m, :, :k].to(torch.int64)  # (T,k)
            pick = torch.randint(0, T, (R, Kmax), generator=gen, device=device)
            counts = torch.zeros(R, N, dtype=torch.float32, device=device)
            ptr = 0
            ones = None
            for K in K_VOTE_GRID:
                add = memb[pick[:, ptr:K].reshape(-1)].reshape(R, -1)  # (R,(K-ptr)*k)
                if ones is None or ones.shape != add.shape:
                    ones = torch.ones_like(add, dtype=torch.float32)
                counts.scatter_add_(1, add, ones)
                ptr = K
                noise = torch.rand(R, N, generator=gen, device=device) * 0.5
                top = torch.topk(counts + noise, k, dim=1).indices
                # the voted set equals the true top-k iff all k picked ranks are < k
                ok = (top < k).all(dim=1)
                acc_by_K[K] += ok.float().mean().item()
            del counts
        for K in K_VOTE_GRID:
            acc_by_K[K] /= M
        k09 = next((K for K in K_VOTE_GRID if acc_by_K[K] >= 0.90), None)
        k099 = next((K for K in K_VOTE_GRID if acc_by_K[K] >= 0.99), None)
        res[k] = {
            "acc_by_K": acc_by_K,
            "K90": k09,
            "K99": k099,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="race_topk_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--micheck", action="store_true")
    ap.add_argument("--fullmi", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    dev = args.device
    print(f"device={dev} torch={torch.__version__}")
    if dev == "cuda":
        print(torch.cuda.get_device_name(0))
    if args.micheck:
        mi_check_small(dev)
        return
    if args.fullmi:
        full_mi(dev)
        return
    t0 = time.time()
    results = []
    for N in N_GRID:
        for ens in ENSEMBLES:
            ts = time.time()
            results.append(run_cell(N, ens, dev))
            print(f"    ({time.time()-ts:.1f}s, total {time.time()-t0:.1f}s)")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out} in {time.time()-t0:.1f}s")




# ----------------------------------------------------------------------------
# Uncapped plug-in MI check at N = 32 (alphabet N*(N-1) = 992 for k = 2, small
# enough to estimate an entropy from 10^5 samples).  Validates the exact
# estimator of §0.6 against a binning-based one.  Run: --micheck
# ----------------------------------------------------------------------------
def mi_check_small(device="cuda", N=32, T=200_000, M=8):
    print(f"uncapped plug-in MI check, N={N}, T={T}, M={M}")
    for ens in ENSEMBLES:
        gen = torch.Generator(device=device)
        gen.manual_seed(SEED0 + 7717 + ENSEMBLES.index(ens))
        z = make_logits(ens, N, M, gen, device)
        r = torch.exp(BETA * z)
        Zp = r.sum(dim=1)
        true_order = torch.argsort(z, dim=1, descending=True)
        rank_of = torch.empty_like(true_order)
        rank_of.scatter_(
            1, true_order, torch.arange(N, device=device).unsqueeze(0).expand(M, N)
        )
        for k in (1, 2):
            mi_plug, mi_exact = [], []
            for m in range(M):
                E = torch.empty(T, N, dtype=DTYPE, device=device).exponential_(generator=gen)
                idx = torch.topk(E / r[m], k, dim=1, largest=False, sorted=True).indices
                rk = rank_of[m][idx]  # (T,k) true ranks, uncapped
                code = torch.zeros(T, dtype=torch.int64, device=device)
                for j in range(k):
                    code = code * N + rk[:, j]
                cnt = torch.bincount(code)
                cnt = cnt[cnt > 0].to(DTYPE)
                p = cnt / cnt.sum()
                H = float((-(p * torch.log2(p)).sum()).item())
                H += (cnt.numel() - 1) / (2 * T * math.log(2))  # Miller-Madow
                Hm = (math.lgamma(N + 1) - math.lgamma(N - k + 1)) / math.log(2)
                mi_plug.append(Hm - H)
                rsel = r[m][idx]
                cum = torch.cumsum(rsel, dim=1)
                nll = (-torch.log(rsel / (Zp[m] - (cum - rsel)))).sum(dim=1).mean().item()
                mi_exact.append(Hm - nll / math.log(2))
            a, b = sum(mi_plug) / M, sum(mi_exact) / M
            print(f"  {ens:<7} k={k}  MI_plugin={a:7.4f}  MI_exact={b:7.4f}  diff={a-b:+.4f} bits")


# ----------------------------------------------------------------------------
# Full-permutation information per race: I(z ; Pi) = log2(N!) - H(Pi | z).
# This is the quantity the paper's utilization argument (§8) compares against.
# Run: --fullmi
# ----------------------------------------------------------------------------
def full_mi(device="cuda"):
    print("full-permutation MI per race (bits)")
    print(f"{'N':>6} {'ens':<7} {'log2(N!)':>12} {'H(Pi|z)':>12} {'I(z;Pi)':>10} {'util%':>7} {'log2N':>7}")
    for N, T, M in [(32, 20000, 16), (256, 4000, 16), (1024, 2000, 16), (32768, 200, 8)]:
        for ens in ENSEMBLES:
            gen = torch.Generator(device=device)
            gen.manual_seed(SEED0 + 313 + N + ENSEMBLES.index(ens))
            z = make_logits(ens, N, M, gen, device)
            r = torch.exp(BETA * z)
            Zp = r.sum(dim=1)
            nll = 0.0
            B = max(1, min(T, int(2e8 / (8 * N))))
            for m in range(M):
                done = 0
                acc = 0.0
                while done < T:
                    b = min(B, T - done)
                    E = torch.empty(b, N, dtype=DTYPE, device=device).exponential_(generator=gen)
                    idx = torch.argsort(E / r[m], dim=1)
                    rsel = r[m][idx]
                    cum = torch.cumsum(rsel, dim=1)
                    denom = (Zp[m] - (cum - rsel)).clamp_min(1e-300)
                    acc += float((-torch.log(rsel / denom)).sum().item())
                    done += b
                    del E, idx, rsel, cum, denom
                nll += acc / T
            H = nll / M / math.log(2)
            lg = math.lgamma(N + 1) / math.log(2)
            print(f"{N:>6} {ens:<7} {lg:12.1f} {H:12.1f} {lg-H:10.2f} {100*(lg-H)/lg:7.3f} {math.log2(N):7.1f}")


if __name__ == "__main__":
    main()
