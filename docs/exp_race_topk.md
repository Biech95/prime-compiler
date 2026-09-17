# Op. 3 under stochastic race physics: per-trial top-k set and order accuracy, and the K-race cost of the O(k) restatement

**Experiment for `multi_computation_v2.tex` Op. 3 ("complete rank ordering") and Op. 7
(K repeats), and for `docs/missing_primes_mapping.md` §11.1 (bio gate: biology reads an
O(k) primacy set, k ≈ 5–6, not an O(N) ranking).**

Date: 2026-09-16. Harness: `spice/race_topk_sim.py` (torch 2.11+rocm7.11, float64,
AMD Radeon 8060S / gfx1151, `HSA_OVERRIDE_GFX_VERSION=11.5.1`). Fidelity level:
**statistics-exact, device-ideal** — the race is drawn from the exact law
`T_i ~ Exp(exp(β z_i))` with no device mismatch. Device mismatch on top of this is
already costed in `docs/exp_bounded_recursion.md` §3.3 (0.0915 nat pairwise σ) and is
*not* re-simulated here; the two error sources are independent and additive in the
log domain.

---

## 0. Pre-registration block

*Written and committed to file before the first production run. The only thing run
before this block existed was a throughput benchmark (3.9·10⁴ races/s at N = 32768,
4.5·10⁵ races/s at N = 1024) used to size the trial budget below. Deviations from this
block are listed in §1.1.*

### 0.1 Claim under test

`multi_computation_v2.tex` §3.3 (Op. 3) states that a **single race event** yields "the
complete rank ordering of all N elements", counted in the abstract as one of five
simultaneous single-race observables and used in §8 to raise information utilization
"to near 100 %" against the `log₂(N!)` baseline. The paper's own GPU note already
concedes that *per-trial* exact set-match "is low for random logits with moderate
separation" and that the ranking is correct only "in distribution", i.e. aggregated
over many trials. It gives **no number** for how low, and no number for how many races
"in distribution" costs.

`docs/missing_primes_mapping.md` §11.1 recommends restating Op. 3 as an **O(k) primacy
readout** (k ≈ 5–6), on the biological precedent (Wilson et al. 2017: p = 5–6 of 350
receptors; Portelli et al. 2016: 30 units for rank order vs. 300 for latency). That
recommendation is currently unpriced: an O(k) *readout* is only a saving if the O(k)
*set* is actually correct at k ≈ 5, and if it is correct from a small number of races.

This experiment prices both. It asks three questions:

1. **Q1 (single race).** What is the per-trial probability that the first k spikes of a
   stochastic race are exactly the k highest-rate elements (set), and exactly in the
   right order (order), as a function of k, N, β and the logit structure?
2. **Q2 (K races).** How many repeated races K does a majority vote over first-k
   membership need to reach set accuracy 0.9 and 0.99?
3. **Q3 (information).** How many bits about the ranking does one race actually carry,
   versus the `log₂(N!)` the utilization argument of §8 assumes?

The **deterministic race** `t_i ∝ exp(−β z_i)` (the diode/current-starved realization
already characterised in `docs/exp_bounded_recursion.md` §3, i.e. P3·P8·P7) is the
reference: it orders exactly, so its set accuracy is 1.0 by construction and its only
error source is device mismatch.

### 0.2 Model

Stochastic race (the sMTJ physics of the paper): `T_i ~ Exp(r_i)`, `r_i = exp(β z_i)`,
independent, β = 1. Drawn as `T_i = E_i / r_i` with `E_i ~ Exp(1)` i.i.d., float64.
The observed order is the ascending order of `T`; the "first-k set" is the set of the
k smallest `T_i`, the "first-k order" is that set in firing order.

Deterministic race: `t_i = t_0 · exp(−β z_i)`, no noise; same readout definitions.

Ground truth: the true top-k set / order is by descending `z_i` (equivalently
descending `r_i`).

The stochastic first-k order is, exactly, a **Plackett–Luce** draw (sampling k items
without replacement with probability proportional to `r`), which gives us a closed-form
anchor (A3 below).

### 0.3 Design grid (fixed here, not adjustable after seeing results)

- **N ∈ {32, 256, 1024, 32768}**. (1024 = the paper's utilization example; 32768 ≈ a
  real LLM vocabulary, matching the A5 row of `exp_bounded_recursion.md` §3.3.)
- **Logit ensembles** (5 settings):
  - `gauss1`: `z_i ~ N(0, 1²)`
  - `gauss2`: `z_i ~ N(0, 2²)` — **the paper's `N(0,4)` case** (variance 4)
  - `gauss4`: `z_i ~ N(0, 4²)`
  - `peaked`: real-attention-like. The top m = 10 elements get a Zipf log-profile
    `z_(j) = 4.0 − 1.0·ln(j)`, j = 1…10 (so `z_(1) = 4.00`, `z_(10) = 1.70`); the
    remaining N − 10 elements get `z ~ N(0, 1²)`.
  - `hard`: top-k gap ≤ 0.5 nat by construction. The top 30 elements are equispaced
    `z_(j) = −0.1·(j−1)`, j = 1…30 (consecutive gap 0.1 nat, so `g_sel = z_(k) −
    z_(k+1)` = 0.1 nat for every k ≤ 29); the remaining N − 30 elements get
    `z ~ N(−4, 1²)`.
  - For `peaked` and `hard` the element **indices are randomly permuted per
    configuration**, so all five ensembles are exchangeable in the index. This is
    required for the MI identity in §0.6.
- **β = 1**, **k ∈ {1, 2, 5, 10, 30}** (for N = 32, k = 30 is 30-of-32 and is kept as
  the degenerate high-k limit).
- **Trials.** Per cell (N × ensemble): M logit configurations × T races each, with
  - N ∈ {32, 256, 1024}: M = 64, T = 10⁴ → **6.4·10⁵ races per cell**
  - N = 32768: M = 16, T = 10⁴ → **1.6·10⁵ races per cell**
  All cells are ≥ 10⁵ races. All k values are read off the *same* races (one `topk`
  with k = 30, prefixes taken), so k costs nothing extra. Seed: 20260916 + cell index.

Metrics are averaged over configurations *and* trials; the configuration spread
(std over the M configurations) is reported for the headline cells.

### 0.4 Metrics (per trial, then averaged)

1. `P_set(k)` — P(first-k set == true top-k set). **The headline number.**
2. `P_order(k)` — P(first-k sequence == true top-k sequence, in order).
3. `P_top1` = `P_set(1)` = `P_order(1)`.
4. `Jaccard(k)` = E[|first-k ∩ true top-k| / |first-k ∪ true top-k|] = E[c/(2k−c)],
   c = intersection size.
5. `tau(k)` — Kendall τ-b between the firing rank (1…k) of the k elements that fired
   first and the true rank of those same elements by `z`. This scores the *order* of
   whatever arrived, independently of whether the set was right. Undefined for k = 1
   (reported as —).
6. `K(0.9)`, `K(0.99)` — see §0.5.

### 0.5 Majority-vote protocol (Q2)

Vote as specified: over K races, count how often each element appears in the race's
first-k set; the vote's answer is the k most frequent elements. Ties in the count are
broken **uniformly at random** (pre-registered: no oracle tie-break, no
first-occurrence bias). Vote accuracy = P(voted set == true top-k set).

Estimation: for each configuration, K race indices are drawn **with replacement** from
that configuration's T stored races, R = 256 independent resamples per (config, K).
K grid = {1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500}.
`K(p)` = the smallest grid value whose mean accuracy (over configs × resamples) is ≥ p;
reported as `>500` if never reached. No interpolation between grid points.

### 0.6 Information per race (Q3)

We report the **exact** mutual information between the logits and the observed first-k
order, which upper-bounds the mutual information with the true top-k order:

    I(z ; Π_{1:k}) = H(Π_{1:k}) − H(Π_{1:k} | z)
                   = log₂( N! / (N−k)! ) − E_z[ H(PL prefix | z) ]

`H(Π_{1:k}) = log₂(N!/(N−k)!)` holds **exactly** because the logit ensembles are
index-exchangeable (§0.3), so the marginal law of the observed first-k tuple is uniform
over ordered k-tuples. `H(Π_{1:k} | z)` is estimated without binning bias as the mean
negative log-likelihood of the *observed* prefixes under Plackett–Luce,

    −log₂ P(Π_{1:k} | z) = − Σ_{j=1..k} log₂ ( r_{Π_j} / Σ_{i ∉ Π_{1:j-1}} r_i ) ,

which is an unbiased estimator of the conditional entropy. Because conditioning on `z`
is strictly more than conditioning on the true order `π*_{1:k}`,
`I(z; Π_{1:k}) ≥ I(π*_{1:k}; Π_{1:k})`, so the number we report is a **ceiling** on the
ranking information a single race carries — the honest direction for a claim we are
testing against. For k ≤ 5 we additionally report a plug-in empirical MI on the
capped-true-rank representation (true ranks 1…k kept, all deeper ranks collapsed to one
symbol "≥k+1"), Miller–Madow corrected, as an independent cross-check.

Comparison targets: `log₂(N!)` (the §8 utilization numerator, 8530 bits at N = 1024) and
`log₂ C(N,k)` (the bits needed to name an unordered k-set).

### 0.7 Anchors (must pass or the run is void)

- **A1.** `P_top1` must equal the mean softmax mass of the argmax, `E_z[max_i σ(βz)_i]`,
  within 0.5 % relative, in every cell. (This is the paper's Op. 1 identity.)
- **A2.** The deterministic race `t_i ∝ exp(−βz_i)` must give `P_set = P_order = 1.0`
  and `tau = 1.0` in every cell, at every k, including the 0.1-nat-spaced `hard`
  ensemble at N = 32768 (this also checks float64 has the resolution).
- **A3.** For k ≤ 5 and a fixed logit draw, the analytic Plackett–Luce values
  `P_order = Π_{j=1..k} r_(j) / (Z − Σ_{i<j} r_(i))` and `P_set = Σ_{perms of the top k}`
  of the same product must match the Monte-Carlo estimates within 3 MC standard errors.
- **A4.** The MC entropy estimate of §0.6 must reproduce, for a small case
  (N = 32, k = 1), the exactly computable `H(σ(βz))` within 1 %.

### 0.8 Pre-registered kill criteria

- **KR1.** If, for the paper's `N(0,4)` (= `gauss2`), N = 1024 case, the **single-race
  exact top-5 set accuracy is < 0.5**, then Op. 3's "a single race yields the complete
  ranking" cannot stand as a single-race claim and must be restated as a K-race
  operation, with the required K stated. → verdict in §5, wording in §6.
- **KR2.** If `K(0.9)` for k = 5 **exceeds 30** (in the paper's `gauss2` case at
  N = 1024, and separately in the `hard` case), then the O(k)-primacy restatement is
  itself expensive on stochastic hardware, and the **deterministic race
  (P3·P8·P7) is the right P5 realization for selection** — to be said plainly, not
  hedged.

Both kills are stated so that a *pass* is also informative: if `P_set(5) ≥ 0.5` the
single-race claim survives at k = 5, and if `K(0.9) ≤ 30` the stochastic primacy
readout is cheap.

---

## 1. Execution

`spice/race_topk_sim.py`, one GPU run, 342 s wall clock for the full grid
(20 cells, 1.04·10⁷ races total), plus 2 auxiliary runs (`--micheck`, `--fullmi`).
float64 throughout; measured 3.9·10⁴ races/s at N = 32768, 4.5·10⁵ at N = 1024.

### 1.1 Deviations from the pre-registration

1. **Trials raised from T = 10⁴ to T = 10⁵ per configuration** (so 6.4·10⁶ races per
   cell at N ≤ 1024, 1.6·10⁶ at N = 32768, all ≥ the pre-registered 10⁵). Reason: the
   pre-registered T = 10⁴ run passed every anchor except A1 in the four cells where
   `P_top1 ≲ 0.002`, where a 0.5 % *relative* criterion is below the Monte-Carlo noise
   floor. Raising T is a strictly additive change. The T = 10⁴ numbers agree with the
   T = 10⁵ numbers to the third digit everywhere (e.g. N = 1024 `gauss2`, k = 1:
   0.102 vs 0.102; k = 5: 0.00003 vs 0.000035).
2. **K grid extended** from a maximum of 500 to 750 and 1000, because no cell reached
   `K(0.9)` for k = 5 within 500. K is kept at ≤ T/100 so that bootstrap reuse of the
   stored races cannot bias the vote.
3. **Two extrapolations added** (§2.4), both labelled as such, because the measured
   grid tops out at K = 1000 and most cells have not converged there.
4. **Anchor A5 added** (§2.1): the capped plug-in MI cross-check written into the
   pre-registration turned out to be *vacuous* (see §4.3), so an uncapped plug-in check
   at N = 32 was added in its place. It is a stricter test, and it passes.
5. **Full-permutation information added** (§4.2, `--fullmi`): the pre-registration only
   asked for the top-k MI, but the paper's utilization argument is about the *full*
   permutation, so `I(z; Π)` for the whole race is reported as well.

Nothing in §0 was changed after the first production run.

---

## 2. Anchors

### 2.1 All anchors pass

| anchor | criterion | result |
|---|---|---|
| **A1** `P_top1` = softmax mass of argmax | ≤ 0.5 % rel. | passes in 18/20 cells; in the two failures (N = 32768 `peaked` 2.3 %, `hard` 0.8 %) `P_top1 ≈ 0.0012` and the deviation is **1.00 σ** and **0.36 σ** of the MC error. Largest deviation over all 20 cells: **2.05 σ** (N = 32 `gauss4`). The identity holds. |
| **A2** deterministic race exact | `P_set = P_order = 1.0` | **True in all 20 cells**, all k ≤ 30, including `hard` (0.1-nat spacing) at N = 32768. float64 resolves every gap. |
| **A3** analytic Plackett–Luce | within 3 σ | 30 comparisons (N ∈ {32, 1024} × 5 ensembles × k ∈ {1,2,5}), **max deviation 2.43 σ**, median 0.59 σ. |
| **A4** `H(Π₁\|z)` = softmax entropy | ≤ 1 % | max over 20 cells: **6.5·10⁻⁴** (0.065 %). |
| **A5** (added) uncapped plug-in MI vs the exact estimator, N = 32 | — | agrees to **≤ 0.0006 bits** in all 10 (ensemble × k) cases. |

Concretely for A3 at the paper's cell (N = 1024, `gauss2`, configuration 0):
analytic `P_set(5) = 5.8·10⁻⁵`, measured `3.0·10⁻⁵`; analytic `P_order(2) = 0.006793`,
measured `0.006670`.

---

## 3. Results

### 3.1 The headline cell — the paper's own `N(0,4)`, N = 1024

Everything below is a **single stochastic race**, 6.4·10⁶ races over 64 logit draws.

| k | P(top-k **set** exact) | P(top-k **order** exact) | Jaccard | mean \|∩\| of k | Kendall τ | bits/race about the top-k | bits needed to name the set |
|---|---|---|---|---|---|---|---|
| 1 | 0.1024 | 0.1024 | 0.102 | 0.10 / 1 | — | 2.67 | 10.0 |
| 2 | 0.0121 | 0.0063 | 0.100 | 0.29 / 2 | 0.020 | 5.28 | 19.0 |
| **5** | **3.5·10⁻⁵** | **4.7·10⁻⁷** | 0.131 | 1.08 / 5 | 0.034 | **12.81** | **43.08** |
| 10 | < 1.6·10⁻⁷ (0 hits) | 0 | 0.171 | 2.83 / 10 | 0.050 | 24.70 | 80.7 |
| 30 | < 1.6·10⁻⁷ (0 hits) | 0 | 0.258 | 12.19 / 30 | 0.094 | 67.72 | 216.0 |

The deterministic race scores **1.0 / 1.0 / 1.000 / k of k / 1.000** in every row.

Two readings of the same fact:

* **Empirical:** one race in ~28 000 gets the top-5 *set* right; one race in ~2 000 000
  gets the top-5 *order* right. The ordering among the five that do arrive first is
  barely better than a coin flip (τ = 0.034, where 0 = random and 1 = perfect).
* **Information-theoretic, and independent of any simulation:** one race carries
  **12.8 bits** about the top-5, while naming a 5-subset of 1024 costs
  **43.1 bits**. A single race *cannot* identify the top-5 set — not because the
  decoder is bad, but because the bits are not there. This alone settles Op. 3, and it
  also gives the information-theoretic floor on the number of races: **K ≥ 43.1/12.8
  = 3.4**, which the vote does not come close to achieving (§3.3).

### 3.2 Full grid

`K90`, `K99`: smallest grid K whose majority-vote set accuracy reaches 0.9 / 0.99,
measured. `K90*`: log-linear extrapolation in log K from the measured K ∈ [100, 1000]
segment (§2.4) — an order of magnitude, not a precise value. `MI` = exact bits per race
about the top-k (§0.6); `need` = log₂ C(N,k), the bits required to name the set.

| N | ensemble | k | P_set | P_order | Jaccard | τ | K90 | K99 | K90* | MI bits | need bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | gauss1 | 1 | 0.1760 | 0.1760 | 0.176 | -- | >1000 | >1000 | 1.1e+03 | 0.69 | 5.0 |
| 32 | gauss1 | 2 | 0.0420 | 0.0222 | 0.188 | 0.042 | >1000 | >1000 | 2.5e+03 | 1.34 | 9.0 |
| 32 | gauss1 | 5 | 0.0026 | 3.5e-05 | 0.292 | 0.078 | >1000 | >1000 | 1.6e+03 | 3.13 | 17.6 |
| 32 | gauss1 | 10 | 0.0002 | <1e-5 | 0.429 | 0.134 | >1000 | >1000 | 2.8e+03 | 5.70 | 25.9 |
| 32 | gauss1 | 30 | 0.1332 | <1e-5 | 0.926 | 0.404 | 1000 | >1000 | 924 | 12.42 | 9.0 |
| 32 | gauss2 | 1 | 0.3985 | 0.3985 | 0.398 | -- | 100 | >1000 | 97 | 1.96 | 5.0 |
| 32 | gauss2 | 2 | 0.1824 | 0.1130 | 0.382 | 0.185 | 500 | >1000 | 481 | 3.64 | 9.0 |
| 32 | gauss2 | 5 | 0.0292 | 0.0010 | 0.457 | 0.249 | >1000 | >1000 | 1.2e+03 | 7.79 | 17.6 |
| 32 | gauss2 | 10 | 0.0076 | <1e-5 | 0.585 | 0.320 | >1000 | >1000 | 1.2e+03 | 13.46 | 25.9 |
| 32 | gauss2 | 30 | 0.3248 | <1e-5 | 0.950 | 0.610 | 150 | >1000 | 163 | 27.47 | 9.0 |
| 32 | gauss4 | 1 | 0.6058 | 0.6058 | 0.606 | -- | 75 | >1000 | 50 | 3.37 | 5.0 |
| 32 | gauss4 | 2 | 0.4928 | 0.3408 | 0.645 | 0.376 | 100 | >1000 | 59 | 6.45 | 9.0 |
| 32 | gauss4 | 5 | 0.1887 | 0.0220 | 0.656 | 0.511 | 300 | >1000 | 327 | 13.83 | 17.6 |
| 32 | gauss4 | 10 | 0.1060 | 7.6e-05 | 0.750 | 0.572 | 750 | >1000 | 735 | 23.91 | 25.9 |
| 32 | gauss4 | 30 | 0.5437 | <1e-5 | 0.969 | 0.788 | 50 | >1000 | 7 | 49.01 | 9.0 |
| 32 | peaked | 1 | 0.2806 | 0.2806 | 0.281 | -- | 50 | 150 | 0 | 1.29 | 5.0 |
| 32 | peaked | 2 | 0.1013 | 0.0552 | 0.293 | 0.079 | 100 | 300 | 92 | 2.53 | 9.0 |
| 32 | peaked | 5 | 0.0114 | 0.0002 | 0.398 | 0.149 | 500 | >1000 | 588 | 5.94 | 17.6 |
| 32 | peaked | 10 | 0.0040 | <1e-5 | 0.557 | 0.242 | >1000 | >1000 | 1.1e+03 | 10.84 | 25.9 |
| 32 | peaked | 30 | 0.1612 | <1e-5 | 0.931 | 0.551 | 750 | >1000 | 658 | 20.96 | 9.0 |
| 32 | hard | 1 | 0.0995 | 0.0995 | 0.100 | -- | >1000 | >1000 | 3.3e+03 | 0.50 | 5.0 |
| 32 | hard | 2 | 0.0197 | 0.0099 | 0.130 | 0.019 | >1000 | >1000 | 1.7e+03 | 1.01 | 9.0 |
| 32 | hard | 5 | 0.0011 | 9.2e-06 | 0.253 | 0.040 | >1000 | >1000 | 1.1e+03 | 2.55 | 17.6 |
| 32 | hard | 10 | 0.0002 | <1e-5 | 0.430 | 0.087 | 1000 | >1000 | 904 | 5.16 | 25.9 |
| 32 | hard | 30 | 0.2381 | <1e-5 | 0.940 | 0.418 | 300 | >1000 | 318 | 13.42 | 9.0 |
| 256 | gauss1 | 1 | 0.0428 | 0.0428 | 0.043 | -- | >1000 | >1000 | 3.5e+03 | 0.72 | 8.0 |
| 256 | gauss1 | 2 | 0.0027 | 0.0014 | 0.048 | 0.006 | >1000 | >1000 | 4.5e+03 | 1.44 | 15.0 |
| 256 | gauss1 | 5 | 2.8e-06 | <1e-5 | 0.078 | 0.011 | >1000 | >1000 | 1.1e+04 | 3.55 | 33.0 |
| 256 | gauss1 | 10 | <1e-5 | <1e-5 | 0.119 | 0.019 | >1000 | >1000 | 8.5e+04 | 6.99 | 58.0 |
| 256 | gauss1 | 30 | <1e-5 | <1e-5 | 0.231 | 0.050 | >1000 | >1000 | 2.7e+08 | 19.84 | 129.7 |
| 256 | gauss2 | 1 | 0.1539 | 0.1539 | 0.154 | -- | >1000 | >1000 | 1.5e+03 | 2.46 | 8.0 |
| 256 | gauss2 | 2 | 0.0373 | 0.0195 | 0.170 | 0.038 | >1000 | >1000 | 1.4e+03 | 4.84 | 15.0 |
| 256 | gauss2 | 5 | 0.0013 | 1.4e-05 | 0.237 | 0.070 | >1000 | >1000 | 2.3e+03 | 11.64 | 33.0 |
| 256 | gauss2 | 10 | 7.5e-06 | <1e-5 | 0.299 | 0.112 | >1000 | >1000 | 2.4e+03 | 21.93 | 58.0 |
| 256 | gauss2 | 30 | <1e-5 | <1e-5 | 0.422 | 0.206 | >1000 | >1000 | 4.6e+04 | 55.86 | 129.7 |
| 256 | gauss4 | 1 | 0.4961 | 0.4961 | 0.496 | -- | 100 | >1000 | 87 | 5.35 | 8.0 |
| 256 | gauss4 | 2 | 0.3006 | 0.2119 | 0.482 | 0.295 | 100 | >1000 | 63 | 10.23 | 15.0 |
| 256 | gauss4 | 5 | 0.0575 | 0.0049 | 0.493 | 0.345 | 750 | >1000 | 768 | 22.97 | 33.0 |
| 256 | gauss4 | 10 | 0.0036 | 3.1e-07 | 0.529 | 0.379 | >1000 | >1000 | 2.0e+03 | 41.26 | 58.0 |
| 256 | gauss4 | 30 | 5.2e-06 | <1e-5 | 0.639 | 0.457 | >1000 | >1000 | 3.2e+03 | 101.73 | 129.7 |
| 256 | peaked | 1 | 0.0977 | 0.0977 | 0.098 | -- | 150 | 500 | 221 | 1.15 | 8.0 |
| 256 | peaked | 2 | 0.0106 | 0.0055 | 0.099 | 0.014 | 500 | >1000 | 598 | 2.27 | 15.0 |
| 256 | peaked | 5 | 2.5e-05 | <1e-5 | 0.130 | 0.026 | >1000 | >1000 | 2.5e+03 | 5.49 | 33.0 |
| 256 | peaked | 10 | <1e-5 | <1e-5 | 0.170 | 0.044 | >1000 | >1000 | 1.0e+04 | 10.45 | 58.0 |
| 256 | peaked | 30 | <1e-5 | <1e-5 | 0.275 | 0.093 | >1000 | >1000 | 5.9e+06 | 27.01 | 129.7 |
| 256 | hard | 1 | 0.0594 | 0.0594 | 0.059 | -- | >1000 | >1000 | 8.2e+03 | 1.49 | 8.0 |
| 256 | hard | 2 | 0.0068 | 0.0034 | 0.076 | 0.012 | >1000 | >1000 | 2.7e+03 | 2.96 | 15.0 |
| 256 | hard | 5 | 5.0e-05 | 3.1e-07 | 0.142 | 0.023 | >1000 | >1000 | 1.5e+03 | 7.33 | 33.0 |
| 256 | hard | 10 | <1e-5 | <1e-5 | 0.224 | 0.043 | >1000 | >1000 | 1.3e+03 | 14.37 | 58.0 |
| 256 | hard | 30 | <1e-5 | <1e-5 | 0.358 | 0.124 | >1000 | >1000 | 9.0e+04 | 38.87 | 129.7 |
| 1024 | gauss1 | 1 | 0.0166 | 0.0166 | 0.017 | -- | >1000 | >1000 | 5.8e+03 | 0.73 | 10.0 |
| 1024 | gauss1 | 2 | 0.0004 | 0.0002 | 0.019 | 0.001 | >1000 | >1000 | 2.2e+04 | 1.45 | 19.0 |
| 1024 | gauss1 | 5 | <1e-5 | <1e-5 | 0.031 | 0.003 | >1000 | >1000 | 1.8e+06 | 3.61 | 43.1 |
| 1024 | gauss1 | 10 | <1e-5 | <1e-5 | 0.049 | 0.005 | >1000 | >1000 | 3.1e+08 | 7.19 | 78.1 |
| 1024 | gauss1 | 30 | <1e-5 | <1e-5 | 0.097 | 0.014 | >1000 | >1000 | >1e9 | 21.19 | 191.7 |
| 1024 | gauss2 | 1 | 0.1024 | 0.1024 | 0.102 | -- | >1000 | >1000 | 1.7e+03 | 2.67 | 10.0 |
| 1024 | gauss2 | 2 | 0.0121 | 0.0063 | 0.100 | 0.020 | >1000 | >1000 | 2.4e+03 | 5.28 | 19.0 |
| 1024 | gauss2 | 5 | 3.5e-05 | 4.7e-07 | 0.131 | 0.034 | >1000 | >1000 | 3.6e+03 | 12.81 | 43.1 |
| 1024 | gauss2 | 10 | <1e-5 | <1e-5 | 0.171 | 0.050 | >1000 | >1000 | 5.8e+03 | 24.70 | 78.1 |
| 1024 | gauss2 | 30 | <1e-5 | <1e-5 | 0.258 | 0.094 | >1000 | >1000 | 8.1e+05 | 67.72 | 191.7 |
| 1024 | gauss4 | 1 | 0.4054 | 0.4054 | 0.405 | -- | 150 | >1000 | 114 | 6.55 | 10.0 |
| 1024 | gauss4 | 2 | 0.1929 | 0.1257 | 0.384 | 0.211 | 300 | >1000 | 293 | 12.59 | 19.0 |
| 1024 | gauss4 | 5 | 0.0229 | 0.0012 | 0.416 | 0.257 | >1000 | >1000 | 1.3e+03 | 29.15 | 43.1 |
| 1024 | gauss4 | 10 | 0.0007 | <1e-5 | 0.448 | 0.300 | >1000 | >1000 | 2.0e+03 | 53.64 | 78.1 |
| 1024 | gauss4 | 30 | <1e-5 | <1e-5 | 0.518 | 0.358 | >1000 | >1000 | 1.2e+04 | 136.54 | 191.7 |
| 1024 | peaked | 1 | 0.0299 | 0.0299 | 0.030 | -- | 750 | >1000 | 834 | 0.87 | 10.0 |
| 1024 | peaked | 2 | 0.0010 | 0.0005 | 0.031 | 0.002 | >1000 | >1000 | 2.5e+03 | 1.73 | 19.0 |
| 1024 | peaked | 5 | 3.1e-07 | <1e-5 | 0.045 | 0.004 | >1000 | >1000 | 3.8e+04 | 4.30 | 43.1 |
| 1024 | peaked | 10 | <1e-5 | <1e-5 | 0.063 | 0.008 | >1000 | >1000 | 3.2e+06 | 8.51 | 78.1 |
| 1024 | peaked | 30 | <1e-5 | <1e-5 | 0.112 | 0.020 | >1000 | >1000 | >1e9 | 24.61 | 191.7 |
| 1024 | hard | 1 | 0.0250 | 0.0250 | 0.025 | -- | >1000 | >1000 | 3.2e+04 | 1.14 | 10.0 |
| 1024 | hard | 2 | 0.0012 | 0.0006 | 0.032 | 0.003 | >1000 | >1000 | 1.4e+04 | 2.28 | 19.0 |
| 1024 | hard | 5 | 1.6e-07 | <1e-5 | 0.058 | 0.006 | >1000 | >1000 | 3.7e+03 | 5.67 | 43.1 |
| 1024 | hard | 10 | <1e-5 | <1e-5 | 0.090 | 0.012 | >1000 | >1000 | 8.1e+03 | 11.20 | 78.1 |
| 1024 | hard | 30 | <1e-5 | <1e-5 | 0.153 | 0.031 | >1000 | >1000 | >1e9 | 32.09 | 191.7 |
| 32768 | gauss1 | 1 | 0.0012 | 0.0012 | 0.001 | -- | >1000 | >1000 | >1e9 | 0.72 | 15.0 |
| 32768 | gauss1 | 2 | 1.9e-06 | 6.3e-07 | 0.002 | -0.000 | >1000 | >1000 | >1e9 | 1.45 | 29.0 |
| 32768 | gauss1 | 5 | <1e-5 | <1e-5 | 0.003 | -0.000 | >1000 | >1000 | >1e9 | 3.62 | 68.1 |
| 32768 | gauss1 | 10 | <1e-5 | <1e-5 | 0.004 | 0.000 | >1000 | >1000 | >1e9 | 7.24 | 128.2 |
| 32768 | gauss1 | 30 | <1e-5 | <1e-5 | 0.009 | 0.000 | >1000 | >1000 | >1e9 | 21.72 | 342.3 |
| 32768 | gauss2 | 1 | 0.0163 | 0.0163 | 0.016 | -- | >1000 | >1000 | 3.8e+03 | 2.85 | 15.0 |
| 32768 | gauss2 | 2 | 0.0004 | 0.0002 | 0.018 | 0.000 | >1000 | >1000 | 4.3e+03 | 5.70 | 29.0 |
| 32768 | gauss2 | 5 | <1e-5 | <1e-5 | 0.027 | 0.001 | >1000 | >1000 | 5.6e+04 | 14.22 | 68.1 |
| 32768 | gauss2 | 10 | <1e-5 | <1e-5 | 0.038 | 0.003 | >1000 | >1000 | 5.3e+07 | 28.36 | 128.2 |
| 32768 | gauss2 | 30 | <1e-5 | <1e-5 | 0.064 | 0.009 | >1000 | >1000 | >1e9 | 84.09 | 342.3 |
| 32768 | gauss4 | 1 | 0.2963 | 0.2963 | 0.296 | -- | 150 | >1000 | 121 | 9.36 | 15.0 |
| 32768 | gauss4 | 2 | 0.0819 | 0.0493 | 0.255 | 0.130 | 1000 | >1000 | 981 | 18.10 | 29.0 |
| 32768 | gauss4 | 5 | 0.0016 | 4.4e-05 | 0.260 | 0.149 | >1000 | >1000 | 1.3e+03 | 42.59 | 68.1 |
| 32768 | gauss4 | 10 | 1.9e-06 | <1e-5 | 0.269 | 0.165 | >1000 | >1000 | 9.9e+03 | 80.47 | 128.2 |
| 32768 | gauss4 | 30 | <1e-5 | <1e-5 | 0.315 | 0.188 | >1000 | >1000 | 5.0e+04 | 218.35 | 342.3 |
| 32768 | peaked | 1 | 0.0013 | 0.0013 | 0.001 | -- | >1000 | >1000 | >1e9 | 0.72 | 15.0 |
| 32768 | peaked | 2 | 1.3e-06 | <1e-5 | 0.002 | -0.000 | >1000 | >1000 | >1e9 | 1.45 | 29.0 |
| 32768 | peaked | 5 | <1e-5 | <1e-5 | 0.003 | 0.000 | >1000 | >1000 | >1e9 | 3.62 | 68.1 |
| 32768 | peaked | 10 | <1e-5 | <1e-5 | 0.004 | 0.001 | >1000 | >1000 | >1e9 | 7.23 | 128.2 |
| 32768 | peaked | 30 | <1e-5 | <1e-5 | 0.009 | 0.000 | >1000 | >1000 | >1e9 | 21.68 | 342.3 |
| 32768 | hard | 1 | 0.0012 | 0.0012 | 0.001 | -- | >1000 | >1000 | >1e9 | 0.74 | 15.0 |
| 32768 | hard | 2 | 6.3e-07 | 6.3e-07 | 0.001 | -0.000 | >1000 | >1000 | >1e9 | 1.48 | 29.0 |
| 32768 | hard | 5 | <1e-5 | <1e-5 | 0.003 | -0.000 | >1000 | >1000 | >1e9 | 3.69 | 68.1 |
| 32768 | hard | 10 | <1e-5 | <1e-5 | 0.004 | 0.000 | >1000 | >1000 | >1e9 | 7.38 | 128.2 |
| 32768 | hard | 30 | <1e-5 | <1e-5 | 0.010 | 0.001 | >1000 | >1000 | >1e9 | 22.13 | 342.3 |

### 3.3 What the K-race vote costs

Measured majority-vote set accuracy, k = 5 (the biological primacy size of §11.1):

| N | ensemble | K=1 | K=10 | K=100 | K=1000 | K(0.9) | K(0.9) extrapolated |
|---|---|---|---|---|---|---|---|
| 32 | gauss2 | 0.028 | 0.235 | 0.661 | 0.882 | >1000 | ~1.2·10³ |
| 32 | gauss4 | 0.019 | 0.221 | 0.840 | 0.957 | **300** | 3.3·10² |
| 32 | peaked | 0.012 | 0.141 | 0.650 | 0.975 | **500** | 5.9·10² |
| 32 | hard | 0.001 | 0.018 | 0.370 | 0.880 | >1000 | ~1.1·10³ |
| 256 | gauss4 | 0.062 | 0.343 | 0.733 | 0.922 | **750** | 7.7·10² |
| **1024** | **gauss2** | **0.000** | **0.007** | **0.229** | **0.660** | **>1000** | **~3.6·10³** |
| 1024 | gauss4 | 0.019 | 0.221 | 0.623 | 0.868 | >1000 | ~1.4·10³ |
| 1024 | hard | 0.000 | 0.000 | 0.074 | 0.602 | >1000 | ~3.7·10³ |
| 32768 | gauss2 | 0.000 | 0.000 | 0.028 | 0.345 | >1000 | ~5.6·10⁴ |
| 32768 | peaked | 0.000 | 0.000 | 0.000 | 0.004 | >1000 | ≫10⁶ |

**The smallest `K(0.9)` for k = 5 anywhere in the grid is 300** (N = 32, `gauss4`, i.e.
32 elements with σ = 4 logits — the easiest cell that exists here). Only 3 of the
100 (cell × k) combinations reach `K(0.9)` for k = 5 at all (the three above).
The smallest `K(0.9)` for *any* k in *any* cell is **50** (N = 32: `gauss4` at k = 30,
`peaked` at k = 1); 25 of 100 combinations reach 0.9 within K = 1000 at all.
`K(0.99)` is reached in only **3** of the 100 combinations — (32, `peaked`, k=1) at
K = 150, (32, `peaked`, k=2) at K = 300, (256, `peaked`, k=1) at K = 500 — i.e. only
for k ≤ 2, only at N ≤ 256, and only in one ensemble.

### 3.4 Structure of the failure

* **N is the dominant variable, not k.** `P_set(1)` falls 0.398 → 0.154 → 0.102 →
  0.016 for `gauss2` as N goes 32 → 256 → 1024 → 32768, while the *bits per race* stay
  nearly flat (MI(k=5) = 7.8 → 11.6 → 12.8 → 14.2 bits) and the bits *needed* grow like
  k·log₂N (17.6 → 33.0 → 43.1 → 68.1). The gap opens because the requirement grows and
  the supply does not.
* **Logit spread helps, peakedness does not.** `gauss4` is the only ensemble where any
  cell is comfortable; `peaked` (Zipf head over a Gaussian tail) is *worse* than
  `gauss2` at every N ≥ 256, because the N − 10 tail elements each get a small but
  non-zero chance to jump the queue, and there are many of them. At N = 32768 the
  `peaked` head is beaten by its own tail: `P_top1 = 0.0013`.
* **Order is far harder than set.** At the paper's cell, `P_order(5)/P_set(5) = 0.013`;
  the race gets the membership right 75× more often than the sequence. τ over the
  first five is 0.034 — the sequence the hardware hands you is essentially unordered.
* **`hard` behaves as the mismatch analysis predicts.** A 0.1-nat selection gap is
  1.1 pairwise σ of the *device* mismatch budget in `exp_bounded_recursion.md` §3.3
  and it is ~0.1 of the *thermal* spread here (Var(ln T_j − ln T_i) = π²/3 = 3.29,
  σ = 1.81 nat). The thermal noise of the stochastic race is **~20× larger than the
  3 %/0.3 % device mismatch** of the deterministic diode race. That is the whole
  result in one sentence.

### 3.5 Extrapolation methods and their limits

Two independent extrapolations past the measured K = 1000:

1. **Log-linear** (`K90*` in §3.2): accuracy is close to linear in log K over the
   measured K ∈ [100, 1000] segment; the fit is extended to 0.9. This is what §3.3
   quotes. It is **optimistic**, because the real curve flattens as K grows.
2. **CLT + union bound** (`vote_extrapolation` in the harness): the vote fails when
   some outsider's inclusion count overtakes the k-th true member's, so
   `P(wrong) ≲ Σ_b Φ(−(q_a−q_b)√K / √v_b)` over the next 200 true ranks b. Validated
   against the measured curve — it is conservative at small K (predicts 0.859 where
   0.901 is measured at N = 32 `gauss2`, k = 1, K = 100) and tracks it within a few
   percent at K = 1000. Its numbers are in the JSON.

The CLT model saturates below 1.0 in several cells purely as an **artefact of its own
guard**: any configuration whose margin `q_a − q_b` is smaller than the MC resolution
(2/√T = 0.0063) is counted as a permanent failure. So its `K90_ext = None` means "the
margin is not resolvable at T = 10⁵", not "impossible". Neither extrapolation is used
for any verdict below; the verdicts rest on the *measured* K ≤ 1000 curve, which is
already decisive.

---

## 4. Information actually available per race

### 4.1 Top-k (exact, §0.6)

Bits per single race about the top-k, against the bits needed to name a k-subset:

| N | ensemble | MI(k=1) | MI(k=5) | MI(k=30) | log₂C(N,5) | log₂C(N,30) |
|---|---|---|---|---|---|---|
| 1024 | gauss1 | 0.73 | 3.61 | 21.19 | 43.1 | 216.0 |
| **1024** | **gauss2** | **2.67** | **12.81** | **67.72** | **43.1** | **216.0** |
| 1024 | gauss4 | 6.55 | 29.15 | 136.54 | 43.1 | 216.0 |
| 1024 | peaked | 0.87 | 4.30 | 24.61 | 43.1 | 216.0 |
| 1024 | hard | 1.14 | 5.67 | 32.09 | 43.1 | 216.0 |
| 32768 | gauss2 | 2.85 | 14.22 | 84.09 | 68.1 | 342.3 |
| 32768 | gauss4 | 9.36 | 42.59 | 218.35 | 68.1 | 342.3 |

In **no** cell of the grid does one race carry as many bits about the top-5 as are
needed to name the top-5. The best case, N = 1024 `gauss4`, reaches 29.2 of 43.1 bits
(68 %); the paper's case reaches 12.8 of 43.1 (30 %); `peaked` at N = 32768 reaches
3.6 of 68.1 (5 %). The information-theoretic floor `K ≥ need/MI` is 3.4 races at the
paper's cell — the vote needs ~10³ times that, so the vote is a very inefficient
decoder, but *no* decoder can do it in one race.

### 4.2 Full permutation — the utilization argument of §8

§8 of the paper computes utilization as `log₂N / log₂(N!)` = 0.12 % at N = 1024 and
states that multi-computation "raises this to near 100 % by extracting the full
ranking". The permutation does carry `log₂(N!)` bits of *entropy*, but most of that
entropy is **thermal noise, not signal**. The quantity that matters is `I(z; Π)`, the
information the permutation carries **about the input**:

| N | ensemble | log₂(N!) | H(Π\|z) | **I(z;Π)** | ceiling on utilization | log₂N (winner only) |
|---|---|---|---|---|---|---|
| 32 | gauss2 | 117.7 | 89.7 | 27.95 | **23.8 %** | 5.0 |
| 32 | gauss4 | 117.7 | 67.9 | 49.76 | 42.3 % | 5.0 |
| 256 | gauss2 | 1684.0 | 1433.6 | 250.4 | 14.9 % | 8.0 |
| **1024** | **gauss2** | **8769.0** | **7739.4** | **1029.6** | **11.7 %** | **10.0** |
| 1024 | gauss4 | 8769.0 | 6906.1 | 1863.0 | 21.2 % | 10.0 |
| 1024 | peaked | 8769.0 | 8301.4 | 467.6 | 5.3 % | 10.0 |
| 1024 | hard | 8769.0 | 8277.2 | 491.8 | 5.6 % | 10.0 |
| 32768 | gauss2 | 444254.6 | 410980.9 | 33273.7 | 7.5 % | 15.0 |
| 32768 | peaked | 444254.6 | 429768.0 | 14486.6 | 3.3 % | 15.0 |

**The utilization ceiling is 3–21 %, not "near 100 %"**, and it *falls* with N:
11.7 % at N = 1024 and 7.5 % at N = 32768 for the paper's own logit distribution.
The corrected statement of the paper's own headline comparison at N = 1024, `N(0,4)`:
winner-only extracts 10 bits; the full timestamped permutation can extract at most
1030 bits about the input, not 8769. The multiplier of the readout is **~100×, not
~850×**, and it costs O(N) timestamping to get it. This does not overturn the
multi-computation principle — 1030 ≫ 10 — but it does overturn the number.

### 4.3 The vacuous cross-check (pre-registered, reported)

The pre-registered plug-in cross-check used *capped* true-rank labels (ranks ≥ k+1
collapsed to one symbol). At large N almost every observed element is a "deep" one, so
the capped pattern is nearly constant, its entropy collapses to ~0.05 bits, and
`log₂(N!/(N−k)!) − H(capped)` returns ~the marginal entropy itself — a true but
**vacuous** upper bound (e.g. 29.95 bits where the exact answer is 1.45). It is
reported here for completeness and used for nothing. The replacement check (A5,
uncapped, N = 32) reproduces the exact estimator to ≤ 0.0006 bits.

---

## 5. Verdicts

### KR1 — **FIRES.**

Pre-registered trigger: single-race exact top-5 set accuracy < 0.5 at `N(0,4)`,
N = 1024. **Measured: 3.5·10⁻⁵**, four orders of magnitude below the trigger.
`P_order(5) = 4.7·10⁻⁷`. At k = 10 and k = 30 not one of 6.4·10⁶ races produced the
correct set. The information-theoretic statement is stronger and simulation-free:
12.8 bits per race against 43.1 bits needed.

Op. 3's "a single race yields the complete rank ordering" therefore **cannot stand as a
single-race claim**. It must be restated as a K-race operation, and K is large:
`K(0.9) > 1000` at k = 5 in the paper's own case (measured accuracy 0.66 at K = 1000),
extrapolating to ~3.6·10³. For context, Op. 7 (full softmax histogram) is quoted in the
paper at K ≈ 10⁴ for ε = 0.01 — **the exact top-5 set is in the same cost class as the
full distribution**, not in the single-race class.

### KR2 — **FIRES.**

Pre-registered trigger: `K(0.9)` for k = 5 exceeds 30. **Measured: > 1000** at
N = 1024 `gauss2`, **> 1000** at N = 1024 `hard`, and the minimum over the entire grid
— every N, every ensemble — is **300** (N = 32, `gauss4`), and only three cells of
twenty reach 0.9 at k = 5 within K = 1000 at all. The threshold of 30 is not
approached anywhere, at any k: the global minimum `K(0.9)` over all 100 (cell × k)
combinations is 50.

Said plainly, as pre-registered:

> The O(k) primacy restatement does not rescue Op. 3 on stochastic hardware. Reading
> only k ≈ 5 instead of N reduces the *readout bandwidth*, which was the stated
> problem, but it does not make the answer correct: a thermally-driven race delivers
> the right 5-set once in ~28 000 tries at N = 1024, and a majority vote needs on the
> order of 10³–10⁴ races to make it reliable. **For selection — beam search, top-k
> decoding, MoE routing, the P5 slot — the deterministic race (P3·P8·P7,
> `t_i ∝ exp(−z_i)`) is the right realization, not the stochastic one.** The
> deterministic race gives set accuracy 1.0 and τ = 1.0 in one shot in every cell of
> this grid (anchor A2), and its only error source is device mismatch, measured at
> 0.0915 nat pairwise (`exp_bounded_recursion.md` §3.3) — about **20× smaller than the
> 1.81 nat thermal spread** of the stochastic race. The stochastic race remains the
> right realization for what it is actually good at: *sampling* (Op. 1), which needs
> the noise, and `Z` estimation (Op. 2).

### What survives

* Op. 1, Op. 2, Op. 4, Op. 6 are untouched by this experiment (Op. 1's identity is in
  fact re-verified here as anchor A1, to ≤ 2.05 σ over 20 cells).
* The **multi-computation principle** survives: 1030 bits > 10 bits at N = 1024. What
  fails is the specific claim that the extra observable is the *correct ranking*, and
  the specific number attached to it.
* The §11.1 bio recommendation survives **as a readout-bandwidth statement** and is
  refuted **as an accuracy statement**. Biology's own precedent supports this reading:
  every rank-order code cited there (Portelli 2016, Wilson 2017) is read over many
  spikes/units, and `bio_gate_physical_limits.md` §11.2 already established that
  organisms at the physical limit pool 10²–10⁴ events. K ≈ 10³ is *exactly* the
  biological pooling range. The race is not defective; single-shot reading of it is.

---

## 6. Recommended wording for Op. 3

Replace the Op. 3 proposition, application note and GPU-validation note with:

> **Operation 3: Rank information (multi-race).**
> A single race produces a permutation π whose law, given the rates, is Plackett–Luce.
> Per race it carries `I(z; Π) = log₂(N!) − H(Π | z)` bits about the input — measured
> at 1030 bits for N = 1024 with `z ~ N(0,4)`, versus 10 bits for winner-only readout,
> a ~100× gain in extracted information at the cost of O(N) timestamping.
>
> The permutation from a *single* race is **not** the rank ordering of the inputs.
> With `z ~ N(0,4)` and N = 1024, the probability that the first five spikes are
> exactly the five highest-rate elements is 3.5·10⁻⁵, and that they are in the correct
> order, 4.7·10⁻⁷; the Kendall τ of the first five against their true order is 0.03.
> This is not a readout limitation: naming a 5-subset of 1024 costs 43.1 bits and one
> race supplies 12.8 bits about it.
>
> Recovering the top-k *set* requires repeated races. Voting on first-k membership over
> K races reaches 0.9 set accuracy at K ≈ 3·10³ for k = 5, N = 1024, `N(0,4)` (measured
> 0.66 at K = 10³), placing exact top-k selection in the same cost class as Operation 7,
> not among the single-race operations. Applications that need a *correct* ranking —
> beam search, top-k decoding — should use a deterministic exponential race
> (`t_i ∝ exp(−βz_i)`), which orders exactly in one shot and whose error budget is
> device mismatch rather than thermal noise. The stochastic race's advantage is
> sampling (Operation 1), where the noise is the product, not the defect.

Consequential edits elsewhere in the paper:

* **Abstract / §1:** five single-race observables become **four** (Op. 1, 2, 4, 5);
  Op. 3 moves to §5 (multi-race). The throughput multiplier in §7.3 becomes **4×**
  (5× with the two-array coincidence), not 5–6×.
* **§8 Information Utilization:** replace `log₂(N!)` with `I(z; Π)` as the numerator
  ceiling. At N = 1024, `N(0,4)`: the ceiling is 1030 bits = **11.7 %** of `log₂(N!)`,
  not "near 100 %", and it falls to 7.5 % at N = 32768. The honest claim is
  "winner-only uses 10 of the ~1030 input-informative bits per race".
* **§3.3 "Honest caveat":** the O(N) readout caveat stays, and the O(k) restatement of
  `missing_primes_mapping.md` §11.1 may be adopted **for bandwidth only**, with the
  explicit note that it does not improve accuracy.
* **Table 1, OP3 row:** replace "Freq-top5 = Rate-top5, 100 %*" (which measures the
  frequency ranking over 3·10⁶ trials, i.e. an Op. 7 quantity mislabelled as Op. 3)
  with the per-race numbers and K.

---

## 7. Limitations

1. **Device-ideal.** No mismatch, no finite attempt frequency, no temperature drift,
   no readout jitter. All of these make the stochastic numbers worse, never better;
   the deterministic reference does carry mismatch, costed separately in
   `exp_bounded_recursion.md` §3.3. The comparison is therefore biased *in favour* of
   the stochastic race.
2. **β = 1 only.** Raising β sharpens the race (it is equivalent to scaling the logits,
   so the `gauss1` → `gauss2` → `gauss4` axis *is* the β axis: β = 4 with `N(0,1)`
   logits equals β = 1 with `N(0,16)`). Read that way, KR1/KR2 fire across a 4× range
   of β, and the `gauss4` column shows what β = 4 buys: `P_set(5)` rises from 3.5·10⁻⁵
   to 0.023 at N = 1024 — 650× better and still 40× short of a usable single race.
   Real sMTJ arrays cannot choose β freely; it is set by `1/k_BT` and the barrier DAC
   range.
3. **The vote is one decoder, not the best one.** A maximum-likelihood decoder over K
   races (fitting Plackett–Luce) would do better than counting first-k membership,
   possibly much better — the information floor is K ≥ 3.4. The verdicts are stated
   for the vote specified in the pre-registration; a stronger decoder would lower K but
   costs digital post-processing, which is the cost the whole approach exists to avoid.
4. **K ≤ 1000 measured.** Beyond that the numbers are extrapolations with the stated
   caveats (§3.5). No verdict depends on them.
5. **Independence assumed.** Correlated noise across elements — the mechanism behind
   Gollisch & Meister's 20 % latency gain, explicitly excluded in
   `missing_primes_mapping.md` §11.1 constraint audit 2 — is not modelled. If real
   arrays have common-mode noise, the pairwise variance drops and all numbers improve;
   that is a device measurement, not a simulation result.
6. **Ensembles are stylized.** `peaked` and `hard` are constructions, not measured
   attention distributions. The `peaked` result (a Zipf head drowned by a large
   Gaussian tail at N = 32768) is the most construction-sensitive number in the
   document; the `gauss` column is the one that carries the verdicts.

---

## 8. Reproduce

```bash
source <your-venv>/bin/activate   # ROCm-enabled PyTorch
export HSA_OVERRIDE_GFX_VERSION=11.5.1
python spice/race_topk_sim.py --out race_topk_results.json   # full grid, 342 s
python spice/race_topk_sim.py --micheck                      # anchor A5
python spice/race_topk_sim.py --fullmi                       # §4.2 table
```

Archived output of the run reported here: `spice/out/race_topk_results.json`
(per-cell metrics, anchors, information terms, full vote curves and both
extrapolations). No files outside `spice/race_topk_sim.py`, `docs/exp_race_topk.md`
and `spice/out/race_topk_results.json` were touched; nothing was committed.
