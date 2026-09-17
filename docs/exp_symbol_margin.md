# Kill-gate 2: symbol error rate vs. noise margin (X1 quantitative row)

*Pre-registration written 2026-09-15 **before** any code was run; results appended
below it. Target: `docs/missing_primes_mapping.md` §1 (X1), §5 (Symbol row),
§8 kill-gate 2. Code: `spice/symbol_margin_sim.py`. Raw numbers:
`spice/out/symbol_margin_results.json`. Tools: ngspice-46, numpy 2.4, scipy 1.17.*

---

## 0. Claim under test

§1 of the mapping document asserts that a symbol is not a missing prime but
**P10 suppressed by a barrier**:

> `symbolic-if:  ΔE / kT  >  ln( f0 · t_hold / ε )`

and §1/L2 asserts that the symbolic operation *exact match / equality* is the
P-composite **P1·P2·P7** — a dot product against a stored code plus a threshold
(analog CAM row; Hopfield attractor). §5 turns this into a Table-4 row: "Symbol
(X1): physics if ΔE/kT > ln(f0·t_hold/ε), alphabet bounded, CAM rows ≤ area;
digital if ε below platform noise margin, unbounded alphabet."

The **qualitative** part of that claim is not in doubt — a Gaussian tail and an
Arrhenius rate are both exponential by construction. What is actually at stake,
and what §8 kill-gate 2 names, is:

1. that a *circuit-level* match really produces the Gaussian-margin statistics
   the row assumes (**prefactor**, tail shape), and
2. whether at **realistic** device mismatch the margin the row demands is
   affordable — i.e. where the "digital if" clause of the Table-4 row actually
   fires.

---

## 1. Pre-registration (fixed before any simulation)

### 1.1 Model

**L1 — k-bit code match (numpy, mandatory).** Stored pattern `w ∈ {±1}^k` held
as conductances with static mismatch `(1+ε_i)`, `ε_i ~ N(0, σ_G)`; query
`x ∈ {±1}^k` as voltages; match score

    s = Σ_i w_i x_i (1 + ε_i) + n,        n ~ N(0, σ_th·√k)

Decision: *exact match* iff `s > θ = k − m`. One bit flip moves the mean of `s`
by exactly **2**, so the margin `m` is quoted in those units ("m = margin in
units of one bit-flip = 2"). Adversary: the nearest non-match, Hamming
distance 1, mean score `k − 2`.

Sweep: `k ∈ {8, 16, 64, 256}`, `σ_G ∈ {0.5, 1, 2, 5} %` (plus the MOS-typical
3 % used by K3), `σ_th` fixed by `σ_th·√k / k ∈ {0.5, 1, 2} %`.

Measured separately: **false reject** `ε_FR = P(s ≤ θ | x = w)` and **false
accept** `ε_FA = P(s > θ | d(x,w) = 1)`. Monte Carlo ≥ 10⁶ trials per point,
with exponential-tilting importance sampling where plain MC cannot reach the
tail; analytic Gaussian prediction alongside; report MC/analytic.

**L2 — Hopfield attractor (numpy, mandatory).** `N` neurons, `P` Hebbian
patterns, `J_ii = 0`, `α = P/N ∈ {0.05, 0.10, 0.14}`, asynchronous Glauber
updates at temperature `T`, optional multiplicative synaptic mismatch
`J_ij → J_ij(1+ε_ij)`. Measured: retrieval error vs. `ΔE/kT` where
`ΔE_i = 2|h_i|` is the flip barrier of neuron `i`, and vs. `σ_G`.
**Calibration anchor:** the Amit–Gutfreund–Sompolinsky capacity
`α_c ≈ 0.138` at `T = 0`. *If the engine does not reproduce α_c within 10 %
(α_c ∈ [0.1242, 0.1518]) the run stops and the engine is fixed.*

**L3 — ngspice aCAM row (≤ 10 min).** One row, `k = 16` conductances with
**lognormal** mismatch at the MOS-typical corner `σ_G = 3 %`, summed into a
transimpedance node with *finite* amplifier gain, behavioral comparator,
Monte Carlo 200 draws. The **same 200×16 mismatch draws** are fed to the L1
model, so the comparison is paired, not distributional-only. Purpose: check
that L1's noise model is what a circuit produces — nothing more.

**Energy.** Per restoration, report the thermodynamic bound `kT·ln(1/ε)`
(T = 300 K) against the circuit cost at the margin actually needed, under
explicitly stated assumptions: match-line capacitance `C = 10 fF`, full-scale
match-line swing `V_FS = 1 V` for a full `k`-bit match (so one score unit is
`V_FS/k`). Report both `E_margin = C·(m*·V_FS/k)²` (the part attributable to
the margin) and `E_line = C·V_FS²` (what the row actually burns), each as a
ratio to the bound.

### 1.2 Kill criteria

**K1.** If `ε` does not fall at least exponentially in `(m/σ)²` — i.e. `ln ε`
vs. `(m/σ)²` is not linear with slope within 30 % of `−1/2` over ≥ 3 decades of
`ε` — in L1, the "symbol = margin" quantitative row of §5 is **KILLED**.

**K2.** If the MC/analytic prefactor ratio exceeds 3× at any point (correlations
or tails not captured by the Gaussian model), record the prefactor as a **design
correction**, not a kill.

**K3.** If at `k = 256`, `σ_G = 3 %` (MOS-typical) the margin required for
`ε = 10⁻⁹` (tokenizer-grade) exceeds 25 % of the full score range `k`, analog
exact-match is **impractical at MOS-typical mismatch** for large alphabets;
record that the X1 row's "digital if" clause fires, and give the `σ_G` at which
it does not.

**K4.** If L3's SPICE score distribution deviates from L1's by more than 20 % in
`σ`, L1's noise model is wrong; say so.

### 1.3 Declared in advance

- Static mismatch only (one draw per row, then many queries); no drift, no
  RTN, no temperature coefficient.
- One CAM row; row-to-row and match-line-to-match-line coupling not modelled.
- No calibrated device model (no PDK); ngspice ideal resistors + behavioral
  gain stage.
- The adversary is fixed at Hamming distance 1 (worst case for a dense code);
  a separate, clearly-labelled sub-analysis reports the minimum-distance-`d_min`
  code case, because that is the available design fix.

---

*Everything above was written before the first line of `symbol_margin_sim.py`
existed. Everything below is the run.*

---

## 2. Method and calibration

Everything below comes from one script, `spice/symbol_margin_sim.py`
(`--seed 2026` fixed), total compute **151 s** on 32 cores.

**Tail estimation.** Plain Monte Carlo cannot reach ε = 10⁻⁹ at 10⁶ trials, so
the deep tails are estimated by **exponential tilting**: the per-line mismatch
variables and the thermal variable are each given a mean shift, split in
proportion to their variance contribution, and the estimate is reweighted by the
exact likelihood ratio. This is unbiased and gives a relative standard error
≤ 4.5 × 10⁻³ at every point, with effective-sample-size fraction ≥ 0.11.

*Cross-check of the estimator against plain MC (10⁶ explicit k-term trials):*

| k | σ_G | z | plain MC (counts) | importance-sampled | analytic Q(z) | IS/plain |
|---|---|---|---|---|---|---|
| 16 | 1 % | 3 | 1.358e-3 (1358) | 1.350e-3 | 1.350e-3 | 0.994 |
| 16 | 1 % | 4 | 3.80e-5 (38) | 3.159e-5 | 3.167e-5 | 0.831 |
| 16 | 5 % | 3 | 1.388e-3 (1388) | 1.346e-3 | 1.350e-3 | 0.970 |
| 256 | 1 % | 3 | 1.351e-3 (1351) | 1.350e-3 | 1.350e-3 | 0.999 |
| 256 | 5 % | 3 | 1.281e-3 (1281) | 1.355e-3 | 1.350e-3 | 1.057 |
| 256 | 5 % | 4 | 3.70e-5 (37) | 3.164e-5 | 3.167e-5 | 0.855 |

At z = 3 the two agree to < 6 %; at z = 4 the plain-MC counts are 33–38, so the
17 % Poisson spread covers the difference. The estimator is sound.

**AGS anchor (mandatory, pre-registered stop condition).** T = 0 asynchronous
descent started at the stored pattern, retrieval declared when the final overlap
exceeds 0.9 (the AGS retrieval state has m ≈ 0.967 just below capacity),
24 realisations per point:

| N | α_c (overlap > 0.9) | deviation from 0.138 |
|---|---|---|
| 1000 | 0.1727 | 25.2 % |
| 2000 | 0.1578 | 14.3 % |
| 4000 | 0.1493 | 8.2 % |
| 8000 | **0.1460** | **5.8 %** |
| 1/√N extrapolation | **0.1295** | **6.2 %** |

Monotone in N, as finite-size theory requires (pattern-initialised descent
over-estimates capacity at finite N). Both readings that are entitled to be
compared with the thermodynamic-limit value — the largest N and the 1/√N
extrapolation — land inside the ±10 % band. **Anchor passes; the run proceeds.**
Reported honestly: N ≤ 2000 alone would have *failed* the anchor, so a
small-N-only Hopfield engine is not fit for this purpose.

---

## 3. K1 — does ε fall exponentially in (m/σ)²?  **PASS (not killed)**

Over all 48 (k, σ_G, σ_th) cells, fitting ln ε_FR against (m/σ_tot)² in the
window z ∈ [3, 6] — a span of **6.14 decades** of ε, twice the pre-registered
requirement:

| quantity | value |
|---|---|
| slope, importance-sampled MC | −0.5228 (min −0.5232, max −0.5223 over 48 cells) |
| slope, analytic Gaussian tail | −0.5228 |
| deviation from −1/2 | **4.6 %** (criterion: ≤ 30 %) |
| decades of ε spanned | 6.14 (criterion: ≥ 3) |

The 4.6 % excess is not error: it is the exact log-correction of the Gaussian
tail, `ln Q(z) = −z²/2 − ln(z√2π) + O(z⁻²)`, whose local slope in z² is
−1/2 − 1/(2z²)·(1+…). MC and analytic agree to four digits, i.e. the simulation
adds nothing the closed form does not already contain — which is the correct
outcome for a model that is Gaussian by construction, and is why the *interesting*
results are K2–K4 and not K1.

**Verdict: the "symbol = margin" quantitative row of §5 is NOT killed.** ε is
exponential in (margin/σ)² with the Arrhenius/Gaussian slope, as claimed.

---

## 4. K2 — prefactor  **TRIGGERED (design correction, not a kill)**

### 4.1 Gaussian mismatch: no prefactor

336 sweep points, MC/analytic ratio: **min 0.990, median 1.0001, max 1.010**.
The Gaussian model is exact, as it must be.

### 4.2 Lognormal mismatch (the physical conductance law): prefactor up to 4×

Memristive and MOS conductances are lognormal, not Gaussian. Holding the
coefficient of variation at σ_G and replacing `1+ε_i` with `exp(ν_i − s²/2)`,
`ν ~ N(0, s²)`, `s² = ln(1+σ_G²)` (so mean 1 and std σ_G are unchanged), the
*tails* change because the sum of lognormals is skewed:

**False accept (adversary side, the dangerous one) — ratio ε_lognormal/ε_Gaussian:**

| k \ z | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 8, σ_G = 3 % | 1.027 | 1.104 | 1.279 | 1.632 | 2.356 |
| 8, σ_G = 5 % | 1.046 | 1.188 | 1.518 | 2.267 | **4.051** |
| 16, σ_G = 5 % | 1.041 | 1.151 | 1.397 | 1.932 | **3.123** |
| 64, σ_G = 5 % | 1.024 | 1.087 | 1.202 | 1.444 | 1.877 |
| 256, σ_G = 3 % | 1.006 | 1.022 | 1.058 | 1.117 | 1.225 |
| 256, σ_G = 5 % | 1.011 | 1.044 | 1.104 | 1.212 | 1.383 |

**False reject (stored-word side) — same ratio:** the skew works the *other* way,
down to **0.12** (k = 8, σ_G = 5 %, z = 6) and 0.295 at k = 8, σ_G = 3 %, z = 6.

Two points exceed the pre-registered 3× threshold, both at z = 6 (ε ≈ 10⁻⁹) with
σ_G = 5 % and short codes: **k = 8 → 4.05×**, **k = 16 → 3.12×**. K2 therefore
triggers, and as pre-registered this is recorded as a design correction:

> **Design correction (X1 row).** At tokenizer-grade ε the Gaussian margin
> formula *under-predicts the false-accept rate by up to 4×* and
> *over-predicts the false-reject rate by up to 8×*, for short codes at large
> mismatch. The deviation is governed by `σ_G/√k` (skewness of the aggregate)
> and is monotone in both: it is ≤ 1.38× everywhere at k = 256 and ≤ 1.23× at
> k = 256, σ_G = 3 %. **Budget a factor 4 on false accepts for k ≤ 16; the
> central limit theorem removes the correction for large k.**

Note the asymmetry is a *feature* for design: lognormal mismatch makes exact
match cheaper on the reject side and more expensive on the accept side, so the
threshold should be set below the symmetric point.

### 4.3 The prefactor that is not in the model: gain error

L3 (below) exposed a term the pre-registration did not anticipate and that
dominates both prefactors above. A fractional **common-mode gain error** `g` on
the match line shifts the score mean by `g·k`, and `k/σ_tot = √k/σ_G` is
enormous, so

    d(ln ε)/dg  =  λ(z) · k/σ_tot ,     λ(z) = φ(z)/Q(z)

Requiring ε to stay within 2× of nominal gives `g_max = ln2·σ_tot/(k·λ(z))`
(verified numerically: the closed form lands at ε ratios 1.961/1.981/1.988 for
the three ε targets, against an intended 2.000):

| k | σ_G | ε | d ln ε/dg | g_max | equivalent gain accuracy |
|---|---|---|---|---|---|
| 8 | 3 % | 1e-9 | 580 | 1.19e-3 | 9.7 bits |
| 16 | 3 % | 1e-9 | 821 | 8.44e-4 | 10.2 bits |
| 64 | 3 % | 1e-9 | 1642 | 4.22e-4 | 11.2 bits |
| 256 | 3 % | 1e-9 | 3283 | 2.11e-4 | **12.2 bits** |
| 256 | 1 % | 1e-9 | 9850 | 7.04e-5 | **13.8 bits** |

> **New condition for the §5 Symbol row.** The row currently constrains only
> mismatch and barrier. It also needs a **gain-accuracy** term: the match line's
> transimpedance gain must be accurate to `ln2·σ_G/(√k·z_ε)` — 10–14 bits for
> the cells above. This is a *systematic*, calibratable error, unlike mismatch,
> but it is a real analog spec and it is currently missing from Table 4.

---

## 5. K3 — is analog exact match practical at MOS-typical mismatch?

### 5.1 The pre-registered criterion is mis-specified; it never fires

Literal K3 asks whether `m*(ε = 10⁻⁹) > 0.25·k`. Across the entire grid the
largest value of `m*/k` is **0.136** (k = 8, σ_G = 3 %, σ_th = 2 %), so **the
literal criterion never fires anywhere.**

That is a fault in the criterion, not a result. The margin `m` lives in score
units where **one bit flip = 2 units, independent of k**. The separation
available between a stored word and its nearest Hamming-1 neighbour is therefore
**2 units at every k**, while `0.25·k` grows without bound. Comparing the
required margin to a fraction of `k` compares it to a range that the decision
never has access to. Recorded as an error in the pre-registration, and replaced
by the test the physics actually imposes.

### 5.2 The feasibility test that does fire

The threshold must sit between the match mean (`k`) and the adversary mean
(`k − 2`). Both sides must clear ε, so with the threshold at the symmetric
point the requirement is

    Q(1/σ_tot) ≤ ε      ⟺      σ_tot ≤ 1/z_ε

with `z_ε = Q⁻¹(ε)` = 3.090 / 4.753 / 5.998 for ε = 10⁻³ / 10⁻⁶ / 10⁻⁹. This is
a constraint on σ alone: **no margin setting whatsoever rescues a row that
violates it.**

**At σ_G = 3 % (MOS-typical), mismatch only:**

| k | σ_tot | m*(1e-3) | m*(1e-6) | m*(1e-9) | m*/k at 1e-9 | feasible at 1e-9? |
|---|---|---|---|---|---|---|
| 8 | 0.0849 | 0.262 | 0.403 | 0.509 | 6.4 % | **yes** |
| 16 | 0.1200 | 0.371 | 0.570 | 0.720 | 4.5 % | **yes** |
| 64 | 0.2400 | 0.742 | 1.141 | 1.439 | 2.2 % | **NO** (1.44×) |
| 256 | 0.4800 | 1.483 | 2.282 | **2.879** | 1.1 % | **NO** (2.88×) |

**Answer to K3 as intended:** at k = 256 and σ_G = 3 % the required margin for
ε = 10⁻⁹ is **2.879 score units against 1.0 available** — analog exact match is
**impractical at MOS-typical mismatch for large alphabets**. The X1 row's
"digital if" clause fires.

**The σ_G at which it does not fire** (the answer the gate asks for):

| ε | σ_tot,max | σ_G,max at k=8 | k=16 | k=64 | k=256 | largest workable k at σ_G = 3 % |
|---|---|---|---|---|---|---|
| 1e-3 | 0.324 | 11.4 % | 8.1 % | 4.1 % | 2.0 % | **k ≤ 116** |
| 1e-6 | 0.210 | 7.4 % | 5.3 % | 2.6 % | 1.3 % | **k ≤ 49** |
| 1e-9 | 0.167 | 5.9 % | 4.2 % | 2.1 % | **1.04 %** | **k ≤ 30** |

So: `σ_G ≤ 1.04 %` makes k = 256 work at ε = 10⁻⁹; at the 3 % corner the
code width is capped at **k ≈ 30**.

**Thermal noise caps k independently of mismatch.** With the noise
parameterised as `σ_th·√k/k = σ_rel`, the score noise is `σ_rel·k` and the
feasibility bound `σ_rel·k ≤ 1/z_ε` gives a *hard* ceiling on k with no
mismatch at all:

| σ_rel | k_max at ε=1e-3 | 1e-6 | 1e-9 |
|---|---|---|---|
| 0.5 % | 65 | 42 | **33** |
| 1 % | 32 | 21 | **17** |
| 2 % | 16 | 10 | **8** |

At k = 256 the thermal term alone is σ_th = 0.005·256 = **1.28** against a budget
of 0.167 — i.e. **k = 256 at ε = 10⁻⁹ is out of reach at any mismatch** once
0.5 % match-line noise is admitted (with σ_G = 3 % on top, σ_tot = 1.367 and the
required margin is 8.20 units of the 2.0 that exist).

### 5.3 The design fix the gate should point at

The bound is `σ_tot ≤ d_min/z_ε`, where `d_min` is the code's minimum Hamming
distance: it is the *code*, not the row width, that buys margin. The
"σ_G ≤ 1.04 % at k = 256" wall is an artefact of using dense (d_min = 1) codes.

| k | d_min = 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 8 | 5.9 % | 11.8 % | 23.6 % | — |
| 16 | 4.2 % | 8.3 % | 16.7 % | 33.4 % |
| 64 | 2.1 % | 4.2 % | 8.3 % | 16.7 % |
| 256 | 1.04 % | 2.1 % | 4.2 % | 8.3 % |

(σ_G,max for ε = 10⁻⁹.) A d_min = 8 code at k = 256 tolerates **8.3 %**
mismatch — comfortably past the MOS corner — at the cost of alphabet size. This
is von Neumann's restoring-margin argument in its coding form, and it is the
honest resolution of the X1 row: *the "digital if" clause fires for dense codes
over large alphabets, and is bought off by code distance, not by device
quality.*

### 5.4 Static mismatch makes ε a yield, not a rate

The pre-registration declared static (per-row, draw-once) mismatch. That has a
consequence the aggregate numbers hide: conditioned on a row, the mismatch is a
fixed score offset, so each row has its *own* error rate and the fleet average
is set by the bad tail. Per-row ε_FR distribution at m = 4σ_tot, 200 000 rows:

| k | σ_G | σ_rel | fleet mean | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| 8 | 3 % | 0.5 % | 3.03e-5 | **3.3e-21** | 1.4e-11 | 3.9e-6 | 2.3e-3 |
| 16 | 3 % | 0.5 % | 3.28e-5 | 2.7e-13 | 6.2e-8 | 9.8e-5 | 5.2e-3 |
| 64 | 3 % | 0.5 % | 3.25e-5 | 2.8e-7 | 2.7e-5 | 5.7e-4 | 3.8e-3 |
| 256 | 1 % | 1 % | 3.17e-5 | 3.06e-5 | 4.3e-5 | 5.6e-5 | 6.8e-5 |

When mismatch dominates (top row) the median row is **16 orders of magnitude**
better than the fleet mean; when thermal noise dominates (bottom row) the
distribution is flat and the mean is the rate. **Consequence for the compiler:
in a mismatch-dominated aCAM the "error rate" is a yield fraction, and
per-row trimming or redundancy buys far more than improving the average
device.** Fleet means themselves match the pooled Gaussian analytic exactly
(3.03–3.28e-5 vs 3.167e-5), confirming that the marginalisation is right and
only the *distribution* is the news.

---

## 6. L2 — Hopfield attractor: barrier, retention, and mismatch immunity

### 6.1 ε ≈ exp(−ΔE/kT) at the single-bit level: confirmed

Binning stored bits by their barrier in the clean target state
(`ΔE_i = 2·h_i^ref·ξ_i`, evaluated once, so the barrier is not conditioned on the
outcome) and measuring the time-averaged misalignment over 200 Glauber sweeps,
N = 1500, α = 0.05:

| T | ΔE/kT bin | n bits | measured ε | Glauber 1/(1+e^{ΔE/kT}) | ratio |
|---|---|---|---|---|---|
| 0.30 | 3.64 | 37 | 3.19e-2 | 2.63e-2 | 1.21 |
| 0.30 | 4.57 | 139 | 1.26e-2 | 1.07e-2 | 1.18 |
| 0.30 | 5.58 | 312 | 4.07e-3 | 3.89e-3 | 1.05 |
| 0.30 | 6.94 | 761 | 1.14e-3 | 1.12e-3 | 1.02 |
| 0.30 | 8.67 | 236 | 3.18e-4 | 1.94e-4 | 1.64 |
| 0.40 | 2.67 | 31 | 1.09e-1 | 6.72e-2 | 1.62 |
| 0.40 | 4.52 | 486 | 1.38e-2 | 1.12e-2 | 1.24 |
| 0.40 | 6.49 | 236 | 1.63e-3 | 1.63e-3 | 1.00 |

Fitted Arrhenius slope of ln ε vs ΔE/kT: **−0.93 (T = 0.3), −1.11 (T = 0.4)**
against the exact value −1. The §1 form `ε ≈ exp(−ΔE/kT)` holds to within a
factor 1.0–1.6 over two decades. **Confirmed.**

### 6.2 Where it stops holding: collective escape

At α ≥ 0.10 and T ≥ 0.3 the fitted slope collapses to −0.13…−0.25 and the bit
error rate saturates at 0.39–0.48 — the network has left the retrieval basin
entirely. The per-bit Arrhenius picture is a *single-bit* picture; near capacity
the relevant barrier is the attractor's, not the bit's, and the failure is
collective. An earlier pass of this experiment with only 5 sampling sweeps
reported BER = 0.026 for exactly the cell that gives 0.388 at 200 sweeps:
**short observation windows flatter retention**, which is the `t_hold` term of
the §1 formula showing up as a measurement artefact.

### 6.3 The t_hold term measured directly

Escape time from the target attractor (first sweep at which overlap < 0.5),
N = 1000, 12 runs per point, cap 4000 sweeps, censoring handled by the
Poisson MLE `τ = (total observed sweeps)/(number of escapes)` — a median over
escaped runs only is badly biased and the first pass of this measurement was
wrong for that reason:

| α | T | escaped | τ [sweeps] |
|---|---|---|---|
| 0.05 | 0.50 | 3/12 | 12 049 |
| 0.05 | 0.53 | 11/12 | 1 360 |
| 0.05 | 0.56 | 10/12 | 960 |
| 0.05 | 0.60 | 12/12 | 32.5 |
| 0.05 | 0.64 | 12/12 | 12.6 |
| 0.10 | 0.25 | 5/12 | 6 228 |
| 0.10 | 0.32 | 7/12 | 2 890 |
| 0.10 | 0.36 | 11/12 | 520 |
| 0.10 | 0.40 | 12/12 | 26.4 |
| 0.10 | 0.50 | 12/12 | 9.2 |

Arrhenius fits of ln τ against 1/T:

| α | barrier ΔE | ln(1/f0) | R² |
|---|---|---|---|
| 0.05 | **14.12** | −19.10 | 0.944 |
| 0.10 | **4.01** | −5.83 | 0.880 |

The escape time is Arrhenius (R² = 0.88–0.94) and the barrier **shrinks 3.5×**
as the load doubles from α = 0.05 to 0.10. This is the §1 row
`ΔE/kT > ln(f0·t_hold/ε)` measured rather than asserted, with the extra term the
row does not carry: **for an attractor memory ΔE is not a device constant, it is
a function of the load α.** A CAM row holding P patterns at k lines has a
retention barrier that degrades with occupancy — the "60 kT MRAM bit" analogy in
§1/L1 applies to an isolated bistable element, not to an attractor array.

### 6.4 The attractor absorbs synaptic mismatch almost completely

Multiplicative mismatch on the synapses, `J_ij → J_ij(1+ε_ij)`, N = 1500:

| α \ σ_G | 0 | 1 % | 3 % | 5 % | 10 % | 20 % | 40 % |
|---|---|---|---|---|---|---|---|
| 0.05 (BER) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.10 (BER) | 0.0006 | 0.0013 | 0.0012 | 0.0024 | 0.0013 | 0.0012 | 0.0010 |
| 0.14 (BER) | 0.0184 | 0.0082 | 0.0442 | 0.0338 | 0.0122 | 0.0402 | 0.0530 |

**At α = 0.05 retrieval is perfect at 40 % synaptic mismatch.** Capacity
measured directly (N = 4000, 16 reps):

| σ_G (nominal) | α_c measured | α_c / α_c(0) |
|---|---|---|
| 0 | 0.152 | 1.000 |
| 0.20 | 0.154 | 1.013 |
| 0.40 | 0.147 | 0.969 |
| 0.60 | 0.142 | **0.934** |

A 60 % synaptic mismatch costs **6.6 % of capacity**. Compare the open-loop CAM
row of §5, where 3 % conductance mismatch already forbids k = 64 at ε = 10⁻⁹.
This is the same P7-feedback effect the repo's X4 experiment found for the AGC
(`docs/exp_mismatch_absorption.md`): closing the loop turns per-element mismatch
into a small perturbation of a collective variable.

**A side law was pre-stated and refuted.** The naive estimate is that mismatch
adds in quadrature to the AGS crosstalk noise, renormalising the load to
α_eff = α(1+σ_G²) and giving α_c(σ_G) = α_c(0)/(1+σ_G²). Measured deviations:
5.4 % (σ_G = 0.2), 12.4 % (0.4), **27.1 % (0.6)** — the law over-predicts the
damage by roughly 2–5×. Part of that is a modelling artefact I introduced: the
mismatched J is symmetrised as ½(J+Jᵀ) to keep the energy function defined,
which halves the effective mismatch variance, so the *effective* independent
mismatch is σ_G/√2 and the nominal 60 % is really 42 %. With that correction the
law still over-predicts by ~2×; the residual is unexplained at this fidelity and
is **not** claimed. What is claimed is the measured table above.

---

## 7. L3 — ngspice aCAM row  **K4 NOT TRIGGERED**

One row, k = 16 memristive conductances with lognormal mismatch at σ_G = 3 %,
summed into a virtual ground formed by a **finite-gain** (A₀ = 10⁴) inverting
stage with R_f = 100 kΩ, behavioral comparator at θ = k − m. Cell currents are
held at 1 µA (V_ref = 0.1 V into R₀ = 100 kΩ) per the `spice/README.md` gotcha;
GMIN is left at its 1e-12 S default, which at 1 µA is a 10⁻⁶ relative
contribution. 200 mismatch draws, each instantiated **twice** — once queried
with the stored word and once with its Hamming-1 neighbour — for 400 rows in one
`.op`. The identical 200×16 draws are pushed through the L1 model, so the
comparison is **paired, row by row**, not distributional. Runtime **1.9 s**.

| quantity | SPICE | L1 (same draws) | ratio |
|---|---|---|---|
| match score, mean | 15.9575 | 15.9846 | 0.99830 |
| match score, σ | 0.122677 | 0.123082 | **0.99671** |
| adversary score, mean | 13.9540 | 13.9777 | 0.99830 |
| adversary score, σ | 0.116328 | 0.116675 | 0.99703 |
| paired relative RMS deviation | — | — | 1.70e-3 |
| paired max absolute deviation | — | — | 0.0282 score units |

**K4: the σ deviation is 0.33 %, against a 20 % trigger. NOT TRIGGERED — L1's
noise model is what the circuit produces.**

The entire deviation is a **gain error, not a noise error**: both means are low
by exactly 0.170 %, which is the predicted noise gain of a finite-A₀
transimpedance stage with 16 parallel inputs, `(1 + k·R_f/R₀)/A₀ = 17/10⁴ =
0.170 %`. Measured 0.1698 %. σ is untouched.

**But that 0.17 % gain error is not harmless.** Counted false accepts at the
chosen margin (m = 1.846, nominal ε_FA = 0.10):

| source | false accepts / 200 |
|---|---|
| SPICE (with 0.17 % gain error) | **11** |
| L1 on the same draws (no gain error) | **13** |
| Gaussian analytic at nominal σ | 20 |

The 20 → 13 step is finite-sample (the 200 empirical draws have σ 2.9 % below
nominal and a mean 0.16 % below `k`); the 13 → 11 step is the gain error, and
the analytic sensitivity of §4.3 predicts 13 → 9.1 for a 0.17 % shift at this
operating point. A gain error of one part in six hundred moves the false-accept
rate by 15–30 %. Deep tails are not countable at n = 200 and are reported
analytically as pre-registered: the margin for ε_FA = 10⁻³ at this row is
m = 1.629 of the 2.0 available.

---

## 8. Energy: both sides scale as ln(1/ε), so the ratio is a technology constant

Assumptions stated as required: match-line capacitance **C = 10 fF**, full-scale
match-line swing **V_FS = 1 V for s = k** (so one score unit is V_FS/k volts),
**T = 300 K**, kT = 4.142e-21 J. Two costs are reported: `E_margin = C·ΔV²` with
ΔV = m*·V_FS/k (the part attributable to the margin) and `E_line = C·V_FS²`
(what the row actually burns to drive the line). The bound is kT·ln(1/ε)
(85.8 zJ at ε = 10⁻⁹).

Because `m* = z_ε·σ_tot` and `z_ε² ≈ 2 ln(1/ε)`, the margin energy is

    E_margin = C·V_FS²·σ_tot²·z_ε²/k²  ≈  2·C·V_FS²·σ_tot²·ln(1/ε)/k²

which carries the **same ln(1/ε)** as the thermodynamic bound. The ratio is
therefore **independent of ε**:

    E_margin / (kT·ln(1/ε))  →  2·C·V_FS²·σ_G² / (k·kT)      (mismatch-limited)

Confirmed numerically (closed form vs exact, mismatch only):

| k | σ_G | ε | m* | ΔV | E_margin | E_margin/bound | closed form |
|---|---|---|---|---|---|---|---|
| 256 | 1 % | 1e-3 | 0.494 | 1.93 mV | 37.3 zJ | 1.30 | 1.89 |
| 256 | 1 % | 1e-9 | 0.960 | 3.75 mV | 141 zJ | **1.64** | 1.89 |
| 256 | 3 % | 1e-9 | 2.879 | 11.25 mV | 1265 zJ | 14.7 | 17.0 |
| 64 | 1 % | 1e-9 | 0.480 | 7.50 mV | 562 zJ | 6.55 | 7.55 |
| 64 | 3 % | 1e-9 | 1.439 | 22.49 mV | 5059 zJ | 58.9 | 67.9 |
| 16 | 3 % | 1e-9 | 0.720 | 44.98 mV | 20 240 zJ | 236 | 272 |
| 8 | 3 % | 1e-9 | 0.509 | 63.62 mV | 40 470 zJ | 471 | 543 |

(The exact ratio sits ~13 % below the closed form because z_ε² = 35.97 against
2 ln(10⁹) = 41.45; the asymptotic form is the ε → 0 limit.)

**Two readings, both worth carrying:**

1. **The margin itself is nearly free.** At k = 256, σ_G = 1 %, one restoration
   to ε = 10⁻⁹ costs 141 zJ against an 86 zJ bound — a factor **1.6 from the
   Landauer-class limit**. The `kT·ln(1/ε)` price of exactness that §1/L0 names
   is genuinely the dominant term in this regime, which is a strong vindication
   of the L0 framing.
2. **The line drive is not.** `E_line = C·V_FS² = 10 fJ` regardless, i.e.
   **1.17×10⁵ times the bound** at ε = 10⁻⁹ (3.50×10⁵ at ε = 10⁻³). The
   overhead of an analog CAM restoration is not the margin, it is charging the
   match line. Any energy argument for analog symbols must attack V_FS and C,
   not the noise margin.

Note the tension with §5: `E_margin/bound ∝ σ_G²/k` **falls** with k, so a wide
row is more energy-efficient per restoration — while the feasibility bound
`σ_G√k ≤ 1/z_ε` **forbids** wide rows. The energetically attractive design is
the one the mismatch forbids, and code distance (§5.3) is what reconciles them.

---

## 9. Verdicts

| gate | verdict | number |
|---|---|---|
| **K1** — ε exponential in (m/σ)² | **PASS, not killed** | slope −0.5228 vs −0.5, 4.6 % dev, over 6.14 decades, all 48 cells |
| **K2** — prefactor > 3× | **TRIGGERED → design correction** | lognormal false-accept 4.05× (k = 8) and 3.12× (k = 16) at σ_G = 5 %, ε ≈ 10⁻⁹; ≤ 1.38× at k = 256 |
| **K3** — impractical at MOS-typical σ_G | **literal criterion mis-specified (never fires); intended test FIRES** | at k = 256, σ_G = 3 %, ε = 10⁻⁹ the required margin is 2.879 units of the 2.0 that exist. Workable: σ_G ≤ 1.04 % at k = 256, or k ≤ 30 at σ_G = 3 % |
| **K4** — SPICE vs L1 σ deviation > 20 % | **NOT TRIGGERED** | 0.33 % (0.99671), paired over 200 draws |
| **AGS anchor** | **PASS** | α_c = 0.1460 at N = 8000 (5.8 %), 0.1295 extrapolated (6.2 %), band ±10 % |

**Net effect on `docs/missing_primes_mapping.md`:**

1. **§5 Symbol row: not killed, but under-specified.** The row's physics-if
   clause should read
   `ΔE/kT > ln(f0·t_hold/ε)` **and** `σ_tot ≤ d_min/z_ε` **and**
   `gain error ≤ ln2·σ_G/(√k·z_ε)`. The second and third conditions are new and
   both bind before the barrier condition does at MOS-typical mismatch.
2. **§5 "digital if" clause fires for dense codes over large alphabets.** At the
   3 % corner, tokenizer-grade exact match over a dense k-bit code is capped at
   **k ≈ 30**; thermal noise alone caps it at k ≈ 33 / 17 / 8 for 0.5 / 1 / 2 %
   match-line noise. The fix is code distance, not device quality: d_min = 8 at
   k = 256 tolerates 8.3 % mismatch.
3. **§1/L1's "ΔE ≈ 60 kT ⇒ symbol" is right for an isolated bistable element and
   wrong for an attractor array.** Measured barrier ΔE = 14.1 at α = 0.05 and
   4.0 at α = 0.10: **retention degrades with occupancy**. Add a load column.
4. **§1/L2's "exact match = P1·P2·P7 in feedback" gains quantitative support
   the document did not claim:** the feedback form absorbs 60 % synaptic
   mismatch for 6.6 % of capacity, where the open-loop CAM row fails at 3 %.
   This is the same absorption mechanism as the repo's X4 result and should be
   cross-referenced.
5. **Gate 2's own framing is corrected:** the cost is not the margin. The margin
   is within 1.6× of kT·ln(1/ε); the match-line drive is 10⁵× the bound.

---

## 10. Limitations

- **No real device model.** ngspice ideal resistors, no PDK, no memristor
  compact model, no non-linear conductance, no read disturb.
- **Single row.** Row-to-row coupling, match-line loading by other rows, and
  sense-amplifier offset are absent; a real aCAM array adds all three.
- **Static mismatch only.** No drift, no random telegraph noise, no temperature
  coefficient, no retention loss of the stored conductance itself. §5.4 shows
  the static case already changes the meaning of "error rate"; a drifting case
  would change it again.
- **Thermal noise is a parameter, not a circuit result.** L3 measures the
  mismatch statistics against L1 but injects no physical noise source; the
  σ_th·√k term is asserted, and its parameterisation (noise scaling linearly
  with k, so that σ_th√k/k is fixed) is a modelling choice that makes thermal
  noise dominant at large k. A per-line-independent noise model would scale as
  √k instead and would move the k_max numbers of §5.2 upward.
- **The adversary is a single Hamming-1 neighbour.** A CAM with R rows faces R
  adversaries and picks up a union-bound factor up to R on the false-accept
  rate, which is not included; at R = 10³ that is another 3 decades of margin.
- **L2 uses the standard Hebb rule.** Pseudo-inverse or Storkey learning changes
  α_c and would change §6.4; the mismatch-absorption result is reported for Hebb
  only.
- **The symmetrisation ½(J+Jᵀ)** halves the effective synaptic mismatch
  variance (§6.4); the quoted σ_G values are nominal, the effective ones are
  σ_G/√2.
- **§4.3's gain-accuracy requirement is analytic plus one SPICE confirmation at
  one operating point.** It has not been swept.
- Nothing here is silicon.
