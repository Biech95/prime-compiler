# Experiment: Mismatch absorption by physically measured gradients

Kill-gate 3 of `missing_primes_mapping.md` §8, with the feedback-alignment
relaxation proposed in §10 (X4 row). Runner: `spice/train_mismatch_absorption.py`.

Status: **pre-registration written 2026-09-15 before any run**; results appended
below after execution.

---

## 1. Pre-registration (written before running)

### 1.1 Question

Table 7 of `prime_compiler_v2` reports that the AGC loop removes the
*common-mode* part of VGA mismatch but leaves a **per-channel residual of
≈ 0.7·σ_VGA** (0.69 % at σ_VGA = 1 %). §4 of the mapping doc claims that a
gradient *measured on the mismatched physical circuit* contains that mismatch and
therefore lets the trainable RMSNorm gain γ absorb it, while a gradient taken
from an idealized digital model does not. §10 further claims, on the biological
precedent of feedback alignment, that the backward path need **not** be the exact
adjoint ("reciprocity is nice, not necessary").

Both claims are tested here on the RMSNorm+AGC stage of the existing SPICE
harness.

### 1.2 Pre-registered kill criteria

| ID | Criterion | Consequence |
|---|---|---|
| **K1** | Condition A (train-in-simulation, deploy-on-circuit) already reduces the per-channel residual below **0.35·σ_VGA** | **KILL-uninformative** — the residual was not a mismatch effect and the experiment says nothing |
| **K2** | Condition B (physics-aware: mismatched forward, ideal backward) does **not** reduce the residual below **0.2·σ_VGA** | the §4/§10 claim that a physically measured gradient absorbs the mismatch is **KILLED** |
| **K3** | Condition C (fixed random backward matrix **B** in place of Wᵀ, plus a mismatched backward gain 1+δ, δ ~ N(0, 5 %)) ends with residual **> 2× that of B** | the feedback-alignment relaxation is **KILLED for this setting** |
| **K4** | Condition D (perturbative / extremum-seeking, no backward at all) needs **> 200×** the forward evaluations of B to reach B's residual within 20 % | not a kill of the claim: a **cost statement** — the perturbative route is impractical at this N |

Secondary, not a kill criterion: condition E (0.5 % multiplicative output noise
added to C and D) — does convergence survive?

### 1.3 Model (fixed before running)

Static network, per Monte-Carlo mismatch draw:

```
x ∈ R^N                       Gaussian, x ~ 1.5·N(0,1), |x_i| floored at 0.05
                              (identical clipping to agc_mismatch_sweep.run_cell)

analog RMSNorm via AGC:       x_eff = x·(1+ε),  ε_i ~ N(0, σ_VGA)   [VGA array]
                              G     = Γ / RMS(x_eff),  Γ = 1        [loop equilibrium]
                              y     = G·x_eff
learnable per-channel gain:   z     = γ ⊙ y                         [RMSNorm affine]
linear read-out:              o     = W z,   W ∈ R^{M×N}, M = 4
loss:                         L     = (1/M)·‖o − t‖²
target (ideal network):       t     = W·(x / RMS(x))                [no mismatch, γ = 1]
```

`γ` is applied **after** the AGC loop, so the loop equilibrium `G` does not depend
on `γ`. The ideal reference used by the error metric is `y_ref = x / RMS(x)`.

**Detector mismatch is set to 0 in the surrogate** (stated choice, per the brief).
Justification: `agc_mismatch_sweep.py` block [A] shows detector mismatch moves only
the common-mode scalar, which a trainable γ absorbs trivially and which the
`decompose_error()` α-fit removes anyway; including it would only add noise to the
quantity under test. The SPICE verification (§1.6) runs **both** an ideal-detector
and a MOS-typical-detector flavour, so the structural claim is checked, not assumed.

### 1.4 Metric

Exactly `decompose_error()` of `agc_mismatch_sweep.py`, applied per test sample to
`z = γ ⊙ y` against `y_ref`:

```
α        = ⟨z, y_ref⟩ / ⟨y_ref, y_ref⟩        common-mode gain shift
resid    = z/α − y_ref
per-chan = RMS_i(resid)                       per-channel residual
```

Reported in % of signal (`y_ref` has unit RMS by construction). Per mismatch draw
we take the median over a held-out test set of 500 inputs; the tables report
p50/p95 **over the ≥20 Monte-Carlo mismatch draws**.

### 1.5 Conditions (identical mismatch draws, data and seeds across A–D)

| | forward | backward | trains |
|---|---|---|---|
| **A** | **ideal** model | exact gradient of the ideal model | γ; then the learned γ is *deployed* on the mismatched forward and measured there |
| **B** | **mismatched circuit** | exact gradient of the **ideal** model (local activation `y⁰`, transpose `Wᵀ`) — Wright et al. 2022 physics-aware training | γ |
| **C** | **mismatched circuit** | **fixed random** `B ∈ R^{M×N}` in place of `Wᵀ`, **and** a per-channel backward gain `(1+δ_i)`, `δ_i ~ N(0, 5 %)`; local activation is the **physical** `y` — Lillicrap et al. 2016 | γ |
| **D** | **mismatched circuit** | **none**. Two-sided Rademacher perturbation `Δγ = ±σ_p` (σ_p = 0.02), `ĝ = (L⁺−L⁻)/(2σ_p)·s` — Cauwenberghs 1992 / extremum seeking | γ |
| **E** | as C and D, plus 0.5 % i.i.d. multiplicative output noise per evaluation | " | γ |

Each condition is additionally run in a **W-learnable** variant (W updated by its
own exact gradient, which is what feedback alignment assumes: only the *hidden*
backward path is randomized), because the alignment mechanism of Lillicrap et al.
requires a forward weight that can rotate towards `B`. Both variants are reported;
the fixed-W variant is the harsher test.

Cells: σ_VGA ∈ {1 %, 2 %} × N ∈ {8, 32}. MC draws per cell: 20. Master seed 2026;
per-cell/per-draw streams spawned deterministically from it so that A/B/C/D/E see
the *same* ε, the same training set and the same W.

### 1.6 Mandatory SPICE verification

After training, for ≥5 mismatch draws per cell the trained γ is pushed through the
**real junction-exact loop**: `agc_mismatch_sweep.settle_gain()` /
`detector_ms()` imported as a module, bisecting the SPICE-measured detector output
to the loop equilibrium, then `decompose_error()` on `γ ⊙ (G_spice · x_eff)`.
Reported before (γ = 1) and after training, for an ideal detector and for the
MOS-typical detector corner (IS 3 %, n 0.3 %, mirror 1 %).
**If SPICE and surrogate disagree by more than 20 % relative on the per-channel
residual, this is stated loudly.**

---

## 2. Method as actually run

Runner: `spice/train_mismatch_absorption.py`. Environment: ngspice-46
(`~/.local/bin/ngspice`, VT = 0.025864890 V), Python 3.14.3, numpy 2.4.2,
scipy 1.17.1, 32 cores, `OPENBLAS_NUM_THREADS=1`, multiprocessing over the
(cell, draw) grid. Command and wall clock:

```
OPENBLAS_NUM_THREADS=1 python3 train_mismatch_absorption.py \
    --mc-runs 20 --spice-draws 5 --spice-inputs 4 --workers 32
# 80 (cell, draw) tasks; training 503.7 s, SPICE 1.1 s, total 504.8 s
```

Raw results: `spice/out/mismatch_absorption.json`, console log
`spice/out/run_main.log`.

### 2.0 Deviations from the pre-registration (declared)

1. **Condition C2 was added after C failed.** It is a post-hoc control, not a
   pre-registered condition, and is reported as such: it does not affect the K3
   verdict (which is decided by C alone), it only diagnoses *why* C failed. Its
   result must be treated as exploratory until re-run as a pre-registered test.
2. **The W-learnable variant was run for A, B, C and C2 only**, not for D (which
   perturbs γ only, by construction) and not for the noise condition E. §1.5 said
   "each condition"; this is a reduction in scope, not a change of design. The
   variant turned out to be uninformative anyway (§4.1 item 2).
3. **Optimiser fixed after a pilot.** §1.5 did not name an optimiser. Adam was
   tried first and disadvantaged condition D badly; plain cosine-decayed SGD was
   then adopted for *all* conditions before the production run. Pilot numbers are
   not included in the tables.

### 2.1 Equations

Per mismatch draw (`ε` fixed, `N_train` = 2000, `N_test` = 500, `M` = 4):

```
ideal      y⁰(x) = x / RMS(x)
mismatched y (x) = Γ·x·(1+ε) / RMS(x·(1+ε)),  Γ = 1
output     o     = W (γ ⊙ y),      target t = W y⁰(x)
loss       L     = (1/M) ‖o − t‖²
```

Gradients used for γ (with `e = (2/M)(o − t)`):

| cond | `∂L/∂γ` as applied |
|---|---|
| A | `(e Wᵀ) ⊙ y⁰` and the forward is `y⁰` too |
| B | `(e Wᵀ) ⊙ y⁰`, forward is `y` (mismatched) |
| C | `(e B) ⊙ y ⊙ (1+δ)`, `B` fixed random `M×N`, `δ ~ N(0,5 %)` |
| C2 | as C, but `B` has **random magnitudes and the signs of W** (sign concordance, Liao et al. 2016 / Xiao et al. 2018) — a control added after C failed, to locate *which* property of the adjoint is load-bearing |
| D | `ĝ = (L(γ+Δ) − L(γ−Δ))/(2σ_p)·s`, `Δ = σ_p s`, `s` Rademacher, `σ_p = 0.02`; the two probes share the minibatch (common random numbers) |

**Optimiser: plain SGD with a cosine-decayed learning rate in every condition**
(grad: 2.0 → 1e-3, batch 64; perturbative: 0.5 → 1e-3, batch 16). Adam was tried
first and is markedly *worse* for the perturbative estimator, whose
per-coordinate variance the Adam normaliser amplifies; using it would have
handicapped condition D and turned K4 into an optimiser artefact. Using one
optimiser everywhere keeps A/B/C/D a comparison of *gradient sources*.

`γ` is clipped to [1e-3, 20]; touching the box, or ending worse than untrained,
is recorded as `div` (divergence) in the tables.

### 2.2 Cost measurement (K4)

`smallest budget` = the smallest budget on a ladder of **independent runs**, each
with its own cosine schedule, whose *final* residual reaches
`1.2 × B_final`. Ladders: gradient conditions 250…32000 steps × 64
(= 16 000 … 2 048 000 forward evaluations), perturbative 6250…800000 steps × 2 × 16
(= 200 000 … 25 600 000). Independent runs per budget avoid the artefact that a
decaying schedule only reaches its best value at the very end.

A reference row **LS** is included: the *analytic* minimiser of the training loss
over γ for fixed W (least squares, `(Σ_k J_kᵀJ_k) γ = Σ_k J_kᵀ t_k`, `J_k = W ⊙ y_k`).
It is the attainable floor on this training set and shows how much of the
remaining residual is finite-sample estimation error rather than optimisation error.

### 2.3 Baseline cross-check against the existing harness

Running the unmodified `agc_mismatch_sweep.run_cell` (block [B], VGA-only
mismatch, ideal detector, 20 MC, junction-exact SPICE loop) gives per-channel
p50 = 0.842 σ (N=8, σ=1 %), 0.726 σ (N=8, 2 %), 0.986 σ (N=32, 1 %), 0.942 σ
(N=32, 2 %). Our untrained surrogate baseline is 0.756 / 0.685 / 0.923 / 0.925 σ
— the same quantity within MC scatter. Table 7's headline 0.69 σ sits at the low
end of this range; the mild N-dependence (`E[resid²] ≈ σ²(1 − Σ_i w_i²)`,
`w_i = x_i²/Σx_j²`, so ≈ 0.79 σ at N=8 and ≈ 0.95 σ at N=32) is reproduced.
**The residual is therefore 0.7–0.95 σ_VGA, not a flat 0.7 σ.**

---

## 3. Results

20 Monte-Carlo mismatch draws per cell, identical ε / training set / W / B / δ
across conditions. Per-channel residual and common-mode in **% of signal**
(`y_ref` has unit RMS); `pc/σ` is the residual in units of σ_VGA; `div` = number
of the 20 draws flagged divergent.

### 3.1 σ_VGA = 1 %, N = 8   (0.7 σ = 0.70 %)

| cond | pc p50 | pc p95 | pc/σ | cm p50 | task MSE | fwd evals | div |
|---|---|---|---|---|---|---|---|
| untrained | 0.756 | 1.079 | 0.756 | 0.003 | 4.94e-05 | – | – |
| LS (analytic floor) | 0.155 | 0.220 | 0.155 | 0.224 | 1.10e-05 | 0 | 0 |
| **A** train-in-sim | 0.756 | 1.079 | **0.756** | 0.003 | 4.94e-05 | 2.05e6 | 0 |
| **B** physics-aware | 0.155 | 0.220 | **0.155** | 0.224 | 1.10e-05 | 2.05e6 | 1 |
| **C** feedback alignment | 81.4 | 164.1 | **81.4** | 995.2 | 1.73e+02 | 2.05e6 | **20** |
| C2 sign-concordant | 0.153 | 0.226 | 0.153 | 0.224 | 1.10e-05 | 2.05e6 | 0 |
| **D** perturbative | 0.152 | 0.220 | **0.152** | 0.223 | 1.10e-05 | 2.56e7 | 0 |
| E: C + 0.5 % noise | 82.2 | 167.2 | 82.2 | 992.7 | 1.62e+02 | 2.05e6 | 20 |
| E: C2 + 0.5 % noise | 0.154 | 0.226 | 0.154 | 0.223 | 1.10e-05 | 2.05e6 | 0 |
| E: D + 0.5 % noise | 0.144 | 0.224 | 0.144 | 0.229 | 1.10e-05 | 2.56e7 | 0 |
| A (W learnable) | 0.756 | 1.079 | 0.756 | 0.003 | 4.94e-05 | 2.05e6 | 0 |
| B (W learnable) | 4.310 | 31.09 | 4.310 | 7.461 | 1.10e-05 | 2.05e6 | 20 |
| C (W learnable) | 0.816 | 1.185 | 0.816 | 0.115 | 1.10e-05 | 2.05e6 | 15 |
| C2 (W learnable) | 0.584 | 0.929 | 0.584 | 0.118 | 1.19e-05 | 2.05e6 | 1 |

Smallest budget reaching 1.2·B_final: **B = 1.6e4**, D = 2.0e5 (20/20),
C = never (0/20), C2 = 1.6e4 (20/20). **D/B = 12.5× (p50), 26× (p95).**
Backward loop-gain sign: fraction of channels with `diag(BᵀW) > 0` = **0.469**
for random B, 1.000 for sign-concordant B.

### 3.2 σ_VGA = 2 %, N = 8   (0.7 σ = 1.40 %)

| cond | pc p50 | pc p95 | pc/σ | cm p50 | task MSE | fwd evals | div |
|---|---|---|---|---|---|---|---|
| untrained | 1.371 | 2.357 | 0.685 | 0.009 | 2.45e-04 | – | – |
| LS (analytic floor) | 0.278 | 0.463 | 0.139 | 0.393 | 4.62e-05 | 0 | 0 |
| **A** | 1.371 | 2.357 | **0.685** | 0.009 | 2.45e-04 | 2.05e6 | 0 |
| **B** | 0.280 | 0.461 | **0.140** | 0.394 | 4.62e-05 | 2.05e6 | 0 |
| **C** | 82.8 | 150.7 | **41.4** | 974.6 | 1.68e+02 | 2.05e6 | **20** |
| C2 | 0.280 | 0.471 | 0.140 | 0.396 | 4.64e-05 | 2.05e6 | 0 |
| **D** | 0.277 | 0.472 | **0.139** | 0.395 | 4.62e-05 | 2.56e7 | 0 |
| E: C + noise | 85.8 | 151.2 | 42.9 | 971.1 | 1.81e+02 | 2.05e6 | 20 |
| E: C2 + noise | 0.279 | 0.477 | 0.140 | 0.396 | 4.65e-05 | 2.05e6 | 0 |
| E: D + noise | 0.281 | 0.471 | 0.140 | 0.394 | 4.63e-05 | 2.56e7 | 0 |
| A (W learnable) | 1.371 | 2.357 | 0.685 | 0.009 | 2.45e-04 | 2.05e6 | 0 |
| B (W learnable) | 23.02 | 38.58 | 11.51 | 19.87 | 4.63e-05 | 2.05e6 | 20 |
| C (W learnable) | 1.564 | 2.452 | 0.782 | 0.320 | 4.63e-05 | 2.05e6 | 14 |
| C2 (W learnable) | 1.400 | 10.40 | 0.700 | 0.572 | 4.72e-05 | 2.05e6 | 8 |

Smallest budget: B = 1.6e4, D = 2.0e5 (20/20), C = never, C2 = 1.6e4 (20/20).
**D/B = 12.5× (p50), 13.1× (p95).** `diag(BᵀW) > 0` fraction = **0.512** (random).

### 3.3 σ_VGA = 1 %, N = 32   (0.7 σ = 0.70 %)

| cond | pc p50 | pc p95 | pc/σ | cm p50 | task MSE | fwd evals | div |
|---|---|---|---|---|---|---|---|
| untrained | 0.923 | 1.092 | 0.923 | 0.004 | 8.19e-05 | – | – |
| LS (analytic floor) | 0.057 | 0.065 | 0.057 | 0.155 | 4.88e-06 | 0 | 0 |
| **A** | 0.923 | 1.092 | **0.923** | 0.004 | 8.19e-05 | 2.05e6 | 1* |
| **B** | 0.058 | 0.065 | **0.057** | 0.155 | 4.87e-06 | 2.05e6 | 0 |
| **C** | 90.2 | 114.1 | **90.2** | 911.9 | 1.59e+02 | 2.05e6 | **20** |
| C2 | 0.060 | 0.076 | 0.060 | 0.152 | 4.89e-06 | 2.05e6 | 0 |
| **D** | 0.058 | 0.066 | **0.058** | 0.153 | 4.88e-06 | 2.56e7 | 0 |
| E: C + noise | 91.4 | 114.8 | 91.4 | 918.4 | 1.59e+02 | 2.05e6 | 20 |
| E: C2 + noise | 0.061 | 0.080 | 0.061 | 0.153 | 4.88e-06 | 2.05e6 | 0 |
| E: D + noise | 0.060 | 0.075 | 0.060 | 0.152 | 4.89e-06 | 2.56e7 | 0 |
| A (W learnable) | 0.923 | 1.092 | 0.923 | 0.004 | 8.19e-05 | 2.05e6 | 12 |
| B (W learnable) | 0.964 | 1.375 | 0.964 | 0.286 | 4.93e-06 | 2.05e6 | 13 |
| C (W learnable) | 0.932 | 1.111 | 0.932 | 0.016 | 4.93e-06 | 2.05e6 | 14 |
| C2 (W learnable) | 0.863 | 1.020 | 0.863 | 0.016 | 4.94e-06 | 2.05e6 | 0 |

Smallest budget: B = 1.6e4, D = 4.0e5 (20/20), C = never, C2 = 1.6e4 (18/20).
**D/B = 25× (p50), 200× (p95).** `diag(BᵀW) > 0` fraction = **0.505** (random).

\* For condition A the trained γ converges to 1 and the residual equals the
untrained one bit-for-bit; the single `div` flag is a floating-point tie, not a
divergence.

### 3.4 σ_VGA = 2 %, N = 32   (0.7 σ = 1.40 %)

| cond | pc p50 | pc p95 | pc/σ | cm p50 | task MSE | fwd evals | div |
|---|---|---|---|---|---|---|---|
| untrained | 1.851 | 2.242 | 0.925 | 0.017 | 3.38e-04 | – | – |
| LS (analytic floor) | 0.115 | 0.141 | 0.058 | 0.288 | 2.12e-05 | 0 | 0 |
| **A** | 1.851 | 2.242 | **0.925** | 0.017 | 3.38e-04 | 2.05e6 | 1* |
| **B** | 0.115 | 0.140 | **0.057** | 0.289 | 2.12e-05 | 2.05e6 | 0 |
| **C** | 90.5 | 118.9 | **45.2** | 942.4 | 1.89e+02 | 2.05e6 | **20** |
| C2 | 0.120 | 0.172 | 0.060 | 0.286 | 2.14e-05 | 2.05e6 | 0 |
| **D** | 0.118 | 0.142 | **0.059** | 0.286 | 2.12e-05 | 2.56e7 | 0 |
| E: C + noise | 90.1 | 123.8 | 45.0 | 928.1 | 1.88e+02 | 2.05e6 | 20 |
| E: C2 + noise | 0.119 | 0.170 | 0.060 | 0.287 | 2.14e-05 | 2.05e6 | 0 |
| E: D + noise | 0.121 | 0.159 | 0.060 | 0.288 | 2.12e-05 | 2.56e7 | 0 |
| A (W learnable) | 1.851 | 2.242 | 0.925 | 0.017 | 3.38e-04 | 2.05e6 | 11 |
| B (W learnable) | 2.493 | 3.421 | 1.246 | 1.274 | 2.14e-05 | 2.05e6 | 20 |
| C (W learnable) | 1.864 | 2.301 | 0.932 | 0.055 | 2.14e-05 | 2.05e6 | 16 |
| C2 (W learnable) | 1.748 | 2.114 | 0.874 | 0.060 | 2.14e-05 | 2.05e6 | 0 |

Smallest budget: B = 1.6e4, D = 8.0e5 (20/20), C = never, C2 = 3.2e4 (18/20).
**D/B = 37.5× (p50), 105× (p95).** `diag(BᵀW) > 0` fraction = **0.508** (random).

### 3.5 SPICE verification (mandatory)

Trained γ from condition B, 5 mismatch draws × 4 test inputs per cell = 20 rows
per entry, pushed through the real junction-exact loop
(`agc_mismatch_sweep.settle_gain`, bisection on the SPICE-measured detector
current, 16 ngspice solves per equilibrium; 40 cells, 1.1 s). `det = ideal`:
junctions present but unmismatched. `det = mos`: IS 3 %, n 0.3 %, mirror 1 %.
All numbers in % of signal, medians.

| cell | det | pc SPICE before | pc SPICE after | pc surrogate after | rel. dev. | total err. before | total err. after | G_SPICE vs G_surrogate |
|---|---|---|---|---|---|---|---|---|
| N=8, σ=1 % | ideal | 0.627 | 0.129 | 0.129 | 0.00 % | 0.627 | 0.262 | 0.0053 % |
| N=8, σ=1 % | mos | 0.627 | 0.129 | 0.129 | 0.00 % | 3.597 | 3.494 | 3.50 % |
| N=8, σ=2 % | ideal | 1.599 | 0.322 | 0.322 | 0.00 % | 1.599 | 0.465 | 0.0030 % |
| N=8, σ=2 % | mos | 1.599 | 0.322 | 0.322 | 0.00 % | 4.507 | 3.771 | 4.10 % |
| N=32, σ=1 % | ideal | 0.952 | 0.058 | 0.058 | 0.00 % | 0.952 | 0.138 | 0.0058 % |
| N=32, σ=1 % | mos | 0.952 | 0.058 | 0.058 | 0.00 % | 4.279 | 3.920 | 4.14 % |
| N=32, σ=2 % | ideal | 1.895 | 0.119 | 0.119 | 0.00 % | 1.894 | 0.491 | 0.0042 % |
| N=32, σ=2 % | mos | 1.895 | 0.119 | 0.119 | 0.00 % | 2.605 | 1.434 | 1.45 % |

**Surrogate/SPICE agreement: max relative deviation 0.000 % on the per-channel
residual, far inside the 20 % threshold — but this agreement is partly structural
and must not be over-read.** The AGC loop gain `G` is a *scalar*, and the
per-channel residual is defined after `decompose_error()` divides out the
best-fit common factor α, so any scalar `G` gives the identical per-channel
number. The genuinely scale-sensitive checks are:

* **G itself**: SPICE vs surrogate agree to 0.003–0.006 % with an ideal detector,
  i.e. the bisected junction-exact loop equilibrium *is* `Γ/RMS(x(1+ε))` to five
  digits. The surrogate is validated, not assumed.
* **Total error** (no common-mode removal): with an ideal detector, training cuts
  it from 0.63/1.60/0.95/1.89 % to 0.26/0.47/0.14/0.49 % — the trained γ survives
  the real loop.
* **MOS detector**: `G` shifts by 1.4–4.1 %, and that shift dominates the total
  error (3.5/3.8/3.9/1.4 % after training), essentially unchanged by training.
  This is the expected behaviour and confirms the paper's decomposition: detector
  mismatch is a pure common-mode term. A static per-channel γ neither can nor
  should absorb it (it is input-dependent, since `G·RMS(x)` varies with x); in a
  full network it is removed by the next stage's gain, and it is removed here by
  the α fit.

**Caveat, stated plainly:** in this harness the VGA array is a per-channel
multiplier applied in Python, and only the RMS *detector* is inside the ngspice
netlist. The SPICE check therefore validates the loop gain and the common-mode
structure at junction-exact fidelity; it does **not** independently simulate the
per-channel VGA path. A per-channel-exact test needs a VGA netlist that does not
exist in this repo.

---

## 4. Verdict per pre-registered kill criterion

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **K1** | A below 0.35 σ ⇒ uninformative | A = **0.756 / 0.685 / 0.923 / 0.925 σ** (identical to untrained; γ converges to 1) | **NOT triggered.** The residual is a genuine mismatch effect and the experiment is informative. |
| **K2** | B not below 0.2 σ ⇒ §4/§10 claim killed | B = **0.155 / 0.140 / 0.057 / 0.057 σ**, all < 0.2 σ, in 4/4 cells | **PASSED — claim survives.** A gradient measured on the mismatched forward absorbs the mismatch; a gradient of the ideal model does not (5–16× reduction; B lands exactly on the analytic least-squares floor, so what remains is finite-training-set estimation error, not a mismatch residue). |
| **K3** | C above 2× B ⇒ FA relaxation killed | C = **81 / 41 / 90 / 45 σ**, i.e. 500–1600× B; **divergent in 80/80 draws** | **KILLED.** A *fully* random backward path does not work in this setting. |
| **K4** | D above 200× B's forward evaluations ⇒ impractical | D/B = **12.5× / 12.5× / 25× / 37.5×** (p50); p95 = 26 / 13 / 200 / 105 | **NOT triggered.** The perturbative route is practical at these N, with cost growing roughly ∝ N as SPSA theory predicts. D reaches B's residual (and the analytic floor) in all 80 draws. |
| E | does convergence survive 0.5 % output noise? | C2: 0.154 / 0.140 / 0.061 / 0.060 σ; D: 0.144 / 0.140 / 0.060 / 0.060 σ — indistinguishable from the noiseless runs | **Yes.** Both the sign-concordant-backward and the perturbative routes are unaffected; C stays divergent with or without noise. |

### 4.1 Why K3 failed, and what replaces it

This is a mechanism, not an optimiser accident, and it is measured:

1. **The trainable stage is diagonal, so nothing can align.** Lillicrap et al.'s
   alignment works because the *forward weight matrix* rotates towards `B`. Here
   the trainable parameter is a per-channel gain (the RMSNorm affine γ). Averaging
   the update over the input distribution leaves a **diagonal** dynamics matrix
   with entries ∝ `diag(BᵀW)_i`; channels with `diag(BᵀW)_i < 0` run *up* the loss.
   Measured fraction of channels with the correct sign: **0.469 / 0.512 / 0.505 /
   0.508** — i.e. a coin flip, exactly as a random `B` predicts. Half the channels
   diverge, in 80/80 draws.
2. **Making W learnable does not rescue it — it destroys the observable.** With a
   diagonal hidden stage, `γ_i → λ_i γ_i`, `W_{:,i} → W_{:,i}/λ_i` leaves the
   output unchanged, so per-channel γ is *gauge-unidentifiable* from the loss. The
   W-learnable rows confirm this: task MSE still falls to the same 1e-5/2e-5, yet
   the per-channel residual returns to ≈ the untrained level (0.76–1.86 σ) or
   worse — including for condition **B**, whose backward path is exact. The
   W-learnable variant is therefore uninformative for this metric, and is reported
   only to document the degeneracy.
3. **What the backward path actually needs is sign concordance, not reciprocity.**
   Condition C2 keeps everything that C relaxed — a *separate* backward matrix,
   *random* magnitudes, and the same mismatched per-channel backward gain
   (1+δ), δ ~ N(0,5 %) — and only restores the **signs** of W. It then matches B
   to within 4 % on every cell (0.153 / 0.140 / 0.060 / 0.060 σ) at the **same**
   forward-evaluation budget (1.6e4, i.e. 1–2× B), and survives the 0.5 % noise.

So §10's claim should be restated. "A separate, random, mismatched backward
crossbar suffices" is **false as written for a diagonal trainable stage**. What
the evidence supports is: *exact transposition (reciprocity) is not required —
magnitude mismatch of tens of percent and an independent backward array are
harmless — but sign agreement between the forward and backward paths is.* That is
the sign-symmetry result of Liao et al. 2016 / Xiao et al. 2018 rather than
Lillicrap-style feedback alignment, and it is a *stronger* hardware requirement
than §10 assumed: a backward crossbar may be uncalibrated, but it may not be
independently programmed.

Note also that condition **D removes this problem entirely**: it needs no backward
path at all, reaches the same floor as the exact-gradient condition in 80/80
draws, and costs only 12–38× more forward evaluations. For this stage the
perturbative/extremum-seeking route is the more robust mapping of X4, not the
fallback.

---

## 5. Limitations

1. **Surrogate vs SPICE.** Only the RMS detector is inside ngspice; the VGA array
   is a Python per-channel multiplier in this harness as well as in ours. The
   SPICE check validates the loop equilibrium (`G` to 0.006 %) and the
   common-mode/per-channel split, not the per-channel VGA path itself. §3.5.
2. **Scale-invariance of the metric.** The per-channel residual is invariant under
   any scalar loop gain, which is why SPICE and surrogate agree to 0.000 % there.
   The 20 %-disagreement test was consequently never at risk of firing on that
   column; the informative columns are `G` and the total error.
3. **Static graph, one trainable stage.** A single RMSNorm affine γ followed by one
   linear layer, M = 4 outputs, no non-linearity, no depth. Conclusions about
   feedback alignment are specific to a *diagonal* trainable stage; a deep network
   with full weight matrices could align and might behave differently. This
   experiment does not test that.
4. **Targets come from the ideal network**, so the task is exactly "undo the
   mismatch". Real training has a task objective that may not identify γ as
   sharply.
5. **Thermal noise** only as 0.5 % multiplicative i.i.d. output noise (condition E),
   applied during training in C/C2/D. No 1/f noise, no drift, no temperature
   sweep, no noise inside the SPICE detector.
6. **No mismatch drift / retraining cadence.** ε is fixed per draw; the experiment
   says nothing about how often a physically trained γ must be re-learned.
7. **Finite training set.** B, C2 and D all stop at the analytic least-squares
   floor (0.14–0.16 σ at N=8, 0.057 σ at N=32) set by `N_train` = 2000. The floor,
   not the gradient source, is what limits the final number; with more data all
   three would go lower together.
8. **Optimiser.** One SGD schedule per condition family, chosen after checking that
   it does not disadvantage D (Adam does). No per-condition hyper-parameter
   search; C's divergence was verified to be a sign-structure effect (§4.1) rather
   than a step-size effect, but no small-step-size run was made to show C merely
   stalling instead of diverging.
