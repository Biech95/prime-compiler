# Experiment: can physics-aware training absorb the GELU mismatch, and how does the perturbative route scale with N?

Two questions in one runner (`spice/train_gelu_absorption.py`):

* **Part A** — `exp_mismatch_calculus.md` §10.2 closes with a *design consequence*:
  "a GELU stage therefore needs either per-device trimming, a sigmoid-offset
  cancellation (chopping/autozero), or the physics-aware training of
  `exp_mismatch_absorption.md` — and the offset **is** static, so R9 says training
  can eat it." That last clause is an untested assertion. Part A tests it.
* **Part B** — `exp_mismatch_absorption.md` §4 (K4) measured the perturbative
  route's cost at **two** values of N only (8 and 32) and asserted "cost growing
  roughly ∝ N as SPSA theory predicts" from a single ratio. Part B measures the
  exponent over **four** values of N.

Status: **§1 (pre-registration) written 2026-09-16 before any run of the new
script.** §2 onwards appended after execution.

---

## 1. Pre-registration (written before running)

### 1.1 Part A — question and prior

The GELU stage of `predict_gelu_layernorm_sim.py` (`P11 · P6`: a translinear
Gilbert multiplier driven by a differential-pair sigmoid) has a per-channel error
of **8.5 %** at the MOS-typical corner — 12× the AGC-normalised VGA residual —
and §10.2's ablation attributes **half** of it to the new rule **R12**, the
sigmoid's input-referred offset entering the output as `δu·(1 − σ(u))`.

`exp_mismatch_absorption.md` showed that a per-channel trainable gain γ, trained
with a gradient measured on the mismatched forward (condition B), removes the VGA
mismatch of the RMSNorm/AGC stage down to the analytic least-squares floor
(0.7 σ → 0.14 σ). The calculus doc extrapolates that result to GELU. The
extrapolation is **not obviously valid**, and the reason is structural:

* At the AGC stage the mismatch is `y_i = γ_i^true · y⁰_i` with a *constant*
  per-channel factor. A per-channel gain is exactly the right inverse: the error
  lives in the span of the parameter.
* At the GELU stage the R12 term is `δu_i·(1 − σ(1.702 x_i))`, i.e. a
  per-channel error whose *size depends on the input x*. A per-channel affine
  `γ_i h_i + β_i` can only match the component of the error lying in
  `span{h_ref(x), 1}`. Whatever part of `x·σ(u)·(1 − σ(u))·δu` is orthogonal to
  `span{x·σ(u), 1}` over the input distribution **cannot be absorbed**.
* The same argument applies to the R1 (translinear) terms, but only partly:
  `b_j = ν_j·λ_j − ι_j` with `λ_j = ln(I_j/I_s)`. The **ι** part is a constant
  per-channel log-gain (absorbable by γ, exactly like the VGA mismatch); the
  **ν** part multiplies `ln|x|` and is therefore an x-dependent *slope* error
  (not absorbable).

So the pre-registered expectation is *not* "training eats it". The expectation is
a **partial** absorption, with the absorbable fraction being the constant-gain
part (ι terms, `g_out`, `g_tail`, `ε_VGA`) and the surviving part being the
shape-dependent terms (R12's offset lever and R1's ν slope lever). The
experiment's job is to measure the split.

### 1.2 Part A — model (frozen before running)

Per Monte-Carlo device draw `d` (drawn **exactly** as
`mismatch_calculus.gelu_draw(rng, N, MOS_TYP)`: `σ_Vt = 10 mV` input-referred,
`σ_K' = 3 %` on the pair's `I_s`, `σ_Is = 3 %` and `σ_n = 0.3 %` on every
translinear junction, `σ_mir = 1 %` on `g_tail`/`g_out`, `σ_VGA = 1 %`; the
reference junction `ι_r, ν_r` is **shared** across channels):

```
x ∈ R^N             x_i ~ N(0,1), then x_i ← sign(x_i)·clip(|x_i|, 0.1, 2.5)
                    (identical to mismatch_calculus.gelu_inputs)

analog GELU stage   h  = gelu_map(x, d)            mismatched, R1 × R3 × R12
ideal reference     h⁰ = x · σ(1.702 x)
per-channel affine  z_i = γ_i · h_i + β_i          trainable, init γ = 1, β = 0
frozen read-out     o  = W z,   W ∈ R^{M×N}, M = 4, W_ij ~ N(0,1)/√N, frozen
loss                L  = (1/M) ‖o − t‖²
target              t  = W h⁰(x)                   the ideal network
```

`W` is frozen in every condition. `exp_mismatch_absorption.md` §4.1 item 2 showed
that making `W` learnable makes a per-channel hidden stage
*gauge-unidentifiable* and destroys the observable; that variant is therefore not
repeated here.

Boxes: `γ ∈ [1e-3, 20]`, `β ∈ [−5, 5]`. Touching a box, or ending with a
per-channel residual worse than untrained, is recorded as `div`.

### 1.3 Part A — surrogate and its validation

The forward surrogate is **`mismatch_calculus.gelu_map` itself**, batched over
inputs (`gelu_map_batch`). It is *not* a new fit: it is the pre-existing calculus
map whose prediction (per-channel p50 = 8.483 %) was already confirmed
out-of-sample by the ngspice bench (8.518 %, ratio 0.996) in
`exp_mismatch_calculus.md` §9.1. Fitting free offset/gain/slope terms to the
bench would *add* free parameters to a map that already agrees to 0.4 %, so no
fit is performed; instead three checks are run and reported before any training
number is quoted:

* **V1 — batching equivalence.** `gelu_map_batch(X, d)[k]` equals
  `mismatch_calculus.gelu_map(X[k], d)` to machine precision (max relative
  deviation < 1e-12), on ≥ 200 random rows.
* **V2 — paired SPICE check, N = 8.** ≥ 5 (we will run 30) device draws, each
  evaluated *both* through ngspice (`predict_gelu_layernorm_sim.build_gelu_netlist`,
  unmodified, imported as a module) and through the surrogate, on the bench's own
  fixed input vector. Reported: per-draw paired relative deviation of the
  per-channel residual, and the surrogate's per-channel **p50**. **Gate: the
  surrogate p50 must be within 20 % of the SPICE p50 and within 20 % of the
  8.5 % headline.** If not, Part A switches to SPICE-in-the-loop at N = 8.
* **V3 — paired SPICE check, N = 32**, 10 draws, same protocol, plus an
  ideal-device control (all deviations zero ⇒ the netlist must reproduce
  `x·σ(1.702x)` to < 1e-4 relative).

### 1.4 Part A — metric

`mismatch_calculus.decompose(y_hat, y_ref)`, applied **per test sample** to
`z = γ ⊙ h + β` against `h⁰`:

```
α        = ⟨z, h⁰⟩ / ⟨h⁰, h⁰⟩                     common-mode gain shift
pc       = RMS_i(z/α − h⁰) / RMS_i(h⁰)            per-channel residual (relative)
```

Per device draw: the **median over a held-out test set of 500 inputs**. Tables
report p50/p95 **over the 20 Monte-Carlo device draws**. Secondary metric: task
MSE `⟨(o − t)²⟩` on the test set, and the common mode `α − 1`.

### 1.5 Part A — conditions

Identical device draws, training data, test data and `W` across A–D within a
draw (seeds spawned from one `SeedSequence`).

| | forward | backward | trains |
|---|---|---|---|
| **A** train-in-sim | **ideal** `h⁰` | exact gradient of the ideal model | (γ, β), then *deployed* on the mismatched forward and measured there |
| **B** physics-aware | **mismatched** `h` | exact gradient of the **ideal** model: `∂L/∂γ = (e Wᵀ)⊙h⁰`, `∂L/∂β = e Wᵀ` (Wright et al. 2022) | (γ, β) |
| **C2** sign-concordant | **mismatched** `h` | `B_sc = |randn|/√N ⊙ sign(W)` in place of `Wᵀ`, local activation the **physical** `h`, and a per-channel backward gain `(1+δ)`, **δ ~ N(0, 20 %)** | (γ, β) |
| **D** perturbative | **mismatched** `h` | **none**. Two-sided Rademacher SPSA on the joint 2N-vector `(γ, β)`, `σ_p = 0.02`, probes share the minibatch | (γ, β) |
| **LS** | — | analytic least-squares minimiser of the training loss over `(γ, β)` for frozen `W` (the loss is quadratic: `o = [W·diag(h) | W]·[γ;β]`) | — |

Condition **C** (fully random backward) is **not** run: it was killed in 80/80
draws in `exp_mismatch_absorption.md` and re-killed in `exp_sign_concordance.md`;
the brief's condition set replaces it with C2, as the task specifies.

Optimiser: plain SGD, cosine-decayed lr, batch 64, 32 000 steps (2.05e6 forward
evaluations) for A/B/C2; SPSA batch 16, 800 000 steps (2.56e7 forward
evaluations) for D — the same budgets as `exp_mismatch_absorption.md`.

**lr-robustness arm (pre-registered).** The lr schedule of the previous
experiment was tuned for the RMSNorm stage, whose signal scale differs. To keep a
KA1 verdict from being a step-size artefact, every gradient condition is run at
`lr0 ∈ {2.0, 0.5, 0.2}` and every SPSA condition at `lr0 ∈ {0.5, 0.1}` (all with
`lr1 = lr0/2000`), and the arm with the **lowest final training loss** is
selected per condition per draw. Selection is on the *training* loss, never on
the test-set residual, so the held-out metric is not peeked at.

Cells: **N ∈ {8, 32}**, 20 Monte-Carlo device draws each. Master seed **5171**
(no seed reused from the previous experiments). `N_train = 2000`, `N_test = 500`.

### 1.6 Part A — pre-registered kill criteria

| ID | Criterion | Consequence |
|---|---|---|
| **KA0** | Condition A (train-in-sim, deploy-on-circuit) already reaches below 0.25× untrained | **KILL-uninformative** — the residual was not a mismatch effect |
| **KA1** | Condition **B** does **not** bring the per-channel residual below **0.25× the untrained value** (8.5 % → < 2.1 %) in both cells | the offset-type GELU error is **NOT absorbable by a per-channel affine**, contrary to §10.2's "the offset is static, so R9 says training can eat it". The report must then name **which component survives** (R12 offset lever vs R1 ν slope lever vs R3 gains) |
| **KA2** | Condition **C2** ends **> 2× B** | sign concordance is insufficient for this stage |

**Component attribution (run whether or not KA1 fires).** Condition B is repeated
on five *masked* device draws in which only one mechanism group is non-zero, the
rest set to zero — the same ablation style as §10.2:

| mask | non-zero entries | calculus rule |
|---|---|---|
| `R3` | `eps_vga`, `g_out`, `g_tail` | pure per-channel gain |
| `R1_iota` | `iota_x, iota_s, iota_o, iota_r` | translinear **constant** log-gain |
| `R1_nu` | `nu_x, nu_s, nu_o, nu_r` | translinear **slope** (∝ ln I) |
| `R12_off` | `vos`, `iota_a`, `iota_b` | sigmoid input offset |
| `R12_slope` | `nu_a`, `nu_b` | sigmoid slope (∝ \|u\|) |

Reported: untrained and post-B per-channel residual per mask. The pre-registered
prediction is `R3` and `R1_iota` → ≈ 0 after training; `R12_off` and `R1_nu` →
largely surviving.

### 1.7 Part A — mandatory SPICE verification

After training, for **≥ 5 device draws per cell** (we will run 5 draws × 4 test
inputs = 20 rows per cell) the trained `(γ, β)` of condition B are pushed through
the **real ngspice GELU bench** — `predict_gelu_layernorm_sim.build_gelu_netlist`
/ `gelu_one`, imported unmodified — with the *same* device draw `d` used in
training, and `decompose()` is applied to `γ ⊙ i_out + β`. Reported **before**
(γ = 1, β = 0) and **after** training, next to the surrogate's own numbers.
**If SPICE and surrogate disagree by more than 20 % relative on the per-channel
residual, this is stated loudly.** Unlike the AGC check of
`exp_mismatch_absorption.md` §3.5, this quantity is **not** scale-degenerate: the
GELU stage has no loop pinning a scalar gain, and the per-channel error is not a
single common factor, so the comparison is informative.

### 1.8 Part B — question, protocol and kill criterion

Runner: same script, `--part b`. It imports `spice/train_mismatch_absorption.py`
as a module and reuses `make_inputs`, `norm_ideal`, `norm_mis`, `train_gradient`,
`train_perturbative`, `evaluate`, `ls_optimum` and the ladders **unchanged**; the
stage is the RMSNorm/VGA stage at **σ_VGA = 1 %**, the model, metric, optimiser
and `σ_p = 0.02` are those of `exp_mismatch_absorption.md` §1.3/§1.4/§2.1.

* Cells: **N ∈ {8, 32, 128, 512}**, **10 draws each**, master seed **4127**
  (fresh).
* Only conditions **B** (exact ideal backward) and **D** (SPSA) are run — the
  cost ratio is the whole question.
* Budget ladders: B = `[250, 500, …, 32000] × 64` forward evaluations;
  D = `[6250, …, 800000] × 2 × 16`, **extended by two rungs** (1.6e6, 3.2e6
  steps ⇒ up to 1.024e8 evaluations) so that large N has headroom. Each budget is
  an **independent run with its own cosine schedule**, exactly as in §2.2 of the
  previous experiment.
* `target = 1.2 × B_final`, `B_final` = B's residual at its largest budget.
  `cost_X` = smallest budget on X's ladder whose final residual ≤ target.
* **Primary cost estimate: log-interpolated.** The rung ladder is a factor-2 grid,
  which quantises any exponent estimate into steps of 1/log₂(N-ratio); the
  previous experiment's "∝ N" claim rests on exactly such a quantised pair. We
  therefore interpolate the crossing budget log-linearly in
  (log budget, log residual) between the two bracketing rungs. The rung-based
  number is reported alongside.
* Ratio `R(N) = median over draws of (cost_D / cost_B)`. Fit
  `log R = a + p·log N` by least squares over the four N.

| ID | Criterion | Consequence |
|---|---|---|
| **KB** | the fitted exponent `p` lies **outside [0.7, 1.3]** | the "cost ∝ N as SPSA theory predicts" statement of `exp_mismatch_absorption.md` §4 (K4) is **wrong**; the measured exponent replaces it |

If the interpolated and rung-based exponents straddle the [0.7, 1.3] boundary,
that is stated loudly and the verdict is declared **undecided** rather than
forced. If N = 512 does not finish inside the wall-clock budget, the
extrapolation is reported **and labelled as an extrapolation**.

Wall clock, worker count and whether every cell completed are reported.

---

## 2. Method as actually run

Runner: `spice/train_gelu_absorption.py` (new; imports `mismatch_calculus.py`,
`predict_gelu_layernorm_sim.py` and `train_mismatch_absorption.py` unmodified).
Environment: ngspice-46 (`~/.local/bin/ngspice`, VT = 0.025864890 V),
Python 3.14.3, numpy 2.4.2, `OPENBLAS_NUM_THREADS=1`, **6 workers**,
`multiprocessing` start method forced to `fork` (3.14 defaults to `forkserver`,
which re-imports the module per worker).

```
OPENBLAS_NUM_THREADS=1 python3 train_gelu_absorption.py --part a --workers 6
# 40 (N, draw) tasks; validation + training 1557 s, SPICE verification 0.1 s,
# total 1557 s
OPENBLAS_NUM_THREADS=1 python3 train_gelu_absorption.py --part b --workers 6
```

Raw results: `spice/out/gelu_absorption_A.json`, `spice/out/gelu_absorption_B.json`;
console logs `spice/out/run_gelu_A.log`, `spice/out/run_gelu_B.log`.

### 2.0 Deviations from the pre-registration (declared)

1. **Condition D2 was added after D failed, as a post-hoc control** (exactly the
   role C2 played in `exp_mismatch_absorption.md`). It is condition D with a
   decay term `0.01·mean(β²)` added to the objective. Reason: a pilot draw showed
   D reaching the *same task MSE* as B while its per-channel residual exploded to
   84 %, with 100 % of `β` lying in `null(W)`. `β` enters the loss only through
   `Wβ`, and `null(W)` has dimension `N − M` (4 at N = 8, 28 at N = 32), so a
   backward-free estimator random-walks there. D2 is exploratory and does **not**
   affect any pre-registered verdict; it only diagnoses D's failure. §1.6's
   criteria are decided by A, B and C2 alone.
2. **Part B ladders were extended downward as well as upward**, before the
   production run, after a pilot showed that at N = 8 *both* B and D already
   reach the target at `train_mismatch_absorption`'s **lowest** rung (250 and
   6250 steps). With the original ladder the cost estimate is left-censored at
   both N = 8 and N = 32 — which is precisely the pair from which
   `exp_mismatch_absorption.md` §4 inferred "cost ∝ N". Rungs added:
   `[2, 4, 8, 15, 30, 60, 125]` steps below the gradient ladder and
   `[25, 50, 100, 200, 400, 800, 1600, 3200]` below the perturbative one, plus
   `[1.6e6, 3.2e6]` above it. Nothing else in the harness was changed.
3. **Part B: a second learning-rate arm for D** (`lr0 = 0.1` beside the harness
   default 0.5), declared post-hoc after the same pilot showed the SPSA ladder at
   N = 128 going *non-monotone* (residual rising to 27 % at intermediate budgets
   before recovering) — i.e. the harness's fixed step size is too large at large
   N. The **pre-registered** arm (harness default) decides KB; the best-of-two
   arm is reported beside it so that "SPSA needs more evaluations" is separated
   from "SPSA needs a smaller step".
4. **Part A baseline is 11.1 % / 13.8 %, not 8.5 %.** The 8.5 % headline of
   `exp_mismatch_calculus.md` §9.1 is measured on **one fixed input vector**
   (`gelu_inputs`, seed 2026) with the median taken over device draws. The
   training harness needs an input *distribution*, and the per-channel residual
   depends on x (the R12 lever `1 − σ(1.702x)` is large exactly where the output
   is small), so taking the median over inputs first and over draws second gives
   a larger number. Both are reported in every table; the bench protocol
   reproduced inside this harness gives 9.19 % (N = 8), within 8 % of the
   published 8.5 %. **The KA1 threshold is computed from this harness's own
   untrained value** (0.25 × 11.078 % = 2.769 % at N = 8), which is the
   conservative reading — using 0.25 × 8.5 % = 2.125 % would make KA1 fire harder.

---

## 3. Part A — results

### 3.1 Surrogate: equivalence and validation against the real bench

| check | result | gate |
|---|---|---|
| **V1** batching: `gelu_map_batch` vs `mismatch_calculus.gelu_map`, 200 rows | max relative deviation **0.000e+00** (bit-identical) | < 1e-12 ✓ |
| **V0** ngspice ideal-device control, N = 8 | max relative deviation from `x·σ(1.702x)` = **2.59e-06** | < 1e-4 ✓ |
| **V2** paired SPICE vs surrogate, N = 8, **30 draws** | per-channel p50 **SPICE 9.003 %** vs **surrogate 8.645 %** (deviation **4.14 %**); p95 16.26 vs 16.67 %; paired per-draw relative deviation p50 **5.55 %**, max 42.2 % | < 20 % ✓ |
| **V3** paired SPICE vs surrogate, N = 32, **10 draws** | per-channel p50 **SPICE 12.918 %** vs **surrogate 12.958 %** (deviation **0.31 %**); paired per-draw deviation p50 2.85 %, max 11.8 % | < 20 % ✓ |
| headline check | surrogate p50 at N = 8 = 8.645 % vs the published **8.5 %** → 1.7 % | < 20 % ✓ |

**The surrogate gate passes; no fit was needed and none was performed, and
SPICE-in-the-loop was not required.** Note the paired *per-draw* spread (max
42 % at N = 8): the calculus map is a first-order algebra, the netlist solves the
junction physics exactly, so individual draws can differ by tens of percent even
though the distributions agree to 0.3–4 %. This matters in §3.4.

### 3.2 Training, N = 8 (20 device draws)

Per-channel residual in % of signal; `× untr` = fraction of the untrained value
(**KA1 threshold = 0.25**); `div` = divergent draws out of 20; `<thr` = fraction
of draws individually below the threshold; `β_null` = median fraction of `β`'s
norm lying in `null(W)`.

untrained: **pc p50 = 11.078 %**, p95 15.002 %, common mode 12.156 %,
task MSE 9.52e-03. (Bench fixed-x protocol on the same draws: **9.192 %**.)
KA1 threshold = **2.769 %**.

| cond | pc p50 | pc p95 | × untr | cm p50 | task MSE | fwd evals | div | `<thr` | β_null |
|---|---|---|---|---|---|---|---|---|---|
| untrained | 11.078 | 15.002 | 1.000 | 12.156 | 9.52e-03 | – | – | – | – |
| **LS** (analytic floor) | **3.881** | 5.536 | **0.350** | 1.294 | 4.07e-04 | 0 | 0 | 0.10 | 0.051 |
| LS, γ only (no β) | 3.915 | 5.740 | 0.353 | 1.452 | 4.58e-04 | 0 | 0 | 0.10 | 0.000 |
| **A** train-in-sim | 11.078 | 15.002 | **1.000** | 12.156 | 9.52e-03 | 6.14e6 | 0 | 0.00 | 0.000 |
| **B** physics-aware | **4.376** | 112.11 | **0.395** | 1.467 | 4.07e-04 | 6.14e6 | 7 | 0.10 | 0.000 |
| **C2** sign-concordant (δ 20 %) | 4.172 | 221.28 | 0.377 | 1.369 | 4.59e-04 | 6.14e6 | 3 | 0.05 | 0.684 |
| **D** perturbative | **79.98** | 186.55 | **7.220** | 19.482 | 4.07e-04 | 5.12e7 | **20** | 0.00 | **1.000** |
| D2 = D + β decay *(post-hoc)* | 3.940 | 5.640 | 0.356 | 1.384 | 4.07e-04 | 5.12e7 | 4 | 0.10 | 0.642 |

lr arms selected (by *training* loss): B {2.0: 10, 0.5: 6, 0.2: 4};
C2 {2.0: 5, 0.5: 7, 0.2: 8}; D {0.1: 17, 0.5: 3}; D2 {0.1: 15, 0.5: 5}.
Sign-concordant backward loop gain `diag(BᵀW) > 0` fraction = **1.000**.

### 3.3 Training, N = 32 (20 device draws)

untrained: **pc p50 = 13.761 %**, p95 16.623 %, common mode 6.496 %,
task MSE 1.08e-02. (Bench fixed-x protocol: **12.665 %**.)
KA1 threshold = **3.440 %**.

| cond | pc p50 | pc p95 | × untr | cm p50 | task MSE | fwd evals | div | `<thr` | β_null |
|---|---|---|---|---|---|---|---|---|---|
| untrained | 13.761 | 16.623 | 1.000 | 6.496 | 1.08e-02 | – | – | – | – |
| **LS** (analytic floor) | **4.400** | 5.606 | **0.320** | 0.870 | 6.81e-04 | 0 | 0 | 0.05 | 0.214 |
| LS, γ only (no β) | 4.769 | 5.765 | 0.347 | 0.900 | 7.60e-04 | 0 | 0 | 0.05 | 0.000 |
| **A** train-in-sim | 13.761 | 16.623 | **1.000** | 6.496 | 1.08e-02 | 6.14e6 | 0 | 0.00 | 0.000 |
| **B** physics-aware | **4.376** | 5.500 | **0.318** | 0.833 | 6.82e-04 | 6.14e6 | 0 | 0.05 | 0.000 |
| **C2** sign-concordant (δ 20 %) | 4.709 | 5.705 | 0.342 | 0.908 | 6.84e-04 | 6.14e6 | 0 | 0.05 | 0.792 |
| **D** perturbative | **76.95** | 552.97 | **5.592** | 10.768 | 6.81e-04 | 5.12e7 | **20** | 0.00 | **1.000** |
| D2 = D + β decay *(post-hoc)* | 6.016 | 7.932 | 0.437 | 1.026 | 6.83e-04 | 5.12e7 | 2 | 0.00 | 0.981 |

lr arms: B {2.0: 8, 0.5: 2, 0.2: 10}; C2 {2.0: 10, 0.5: 3, 0.2: 7};
D {0.1: 18, 0.5: 2}; D2 {0.1: 18, 0.5: 2}. `diag(BᵀW) > 0` fraction = 1.000.

### 3.4 Component attribution — which mechanism survives the affine

Per-channel residual p50 [%] with **only one mechanism group non-zero**, before
and after fitting the *optimal* per-channel affine (the LS floor, so this is the
parametrisation's limit and not an optimiser statement):

| mask | rule | N = 8 untrained → trained | absorbed | N = 32 untrained → trained | absorbed |
|---|---|---|---|---|---|
| `R3` `eps_vga, g_out, g_tail` | per-channel gain | 0.980 → 0.064 | **93.5 %** | 1.447 → 0.180 | **87.5 %** |
| `R1_iota` `ι_x, ι_s, ι_o, ι_r` | translinear constant log-gain | 2.878 → 0.060 | **97.9 %** | 4.320 → 0.182 | **95.8 %** |
| `R1_nu` `ν_x, ν_s, ν_o, ν_r` | translinear slope (∝ ln I) | 7.404 → 0.253 | **96.6 %** | 10.401 → 0.295 | **97.2 %** |
| **`R12_off`** `V_os, ι_a, ι_b` | **sigmoid input offset** | **5.342 → 3.812** | **28.6 %** | **5.912 → 4.366** | **26.1 %** |
| `R12_slope` `ν_a, ν_b` | sigmoid slope (∝ \|u\|) | 0.106 → 0.095 | 10.2 % | 0.101 → 0.198 | −96 %\* |

\* the `R12_slope` term is ~0.1 %, i.e. two orders below the others; its
"absorption" is numerical noise around zero and carries no information.

This is the answer KA1 asks for. The **pre-registered prediction was half right**:
`R3` and `R1_iota` are absorbed as predicted (93–98 %), the sigmoid offset
`R12_off` survives as predicted (26–29 % absorbed) — but `R1_nu`, predicted to
survive as a slope lever, is absorbed to 97 %. The reason is quantitative: the
translinear lever is `λ = ln(I/I_s) ≈ 23`, while `ln|x|` varies by only
`ln(2.5/0.1) = 3.2` over the input range, so `ν·λ` is ~93 % a *constant*
per-channel log-gain and only ~7 % an x-dependent slope. R1's ν term is the
largest single untrained contributor (7.4 / 10.4 %) and almost all of it is
learnable.

### 3.5 Mandatory SPICE verification (condition B, 5 draws × 4 inputs per cell)

Trained `(γ, β)` from condition B pushed through the **real ngspice GELU
netlist** with the same device draw. All numbers % of signal, medians over the
20 rows. `tot` = residual without common-mode removal.

| cell | pc SPICE **before** | pc surr before | dev | pc SPICE **after** | pc surr after | dev | tot SPICE before | tot SPICE after | SPICE after/before |
|---|---|---|---|---|---|---|---|---|---|
| N = 8 | **11.226** | 11.358 | 1.17 % | **5.710** | 4.821 | **18.43 %** | 16.870 | 5.706 | **0.509** |
| N = 32 | **15.974** | 15.654 | 2.04 % | **4.939** | 4.242 | **16.42 %** | 18.788 | 4.997 | **0.309** |

Max relative SPICE/surrogate deviation on the per-channel residual: **18.43 %**,
inside the 20 % threshold but **not comfortably so, and it must be read
carefully**: before training the two agree to 1–2 %, after training to only
16–18 %. That asymmetry is expected and is *not* a surrogate failure — it is the
per-draw spread of §3.1 becoming visible once the absorbed part has been
subtracted. The trained affine cancels the surrogate's version of the constant
per-channel gain; what remains (≈ 5 %) is of the same order as the per-draw
surrogate/SPICE difference itself, so the relative deviation grows even though
the absolute one does not. **The verdict is unaffected: on the real circuit the
residual falls only to 0.51× (N = 8) and 0.31× (N = 32) of its untrained value,
both above the 0.25× threshold.** KA1 fires on SPICE as well as on the surrogate.

---

## 4. Part A — verdicts

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **KA0** | A below 0.25× untrained ⇒ uninformative | A = **11.078 / 13.761 %** = **1.000×** untrained in both cells (γ → 1, β → 0 bit-for-bit) | **NOT triggered.** The residual is a genuine mismatch effect; the experiment is informative. |
| **KA1** | B not below 0.25× untrained ⇒ the offset-type GELU error is not absorbable by a per-channel affine | B = **4.376 % (0.395×)** at N = 8 and **4.376 % (0.318×)** at N = 32, against thresholds 2.769 % / 3.440 %. The **analytic least-squares floor** — the best per-channel affine that exists — is **3.881 % (0.350×)** and **4.400 % (0.320×)**, also above threshold. On the real ngspice bench: **0.509× / 0.309×**. | **TRIGGERED in both cells, on the surrogate, at the analytic floor, and in SPICE.** |
| **KA2** | C2 above 2× B ⇒ sign concordance insufficient | C2/B = **0.953** (N = 8), **1.076** (N = 32) | **NOT triggered.** A sign-concordant backward with random magnitudes and a **20 %** per-channel backward gain mismatch matches the exact-adjoint condition to within 8 %. |

### 4.1 What KA1 means, and which component remains

**The claim in `exp_mismatch_calculus.md` §10.2 — "the offset *is* static, so R9
says training can eat it" — is false as written for a per-channel affine.**
"Static" is necessary but not sufficient. What a per-channel affine can absorb is
the component of the error lying in `span{h⁰(x), 1}` over the input
distribution; the sigmoid's input-referred offset does not lie there.

Concretely, the mismatched stage is `h_i(x) ≈ (1+g_i)·x·σ(1.702x + δu_i)`, so to
first order the error is

```
h_i − h⁰ = g_i·h⁰(x)            [absorbable: exactly γ_i]
         + δu_i · x·σ(u)(1 − σ(u))   [R12: NOT in span{x·σ(u), 1}]
```

`x·σ(u)(1 − σ(u))` is a bump peaking near |x| ≈ 1 and decaying to zero at both
ends, while `h⁰ = x·σ(u)` grows linearly for x > 0 and vanishes for x ≪ 0. The
two are close to orthogonal over the input distribution, and the measurement
confirms it: **the sigmoid-offset mask is absorbed only 26–29 %, while every
gain-like mask is absorbed 88–98 %** (§3.4). Adding `β` buys almost nothing over
`γ` alone (3.881 vs 3.915 % at N = 8; 4.400 vs 4.769 % at N = 32) — the surviving
error is not a constant.

**Design consequence, replacing §10.2's.** Of the 8.5–13.8 % per-channel GELU
error, roughly **two thirds is learnable** by a per-channel affine and
**one third — the sigmoid pair's input offset, the R12 term — is not**. A GELU
stage built this way therefore needs offset cancellation at the *pair itself*
(chopping / autozero / trimming, i.e. removing `δu` where it is generated), or a
trainable element with more than two parameters per channel (e.g. a trainable
input offset `σ(1.702x + b_i)`, which *is* in the right span and would absorb
`δu` exactly). Physics-aware training alone leaves ≈ 4–5 % per-channel error.
The claim that survives is the weaker one: training removes the **gain-type**
half of the GELU error, including the large translinear `ν·ln I` term that the
calculus classified as a slope lever but which is 93 % constant in practice.

### 4.2 Why condition D failed, and what it says about β

Condition D reaches the **same task MSE as B to three digits** (4.07e-04 /
6.81e-04) while its per-channel residual is **80 % / 77 %**, divergent in
**40/40** draws, with the median fraction of `‖β‖` lying in `null(W)` equal to
**1.000**. This is a clean identifiability statement, not an optimiser accident:

* `β` enters the loss only through `Wβ`. With `W ∈ R^{4×N}`, `null(W)` has
  dimension `N − 4` (4 at N = 8, 28 at N = 32), and any `β` component there is
  **invisible to every loss-based method**.
* Gradient conditions are protected *by construction*: the β-update is
  `mean(∂L/∂z)`, which for A/B lies in `row(W)` — measured `β_null = 0.000`. C2
  uses a different backward matrix and picks up a null component (0.68 / 0.79)
  but keeps it small (‖β‖ ≈ 0.02).
* A backward-free estimator has no such projection and **random-walks in the null
  space**: `β_null = 1.000`, `‖β‖ ≈ 1.4`.
* Adding a decay `0.01·mean(β²)` (condition D2 — physically just leakage of an
  analog offset store) restores it: **3.940 % / 6.016 %**, i.e. the same range as
  B, at the same task MSE.

This is the additive-offset analogue of the multiplicative gauge degeneracy of
`exp_mismatch_absorption.md` §4.1 item 2. **Practical rule: a trainable additive
offset in an analog stage must be regularised (or measured), because the task
loss does not determine it.** Note that this also means the per-channel residual
`pc` is *not* a function of the loss alone — two parameter sets with identical
task MSE can differ by a factor 20 in `pc`.

### 4.3 The B tail at N = 8

Condition B's p95 at N = 8 is 112 % with 7/20 draws flagged `div`, while its p50
sits at the analytic floor and N = 32 shows no such tail (p95 5.5 %, 0 div). The
cause is conditioning, not the gradient source: `γ` is identified through
`(WᵀW) ⊙ Σ_k h_k h_kᵀ`, and `WᵀW` has rank 4; at N = 8 that Hadamard product is
close to singular for some draws, so the flat directions let a large-`lr` arm
wander while the *training* loss (the arm-selection criterion) barely moves. The
LS floor, which solves the same system directly with a ridge, has 0/20
divergences. **The KA1 verdict does not rest on this tail** — it is decided by
p50 and by the LS floor, both of which are above threshold. It is reported as a
limitation (§6, item 4).

---
## 5. Part B — perturbative cost scaling

RMSNorm/VGA stage, `σ_VGA = 1 %`, harness `train_mismatch_absorption.py` reused
unchanged, 10 device draws per N, master seed 4127, 6 workers, wall clock
**1713 s** (40 tasks). `cost` = forward evaluations to reach
`1.2 × B_final`; `interp` = log-log interpolated crossing (primary),
`rung` = smallest ladder rung (secondary, the previous experiment's definition).
**All four cells completed, N = 512 included.**

### 5.1 Measured costs

| N | untrained pc % | B pc % | D pc % | LS pc % | cost B (rung / interp) | cost D (rung / interp) | **D/B rung** | **D/B interp** | D/B p95 (interp) | hits D | s / draw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.793 | 0.162 | 0.153 | 0.163 | 1.92e3 / 1.29e3 | 9.60e3 / 6.64e3 | **4.17** | **4.46** | 13.5 | 10/10 | 2.3 |
| 32 | 0.893 | 0.0568 | 0.0604 | 0.0567 | 1.60e4 / 1.25e4 | 4.00e5 / 3.01e5 | **12.5** | **16.75** | 93.6 | 10/10 | 4.1 |
| 128 | 0.991 | 0.0249 | 0.0280 | 0.0248 | 1.92e5 / 1.41e5 | 3.84e7 / 2.44e7 | **200** | **210.5** | 339 | 10/10 | 174.3 |
| 512 | 0.993 | 0.0238 | **1.623** | 0.0237 | 1.02e6 / 8.00e5 | **never** | — | — | — | **0/10** | 691.9 |

At **N = 512 the pre-registered perturbative arm does not converge at all**: at
the harness's fixed step size `lr0 = 0.5` its final residual (1.62 %) is *worse
than untrained* (0.99 %) at every rung up to 1.02e8 forward evaluations. This is
not a budget statement, it is an instability: the ladder is non-monotone, the
residual rising to tens of percent at intermediate budgets before (at N = 128)
recovering. The SPSA gradient estimate has variance ∝ N, so a step size tuned at
N = 8 is too large at N ≥ 128.

### 5.2 Fitted exponent

`log(D/B) = a + p·log N`, least squares:

| arm | cells used | fit | R² | exponent p | KB |
|---|---|---|---|---|---|
| **pre-registered** (harness lr0 = 0.5), interp — *primary* | 8, 32, 128 (512 never converges) | `D/B = 0.202·N^1.390` | 0.968 | **1.390** | **TRIGGERED** |
| pre-registered, rung | 8, 32, 128 | `D/B = 0.173·N^1.396` | 0.941 | **1.396** | **TRIGGERED** |
| post-hoc best-of-lr {0.5, 0.1}, interp | 8, 32, 128, 512 | `D/B = 0.420·N^0.974` | 0.927 | **0.974** | not triggered |
| post-hoc best-of-lr, rung | 8, 32, 128, 512 | `D/B = 0.444·N^0.954` | 0.899 | **0.954** | not triggered |

Median interpolated cost per lr arm (forward evaluations): N = 8 → 6.6e3 (0.5) /
1.2e4 (0.1); N = 32 → 3.0e5 / 2.4e5; N = 128 → 2.4e7 / 1.0e7; N = 512 →
**never** (0.5) / 9.0e7 (0.1). The smaller step wins from N = 32 upward and is
the only arm that converges at N = 512 (8/10 draws).

### 5.3 Verdict on KB, and a correction to the previous experiment

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **KB** | fitted exponent outside [0.7, 1.3] | primary (pre-registered arm, interpolated): **p = 1.390**, R² = 0.968 | **TRIGGERED** |

Both primary and secondary estimates of the pre-registered arm agree (1.390 /
1.396) and both lie clearly outside the interval, so the "straddling" escape
clause of §1.8 does not apply. **`exp_mismatch_absorption.md` §4's statement
"cost growing roughly ∝ N as SPSA theory predicts" is wrong as a description of
the harness as it stands.** The corrected statement has two parts:

1. **At a fixed step size the cost grows as ≈ N^1.4 and then stops working
   entirely** — at N = 512 the perturbative route never reaches the
   exact-gradient quality at any budget on the ladder (up to 1.02e8 forward
   evaluations, 100× B's own cost).
2. **The ∝ N law is recovered only if the step size is annealed with N**:
   best-of-lr gives **p = 0.97 (R² = 0.93)** across all four N including 512.
   So SPSA theory is not violated — but the "practical at these N" conclusion of
   K4 was an artefact of the two N values tested and of a step size that happened
   to suit them.

**A second correction.** The previous experiment reported `D/B = 12.5×` at N = 8.
Re-measured here with a ladder that extends below its lowest rung, `D/B = 4.5×`
at N = 8 — the old number was **left-censored**: both B and D already reached the
target at the *first* rung of the old ladder (250 and 6250 steps), so the ratio
measured the ladder, not the algorithms. The N = 32 value (12.5 rung / 16.8
interp) reproduces. This censoring is precisely why the old two-point "∝ N"
inference was unsafe, and it is why §2.0 deviation 2 extended the ladder
downward before the production run.

---

## 6. Limitations

1. **The GELU surrogate is a first-order algebra.** Distributions agree with the
   junction-exact netlist to 0.3–4 % on the per-channel p50, but *individual*
   draws differ by up to 42 % (N = 8). After training, when the absorbable part
   has been removed, that per-draw spread becomes the dominant term: the
   SPICE/surrogate deviation on the post-training residual is 16–18 %, inside the
   pre-registered 20 % but only just. A tighter statement would need
   SPICE-in-the-loop training, which was not run.
2. **Wiring-ideal fidelity.** As in `spice/README.md`: all exponential physics is
   real device I-V, but current mirrors and the differential copy are ideal
   sources carrying a stated gain error. No parasitics, no finite output
   impedance, no noise, no temperature.
3. **`pc` is not a function of the loss.** `β` is identifiable only modulo
   `null(W)` (dimension N − 4), so two parameter sets with identical task MSE can
   differ 20-fold in per-channel residual (§4.2). All conditions are therefore
   compared at equal task MSE as well as on `pc`.
4. **Condition B has a heavy tail at N = 8** (p95 = 112 %, 7/20 draws flagged)
   caused by the near-singularity of `(WᵀW) ⊙ Σ h hᵀ` at N = 8 with M = 4; the
   lr arm is selected on training loss, which barely distinguishes those flat
   directions. The KA1 verdict rests on the p50 and on the LS floor, neither of
   which has this tail, but the tail means the *reliability* of physics-aware
   training at small N is worse than the p50 suggests.
5. **Static graph, one trainable stage, targets from the ideal network.** One
   GELU stage, one frozen 4×N read-out, no depth, no non-linearity after it. The
   task is exactly "undo the mismatch"; a real objective may identify (γ, β) less
   sharply.
6. **One device corner, one input distribution.** MOS-typical only, `|x|`
   clipped to [0.1, 2.5]. The R12 lever grows in the left tail, so a distribution
   with more mass at negative x would make KA1 fire harder, and one clipped to
   x > 0 would make it fire less.
7. **No mismatch drift, no noise during Part A training.** ε and d are fixed per
   draw; the 0.5 % output-noise condition of `exp_mismatch_absorption.md` was not
   repeated here.
8. **Part B: 10 draws per cell and a two-point lr sweep.** The post-hoc
   best-of-lr exponent (0.97) is the best of two step sizes, not of a tuned
   schedule; a finer sweep could lower it further. The N = 512 cell of that arm
   converges in 8/10 draws, not 10/10, so its cost is a slight underestimate of
   the worst case.
9. **Part A condition D2 and Part B's second lr arm are post-hoc.** Both are
   declared in §2.0 and neither decides a pre-registered verdict, but both should
   be re-run as pre-registered tests before being relied on.

---

## 7. Summary

* **KA0 not triggered**, **KA1 TRIGGERED in both cells**, **KA2 not triggered**,
  **KB TRIGGERED**.
* Physics-aware training removes about **two thirds** of the GELU per-channel
  error (11.1 → 4.4 % at N = 8, 13.8 → 4.4 % at N = 32; on the real ngspice
  bench 11.2 → 5.7 % and 16.0 → 4.9 %) and **cannot do better**: the analytic
  optimum over all per-channel affines is 3.9 / 4.4 %.
* What survives is **the sigmoid pair's input-referred offset (R12)** — absorbed
  only 26–29 % — because `δu·x·σ(u)(1−σ(u))` is close to orthogonal to
  `span{x·σ(u), 1}`. Every gain-like mechanism, including the translinear `ν·ln I`
  term the calculus classed as a slope lever, is absorbed 88–98 %.
* `exp_mismatch_calculus.md` §10.2's "the offset is static, so R9 says training
  can eat it" must be replaced by: *static is necessary, not sufficient — an
  error is absorbable only if it lies in the span of the trainable parameters.*
  For this GELU the fix is offset cancellation at the pair, or a trainable input
  offset rather than a trainable output affine.
* Sign concordance is **sufficient** for this stage too (C2/B = 0.95–1.08) even
  with a 20 % backward gain mismatch — a second, independent confirmation of
  `exp_sign_concordance.md`.
* The backward-free route needs a **regulariser on the additive offset**: without
  one it random-walks in `null(W)` (β_null = 1.000) at unchanged task MSE.
* Perturbative cost scales as **N^1.39** at fixed step size and fails outright at
  N = 512; the ∝ N law (**N^0.97**) is recovered only with an N-annealed step.
  The previous experiment's `D/B = 12.5×` at N = 8 was ladder-censored; the true
  value is 4.5×.
