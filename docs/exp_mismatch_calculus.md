# Experiment: does the mismatch calculus replace per-operation SPICE?

*Companion to `docs/mismatch_calculus.md` (the algebra) and
`spice/mismatch_calculus.py` / `spice/predict_gelu_layernorm_sim.py` (the code).
Environment: ngspice-46 (`~/.local/bin`, VT = 0.025864890 V), numpy 2.4,
scipy 1.17, ≤ 12 cores, `OPENBLAS_NUM_THREADS=1`.*

**Status: §0–§9 written 2026-09-16 BEFORE any line of
`predict_gelu_layernorm_sim.py` was executed on a mismatch draw. §10 onwards
was appended after the run. The two numbers that decide the prospective gate
(per-channel p50 for GELU and for LayerNorm) are printed in §9 below and were
frozen before the arbiter ran.**

---

## 0. Pre-registration

### 0.1 Claim under test

`docs/mismatch_calculus.md` claims that the mismatch information this repo has
been buying with 200-draw ngspice Monte Carlo is carried, to within a factor of
two, by a per-realization sensitivity tuple plus the composition rules R1–R15 —
i.e. that analog prime realizations can be characterized once and composed by
rule, the way Wilkinson/Herbie compose condition numbers and the way standard
cell libraries compose with static timing analysis.

Two things can falsify that:

1. **Retrodictively** — the algebra fails to reproduce numbers that were
   measured before it existed. Then it is a post-hoc description, not a
   calculus.
2. **Prospectively** — the algebra fails on a composite it has never seen.
   Then it is a fit to the seven circuits in the repo, not an algebra.

Both are tested, with pre-registered kill criteria.

### 0.2 Pre-registered kill criteria

| ID | Criterion | Consequence |
|---|---|---|
| **KR** | Of the seven retrodiction items of §1, **more than 2** are missed by more than a factor of 2 | the rules are a **list of anecdotes**, not a calculus. Say so, stop, do not run Part 3. |
| **KP1** | The predicted **per-channel p50** of the GELU composite (P11·P6) is off by more than **2×** against the ngspice Monte Carlo | the calculus **fails prospectively**; report which rule was wrong. |
| **KP2** | The predicted **per-channel p50** of the full LayerNorm composite (P1·P2 + AGC) is off by more than **2×** | as KP1. |
| **KS** (secondary, not a kill) | p95 and common-mode columns off by more than 2× | recorded as a scope limitation of the tuple (the calculus predicts a distribution; the tails are where a first-order tuple is weakest). |

A "miss" for KR is scored per **item** (1…7), not per row: an item counts as
hit if *every* row scored for it lands inside 2×.

### 0.3 Declared in advance

* The retrodiction runs **no ngspice at all**. Where a rule predicts a
  distribution rather than a scalar, the closed-form map is evaluated by cheap
  numpy Monte Carlo over the same mismatch model — and, wherever the original
  harness's RNG stream can be replayed exactly (Table 6, Table 7, Result 5,
  Mamba), it *is* replayed, so the comparison is on identical draws and the
  only difference is that the circuit has been replaced by its sensitivities.
* Three of the seven items are **algebraic identities** rather than
  approximations, and are labelled as such: item 5 (symbol margin) and item 6
  (race) are closed forms already present in the source documents, so
  reproducing them tests that the rule was transcribed correctly, nothing more.
  Item 4 (Mamba) turns out to be an **exact re-derivation** of the netlist's own
  ODE rather than an approximation of it (§3); that is a real result about what
  the SPICE run was worth, but it is not evidence that a *lossy* tuple suffices.
  The items that genuinely test lossiness are **1, 2, 3 and 7**.
* The prospective test uses **two new circuits that no rule was fitted to**:
  a Gilbert/translinear multiplier driven by a differential-pair sigmoid, and a
  mean-subtraction mirror pair in front of the validated AGC loop. Rule R12
  (saturating-stage slope lever) is *derived* in `mismatch_calculus.md` §3 and
  has no anchor in the repo; the GELU test is its first use.
* **Corner, fixed in advance** (the paper's MOS-typical): `σ_Is = 3 %`,
  `σ_n = 0.3 %`, `σ_mirror = 1 %`, `σ_VGA = 1 %`, and for the sigmoid pair
  additionally `σ_Vt = 10 mV` (input-referred) and `σ_K′ = 3 %` (entering as a
  3 % spread on each pair device's saturation current).
* **N = 8 channels, 200 Monte-Carlo draws**, seed 20260916, bisection depth 16
  (identical to `agc_mismatch_sweep.settle_gain`).
* **Metric, fixed in advance**: `decompose_error()` of `agc_mismatch_sweep.py`,
  i.e. a least-squares scalar `α` is removed and the residual RMS is reported,
  normalised by `RMS(reference)`. Reported as p50 and p95 over the 200 draws,
  in % of signal. This is the metric of Table 7, so the two circuits are
  directly comparable to the validated result.
* **Bench calibration, run before the pre-registration and declared here**:
  both netlists were executed **once with zero mismatch** to check that they
  compute the intended function. GELU reproduces `x·σ(1.702x)` to a maximum
  relative deviation of **2.6e-6**; LayerNorm reproduces `(x−mean)/RMS` with a
  per-channel residual of **8e-17** (structurally zero, as R2/R3 require) and a
  common-mode residue of **7.9e-5** (the bisection floor of R15). No mismatch
  draw was simulated before §9 was written.

---

## 1. Retrodiction (Part 2) — the kill-gate of the calculus

Runner: `spice/mismatch_calculus.py`, wall clock **2.0 s** on one core, no
ngspice process started. Raw numbers: `spice/out/mismatch_calculus.json`.
Everything below is predicted from `docs/mismatch_calculus.md` §1–§3 and
compared against the number printed in the paper or in an `exp_*.md`.

| item | quantity | measured | predicted | ratio |
|---|---|---|---|---|
| 1 Table 6 | BJT-grade, mean error | 2.2 % | 2.193 % | 1.00 |
| 1 Table 6 | MOS typical, mean error | 11.4 % | 10.81 % | 0.95 |
| 1 Table 6 | MOS worst, mean error | 18.7 % | 18.02 % | 0.96 |
| 2 Table 7 | detector BJT, common-mode | 1.0 % | 0.889 % | 0.89 |
| 2 Table 7 | detector MOS typ, common-mode | 3.0 % | 3.703 % | 1.23 |
| 2 Table 7 | detector MOS worst, common-mode | 8.1 % | 6.397 % | 0.79 |
| 2 Table 7 | detector, per-channel (all corners) | 0.000 % | 0.000 % | — (structural) |
| 2 Table 7 | VGA σ = 0.5 %, per-channel | 0.37 % | 0.374 % | 1.01 |
| 2 Table 7 | VGA σ = 1 %, per-channel | 0.69 % | 0.692 % | 1.00 |
| 2 Table 7 | VGA σ = 2 %, per-channel | 1.66 % | 1.647 % | 0.99 |
| 2 Table 7 | VGA σ = 0.5 %, common-mode | 0.006 % | 0.0057 % | 0.95 |
| 2 Table 7 | VGA σ = 1 %, common-mode | 0.006 % | 0.0057 % | 0.96 |
| 2 Table 7 | VGA σ = 2 %, common-mode | 0.015 % | 0.0150 % | 1.00 |
| 3 Result 5 | BJT-grade, AGC/open ratio | 3.1× | 3.16× | 1.02 |
| 3 Result 5 | MOS typical, AGC/open ratio | 3.3× | 3.27× | 0.99 |
| 3 Result 5 | MOS worst, AGC/open ratio | 3.4× | 3.26× | 0.96 |
| 3 Result 5 | MOS typical, L1 open-loop | 13.6 % | 13.29 % | 0.98 |
| 3 Result 5 | MOS typical, L1 AGC | 4.1 % | 4.06 % | 0.99 |
| 4 Mamba | triode p50, geo-mean of 9 cells | 8.98 % | 8.97 % | 1.00 |
| 4 Mamba | triode p50, min cell | 3.34 % | 3.34 % | 1.00 |
| 4 Mamba | triode p50, max cell | 18.79 % | 18.78 % | 1.00 |
| 4 Mamba | pair p50, geo-mean of 9 cells | 995 % | 995 % | 1.00 |
| 4 Mamba | pair refit residual, geo-mean (narrow Δ) | 14.96 % | 12.64 % | 0.85 |
| 5 Symbol | m*(k=256, σ_G=3 %, ε=1e-9), of 2.0 available | 2.879 | 2.879 | 1.00 |
| 5 Symbol | σ_G,max at k = 256 | 1.04 % | 1.042 % | 1.00 |
| 5 Symbol | k_max at σ_G = 3 % | 30 | 30 | 1.00 |
| 5 Symbol | d ln ε/dg at k = 256 | 3283 | 3283 | 1.00 |
| 6 Race | 0.3 % ideality → effective-logit error | 0.0573 nat | 0.0573 nat | 1.00 |
| 6 Race | 3 % saturation current → effective-logit error | 0.0300 nat | 0.0296 nat | 0.99 |
| 6 Race | total per element | 0.0647 nat | 0.0645 nat | 1.00 |
| 7 Training | LS floor, N=8, σ=1 % | 0.155 σ | 0.170 σ | 1.09 |
| 7 Training | LS floor, N=8, σ=2 % | 0.140 σ | 0.169 σ | 1.20 |
| 7 Training | LS floor, N=32, σ=1 % | 0.057 σ | 0.058 σ | 1.01 |
| 7 Training | LS floor, N=32, σ=2 % | 0.057 σ | 0.058 σ | 1.02 |
| 7 Training | untrained residual, N = 8 | 0.79 σ | 0.837 σ | 1.06 |
| 7 Training | untrained residual, N = 32 | 0.95 σ | 0.955 σ | 1.01 |
| 7 Training | perturbative cost, scaling N = 8 → 32 | 2.45× | 4.0× | 1.63 |

**36 scored rows, 0 outside a factor of 2. Worst row 1.63× (the SPSA cost
scaling), second worst 1.23× (the MOS-typical detector common-mode).**

### 1.1 Verdict on KR

**KR NOT triggered: 0 of 7 items missed, against a bar of "more than 2".**
The rules reproduce every measured mismatch number in this repository from the
device-parameter spreads alone.

### 1.2 What each item actually cost

| item | calculus | original |
|---|---|---|
| 1 Table 6 | one closed form, < 1 ms | 600 ngspice `.op` runs |
| 2 Table 7 | closed form + 100 numpy draws, < 1 ms | 9 600 ngspice `.op` runs (16 per bisection × 100 × 6) |
| 3 Result 5 | one closed form, < 1 ms | 600 ngspice `.op` runs |
| 4 Mamba | 1.4 s numpy (incl. 90 refits) | ~2 min ngspice transient, 1800 cell-transients |
| 5, 6 | closed forms | 151 s / 362 s of the original campaigns |
| 7 | 80 small least-squares solves, 0.5 s | 504 s of training × 80 draws |

---

## 2. Where the retrodiction is weakest, stated plainly

* **The detector common-mode column (item 2b) is the loosest fit: 0.79–1.23×.**
  Reason, and it is not a defect of the rule: `agc_mismatch_sweep.run_cell`
  draws a *fresh* `x` for every cell, and the prediction depends on that draw
  through `Σ w_i²` (R3). Replaying the RNG stream recovers the exact `x` per
  cell, which is why the three ratios are non-monotone rather than biased. With
  `E[Σ w_i²] = 0.30` instead of the realised value the three predictions would be
  0.83 / 4.03 / 6.73 %, i.e. 0.83 / 1.34 / 0.83 — the same quality. **The
  scatter is in the measurement (100 draws, one input vector), not in the rule.**
* **The SPSA cost scaling (item 7c) is 1.63×** and is the one row that is not a
  near-hit. The rule predicts `D/B ∝ N`; the measurement gives `N^0.65` over a
  single factor-of-4 in N, on a budget ladder quantised in factors of 2 whose
  *bottom rung* was already sufficient for the exact-gradient condition. The
  honest statement is that the measurement cannot resolve an exponent to better
  than ±0.35 and the rule is inside that.
* **Item 4's agreement to four significant figures is not a triumph of
  approximation; it is a demonstration of redundancy.** The triode cell in
  level-1 MOS with a pinned common mode *is* the linear ODE that R10 writes
  down, and the differential-pair cell *is* the tanh ODE that R10 writes down.
  Once the rule is stated, ngspice had nothing left to add at this fidelity
  level — which is exactly the claim of the calculus, but it means item 4 is
  evidence about the *simulation*, not about the *tuple's lossiness*.
* **Items 5 and 6 are transcription checks.** They are closed forms that
  already appear in `exp_symbol_margin.md` and `exp_bounded_recursion.md`.
  They belong in the table because a calculus must contain them, but they carry
  no independent information.
* **The load-bearing rows are items 1, 2 and 3** (13 rows, ratios 0.79–1.23),
  where the calculus replaces a 5.375-coefficient signal-flow sum and a
  200-draw junction Monte Carlo by one line of algebra, and item 7 (7 rows),
  where it replaces a training run.

---

## 3. Rules confirmed, rules added

Confirmed as stated in the brief: **R1** (lever 19–23, items 1/3/6),
**R2** (items 2b/3), **R3** (items 1/2/7), **R4** and **R5** (item 5),
**R9** (item 4). **R6**, **R7**, **R8** are structural gates rather than
numbers and are carried into the calculus unchanged from
`exp_sign_concordance.md` and `exp_bounded_recursion.md`.

Added while deriving the above (`mismatch_calculus.md` §3):

* **R10** terminal count decides what an offset becomes (two-terminal → a
  conductance floor, three-terminal → a state offset). This is the rule that
  makes item 4 exact.
* **R11** dynamic range closes the swing wall and the control-margin wall
  simultaneously, at `1/R` each.
* **R12** saturating-stage slope lever `δ·f'(u)/f(u)`, of which R1 is the
  special case `f = exp`. **No anchor in the repo — first use is §9 below.**
* **R13** what training reaches (the LS floor of the *loss*, not of the metric)
  and the gauge-unidentifiability of the per-channel metric.
* **R14** race latency/energy laws.
* **R15** the measurement's own resolution composes like mismatch. Without it
  the Table-7 common-mode column at σ_VGA = 0.5 % looks like a 7× discrepancy
  instead of a hit; with it, the three entries 0.006/0.006/0.015 % are
  quadrature sums of a 0.0009/0.0035/0.014 % physical term and a 0.0053 %
  bisection floor.

---

## 9. PROSPECTIVE PREDICTION (written before the arbiter ran)

Two composites the calculus has never seen, both at the MOS-typical corner,
N = 8, 200 draws, metric `decompose_error()`.

### 9.1 GELU = P11 · P6

`g(x) = x·σ(1.702x)`, realised as a translinear (Gilbert) multiplier — four
junctions in one loop, `Dx · Ds / Dref → Do`, three of them per channel — whose
second factor is the collector current of a differential pair,
`I_s = I_tail·σ((v_bp − v_bn)/V_T)`, with `I_ref = I_tail = 4 µA` so that
`I_out = |x|·σ(1.702x)` in µA.

Composition (R1 × R3 × R12):
```
 e_i = b_x,i + b_s,i − b_o,i   [per-channel junctions, lever ln(I/I_s) ≈ 20–24]
     − b_ref                   [shared junction  → common mode]
     + ε_VGA,i + g_out,i + g_tail,i        [per-channel mirror/gain, R3]
     + δu_i · (1 − σ(1.702 x_i))           [R12, the new rule]
 δu_i = V_os,i/V_T + (ι_a − ι_b) + (ν_a − ν_b)|u_i|
      ⇒  σ_δu = hypot(10 mV/25.865 mV, 0.03·√2, …) = 0.389
```
The R12 term is the interesting one: `1 − σ(u)` is largest exactly where the
GELU output is smallest, so the sigmoid's offset is a per-channel error
amplifier in its own left tail and **no downstream normalisation can remove
it**.

**Prediction (frozen):**

| | p50 | p95 |
|---|---|---|
| **per-channel** | **8.483 %** | 16.313 % |
| **common-mode** | **9.286 %** | 26.785 % |

### 9.2 Full LayerNorm = P1 · P2 (mean) + AGC RMSNorm

`LN(x)_i = (x_i − mean(x))/RMS(x − mean(x))`. The mean is one shared 1:N mirror
(gain error `g_mir`), copied to each channel by its own mirror (`g_copy,i`);
the result enters the validated AGC loop (junction RMS detector at the
MOS-typical corner, VGA array at σ = 1 %, equilibrium by 16-step bisection).

Composition (R2 × R3 × R15). The mean path sits **upstream of the loop input**,
so R2 gives it no protection; and because the centred signal `c⁰` is orthogonal
to the all-ones vector, *both* the shared and the per-channel mirror error land
entirely in the per-channel residual:
```
 pc² = σ_VGA²·(1 − Σ_i w_i²)  +  2·mean(x)²·σ_mir² / MS(c⁰)
 cm  = the detector's d ln G  (R2: common mode only) ⊕ R15 bisection floor
```
Closed form for this input draw: **0.828 %** per channel, of which 0.80 % is the
VGA term and 0.21 % the mean-subtraction term.

**Prediction (frozen):**

| | p50 | p95 |
|---|---|---|
| **per-channel** | **0.723 %** | 1.350 % |
| **common-mode** | **4.800 %** | 13.835 % |

### 9.3 What would falsify what

* If GELU's per-channel p50 comes out **far above** 8.5 %, R12's `1 − σ(u)`
  weighting under-counts the sigmoid's contribution — most likely because the
  offset is not small (`δu ≈ 0.39` is a third of the sigmoid's own scale) and
  the linear-in-δ picture of the lever is wrong even though the code evaluates
  the exact map.
* If it comes out **far below**, the log-lever count is wrong: three
  per-channel junctions at `L ≈ 22` already give
  `√3 · hypot(0.03, 0.003·22) = 13 %` on their own, so a much smaller answer
  would mean the translinear loop cancels errors the graph walk says it does
  not.
* If LayerNorm's per-channel p50 is **above** ~1.5 %, the mean path is worse
  than a pair of mirror gain errors — e.g. the loop's response to a shifted
  centre is not the pure scalar R2 assumes.
* If it is **at 0.69 %**, i.e. exactly Table 7's VGA-only number, then the
  mean-subtraction term is negligible and R3's orthogonality argument
  (`⟨1, c⁰⟩ = 0`, so the shared mirror error survives) is wrong.
  *(This criterion is mis-specified and is repaired in §10.3 — Table 7's
  0.69 % belongs to a different input draw, so numerical coincidence with it
  proves nothing either way. The test that actually separates the two
  hypotheses is an ablation on the same draw, which was run.)*

---

## 10. SPICE arbitration (Part 3)

Runner: `spice/predict_gelu_layernorm_sim.py`, 200 Monte-Carlo draws per
circuit, N = 8, seed 20260916, 12 workers, ngspice-46. Wall clock **2.9 s**
(200 `.op` solves for GELU, 3 200 for LayerNorm — 16 bisection steps per draw).
Raw numbers: `spice/out/predict_gelu_layernorm.json`.

Bench controls, ideal devices (re-confirmed in the production run):
GELU reproduces `x·σ(1.702x)` to **2.59e-6**; LayerNorm gives a per-channel
residual of **0.0000 %** and a common-mode residue of **0.0079 %** (the R15
bisection floor).

### 10.1 Result

| circuit | quantity | predicted (calculus) | SPICE | ratio | verdict |
|---|---|---|---|---|---|
| **GELU = P11·P6** | **per-channel p50** | **8.483 %** | **8.518 %** | **1.00** | **HIT (KP1)** |
| GELU | per-channel p95 | 16.313 % | 16.386 % | 1.00 | ok |
| GELU | common-mode p50 | 9.286 % | 10.033 % | 0.93 | ok |
| GELU | common-mode p95 | 26.785 % | 27.347 % | 0.98 | ok |
| **LayerNorm = P1·P2 + AGC** | **per-channel p50** | **0.723 %** | **0.690 %** | **1.05** | **HIT (KP2)** |
| LayerNorm | per-channel p95 | 1.350 % | 1.266 % | 1.07 | ok |
| LayerNorm | common-mode p50 | 4.800 % | 5.228 % | 0.92 | ok |
| LayerNorm | common-mode p95 | 13.835 % | 12.660 % | 1.09 | ok |

**KP1 NOT triggered** (1.00×, bar 2×). **KP2 NOT triggered** (1.05×, bar 2×).
**KS NOT triggered**: all eight rows, tails and common mode included, land
between 0.92× and 1.09×.

Monte-Carlo noise control (declared exploratory, run after the arbiter): the
calculus re-evaluated at 20 000 draws instead of 2 000 gives GELU 8.55 % /
16.30 % / 9.44 % / 27.65 % and LayerNorm 0.716 % / 1.331 % / 4.62 % / 13.34 %,
i.e. the prediction moves by ≤ 1.6 % — the agreement above is not a sampling
artefact on either side.

### 10.2 Which rule carried the GELU error

Ablation of the calculus map (4 000 draws each, per-channel p50 / common-mode p50):

| configuration | per-channel | common-mode |
|---|---|---|
| full | 8.53 % | 9.40 % |
| R12 term only (sigmoid offset) | 5.97 % | 1.99 % |
| R1 term only (translinear junctions) | 5.79 % | 9.28 % |
| R12 switched off | 5.79 % | 9.39 % |
| R1 switched off | 6.01 % | 2.16 % |

`√(5.79² + 5.97²) = 8.32` against 8.53 — the two mechanisms add in quadrature
as R3 says. **The new rule R12 carries half of the GELU per-channel error, so
the prospective test was not decided by rules that had already been fitted.**
The common mode is almost entirely the *shared* reference junction of the
translinear loop (R1 × R3): a single device sets a 9 % scale error on the whole
layer, which is benign in the paper's sense (a trained network absorbs it) — but
it is 9 %, not 0.006 %, because unlike the AGC there is no loop pinning it.

**Design consequence, new.** GELU realised this way is *not* protected by
anything: its per-channel error is 8.5 % at the MOS-typical corner, 12× the
AGC-normalised VGA residual of Table 7, and half of it comes from a sigmoid
pair whose offset is amplified exactly on the channels that carry least signal
(R12). A GELU stage therefore needs either per-device trimming, a
sigmoid-offset cancellation (chopping/autozero), or the physics-aware training
of `exp_mismatch_absorption.md` — and the offset *is* static, so R9 says
training can eat it.

### 10.3 The mis-specified LayerNorm falsifier, repaired

§9.3 said that a SPICE answer of 0.69 % would refute R3's orthogonality
argument. That was wrong: Table 7's 0.69 % was measured on a *different* input
vector, and `Σ w_i²` — hence the VGA-only baseline — is a property of the draw.
The test that separates the hypotheses is an ablation on the *same* draw, which
was run (200 draws, mean-path mirrors set to ideal, everything else identical):

| configuration | calculus | SPICE |
|---|---|---|
| full LayerNorm | 0.711 % | **0.690 %** |
| mean path ideal (VGA + detector only) | 0.658 % | **0.615 %** |
| implied mean-path contribution (quadrature) | 0.27 % | **0.31 %** |

The mean-subtraction mirrors contribute **0.31 %** in SPICE against a predicted
0.24–0.30 % (MC and closed form). **R3's orthogonality argument is confirmed,
not refuted:** because `⟨1, c⁰⟩ = 0`, the *shared* 1:N mirror's gain error
cannot be absorbed by the loop or by the α-fit and lands in the per-channel
residual with full weight — a shared device behaving as a per-channel error
source, which is the one place R3's usual "shared ⇒ common mode" reading
inverts. The coincidence with Table 7's 0.69 % is a coincidence.

**Design consequence, new.** The cost of upgrading RMSNorm to LayerNorm is
`√2·|mean(x)|·σ_mirror/RMS(x − mean(x))` of extra per-channel error — here
0.31 % on top of 0.62 %, i.e. it makes the layer ~12 % worse. It scales with
the *input's own mean-to-deviation ratio*, so it is free for zero-mean
activations and expensive for biased ones. That is a compile-time check the
compiler can perform on activation statistics it already has.

---

## 11. Verdicts

| gate | bar | measured | verdict |
|---|---|---|---|
| **KR** retrodiction | > 2 of 7 items off by > 2× | **0 of 7** items; 36/36 rows inside 2×, worst 1.63× | **NOT triggered — it is a calculus, not a list of anecdotes** |
| **KP1** GELU per-channel p50 | > 2× | **1.00×** (8.483 predicted, 8.518 measured) | **NOT triggered — prospective HIT** |
| **KP2** LayerNorm per-channel p50 | > 2× | **1.05×** (0.723 predicted, 0.690 measured) | **NOT triggered — prospective HIT** |
| **KS** tails / common mode | > 2× | 0.92×–1.09× on all 8 rows | **NOT triggered** |

**Consolidated statement.** For the class of circuits this repository has built
— junction-exact, wiring-ideal, static mismatch, one technology model —
per-operation Monte-Carlo SPICE is **redundant**. A per-realization
sensitivity tuple plus thirteen composition rules reproduces every measured
number in the repo and predicts two composites it had never seen to 1.00× and
1.05× on the quantity that matters, at 10⁻³ of the compute (2.0 s of numpy for
the whole retrodiction against roughly 17 000 ngspice runs originally, and 2.9 s
for a 3 400-run arbitration whose answer was already on paper).

The useful consequence is not the speed-up. It is that the compiler can now
carry a **mismatch budget per prime** the way an HLS tool carries a timing
budget per cell: quote `char(P)` once per corner, walk the signal-flow graph,
and read off whether a mapping is per-channel-exposed (needs trimming or
training) or common-mode (free). §10.2 and §10.3 are two such answers that no
one had computed, produced without building either circuit.

---

## 12. What this does NOT establish

1. **Same fidelity level, same limitations.** Junction-exact and
   *wiring-ideal*: mirrors, current copies and the differential injection are
   ideal sources with a stated gain error. The calculus inherits every
   limitation of the harness it reproduces — no bias-network errors, no Early
   effect, no layout parasitics. It cannot be more right than the simulations
   it agrees with.
2. **Static mismatch only.** No thermal noise, no 1/f, no kT/C, no drift, no
   temperature. `exp_symbol_margin.md` §5.2 already shows a case where 0.5 %
   match-line noise moves a feasibility bound further than the mismatch does;
   the calculus has no rule for it, and adding one is not a re-parameterisation
   but a new derivation.
3. **One technology model.** Level-1 MOS, ideal diodes, BJTs standing in for
   subthreshold pairs, `I_s = 1e-16`, µA biases, no PDK. The levers
   `L = ln(I/I_s) ≈ 19–23` are properties of *these* numbers. On a real process
   with a different `I_s` and different biases, every `σ_b` changes — the rules
   survive, the constants do not.
4. **Independent mismatch.** No spatial correlation, no common bias network, no
   `1/√(WL)` area law, no systematic gradients. `exp_sign_concordance.md` §5.4
   flags the correlated case as *worse* than independent, so this is an
   optimistic assumption and is not tested here.
5. **Two composites is two composites.** GELU and LayerNorm are both built from
   primes whose tuples were characterized on *other* circuits, which is the
   point; but a rule set that survives two prospective tests has survived two
   prospective tests. R12 in particular has exactly one use.
6. **Item 4 of the retrodiction is a re-derivation, not an approximation**
   (§2), and items 5 and 6 are transcription checks. The evidence that a *lossy*
   tuple suffices rests on items 1, 2, 3, 7 and on §10.
7. **The metric is the repo's metric.** `decompose_error()` removes a scalar,
   which is the right thing for a normalisation layer and the wrong thing for a
   stage whose absolute gain matters. The GELU common-mode of 9–10 % is real and
   is *not* obviously benign in the way a normalisation layer's is.
8. **Nothing here is silicon.**

---

## 13. Reproduce

```bash
cd spice
OPENBLAS_NUM_THREADS=1 python3 mismatch_calculus.py --mc-runs 100
#   2.0 s, no ngspice; writes out/mismatch_calculus.json
OPENBLAS_NUM_THREADS=1 python3 predict_gelu_layernorm_sim.py \
        --mc-runs 200 --workers 12
#   2.9 s, 3400 ngspice .op solves; writes out/predict_gelu_layernorm.json
```
Total compute for this experiment: **under one minute**.
