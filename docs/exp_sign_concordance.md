# Experiment: what the backward path must preserve — sign concordance, magnitude tolerance, and the diagonal-stage obstruction

Pre-registered re-run of the **exploratory** condition C2 of
`exp_mismatch_absorption.md` (§2.0 deviation 1, §4.1 item 3), with fresh seeds and
a sharper condition set that locates the *boundary* of what the backward path
needs. Runner: `spice/train_sign_concordance.py` (imports
`spice/train_mismatch_absorption.py` as a module; the forward model, data,
optimiser and metric are unchanged).

Status: **§1 written 2026-09-15 before any run of the new script**; §2–§5
appended after execution.

---

## 1. Pre-registration (written before running)

### 1.1 Question

The previous experiment killed condition **C** (fully random backward matrix `B`
in place of `Wᵀ`, plus a per-channel backward gain mismatch `1+δ`, `δ ~ N(0,5 %)`):
divergent in **80/80** draws. A *post-hoc* control **C2** — random magnitudes but
the **signs of `W`**, same 5 % backward gain mismatch — matched the physics-aware
reference **B** to within 4 % in every cell. §2.0 declared C2 exploratory; §4.1
proposed the mechanism (the trainable stage is diagonal, so the averaged update
matrix is `diag(BᵀW)` and a random `B` gets the sign right for only ~half the
channels) and predicted that feedback alignment should be recovered once the
trainable stage is **non-diagonal**, i.e. once there is a matrix that can rotate.

This experiment asks three pre-registered questions:

1. **Is the sign-concordance result real?** (fresh seeds, pre-registered — S1)
2. **How much magnitude and gain error does it tolerate?** (S2)
3. **How many sign errors does it tolerate?** (S3 — this is the number the
   compiler needs: a tolerable sign-error rate for the backward crossbar)

and one prediction test:

4. **Does Lillicrap-style feedback alignment come back when the trainable stage
   is a dense matrix instead of a diagonal gain?** (S4)

### 1.2 Seeds

**Master seed 3031** (the previous run used 2026; no seed is reused). Per
(cell, draw) streams are spawned deterministically as
`np.random.SeedSequence(entropy=3031, spawn_key=(cell_idx, draw))`, exactly as in
`train_mismatch_absorption.run_draw`, so all conditions within a draw see the same
`ε`, training set, test set and `W`. Command to be run:

```
OPENBLAS_NUM_THREADS=1 python3 spice/train_sign_concordance.py \
    --mc-runs 20 --seed 3031 --workers 32
```

### 1.3 Model, data, optimiser, metric — all frozen and unchanged

Identical to `exp_mismatch_absorption.md` §1.3/§1.4/§2.1, reused by import:

```
x ~ 1.5·N(0,1) ∈ R^N, |x_i| floored at 0.05      N_train = 2000, N_test = 500
y⁰ = x / RMS(x)                                   ideal AGC
y  = x(1+ε) / RMS(x(1+ε)),  ε_i ~ N(0, σ_VGA)     mismatched AGC (SPICE-validated)
z  = γ ⊙ y            (S0–S3)                     trainable per-channel gain
z  = Γ y              (S4)                        trainable dense N×N stage
o  = W z,  W ∈ R^{4×N} frozen in every condition
t  = W y⁰,  L = (1/4)‖o − t‖²
```

Optimiser: plain SGD, cosine-decayed lr 2.0 → 1e-3, batch 64, 32000 steps
(2.05e6 forward evaluations) in **every** condition — the same single-budget
setting the previous run used for its headline rows. `γ` clipped to [1e-3, 20];
for the dense stage of S4 the box is `|Γ_ij| ≤ 20` (an off-diagonal entry must be
allowed to be negative). Touching the box, or ending worse than untrained, is
recorded as `div`.

**Two metrics**, both `decompose_error()`-style (common mode removed by a
least-squares scalar `α` per test sample), reported as medians over the test set
and then p50/p95 over the 20 draws:

* **per-channel residual `pc`** (primary for S0–S3), on the hidden signal:
  `α = ⟨z,y⁰⟩/⟨y⁰,y⁰⟩`, `pc = RMS_i(z/α − y⁰)`. Reported in % of signal and in
  units of σ_VGA. Identical to the previous experiment.
* **output residual `pc_out`** (primary for S4, reported for all conditions so
  the comparison is like-for-like), on the observable:
  `α = ⟨o,t⟩/⟨t,t⟩`, `pc_out = RMS_M(o/α − t) / RMS_M(t)`.
  A dense `Γ` is **not** identifiable from the loss (`W` is 4×N with N ≥ 8, so
  `Γ` has a large null direction), therefore S4 must be judged on the output,
  which *is* identifiable. `pc_out` is dimensionless, reported in %.

Cells: σ_VGA ∈ {1 %, 2 %} × N ∈ {8, 32}, **20 MC draws per cell** (80 draws).

### 1.4 Conditions (all with frozen `W`, same draw, same optimiser)

| ID | backward path used for the trainable stage | trainable stage |
|---|---|---|
| **S0** | exact adjoint of the **ideal** model (`Wᵀ`, local activation `y⁰`) — condition **B** of the previous run, the anchor | diagonal `γ` |
| **S1** | `B = |randn|/√N ⊙ sign(W)` (random magnitudes, correct signs) + per-channel backward gain `1+δ`, `δ ~ N(0, 5 %)`; local activation = physical `y` — condition **C2** as before | diagonal `γ` |
| **S2** | sign-concordant with **log-uniform magnitudes over one decade**: `B = sign(W) ⊙ |W| ⊙ 10^U`, `U ~ U(−1, 0)`, i.e. 0.1–1× the true magnitude, **and** backward gain mismatch `δ ~ N(0, 20 %)` | diagonal `γ` |
| **S3(f)** | S1's `B`, with a fraction **f** of its `M·N` entries sign-flipped; `f ∈ {0, 5 %, 10 %, 20 %, 35 %, 50 %}`. The flip sets are **nested**: one random permutation of the `M·N` entries is drawn per mismatch draw and `S3(f)` flips its first `round(f·M·N)` entries, so the curve in `f` is monotone in the flipped set and not re-randomised at every point. `S3(0) ≡ S1` by construction (same `B`, same `δ`, same stream) and is reported once | diagonal `γ` |
| **S4** | **fully random** `B = randn/√N` (no sign information at all) + `δ ~ N(0, 5 %)` — i.e. exactly the killed condition **C** | **dense** `Γ`, `N×N`, init `I` |
| **S4ref** | exact adjoint (`Wᵀ`) — positive control showing the dense parametrisation is trainable at all | dense `Γ`, init `I` |

`S4ref` is a pre-registered positive control: without it, an S4 failure could not
be distinguished from "the dense stage does not train under this optimiser".

**lr-robustness arm for the dense stage.** The frozen lr schedule (2.0 → 1e-3)
was chosen for a *diagonal* stage in the previous experiment. To make sure a K4
verdict is not a step-size artefact, **S4 and S4ref are each run at two
schedules**: the frozen one and a secondary 0.2 → 1e-4 (same 32000 steps, same
batch). Both are reported; **K4 is decided on the better of the two**, which can
only make the FA prediction easier to confirm, never harder to reject.

Diagnostics recorded per draw: fraction of channels with `diag(BᵀW)_i > 0` (the
averaged diagonal-stage loop gain, §4.1 item 1); for S4, the eigenvalues of the
`M×M` matrix `W Bᵀ` that governs the dense-stage dynamics — specifically whether
all four have positive real part, and the minimum real part.

### 1.5 Pre-registered kill criteria

| ID | Criterion | Consequence |
|---|---|---|
| **K1** | S1 (fresh seeds) does **not** match S0 within **25 %** on the per-channel residual p50 in **≥ 3 of 4 cells** | the sign-concordance result is **KILLED** — it was a seed artefact of the post-hoc run |
| **K2** | S2 fails the same 25 %/≥3-of-4 bar | the claim **narrows** to "signs *and* magnitudes within ~5 %" instead of "signs only"; the hardware statement becomes a calibration requirement, not just a polarity requirement |
| **K3** | report `f_max` = the largest sign-flip fraction at which **≥ 18/20** draws converge to within **2× S0** (per draw: `pc ≤ 2·pc_S0` for the *same* draw, and not flagged `div`), per cell. If `f_max < 5 %` the requirement is effectively "exact signs" and the hardware statement hardens to **"the backward crossbar must be sign-programmed from the forward one"** | boundary statement for the compiler |
| **K4** | S4 converges to within **2× S0** on `pc_out` in **≥ 18/20** draws | if yes: Lillicrap-style FA is **recovered** as soon as the trainable stage is non-diagonal, and §4.1's prediction stands. If no: FA is **dead for this circuit class regardless of parametrisation**, and the §10 relaxation cannot be rescued by re-parametrising the trainable stage |

`f_max` is reported per cell and as the minimum over cells (the conservative
number a compiler would use).

### 1.6 What this run does *not* re-do

No SPICE re-verification. The forward surrogate is bit-identical to the one
validated in `exp_mismatch_absorption.md` §3.5 (loop gain `G` agreeing with the
junction-exact ngspice loop to 0.003–0.006 %, per-channel residual to 0.000 %);
nothing in S0–S4 changes the forward path, only the backward path and the
parametrisation of the trainable stage. Conditions D (perturbative), E (noise)
and the W-learnable variants are not repeated.

---

## 2. Method as actually run

Runner: `spice/train_sign_concordance.py`, which imports
`spice/train_mismatch_absorption.py` as a module and reuses its forward
surrogate (`make_inputs`, `norm_ideal`, `norm_mis`), its optimiser
(`train_gradient`, `cos_lr`), its metric (`decompose_batch`) and its analytic
floor (`ls_optimum`) unchanged. Only the backward matrix, the backward gain
mismatch and — for S4 — the parametrisation of the trainable stage are new.

Environment: Python 3.14.3, numpy 2.4.2, 32 cores, `OPENBLAS_NUM_THREADS=1`,
multiprocessing over the 80 (cell, draw) tasks.

```
OPENBLAS_NUM_THREADS=1 python3 train_sign_concordance.py \
    --mc-runs 20 --seed 3031 --workers 32          # 80 tasks, 55.5 s
OPENBLAS_NUM_THREADS=1 python3 train_sign_concordance.py \
    --mc-runs 20 --seed 3031 --workers 32 --addendum   # §3.4, 14.8 s
```

Raw results: `spice/out/sign_concordance.json` (aggregates **and** per-draw
records) and `spice/out/sign_concordance_addendum.json`.

### 2.1 Deviations from §1 (declared)

1. **§3.4 is post-hoc.** The `k = 1` / `k = 2` single-entry flip addendum was
   added *after* seeing that the smallest pre-registered grid point (f = 5 %)
   already fails K3. It is exploratory, labelled as such, and decides nothing —
   K3 is decided on the pre-registered grid alone. It is nevertheless
   reproducible (`--addendum` flag on the same runner, same seeds).
2. **§3.3 (the mechanism cross-tabulation) is an analysis, not a new run**: it
   re-reads the per-draw records of the pre-registered run. No new condition.
3. Nothing else. The optimiser, seeds, conditions, metrics and kill criteria are
   as written in §1.

---

## 3. Results

20 Monte-Carlo mismatch draws per cell; within a draw every condition sees the
same `ε`, training set, test set and `W`. **Every condition uses the same budget,
32000 steps × batch 64 = 2.048e6 forward evaluations.** `pc` = per-channel
residual on the hidden signal (% of signal), `pc_out` = output residual against
the ideal network with the common mode removed (% of `RMS(t)`), `div` = draws
flagged divergent, `conv` = draws converged to within 2× the S0 anchor of the
*same* draw.

### 3.1 Main table, per cell

**N = 8, σ_VGA = 1 %**  (untrained: pc 0.803 %, pc_out 0.649 %)

| cond | pc p50 | pc p95 | pc/σ | pc_out p50 | pc_out p95 | task MSE | div | conv |
|---|---|---|---|---|---|---|---|---|
| LS (analytic floor) | 0.1643 | 0.2054 | 0.164 | 0.1284 | 0.1812 | 1.21e-05 | 0 | – |
| **S0** anchor (exact adjoint) | 0.1650 | 0.2044 | **0.165** | 0.1290 | 0.1823 | 1.21e-05 | 0 | – |
| **S1** signs + random magnitudes, δ 5 % | 0.1640 | 0.2048 | **0.164** | 0.1263 | 0.1855 | 1.21e-05 | 0 | **20** |
| **S2** signs + decade magnitudes, δ 20 % | 0.1649 | 0.2059 | **0.165** | 0.1287 | 0.1867 | 1.21e-05 | 0 | **20** |
| S3 f = 5 % | 0.1935 | 173.34 | 0.194 | 0.1479 | 125.81 | 1.70e-05 | 7 | 13 |
| S3 f = 10 % | 0.2059 | 176.04 | 0.206 | 0.1680 | 131.67 | 2.29e-05 | 9 | 11 |
| S3 f = 20 % | 70.36 | 172.58 | 70.4 | 55.66 | 131.63 | 4.15e+00 | 14 | 6 |
| S3 f = 35 % | 117.20 | 173.23 | 117.2 | 97.32 | 119.81 | 7.94e+01 | 20 | 0 |
| S3 f = 50 % | 89.90 | 158.09 | 89.9 | 83.85 | 120.08 | 1.04e+02 | 20 | 0 |
| **S4** dense Γ, random B, lr 2.0 | 371.0 | 404.4 | 371 | **206.40** | 266.31 | 1.12e+03 | 20 | **0** |
| S4 dense Γ, random B, lr 0.2 | 378.4 | 412.0 | 378 | **195.91** | 253.34 | 1.06e+03 | 20 | **0** |
| S4ref control, exact adjoint, lr 2.0 | 310.3 | 396.4 | 310 | 0.1275 | 0.1817 | 1.22e-05 | 13 | 7 |
| S4ref control, exact adjoint, lr 0.2 | 0.5421 | 0.6772 | 0.542 | 0.1273 | 0.1819 | 1.22e-05 | 0 | **20** |

**N = 8, σ_VGA = 2 %**  (untrained: pc 1.478 %, pc_out 0.988 %)

| cond | pc p50 | pc p95 | pc/σ | pc_out p50 | pc_out p95 | task MSE | div | conv |
|---|---|---|---|---|---|---|---|---|
| LS | 0.2868 | 0.4857 | 0.143 | 0.2095 | 0.4797 | 5.05e-05 | 0 | – |
| **S0** | 0.2877 | 0.4905 | **0.144** | 0.2101 | 0.4771 | 5.04e-05 | 1 | – |
| **S1** | 0.2910 | 0.5245 | **0.145** | 0.2141 | 0.4670 | 5.03e-05 | 0 | **20** |
| **S2** | 0.2896 | 0.5105 | **0.145** | 0.2096 | 0.4741 | 5.04e-05 | 0 | **20** |
| S3 f = 5 % | 0.4425 | 180.39 | 0.221 | 0.3729 | 130.18 | 1.00e-04 | 6 | 13 |
| S3 f = 10 % | 1.2708 | 177.71 | 0.635 | 0.9494 | 128.77 | 3.04e-04 | 10 | 10 |
| S3 f = 20 % | 84.44 | 177.11 | 42.2 | 68.47 | 124.20 | 2.69e+01 | 15 | 4 |
| S3 f = 35 % | 114.56 | 162.65 | 57.3 | 81.20 | 118.84 | 9.24e+01 | 19 | 1 |
| S3 f = 50 % | 95.83 | 138.47 | 47.9 | 78.72 | 118.37 | 1.71e+02 | 20 | 0 |
| **S4** lr 2.0 | 373.2 | 413.4 | 187 | **161.39** | 220.35 | 1.36e+03 | 20 | **0** |
| S4 lr 0.2 | 373.8 | 421.9 | 187 | **162.02** | 220.15 | 1.76e+03 | 20 | **0** |
| S4ref lr 2.0 | 337.3 | 412.4 | 169 | 0.2087 | 0.4809 | 5.05e-05 | 14 | 6 |
| S4ref lr 0.2 | 1.0472 | 1.5560 | 0.524 | 0.2080 | 0.4763 | 5.06e-05 | 0 | **20** |

**N = 32, σ_VGA = 1 %**  (untrained: pc 0.958 %, pc_out 0.780 %)

| cond | pc p50 | pc p95 | pc/σ | pc_out p50 | pc_out p95 | task MSE | div | conv |
|---|---|---|---|---|---|---|---|---|
| LS | 0.0597 | 0.0668 | 0.060 | 0.0484 | 0.0595 | 5.36e-06 | 0 | – |
| **S0** | 0.0596 | 0.0668 | **0.060** | 0.0480 | 0.0598 | 5.36e-06 | 0 | – |
| **S1** | 0.0623 | 0.0730 | **0.062** | 0.0507 | 0.0614 | 5.37e-06 | 0 | **20** |
| **S2** | 0.0606 | 0.0702 | **0.061** | 0.0486 | 0.0606 | 5.37e-06 | 0 | **20** |
| S3 f = 5 % | 167.06 | 191.12 | 167 | 85.24 | 140.16 | 3.36e+00 | 17 | 3 |
| S3 f = 10 % | 176.83 | 209.65 | 177 | 134.14 | 153.50 | 2.08e+01 | 18 | 2 |
| S3 f = 20 % | 153.40 | 195.86 | 153 | 118.78 | 156.95 | 5.24e+01 | 20 | 0 |
| S3 f = 35 % | 111.04 | 146.28 | 111 | 96.69 | 132.74 | 1.14e+02 | 20 | 0 |
| S3 f = 50 % | 82.23 | 110.65 | 82.2 | 75.92 | 104.69 | 1.87e+02 | 20 | 0 |
| **S4** lr 2.0 | 803.1 | 876.7 | 803 | **212.14** | 249.37 | 8.97e+03 | 20 | **0** |
| S4 lr 0.2 | 807.0 | 877.1 | 807 | **207.50** | 259.84 | 8.64e+03 | 20 | **0** |
| S4ref lr 2.0 | 793.7 | 893.8 | 794 | 0.0524 | 0.0646 | 5.43e-06 | 16 | 4 |
| S4ref lr 0.2 | 0.9099 | 0.9678 | 0.910 | 0.0520 | 0.0657 | 5.43e-06 | 0 | **20** |

**N = 32, σ_VGA = 2 %**  (untrained: pc 1.833 %, pc_out 1.522 %)

| cond | pc p50 | pc p95 | pc/σ | pc_out p50 | pc_out p95 | task MSE | div | conv |
|---|---|---|---|---|---|---|---|---|
| LS | 0.1091 | 0.1500 | 0.055 | 0.0908 | 0.1161 | 1.96e-05 | 0 | – |
| **S0** | 0.1096 | 0.1510 | **0.055** | 0.0906 | 0.1163 | 1.96e-05 | 0 | – |
| **S1** | 0.1198 | 0.1583 | **0.060** | 0.0948 | 0.1246 | 1.97e-05 | 0 | **20** |
| **S2** | 0.1085 | 0.1471 | **0.054** | 0.0920 | 0.1158 | 1.96e-05 | 0 | **20** |
| S3 f = 5 % | 169.02 | 192.60 | 84.5 | 92.26 | 145.44 | 4.27e+00 | 14 | 6 |
| S3 f = 10 % | 179.45 | 195.56 | 89.7 | 126.05 | 151.67 | 1.69e+01 | 19 | 0 |
| S3 f = 20 % | 166.80 | 199.88 | 83.4 | 134.40 | 163.19 | 5.27e+01 | 20 | 0 |
| S3 f = 35 % | 115.09 | 151.02 | 57.5 | 105.03 | 127.39 | 1.07e+02 | 20 | 0 |
| S3 f = 50 % | 86.23 | 111.29 | 43.1 | 74.73 | 100.47 | 1.82e+02 | 20 | 0 |
| **S4** lr 2.0 | 801.8 | 886.2 | 401 | **201.00** | 279.35 | 3.79e+03 | 18 | **2** |
| S4 lr 0.2 | 801.4 | 926.9 | 401 | **206.25** | 260.18 | 3.60e+03 | 18 | **2** |
| S4ref lr 2.0 | 792.1 | 881.1 | 396 | 0.1003 | 0.1314 | 1.97e-05 | 17 | 3 |
| S4ref lr 0.2 | 1.7114 | 2.1855 | 0.856 | 0.1020 | 0.1318 | 1.97e-05 | 0 | **20** |

Note on the S4/S4ref `div` column at lr 2.0: the divergence flag fires on the
`|Γ_ij| ≤ 20` box, and for a dense stage most of `Γ` lies in the **null space of
`W`** and is therefore unconstrained by the loss — it drifts into the box while
the *output* stays perfect. That is exactly what the S4ref rows show
(`pc_out` = 0.05–0.21 %, i.e. on the S0 anchor, with 13–17/20 box flags). For
the identifiable observable, `pc_out ≤ 2×S0` holds in **20/20** draws for S4ref
at *both* learning rates, and in **0/20, 0/20, 0/20, 2/20** for S4. **The S4
kill is a kill on the observable, not a box artefact.**

### 3.2 The S3 curve (sign-flip fraction f)

Draws (of 20) converging to within 2× the S0 anchor of the same draw:

| f | flipped entries (N=8, of 32) | (N=32, of 128) | N8 σ1 % | N8 σ2 % | N32 σ1 % | N32 σ2 % | mean fraction of channels with correct `diag(BᵀW)` |
|---|---|---|---|---|---|---|---|
| 0 % (= S1) | 0 | 0 | **20** | **20** | **20** | **20** | 1.000 |
| 5 % | 2 | 6 | 13 | 13 | 3 | 6 | 0.950–0.963 |
| 10 % | 3 | 13 | 11 | 10 | 2 | 0 | 0.917–0.931 |
| 20 % | 6 | 26 | 6 | 4 | 0 | 0 | 0.816–0.869 |
| 35 % | 11 | 45 | 0 | 1 | 0 | 0 | 0.647–0.706 |
| 50 % | 16 | 64 | 0 | 0 | 0 | 0 | 0.469–0.531 |

`f_max` (largest f with ≥ 18/20 converged) = **0 % in all four cells**. The
residual p50 does not degrade gracefully with f: it is on the anchor at f = 0,
still near it at f = 5–10 % for N = 8 (because the *median* draw happens to keep
all channel signs), and 10²–10³× the anchor as soon as a channel's sign flips.
There is no usable plateau.

### 3.3 Mechanism: it is not `f`, it is "does **any** channel flip"

§4.1 of the previous experiment predicted that for a diagonal trainable stage the
averaged dynamics is diagonal with entries ∝ `diag(BᵀW)_i`, so a channel with
`diag(BᵀW)_i < 0` runs *up* the loss. The per-draw records confirm this as an
essentially deterministic predictor:

* Over the 400 (draw × f > 0) records of the S3 sweep: **0 of 327** records with
  at least one wrong-sign channel converged, and **69 of 73** records with all
  `N` channel signs correct did. One wrong-sign channel out of `N` is enough to
  destroy the run — the tolerance is not statistical, it is a hard gate.
* This is why `f_max` collapses with `N`: a channel's sign is the sign of a sum
  of `M = 4` entries, so `P(no channel flips) ≈ (1 − q(f))^N` with `q` the
  per-channel flip probability. At f = 5 % this is 13/20 draws at N = 8 but only
  3–6/20 at N = 32. **The tolerable *fraction* shrinks as the array grows; the
  tolerable *count* of wrong-sign channels is zero at every N.**
* For the dense stage (S4) the corresponding object is the `M×M` matrix `W Bᵀ`
  that governs `d(WΓ)/dt = −η (W Bᵀ)(WΓ − WΓ*) C`. Its eigenvalues must all have
  positive real part. Measured: all-positive in **2 of 80** draws
  (0/20, 0/20, 0/20, 2/20), mean minimum real part −0.41 / −0.49 / −0.25 / −0.20,
  versus +0.20 / +0.22 / +0.58 / +0.61 for the exact adjoint (`W Wᵀ`, positive
  definite by construction). **S4 converged in exactly the 2 draws whose `W Bᵀ`
  had all-positive eigenvalues: agreement 80/80.**

### 3.4 Post-hoc addendum (exploratory, not pre-registered): the absolute floor

Because f = 5 % already fails K3, the same runner was re-run with exactly
`k = 1` and `k = 2` flipped backward entries (`--addendum`, same seeds):

| cell | k = 1 (of 32 / 128 entries) pc p50 | conv | all channel signs OK | k = 2 pc p50 | conv | all signs OK |
|---|---|---|---|---|---|---|
| N=8, σ=1 % | 0.179 % | 17/20 | 17/20 | 0.192 % | 13/20 | 13/20 |
| N=8, σ=2 % | 0.293 % | 18/20 | 19/20 | 0.440 % | 13/20 | 14/20 |
| N=32, σ=1 % | 0.068 % | 15/20 | 15/20 | 0.078 % | 11/20 | 11/20 |
| N=32, σ=2 % | 0.131 % | 16/20 | 16/20 | 0.175 % | 11/20 | 11/20 |

A **single** wrong entry out of 32 (N = 8) or 128 (N = 32) already drops
convergence below the 18/20 bar in 3 of 4 cells, and the convergence count equals
the "all channel signs OK" count cell by cell at k = 1 (17 vs 17, 18 vs 19,
15 vs 15, 16 vs 16; the addendum stores aggregates only, so this is a comparison
of counts, not of individual draws). The tolerable
number of sign errors in the backward crossbar is **zero**, not a small
percentage. (Exploratory: this grid was chosen after seeing the f = 5 % result.)

---

## 4. Verdicts per pre-registered kill criterion

| ID | Threshold | Measured | Verdict |
|---|---|---|---|
| **K1** | S1 not within 25 % of S0 (pc p50) in ≥ 3/4 cells ⇒ sign concordance was a seed artefact | deviation **0.59 % / 1.14 % / 4.57 % / 9.28 %** — within 25 % in **4/4** cells, with **0/80** divergences; S1 sits on the analytic least-squares floor (LS) in every cell | **NOT triggered — the sign-concordance result REPLICATES on fresh seeds (3031).** It is no longer exploratory. |
| **K2** | S2 fails the same bar ⇒ claim narrows to "signs + magnitudes within ~5 %" | deviation **0.06 % / 0.67 % / 1.68 % / 1.07 %**, 4/4 cells, **0/80** divergences — with magnitudes drawn log-uniform over a full decade (0.1–1× the true \|W\|) **and** a 20 % backward gain mismatch | **NOT triggered — the claim stays "signs only".** Backward magnitudes may be wrong by a decade and the backward gain by 20 % with no measurable cost. |
| **K3** | report `f_max`; if `f_max` < 5 % the requirement is effectively exact signs | `f_max` = **0 % in all 4 cells**. At f = 5 % only 13/13/3/6 of 20 draws converge. The post-hoc addendum shows even **k = 1** flipped entry falls below the bar in 3/4 cells | **Requirement is EXACT SIGNS.** Hardware statement hardens to: *the backward crossbar must be sign-programmed from the forward one* — a shared polarity bit per synapse, not an independently written array. |
| **K4** | S4 (dense Γ, fully random backward) within 2× S0 on `pc_out` in ≥ 18/20 | **0/20, 0/20, 0/20, 2/20** (best of two learning rates); `pc_out` p50 = **161–212 %** versus 0.05–0.21 % for the anchor. Positive control S4ref (dense Γ, exact adjoint): **20/20 in all four cells** | **KILLED. §4.1's prediction is WRONG.** Feedback alignment is **not** recovered by making the trainable stage non-diagonal. FA is dead for this circuit class regardless of the parametrisation of the trainable stage. |

### 4.1 Why K4 failed — the prediction was wrong for a diagnosable reason

§4.1 of the previous experiment reasoned that Lillicrap-style alignment needs "a
matrix that can rotate", and that a dense `Γ` supplies one. The measurement says
otherwise, and the per-draw eigenvalue diagnostic says exactly why: in
Lillicrap et al. the matrix that rotates towards `B` is the **downstream** weight
matrix, which is *learned*. Here the downstream matrix `W` is the frozen analog
crossbar — that is the whole point of the mapping — so nothing can rotate towards
`B`. What is left is the linear dynamics
`d(WΓ)/dt = −η (W Bᵀ)(WΓ − WΓ*) C`, stable only if every eigenvalue of the
`4×4` matrix `W Bᵀ` has positive real part. For a random `B` that is a property
of a random real matrix and holds in **2/80** draws — and in precisely those 2
draws, and no others, S4 converged.

So the same obstruction appears at both parametrisations, in two different
guises: for a diagonal stage it is `diag(BᵀW)_i > 0` **for every channel**, for a
dense stage it is `Re λ(W Bᵀ) > 0` **for every eigenvalue**. Both are sign/
positivity conditions that a randomly programmed backward array satisfies with
probability ≈ 2^(−N) resp. ≈ few %. Making the trainable stage richer does not
help, because the obstruction lives in the *frozen* forward matrix's relation to
the backward one, not in the trainable stage.

### 4.2 Consolidated statement for the compiler

1. **Reciprocity is not required.** A physically separate backward array whose
   magnitudes are wrong by up to a decade and whose per-channel gain is
   mismatched by 20 % trains as well as the exact adjoint (S2 = S0 to within
   1.7 %, 80/80 draws).
2. **Sign agreement is required, exactly.** Not "to within a few percent": zero
   wrong-sign channels, and in practice zero wrong-sign *entries* (§3.4). The
   tolerable sign-error fraction shrinks as `(1−q)^N` with array size, so this
   gets stricter, not looser, for large arrays.
3. **Therefore the backward crossbar must share polarity with the forward one at
   the device level** (a shared sign bit / a differential pair written from the
   same programming operation), while its conductance magnitudes may be left
   uncalibrated.
4. **Feedback alignment proper is not available for this circuit class**, at
   either parametrisation of the trainable stage, because the forward crossbar is
   frozen. The two working routes remain: sign-concordant backward (this
   experiment) and the backward-free perturbative route (condition D of the
   previous experiment, 12–38× the forward evaluations).

---

## 5. Limitations

1. **Same surrogate, no new SPICE.** Deliberate (§1.6): the forward path is
   unchanged and was validated against the junction-exact ngspice loop in
   `exp_mismatch_absorption.md` §3.5 (loop gain `G` to 0.003–0.006 %). This run
   therefore inherits that experiment's limitation: only the RMS detector is
   inside ngspice, the per-channel VGA path is a Python multiplier in this
   harness. Nothing here is evidence about the per-channel VGA circuit itself.
2. **Static graph, one trainable stage, M = 4, no non-linearity, no depth.** The
   S4 result rules out FA *for a frozen forward read-out*; it says nothing about
   a deep network in which the downstream matrices are themselves learned — which
   is the setting Lillicrap et al. actually studied. The claim being killed is
   the §10 mapping claim (frozen analog crossbar), not the original FA result.
3. **`M = 4` drives the S4 eigenvalue statistic.** `P(all Re λ > 0)` for a random
   `4×4` matrix is a few percent; it would be even smaller for larger `M`. The
   direction of the conclusion is robust, the exact 2/80 is `M`-specific.
4. **The sign-flip model is entry-wise i.i.d.** A real mis-programmed crossbar may
   have spatially correlated polarity errors (a whole row/column written with the
   wrong polarity), which would be *worse*, not better, since it flips whole
   channels deterministically. Not simulated.
5. **Divergence flag for the dense stage is partly the null space.** `Γ` is
   unidentifiable from the loss (`W` is 4×N), so the `|Γ| ≤ 20` box fires on
   drift in `ker W` even for the perfectly trained control. All S4 verdicts are
   therefore taken on `pc_out`, and are reported alongside the box flag rather
   than through it (§3.1 note).
6. **One optimiser family.** SGD with a cosine-decayed lr, 32000 steps, as frozen
   in §1; the dense stage additionally at a 10× smaller lr. No per-condition
   tuning. A very small step size would let a wrong-sign channel *stall* rather
   than run to the box, but it cannot make it descend — the sign of the averaged
   update is step-size independent.
7. **`f_max` resolution.** The pre-registered grid starts at 5 %; the finer
   `k = 1, 2` probe in §3.4 is post-hoc and is reported as exploratory.
8. **Targets come from the ideal network** (inherited from the previous design),
   so the task is exactly "undo the mismatch"; a real task objective may identify
   `γ` less sharply.
