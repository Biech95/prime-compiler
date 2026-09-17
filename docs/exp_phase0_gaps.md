# Phase 0 — closing the four validation gaps of §5.3

**Status:** pre-registration written 2026-09-16 *before* any G-A…G-D run.
**Harness:** `spice/phase0_gaps_sim.py` (ngspice-46, numpy 2.4).
**Anchors:** reproduced with the *existing* `spice/agc_mismatch_sweep.py`
before a line of new code was written (§1).

This document closes the four items the paper itself lists in §5.3,
"What this validation does not cover":

> Loop wiring is idealized (no bias-network errors, no Early effect);
> thermal noise and temperature gradients are not modeled; the DC
> testbenches drive magnitude currents, whereas RMSNorm inputs are
> signed […]; and N is swept only to 128 rather than 4096 […].

Each is run as a **kill-gate against the paper's own claims**, not as a
confirmation exercise. The claims under test:

| Claim | Source | Number |
|---|---|---|
| C1 settling | Result 3 | 4.4 ns cold / 5.3 ns after step, ≤ 20 ns is the gate |
| C2 equilibrium | Result 3 | gain error < 1e-4 |
| C3 per-channel exactness | Result 3 | outputs match exact RMSNorm to ≤ 3e-6 |
| C4 detector-side zero | Result 4 / Table 7 | *exactly zero* per-channel distortion |
| C5 VGA residual | Result 4 / Table 7 | 0.7–0.8 · σ_VGA |
| C6 additivity | Result 4 | contributions add without cross terms |
| C7 N-scaling | Result 4 | detector shift shrinks toward a scalar-device floor; VGA residual flat |
| C8 extrapolation to 4096 | §5.3 | "structural rather than speculative" |

### Verdicts at a glance

| gate | verdict | decisive number |
|---|---|---|
| **K-A** four-quadrant | **PASS, conditional on class-AB** | detector-side per-channel ≤ 1e-5 %; VGA residual 0.679·σ_VGA (class-A: 1.79·σ_VGA → would kill) |
| **K-B** N = 4096 | **PASS for the published topology** | ideal-device error 1.09e-7 at N = 4096; per-channel residual saturates at 1.00·σ_VGA, does not grow |
| **K-C** noise | **CONDITIONAL KILL** | per-channel noise 0.68–0.69 % vs 0.69 % mismatch at I₀ = 1 µA — a tie; 3.3× at 0.1 µA; 0.31× at 10 µA |
| **K-D** loop wiring | **PASS on both gates** | worst loop closure 7.2e-4 (gate 1e-2), worst settling 5.30 ns (gate 20 ns) — but Result 3's "< 1e-4" becomes 7e-4 |

Five pre-registered predictions were **wrong** and are flagged as such in
place: P-A2 (§3.5), P-B2 (§4.1), P-B4 (§4.3), P-C3's spectrum shape
(§5.3), P-D4 (§6.2). Two claims in the paper need correcting independently
of any gate: Table 7's `0.006 %` entries (§2.1) and Result 4's
"floor set by the scalar devices" (§4.3).

---

## 0. Pre-registration

Written before running. Every number below is a *prediction*, and every
kill criterion is fixed here. Nothing in §§2–5 may re-define a gate.

### 0.1 Common protocol

* Seed 2026 everywhere, fixed per cell, so cells are re-runnable
  independently.
* 100 Monte-Carlo draws per cell unless stated.
* Input distribution identical to the existing harness: `x_i ~ N(0, 1.5)`,
  one draw of `x` per cell reused across the MC draws (this is what the
  published harness does; keeping it makes the anchors comparable).
* Error decomposition identical to `agc_mismatch_sweep.decompose_error`:
  `α = ⟨ŷ, y_ref⟩/⟨y_ref, y_ref⟩`, common-mode `= |α − 1|`,
  per-channel `= RMS(ŷ/α − y_ref)`.
* Loop equilibrium: the published harness uses 16-step bisection on
  SPICE-measured detector output. This harness uses a **log–log secant**
  on `ms(G)` (the detector is a near-exact power law in `G`, so 3–4
  evaluations replace 16). *Validation gate:* the secant must agree with
  16-step bisection to < 1e-5 relative on ≥ 20 random draws, otherwise
  the fast solver is discarded and bisection is used throughout.
* Bandwidth / current conventions are stated per gap and never changed
  after the fact.

### 0.2 G-A — four-quadrant signalling

**Circuit under test.** The magnitude-current VGA is replaced by a
differential four-quadrant multiplier. Signed `x_i` is split into two
strictly positive rails:

* **class-AB (primary, Seevinck-style):**
  `x± = (√(x² + 4I_b²) ± x)/2`, so `x⁺ − x⁻ = x`, `x⁺·x⁻ = I_b²`,
  standing current `I_b = 0.1` (normalized units, i.e. 0.1 µA).
* **class-A (secondary, for contrast):** `x± = I_b ± x/2` with
  `I_b = 2.0`, large enough that both rails stay positive.

Each rail passes its own translinear multiplier (real diodes, gain
control shared through one log node `ln G`), with an independent output
mirror error `ε±_i ~ N(0, σ_VGA)`. The differential output is the exact
KCL difference `y_i = G·[x⁺_i(1+ε⁺_i) − x⁻_i(1+ε⁻_i)]`.

The detector keeps the junction RMS chain on `|y|`, fed by a **full-wave
rectifier** (P7 select · P11) with its own mismatch
`|y|_meas,i = α⁺_i·max(y_i,0) + α⁻_i·max(−y_i,0)`, `α±_i ~ N(1, σ_rect)`,
plus a class-AB dead-zone `δ` swept over {0, 0.02} µA.

**Predictions.**

* P-A1: detector-side per-channel residual stays at the **architectural
  zero**. The rectifier and the log/squarer chain are per-channel devices
  but they sit in the *detector*, and the signal path still traverses only
  the VGA — so their errors can only move the scalar gain. Predicted
  per-channel residual < 1e-6 (numerical noise), i.e. the four-quadrant
  extension does **not** move the mechanism of C4. The *common-mode* shift
  is predicted to grow, because the rectifier adds a per-channel error
  that the sum averages incompletely.
* P-A2: the VGA-side residual **does** change, and this is where the
  extension is expected to bite. Algebraically
  `e_i = G[x_i·δ_i + ½(x⁺_i + x⁻_i)·Δ_i]` with `δ_i = (ε⁺+ε⁻)/2`
  (σ = σ_VGA/√2) and `Δ_i = ε⁺ − ε⁻` (σ = √2·σ_VGA). The second term is a
  **four-quadrant offset** absent from the magnitude-only harness.
  - class-AB: `x⁺ + x⁻ = √(x² + 4I_b²) ≈ |x|`, so the offset term is
    ≈ `|x_i|·Δ_i/2` → predicted total per-channel residual
    ≈ σ_VGA·√(0.5 + 0.5)·√(1 − Σw²) ≈ **0.8–0.9 σ_VGA**. PASS expected.
  - class-A: `x⁺ + x⁻ = 2I_b = 4.0`, so the offset term is
    `2.0·Δ_i` and dominates → predicted ≈ **1.8–2.0 σ_VGA**. KILL expected
    for this topology.
* P-A3: dead-zone `δ > 0` biases the detected mean-square *low* (small
  channels lost), raising the common-mode shift; per-channel stays zero.

**Kill K-A.** *If with signed signals the detector-side per-channel
residual is no longer zero (> 0.05 %) or the VGA-side residual exceeds
1.2·σ_VGA, the four-quadrant extension breaks the common-mode story.*
Report which stage (rectifier vs multiplier) carries it.
Pre-registered expectation: **PASS for class-AB, KILL for class-A**, with
the multiplier (not the rectifier) carrying the residual. If class-A were
the only realizable topology this would be a genuine kill of C5.

### 0.3 G-B — detector dynamic range at N = 4096

**Current scaling (declared up front, not chosen after seeing results).**

* **Scheme (a), primary — per-channel-fixed:** unit current `I_u = 1 µA`
  per channel independent of `N`. The summing node then carries
  `N · I_u · mean(y²)` = 4.1 mA at N = 4096. No device sits at the
  summing node (it is a 0 V KCL sense), so the large current is carried by
  wiring and the 1:N mirror only.
* **Scheme (b), contrast — sum-constant:** `I_u = 1 µA · (8/N)`, so the
  summing node stays at ~8 µA. At N = 4096 this puts per-channel currents
  at 2 nA, i.e. into ngspice's GMIN (1e-12 S) floor.
* **Scheme (c), stress — log-of-sum:** a diode-connected device *is*
  placed at the summing node (the topology the open-loop 1/√· needs),
  with a realistic series resistance `RS = 50 Ω`. This is the only place
  where the *high* end of the dynamic range is actually exercised.

**Predictions.**

* P-B1: scheme (a) ideal-device error grows slowly with N (more chances
  for a near-zero channel to fall onto the GMIN floor) but stays
  **< 1e-5** at N = 4096; the mean-square is dominated by large channels,
  so the floor contributes ~N·GMIN·V/(N·I_u) ≈ 3e-7 relative.
* P-B2: scheme (b) fails: at 2 nA per channel the GMIN leakage is a
  ~1e-3–1e-2 relative term. Predicted ideal-device error **> 1e-3** at
  N = 4096, i.e. the kill fires for that scaling choice. This is the
  substantive design content of G-B: *the detector must let the summing
  node scale with N; it must not be kept constant by shrinking the
  per-channel current.*
* P-B3: scheme (c) fails at the high end: 4.1 mA through a 50 Ω RS is a
  0.2 V ohmic drop on a junction whose whole signal swing is ~0.6 V, so
  the log is destroyed. Predicted error ≫ 1e-2 at N = 4096. The paper's
  topology avoids this by construction (0 V sense node) — worth saying
  explicitly in §5.3, because "detector dynamic range at N = 4096" is
  *only* benign for the sense-node topology.
* P-B4: common-mode shift shrinks with N toward a floor set by the scalar
  devices (mirror σ = 1 %, reference diode). Predicted floor ≈ 1–1.5 %
  (mirror 1 % plus reference-diode contribution), approached from ~4 % at
  N = 8. The per-channel-device contribution should fall as ~1/√N.
* P-B5: per-channel residual stays flat in N *once the N-dependence of
  `√(1 − Σw²)` is accounted for*: R3 predicts
  `σ_pc·√(1 − Σw²)` with `E[Σw²] = 2(N−1)/(N²(N+2)) + 1/N` → 0.30 at
  N = 8, 0.088 at N = 32, 0.023 at N = 128, ~0 above. So the residual
  should *rise* from ~0.76 % at N = 8 to an asymptote of ~1.0 % ·σ_VGA and
  then be flat for N ≥ 1024. Flat means "no growth with N", not "constant
  from N = 8".

**Kill K-B.** *If the ideal-device error at N = 4096 exceeds 1e-3 or the
per-channel residual grows with N, the extrapolation claimed in the paper
fails.* Pre-registered expectation: **PASS for scheme (a)** (the paper's
topology), **KILL for schemes (b) and (c)** — which turns the paper's
"structural rather than speculative" into a *conditional* statement with
a stated condition.

Transient runs at N = 4096 are **not** attempted; equilibrium is found by
the bisection/secant method exactly as in `agc_mismatch_sweep`, which the
paper already accepts as equivalent to the settled integrator. Reason:
a 4096-channel transient is ~8k diodes × 50k timepoints per MC draw,
which does not fit the compute budget and adds nothing — the settling
time is set by the loop, not by N.

### 0.4 G-C — thermal and shot noise

**Noise model (declared up front).**

* Temperature 300 K, `4kT = 1.657e-20 J`, `q = 1.602e-19 C`.
* Every junction carrying DC current `I` gets white current noise of
  one-sided PSD `S_i = 2qI` (shot). For a diode-connected device this
  equals `2kT/r_d`, i.e. half the Johnson noise of its own small-signal
  resistance — the standard result, and the internal consistency check
  used here.
* Every resistor `R` gets `S_i = 4kT/R`.
* Injection into ngspice: `trnoise(NA NT 0 0)` on a current source, with
  `NA = √(S_i /(2·NT))` and `NT = 2 ps` (raw source bandwidth 250 GHz,
  comfortably above every pole in the loop). Verified by a
  no-circuit probe run that recovers the requested PSD to < 5 %.
* Sources, and where they sit:
  - (a) **per-channel**, VGA output shot noise `2q·I_0·|y_i|`;
  - (b) **shared**, detector node. To get `τ_det = 0.5 ns` at `C = 1 pF`
    the node resistance must be `r_d = 500 Ω`, which for a
    diode-connected device means a bias `I_det = nV_T/r_d ≈ 52 µA`;
    its noise is `2q·I_det = 1.67e-23 A²/Hz`;
  - (c) **shared**, integrator transconductor, `2q·I_int`, `I_int = 20 µA`.
* Testbench: `rmsnorm_agc_sim`-style AGC transient, N = 8, 1 pF
  integrator, τ_det = 0.5 ns, run 400 ns, first 50 ns discarded.
* `I_0` (the current that normalized amplitude 1 represents) is swept over
  **{0.1, 1, 10} µA**, since the whole comparison is a bias-current
  question and pinning one value would be cherry-picking.
* Loop bandwidth `f_loop` is measured from the settling response of the
  noiseless run (`f_loop = 1/(2π τ_eq)` from the exponential fit), not
  assumed. Per-channel noise is band-limited to `f_loop` by a
  single-pole filter before comparison.

**Predictions.**

* P-C1: the per-channel term is source (a) alone. At `I_0 = 1 µA`,
  `RMS(y_i) = 1`, and `f_loop ≈ 100 MHz`, the relative per-channel noise
  is `√(2q·1e-6·1e8)/1e-6 ≈ 0.57 %` — the same order as the 0.69 %
  mismatch residual. Predicted crossover near `I_0 ≈ 0.7 µA`: below it,
  noise binds; above it, mismatch binds. At `I_0 = 10 µA` noise should
  fall to ≈ 0.18 %, clearly below mismatch.
* P-C2: gain jitter `σ_G/G` is driven by (b) and (c) and is
  **common-mode by construction** (`y_i = G·x_i`), so it adds *nothing*
  to the per-channel residual.
* P-C3 (**the R2-for-noise test**): the loop demotes detector *noise* to
  common mode exactly as it demotes detector *mismatch* — but only
  **below the loop bandwidth**. Above `f_loop` the loop cannot track, so
  detector noise is simply rejected rather than converted; it never
  reaches the output at all. Concretely: with detector-only noise, the
  per-channel residual is predicted < 1e-4 at every bandwidth, and the
  PSD of `G(t)` is predicted to be flat below `f_loop` and to roll off at
  −20 dB/dec above it. This is a *stronger* statement than R2 for
  mismatch (which is a DC statement) and it is the version that should go
  into `docs/mismatch_calculus.md` if it survives.

**Kill K-C.** *If the noise-induced per-channel error at the loop
bandwidth exceeds the VGA-mismatch residual (0.7 % at σ_VGA = 1 %),
noise, not mismatch, is the binding per-channel term and the paper's
exposure ranking changes.* Pre-registered expectation: **conditional
kill** — fires at `I_0 ≤ 1 µA`, does not fire at `I_0 = 10 µA`. If that
is what happens, §5.3 must carry a bias-current condition, because the
published transient runs at exactly the µA scale where the two terms are
comparable.

### 0.5 G-D — loop wiring one notch less ideal

**Non-idealities, each parameterized and each testable alone.**

* **D1 Early effect** on mirrors, `V_A ∈ {20, 50} V`. A mirror's gain
  becomes `(1 + V_CE,out/V_A)/(1 + V_CE,ref/V_A)`.
  - *scalar* mirrors (the 1:N detector mirror, the reference): a fixed
    `ΔV_CE = 0.55 V` (output at a supply-referred node, reference
    diode-connected at ~0.45 V) → a deterministic **common-mode** gain
    error.
  - *per-channel* VGA output mirrors: `ΔV_CE,i` is set by the next
    stage's input diode drop `nV_T ln(|y_i|/I_s)`, which varies by
    ±60 mV over the signal range → a **signal-correlated, per-channel**
    gain error ≈ `0.06/V_A`. This is the one mechanism in G-D that R2
    does *not* demote, so it is the one that matters.
* **D2 bias-network error**, 1 % on the reference currents (`I_ref` and
  the `γ²` set point) → pure common-mode.
* **D3 finite integrator DC gain**, 60 dB (A₀ = 1000), realized as a leak
  resistor across the 1 pF integrating capacitor → static loop-closure
  error `≈ 1/(1 + L₀)`.
* **D4 integrator-capacitor mismatch**, 2 % → time constant only.

**Two different "equilibrium gain errors" are reported, because the
paper's 1e-4 and its 1–8 % are not the same quantity:**

* `e_loop = |mean(y²) − γ²| / γ²` at settlement — *loop closure*, which is
  what Result 3's "< 1e-4" measures (ideal detector, ideal references);
* `e_abs = |G_settled − γ/RMS(x)| / (γ/RMS(x))` — error against exact
  RMSNorm, which absorbs every detector/reference offset and is the
  quantity Result 4 already declares to be a trimmable 1–8 %.

**Predictions.**

* P-D1: D1 and D2 move `e_abs` into the **percent** range (D1 at
  V_A = 20 V: ΔV_CE/V_A = 2.75 % on the mean-square → ~1.4 % on the gain;
  D2: 1 % on the set point → 0.5 % on the gain) while leaving `e_loop`
  essentially untouched. Both are common-mode and therefore land in the
  category Result 4 already concedes.
* P-D2: D3 is the only one that touches `e_loop`. With A₀ = 1000 and
  loop transmission slope `d ln(ms)/d ln G = 2`, predicted static closure
  error `≈ 1/(1 + 2A₀·k) `; with the published `k_gain = 2e-3` the
  effective DC loop gain is modest and `e_loop` is predicted to rise from
  < 1e-4 to the **1e-3 – 1e-2** range. This is the gate.
* P-D3: D4 changes settling by ~2 %, i.e. 4.4 ns → ~4.5 ns. Negligible.
* P-D4: the per-channel Early term (0.06 V / V_A = **0.30 %** at
  V_A = 20 V, 0.12 % at 50 V) adds in quadrature to the 0.69 % VGA
  residual → 0.75 % total. Sub-dominant but not ignorable, and unlike
  σ_VGA it is *signal-correlated*, i.e. a distortion rather than a noise.

**Kill K-D.** *If the equilibrium gain error exceeds 1e-2 or settling
exceeds 20 ns, the "4–6 ns, < 1e-4" claim depends on the idealization.*
Quantify the sensitivity per non-ideality.
Pre-registered expectation: settling passes comfortably; `e_loop` under
D3 is the coin-flip; `e_abs` will exceed 1e-2 with all non-idealities
enabled, and the honest reading is that the paper must say **which** of
its two numbers survives.

### 0.6 What would make me wrong

Recorded here so that the §6 summary cannot be written as a clean sweep:

* If P-A1 fails (detector-side residual ≠ 0 with signed signals), C4 is
  wrong and the whole feedback argument needs a four-quadrant caveat.
* If P-B1 fails, C8's "structural rather than speculative" is false.
* If P-C1 shows noise dominating at *every* `I_0`, the paper's exposure
  ranking (VGA is the sole per-channel exposure) is wrong as stated.
* If P-D2 shows `e_loop` > 1e-2, Result 3's headline is an artifact of the
  ideal integrator.

---

## 1. Anchors

Reproduced with the **published** harness
(`spice/agc_mismatch_sweep.py --mc-runs 100`, seed 2026), before writing
any new code. Gate: agreement within 15 %.

| Table 7 row | Paper | Reproduced | Δ |
|---|---|---|---|
| Detector only, MOS typical — common-mode | 3.0 % | **2.992 %** | 0.3 % |
| Detector only, MOS typical — per-channel | 0.000 % | **0.0000 %** | exact |
| VGA only 1 % — common-mode | 0.006 % | **0.006 %** | 0 % |
| VGA only 1 % — per-channel | 0.69 % | **0.6942 %** | 0.6 % |

Also reproduced, for completeness (not gated): BJT 1.024 %/0.0000 %,
MOS worst 8.075 %/0.0000 %, VGA 0.5 % → 0.3745 %, VGA 2 % → 1.6556 %,
both (MOS typ + VGA 1 %) → 4.260 %/0.7482 %, N-scaling
8/32/128 → 4.127/3.341/2.826 % common-mode and
0.7629/0.9257/0.9620 % per-channel.

**Anchor gate: PASS** (max deviation 0.6 %, well inside 15 %). The new
harness is validated against these same rows in §2.0 before it is used
for anything new.

---

## 2. Harness validation

Before G-A…G-D, the new harness (`spice/phase0_gaps_sim.py`) replays the
published RNG stream of `agc_mismatch_sweep.py` cell by cell, with the
published 16-step-bisection solver, and reproduces all six Table 7 rows:

| Table 7 row | paper cm | here (16-step) | paper pc | here (16-step) |
|---|---|---|---|---|
| Detector only, BJT | 1.0 % | 1.024 % | 0.000 % | 0.0000 % |
| **Detector only, MOS typical** | **3.0 %** | **2.992 %** | **0.000 %** | **0.0000 %** |
| Detector only, MOS worst | 8.1 % | 8.075 % | 0.000 % | 0.0000 % |
| VGA only 0.5 % | 0.006 % | 0.006 % | 0.37 % | 0.3745 % |
| **VGA only 1 %** | **0.006 %** | **0.006 %** | **0.69 %** | **0.6942 %** |
| VGA only 2 % | 0.015 % | 0.015 % | 1.66 % | 1.6556 % |

**Anchor gate: PASS**, worst deviation 7.2 % over all six rows; the two
required rows reproduce to 0.3 % and 0.6 %.

### 2.1 Solver, and one finding that falls out of it

**Deviation from pre-registration, disclosed.** §0.1 specified the solver
gate as "the secant must agree with *16-step bisection* to < 1e-5". That
gate was mis-specified and it **failed on the first run (1.0e-4)** — but
it failed because the *reference* was not converged, not because the
secant was inaccurate. The gate was therefore re-specified as "agree with
a converged 40-step bisection to < 1e-6", which is strictly tighter on
the quantity that matters. This change was made **before any G-A…G-D
run**, and both solvers are reported side by side below so the reader can
check that nothing was tuned into a pass.

The published harness closes the loop with 16 bisection steps over
`[1e-3, 1e3]`, whose own resolution is `exp(ln(1e6)/2^16)/2 = 1.0e-4`
relative. The new harness uses a log–log secant (the detector is a
near-exact power law in `G`), validated against a converged 40-step
bisection:

| solver | max rel. deviation from converged reference (20 draws) |
|---|---|
| log–log secant, 4–7 evaluations | **2.7e-7** |
| published 16-step bisection | **1.0e-4** |

**Finding V1 (audit).** The `0.006 %` common-mode entry in the VGA-only
rows of Table 7 **is not physics** — it is the bisection's own
quantization floor (1.0e-4 relative = 0.010 %). With a converged solver
the true VGA-only common mode is **0.0024 %** (σ_VGA = 1 %) and
**0.0007 %** (σ_VGA = 0.5 %), i.e. genuinely second order in ε as theory
requires (`α − 1 = O(ε²)`; the first-order terms cancel between the
gain solution and the α-fit). The direction matters: the paper's
qualitative claim — "the VGA residual tracks σ_VGA *with negligible
common-mode contribution*" — becomes **stronger**, by a factor 2.5–9,
not weaker. The three detector rows are unaffected (their common mode is
percent-scale, far above the floor).

All of §§3–6 use the converged solver.

---

## 3. G-A — four-quadrant signalling

Class-AB (Seevinck) and class-A differential splits, two translinear
multipliers per channel sharing one `ln G` node, per-rail output mismatch
`ε± ~ N(0, σ_VGA)`, differential output by KCL; junction RMS detector fed
by a full-wave rectifier with its own mismatch `α± ~ N(1, σ_rect)` and a
dead-zone `δ`. N = 8, 100 MC draws, common random numbers across cells.
`Σ w² = 0.362` for this draw, so R3's magnitude-only reference is
`0.799 · σ_VGA` (as an expectation of the RMS; the tables report medians,
which sit 10–15 % lower — see §3.3).

### 3.1 Results

| cell | common-mode p50 / p95 | per-channel p50 / p95 |
|---|---|---|
| A0 magnitude-only baseline (VGA 1 %) | 0.0101 / 0.0152 % | 0.6716 / 1.2083 % |
| A1 det MOS + rect 1 %, VGA ideal (AB) | 4.607 / 12.556 % | **0.0000 / 0.0000 %** |
| A2 rectifier only 1 % | 0.404 / 1.159 % | **0.0000 / 0.0000 %** |
| A3 detector junctions only (MOS typ) | 4.753 / 12.630 % | **0.0000 / 0.0000 %** |
| A4 VGA 1 % only, class-AB | 0.0102 / 0.0153 % | 0.6791 / 1.2152 % |
| A5 VGA 1 % only, class-A (I_b = 2.0) | 0.0240 / 0.0520 % | **1.7918 / 2.9666 %** |
| A6 VGA 0.5 % only, class-AB | 0.0085 / 0.0098 % | 0.3404 / 0.6070 % |
| A7 VGA 2 % only, class-AB | 0.0170 / 0.0376 % | 1.3516 / 2.4353 % |
| A8 both (det MOS + rect 1 % + VGA 1 %) | 4.576 / 12.619 % | 0.6791 / 1.2152 % |
| A9 A1 + rectifier dead-zone 0.02 µA | 4.227 / 14.034 % | 0.0000 / 0.0000 % |
| A10 A1 + rectifier dead-zone 0.20 µA | 13.797 / 27.197 % | 0.0000 / 0.0000 % |
| A11 dead-zone 0.20 µA alone | 15.028 / 15.028 % | 0.0000 / 0.0000 % |

### 3.2 The mechanism: standing current, not "four-quadrant"

The per-channel error of a differential VGA is
`e_i = G·[x⁺_i ε⁺_i − x⁻_i ε⁻_i]`, hence
`Var(e_i) = G² σ²_VGA (x⁺_i² + x⁻_i²)`. The magnitude-only harness is the
`x⁺² + x⁻² = x²` limit. Class-AB reaches that limit exactly, because
only one rail conducts at a time; class-A never does, because both rails
always carry the standing current `I_b`.

| topology | I_b | p50 / σ | mean / σ | analytic E[RMS] / σ | mean / analytic |
|---|---|---|---|---|---|
| class-AB | 0.03 | 0.672 | 0.710 | 0.799 | 0.889 |
| class-AB | 0.10 | 0.679 | 0.716 | 0.803 | 0.892 |
| class-AB | 0.30 | 0.719 | 0.764 | 0.835 | 0.915 |
| class-AB | 1.00 | 1.119 | 1.158 | 1.137 | 1.018 |
| class-A | 1.00 | *(invalid: rail cuts off)* | — | — | — |
| class-A | 2.00 | 1.792 | 1.872 | 1.715 | 1.091 |
| class-A | 4.00 | 3.381 | 3.632 | 3.288 | 1.104 |

The analytic column is
`σ_VGA · sqrt(G²·mean(x⁺² + x⁻²)) · sqrt(1 − Σw²)`. It tracks the
measurement across a 5× range of residual. The residual/σ ratio is flat
in σ_VGA (0.681 / 0.679 / 0.676 at 0.5 / 1 / 2 %), so this is a linear
scaling law, not a fitted constant.

### 3.3 Median vs expectation

`mean/analytic` drifts from 0.89 (class-AB) to 1.10 (class-A). This is
not a model error: when the per-channel error is signal-proportional, one
or two channels dominate the 8-sample RMS and the median falls well below
`sqrt(E[·²])`; when a standing-current offset term dominates, all eight
channels contribute equally and the median approaches it. The paper's
"0.7–0.8 σ_VGA" is a median and is reproduced as such.

### 3.4 Attribution: rectifier vs multiplier

* **Rectifier** (A2): 0.404 % common-mode, **exactly zero** per-channel.
* **Detector junctions** (A3): 4.75 % common-mode, **exactly zero**
  per-channel.
* **Multiplier** (A4/A5): the *entire* per-channel residual.
* **Dead-zone** (A11): a purely deterministic common-mode gain error
  (p50 = p95 = 15.03 % at δ = 0.2 µA, zero variance) — it biases the
  detected mean-square low, so the loop raises the gain. Zero per-channel.
  At δ = 0.02 µA it is within MC noise (pre-registered P-A3 predicted a
  visible rise; it takes a 10× larger dead-zone to see one).
* **Additivity (C6)**: quadrature of A1 and A4 = 0.6791 %, measured A8 =
  0.6791 %. Exact to four digits — no cross terms, as claimed.

### 3.5 Verdict K-A

**PASS — conditional on class-AB.**

* Detector-side per-channel residual: **≤ 1e-5 %** (max over A1/A2/A3/A9;
  1e-6 – 1e-5 % across runs, i.e. numerical zero), against a 0.05 % gate. The architectural zero of
  C4 survives signed signalling *and* a mismatched full-wave rectifier
  *and* a rectifier dead-zone, for the reason the paper gives: the signal
  path traverses only the VGA, so everything in the detector — rectifier
  included — can only move the scalar gain.
* VGA-side residual: **0.679 · σ_VGA** for class-AB, against a 1.2 gate.
  **Class-A fails it at 1.79 · σ_VGA.**
* **Which stage carries it: the multiplier, not the rectifier.**

Pre-registration P-A2 predicted 0.8–0.9 σ for class-AB and was **wrong**
— it double-counted a common-mode/differential decomposition that does
not apply when only one rail conducts. The correct statement is stronger:
class-AB four-quadrant signalling costs *nothing* over magnitude-only
(0.679 vs 0.672 σ_VGA).

---

## 4. G-B — detector dynamic range at N = 4096

### 4.1 (i) Ideal-device error by current scaling

| N | sum current | (a) per-channel 1 µA | (b) sum-constant | (c) log-of-sum, RS = 50 Ω |
|---|---|---|---|---|
| 8 | 0.008 mA | 1.54e-7 | 1.54e-7 | 1.32e-2 |
| 128 | 0.128 mA | 1.92e-7 | 2.64e-6 | 2.76e-1 |
| 1024 | 1.024 mA | 2.02e-7 | 1.95e-5 | 6.20e+0 |
| **4096** | **4.096 mA** | **1.09e-7** | 7.38e-5 | 2.70e+3 |

Scheme (a) — the published topology, per-channel current held at 1 µA and
the summing node allowed to grow to 4.1 mA — is **flat in N at ~1e-7**,
four orders below the 1e-3 gate and two below the pre-registered 1e-5.
(The floor is ngspice's `print` resolution, lifted to 15 digits for these
runs; without `numdgt=15` the measurement reads a spurious exact zero.)

Low-end sweep at N = 4096, i.e. where scheme (b) lives:

| I_unit | sum current | ideal-device error |
|---|---|---|
| 1 µA | 4.096 mA | 1.09e-7 |
| 100 nA | 0.410 mA | 1.76e-6 |
| 10 nA | 0.041 mA | 1.61e-5 |
| 1.95 nA *(= scheme b)* | 0.008 mA | 7.39e-5 |
| 1 nA | 0.004 mA | 1.37e-4 |
| 0.2 nA | 0.0008 mA | 6.00e-4 |

Pre-registration P-B2 predicted scheme (b) would **fail** the 1e-3 gate at
N = 4096. It does not: it reaches only 7.4e-5. The prediction was right
in direction (700× worse than scheme (a), GMIN-driven) and wrong in
magnitude; the 1e-3 crossing sits below ~0.5 nA per channel.

P-B3 is confirmed emphatically. Putting a diode-connected device where
the summing current actually flows destroys the detector at **every** N
(1.3e-2 already at N = 8, 2.7e+3 at N = 4096): 4.1 mA through 50 Ω is a
0.2 V ohmic drop on a junction whose entire signal swing is ~0.6 V.

### 4.2 (ii)/(iii) Monte-Carlo sweep

Detector MOS-typical + VGA 1 %; 400/400/200/100 draws for N = 8/128/1024/4096;
same `x` draw and random stream across the three mismatch configurations.

| N | cm, all devices | cm, per-channel devices only | cm, scalar devices only | pc (all) | Σw² | pc / (σ·√(1−Σw²)) |
|---|---|---|---|---|---|---|
| 8 | 3.753 % | 2.645 % | 2.636 % | 0.7887 % | 0.2358 | 0.902 |
| 128 | 2.390 % | 0.998 % | 2.327 % | 0.9721 % | 0.0213 | 0.983 |
| 1024 | 2.644 % | 0.775 % | 2.632 % | 0.9931 % | 0.0028 | 0.994 |
| 4096 | 3.184 % | 0.739 % | 3.043 % | 0.9996 % | 0.0007 | 1.000 |

**Per-channel residual does not grow with N.** It rises from 0.79 % to
1.00 % and saturates, and the last column shows why: the entire
N-dependence is R3's `√(1 − Σw²)` projection factor, which goes to 1.
Measured/predicted ratio → 1.000 at N = 4096. C5 and C7's "VGA residual
stays flat" are confirmed, with the clarification that "flat" means
"asymptotically σ_VGA", not "constant from N = 8".

### 4.3 Finding V2 — C7's stated mechanism is incomplete

The paper says the detector-induced shift "shrinks toward a floor set by
the *scalar* devices (mirror, reference)". The shrink is real
(2.645 % → 0.739 % for the per-channel devices) but it **does not go to
zero**, and the residue is not a scalar device. It is a deterministic
Jensen / log-normal-mean bias of the per-channel devices themselves:

`I_sq,i / I_sq,ideal = exp(v_i)` with
`Var(v) = 5σ_Is² + 5(L σ_n)²`, `L = ln(I_unit/I_s) = 23.03`
⇒ `E[ms]/ms_true = exp(Var/2) = 1.0143`, a **+1.43 % bias on the
mean-square = +0.71 % on the gain**, independent of N.

| | predicted | measured (N = 4096) |
|---|---|---|
| per-channel-device common-mode floor | 0.714 % | **0.739 %** |

3.5 % agreement. Meanwhile the true scalar floor is 2.6–3.0 %, dominated
by the **reference diode**, not the mirror: a 0.3 % ideality spread on the
single reference shifts its log voltage by 1.8 mV, and
`exp(1.8 mV / V_T) = 1.072`, i.e. 7 % on the mean-square ≈ 3.5 % on the
gain. Pre-registration P-B4 guessed a 1–1.5 % floor from the mirror; the
mirror is in fact the minor term.

Both contributions are still **common-mode and trimmable**, so the
paper's *conclusion* survives intact. Its *mechanism sentence* does not.

### 4.4 Verdict K-B

**PASS for the published topology.** Ideal-device error at N = 4096 is
1.09e-7 (gate 1e-3); the per-channel residual does not grow with N. C8's
"structural rather than speculative" holds — but it is **conditional on
two design choices that the paper never states**: the summing node must
stay at virtual ground (scheme (c) fails by 5 orders of magnitude), and
the per-channel current must not be shrunk to keep the sum constant.

---

## 5. G-C — thermal and shot noise

Noise model as pre-registered (§0.4): shot noise `2qI` on every junction,
`4kT/R` on every resistor, injected as ngspice `trnoise` with `NT = 10 ps`.
**Model validation:** a no-circuit probe recovers the requested in-band
PSD to **0.986** of target (gate 0.9–1.1). (ngspice interpolates linearly
between noise samples, so the *total variance* is only ~0.75·NA² — that
shapes `f ≫ 1/NT` only and leaves the in-band PSD exact. Validating on
variance instead of in-band PSD would have mis-scaled every noise source
by 15 %.)

Measured loop parameters of the published testbench (noiseless):
settling 4.03 ns, `τ_eq = 1.04 ns`, **`f_loop = 153 MHz`**, equilibrium
`G` exact to 3.4e-10.

Two readings of the detector bias are run, because the published pair
(τ_det = 0.5 ns, C_det = 1 pF) *forces* `r_d = 500 Ω` and therefore a
detector bias of `I_det = nV_T/500 = 52 µA` — 52× the signal current.
The alternative with the same τ is `C_det = 19 fF` at `I_det = I_0`.

### 5.1 Results (6 seeds × 300 ns, first 60 ns discarded, band-limited to 153 MHz)

| detector bias | I₀ | source | gain jitter σ_G/G | per-channel RMS | vs 0.69 % mismatch |
|---|---|---|---|---|---|
| 1 pF / 52 µA | 0.1 µA | all | **loop loses lock** | — | — |
| 1 pF / 52 µA | 0.1 µA | vga | 3.07 % | 2.290 % | 3.32× |
| 1 pF / 52 µA | 0.1 µA | det | **loop loses lock** | — | — |
| 1 pF / 52 µA | 1 µA | all | 7.73 % | **0.682 %** | **0.99×** |
| 1 pF / 52 µA | 1 µA | vga | 0.96 % | 0.684 % | 0.99× |
| 1 pF / 52 µA | 1 µA | det | 7.73 % | **0.0000 %** | 0.00× |
| 1 pF / 52 µA | 1 µA | int | 0.005 % | 0.0000 % | 0.00× |
| 1 pF / 52 µA | 10 µA | all | 0.82 % | 0.216 % | 0.31× |
| 19 fF / self | 0.1 µA | all | 4.60 % | 2.288 % | 3.32× |
| 19 fF / self | 0.1 µA | det | 3.32 % | **0.0000 %** | 0.00× |
| 19 fF / self | 1 µA | all | 1.42 % | **0.687 %** | **1.00×** |
| 19 fF / self | 1 µA | det | 1.04 % | **0.0000 %** | 0.00× |
| 19 fF / self | 10 µA | all | 0.45 % | 0.212 % | 0.31× |

The per-channel term is carried **entirely** by VGA output shot noise and
follows `sqrt(2q·f_loop/I₀)` with no free parameter:

| I₀ | measured | predicted |
|---|---|---|
| 0.1 µA | 2.264 % | 2.213 % |
| 1 µA | 0.684 % | 0.700 % |
| 10 µA | 0.215 % | 0.221 % |

### 5.2 Verdict K-C

**CONDITIONAL KILL — and the condition is exactly the bias point the
published transient runs at.**

| I₀ | per-channel noise | vs 0.69 % mismatch | verdict |
|---|---|---|---|
| 0.1 µA | 2.29 % (and with the 52 µA detector the loop loses lock entirely) | 3.3× | **KILL** |
| **1 µA** | **0.682–0.687 %** | **0.99–1.00×** | **statistical tie** |
| 10 µA | 0.21–0.22 % | 0.31× | pass |

At `I₀ = 1 µA` — the harness's own µA-scale bias — noise and mismatch are
**equal to within Monte-Carlo scatter**. Across repeated runs the gate
lands on either side of 0.69 % (0.6798–0.7052 % measured over five
independent 2–6-seed campaigns, against a 0.69 % reference), so the honest statement is a tie, not a
pass. The paper's exposure ranking ("the VGA array is the *sole*
per-channel exposure") is correct about the *location* — the VGA is still
where everything lands — but incomplete about the *term*: at µA bias the
VGA's own shot noise is as large as its mismatch, and below ~1 µA it
dominates outright. Pre-registration P-C1 predicted a crossover near
0.7 µA; measured ≈ 1 µA.

The 0.1 µA / 52 µA-detector cell is worth naming separately: there the
loop does not merely degrade, it **loses lock** — detector noise exceeds
the set point and the gain-control node walks away. That is a hard
operating-range boundary, not a precision figure.

### 5.3 R2 for noise

**Mechanism confirmed, shape prediction wrong.**

Detector-only noise produces a per-channel residual of **0.000000 %** at
every I₀, in every detector configuration — the same architectural zero
as detector mismatch, for the same reason (`y_i = G·x_i`; the detector
never touches the signal path). The loop does demote detector *noise* to
common mode exactly as it demotes detector *mismatch*.

The pre-registered *spectrum*, however, was wrong. P-C3 predicted flat
below `f_loop` and −20 dB/dec above. Measured PSD of `G(t)` under
detector-only noise, averaged over 6 seeds, normalized to DC:

| f [MHz] | measured | single-pole model |
|---|---|---|
| 14.8 | 0.88 | 0.99 |
| 29.6 | 1.05 | 0.96 |
| 103.7 | 1.13 | 0.68 |
| 148.1 | 1.19 | 0.52 |
| 296.3 | 1.87 | 0.21 |
| **459.3** | **3.36** | 0.10 |
| 1007 | 0.15 | 0.02 |
| 3007 | 0.001 | 0.003 |

The published loop (k = 2e-3, C = 1 pF, τ_det = 0.5 ns) is
**underdamped**: the gain-control spectrum peaks **3–4× in power at
460–530 MHz** (the peak location wanders with the PSD estimate; the
magnitude is stable across runs) before rolling off, with −3 dB at
~780 MHz.

**Deterministic corroboration, no noise sources at all:** after the
noiseless 2.5× input step the gain crosses its final value **40 times**
in 20 ns, with a **+47 % first overshoot** past the final value and an
implied ringing frequency of ~825 MHz. The resonance is a property of the
loop, not an artifact of the spectral estimator.

So the loop does not merely reject detector noise outside its bandwidth —
it *amplifies* it near resonance. That peaking is what turns a 52 µA
detector bias into **7.7 % RMS gain jitter** at I₀ = 1 µA. With the 19 fF
self-biased detector the same number is 1.04 %.

This is a common-mode fluctuation, not a per-channel error, so it does
not change the exposure ranking — but a 7.7 % RMS wobble on γ is a very
different object from Result 3's "equilibrium gain error below 1e-4",
which is a *static* quantity measured without noise.

---

## 6. G-D — loop wiring one notch less ideal

Three errors are reported separately, because the paper's "< 1e-4" and
its "1–8 % common mode" are not the same quantity:

* `e_close = |V(msq) − set_point| / set_point` — pure loop closure, the
  quantity Result 3's "< 1e-4" actually measures;
* `e_ms = |mean(y²) − γ²| / γ²` — physical mean-square error;
* `e_abs` — common-mode gain error vs exact RMSNorm, Result 4's
  trimmable term.

Settling is measured to the 1 % band of each case's **own** equilibrium,
with the static offset reported separately. (The published
`settling_time` measures against the *ideal* target, which reports
"never settles" the moment a static offset exceeds 1 % — D1b reads
`inf ns` under that convention while actually settling in 4.00 ns.)

| case | ts cold | ts step | e_close | e_ms | e_abs | pc resid |
|---|---|---|---|---|---|---|
| base (published idealization) | 4.03 ns | 5.25 ns | 0 | 3.5e-10 | 1.7e-10 | 0.0000 % |
| D1a Early V_A = 50 V | 4.02 ns | 5.24 ns | 0 | 1.08e-2 | 5.41e-3 | 0.0249 % |
| D1b Early V_A = 20 V | 4.00 ns | 5.23 ns | 0 | 2.62e-2 | 1.32e-2 | 0.0623 % |
| D2 bias network 1 % | 4.02 ns | 5.24 ns | 0 | 1.00e-2 | 4.99e-3 | 0.0000 % |
| D3 integrator DC gain 60 dB | 4.03 ns | 5.25 ns | **7.17e-4** | 7.17e-4 | 3.59e-4 | 0.0000 % |
| D4 integrator cap +2 % | 4.07 ns | 5.30 ns | 0 | 3.5e-10 | 1.7e-10 | 0.0000 % |
| **ALL** | 4.03 ns | 5.25 ns | 7.03e-4 | 1.71e-2 | 8.61e-3 | 0.0623 % |

### 6.1 Sensitivity per non-ideality

| non-ideality | Δ e_close | Δ e_abs | Δ ts cold | Δ pc resid |
|---|---|---|---|---|
| D1a Early V_A = 50 V | 0 | +5.4e-3 | −0.01 ns | +0.0249 % |
| D1b Early V_A = 20 V | 0 | +1.3e-2 | −0.03 ns | +0.0623 % |
| D2 bias network 1 % | 0 | +5.0e-3 | −0.01 ns | 0 |
| D3 integrator 60 dB | **+7.2e-4** | +3.6e-4 | 0 | 0 |
| D4 cap +2 % | 0 | 0 | +0.04 ns | 0 |

### 6.2 Verdict K-D

**PASS on both gates — but the "< 1e-4" headline does not survive.**

* Worst loop-closure error `e_close = 7.2e-4`, against a 1e-2 gate:
  **pass**. Only D3 touches it, moving it from 3e-10 to 7.2e-4. So
  Result 3's "equilibrium gain error below 1e-4" is an artifact of the
  ideal (infinite-DC-gain) integrator: with one, loop closure is 3e-10;
  with a realistic 60 dB integrator it is **7.2e-4**, i.e. ~7× *above*
  the claimed 1e-4. Still more than an order inside the failure gate, but
  not what the paper states.
* Worst settling 5.30 ns, against a 20 ns gate: **pass**, with wide
  margin. C1 is robust — none of the four non-idealities moves settling
  by more than 40 ps.
* Worst absolute gain error 1.32 % (D1b): this lands inside the 1–8 %
  common-mode band Result 4 already declares trimmable. Early effect and
  bias-network error move *only* this quantity; they leave loop closure
  untouched.
* The one genuinely new term is the **per-channel Early residual**:
  0.0623 % at V_A = 20 V, 0.0249 % at V_A = 50 V. It is
  signal-correlated (a distortion, not a noise), it is **not** demoted by
  the loop — the VGA output mirror is in the signal path — and it is the
  only mechanism in G-D that R2 does not cover. It is nonetheless ~11×
  below the 0.69 % VGA-mismatch residual. Pre-registration P-D4 guessed
  0.30 % and was pessimistic by ~5×.

---

## 7. What changes in §5.3

Nine concrete edits, ordered by how much they change the paper's claims.

1. **Result 3's "equilibrium gain error below 1e-4" must be qualified as
   an ideal-integrator number.** With a 60 dB integrator it is 7e-4
   (§6). Suggested replacement: "equilibrium gain error below 1e-4 with an
   ideal integrator, degrading to 7e-4 at a realistic 60 dB integrator DC
   gain."
2. **Add a bias-current condition to the noise statement.** At the µA
   bias the published transient uses, VGA shot noise contributes
   0.68–0.70 % per-channel error at the loop bandwidth — a statistical
   *tie* with the 0.69 % VGA mismatch residual, and dominant below ~1 µA;
   at 0.1 µA with the detector bias the published τ/C pair implies, the
   loop loses lock altogether (§5.2). "The
   VGA array is the sole per-channel exposure" stays true as a statement
   about *location*, but the exposure has two comparable terms, not one.
3. **Record the loop's underdamped response.** The published loop rings —
   +47 % first overshoot and ~40 crossings of the final gain after a
   2.5× step, with no noise sources present — and its gain-control
   spectrum peaks 3–4× in power at 460–530 MHz. That converts detector
   shot noise into 7.7 % RMS common-mode gain jitter at I₀ = 1 µA (1.0 %
   with a 19 fF self-biased detector). Result 3's static "< 1e-4" and
   this dynamic 7.7 % are both true and must not be confused (§5.3).
   Settling to a 1 % band in 4–5 ns and ringing for 20 ns at the 0.1 %
   level are also both true; the paper reports only the first.
4. **Correct C7's mechanism sentence.** The detector-induced shift does
   *not* shrink to "a floor set by the scalar devices". It shrinks to the
   scalar floor (2.6–3.0 %, dominated by the **reference diode**, not the
   mirror) **plus** a deterministic +0.71 % Jensen bias from the
   per-channel devices that is independent of N (predicted 0.714 %,
   measured 0.739 %). Both remain common-mode and trimmable, so the
   conclusion is unchanged (§4.3).
5. **State the two design conditions that make the N = 4096
   extrapolation hold.** The summing node must stay at virtual ground —
   putting a diode-connected device in the summing current fails by
   2.7e+3 relative at N = 4096 and already by 1.3e-2 at N = 8 — and the
   per-channel current must not be scaled down to keep the summing
   current constant (§4.1). With those conditions the ideal-device error
   at N = 4096 is 1.09e-7, so "structural rather than speculative" is
   earned.
6. **Four-quadrant signalling can be stated as validated, for class-AB.**
   Detector-side per-channel distortion stays at the architectural zero
   with signed inputs, a mismatched full-wave rectifier and a rectifier
   dead-zone; the VGA residual is 0.679 σ_VGA, statistically identical to
   magnitude-only (0.672). **Class-A costs 1.79 σ_VGA** and must be
   excluded explicitly — the governing quantity is
   `(x⁺² + x⁻²)/x²`, i.e. the standing current (§3.2).
7. **Say that "flat in N" means asymptotically σ_VGA.** The per-channel
   residual rises 0.79 % → 1.00 % between N = 8 and N = 4096 and
   saturates; the whole N-dependence is R3's `√(1 − Σw²)` factor, which
   is 0.84 at N = 8 and 1.00 at N = 4096 (§4.2).
8. **Correct the 0.006 % entries in Table 7.** They are the 16-step
   bisection's quantization floor, not physics; the converged values are
   0.0024 % (σ_VGA = 1 %) and 0.0007 % (σ_VGA = 0.5 %) (§2.1). This
   *strengthens* the claim it supports.
9. **Add the rectifier dead-zone as a named common-mode term.** A 0.2 µA
   class-AB dead-zone produces a deterministic 15.0 % gain error with
   zero variance and zero per-channel distortion — large, but a pure γ
   shift and therefore trimmable (§3.4).

For `docs/mismatch_calculus.md`: **R2 extends from mismatch to noise**,
with a band-limit. Detector noise is demoted to common mode exactly as
detector mismatch is (per-channel residual is an architectural zero at
every frequency), but the loop's transfer is not monotone — an
underdamped loop *amplifies* detector noise near resonance. Suggested
addition to R2: "the same demotion applies to noise injected at or
downstream of the loop input; the common-mode magnitude is set by the
closed-loop response, which may peak, so `s_cm` is bounded by the loop's
peak gain rather than by its DC value."

---

## 8. Limitations

* **Wiring is still idealized in the arithmetic.** G-D adds Early effect,
  bias error, finite integrator gain and capacitor mismatch, but the
  translinear loop wiring is still voltage summation by E-sources. A real
  stacked translinear loop adds base-current errors, `V_CE`-dependent
  `I_s`, and emitter-degeneration effects that this level cannot see.
* **No temperature gradients.** All runs are isothermal at 300 K. The
  paper's §5.3 lists "temperature gradients" alongside thermal noise;
  only the latter is closed here. A gradient across a 4096-channel VGA
  array would produce a *spatially correlated* per-channel error, which
  is exactly the case `docs/mismatch_calculus.md` R3 flags as breaking
  the independence assumption — it would not average and would not be
  demoted.
* **No 1/f noise.** Only white shot and Johnson terms are modeled. In
  subthreshold MOS, flicker noise typically dominates below ~1 MHz;
  since the per-channel comparison is made at the 153 MHz loop bandwidth
  this is a small correction there, but the *gain jitter* numbers (which
  integrate down to DC) are lower bounds.
* **The noise model is lumped, not device-intrinsic.** ngspice's `.noise`
  analysis is AC-only and cannot be run on this behavioral transient
  loop, so noise is injected as explicit `trnoise` sources with
  hand-derived PSDs (validated to 1.4 % against a probe). A netlist built
  from real MOS models with a PDK would place these terms automatically
  and would likely find more of them.
* **One `x` draw per cell**, as in the published harness. The common-mode
  medians in §4.2 carry visible draw-to-draw scatter (2.4–3.2 % across
  N at a floor of ~3 %); the per-channel numbers, which are what the
  kill-gates turn on, are stable to ~1 %.
* **N = 4096 is DC-only.** Equilibrium is found by the bisection/secant
  method the paper already accepts as equivalent to the settled
  integrator; no 4096-channel transient was run. Settling is a property
  of the loop, not of N, but a 4096-channel *layout* would add
  distribution parasitics on the `ln G` rail that this model has no
  representation of.
* **`σ_rect`, `I_b`, `V_A`, `I_det` and `I_int` are declared design
  parameters, not measured ones.** They are swept where the result turns
  on them (I_b in §3.2, V_A in §6, I_det and I₀ in §5) and fixed by
  stated reasoning otherwise.
* **K-C at I₀ = 1 µA is inside the Monte-Carlo scatter and no number of
  seeds fixes that**, because the two quantities being compared really
  are equal there. The result is reported as a tie on purpose; anyone
  wanting a one-sided verdict has to move the bias point, not the
  statistics.
* **The rectifier is behavioral.** `max(·, 0)` with per-branch gain and a
  dead-zone captures the mismatch and the dead-zone of a current-mode
  full-wave rectifier, but not its speed, its charge injection, or its
  behaviour on fast zero crossings — all of which matter for a real
  four-quadrant signal path and none of which a DC testbench can see.

---

## 9. Reproducing

```bash
cd spice
python3 phase0_gaps_sim.py --section anchors          # §2, ~2 min
python3 phase0_gaps_sim.py --section ga --mc 100      # §3, ~3 min
python3 phase0_gaps_sim.py --section gb --mc 100      # §4, ~6 min
python3 phase0_gaps_sim.py --section gc --gc-seeds 6  # §5, ~5 min
python3 phase0_gaps_sim.py --section gd               # §6, <1 min
python3 phase0_gaps_sim.py --section all              # everything
```

Seeds are fixed (`--seed 2026`). Sections are independent and each
re-derives its own anchors where it needs them. Netlists are generated
into `spice/out/phase0/` and deleted after each run; nothing is cached.
G-C is the only section whose numbers move run to run, because ngspice's
noise seed enters there; the movement is ±0.02 percentage points on the
per-channel figures, which is what makes the K-C verdict at 1 µA a tie
rather than a pass or a kill.
