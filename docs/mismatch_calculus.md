# The mismatch calculus

*A composable error-propagation algebra for the prime compiler.
Code: `spice/mismatch_calculus.py`. Retrodiction, prediction and SPICE
arbitration: `docs/exp_mismatch_calculus.md`, `spice/predict_gelu_layernorm_sim.py`.
Written 2026-09-16 against the measured record of this repository.*

---

## 0. What this replaces, and what it is copied from

Every result in `spice/` so far was obtained the same way: build the netlist,
run 200 Monte-Carlo draws, read the number. That is *characterization by
simulation of the whole circuit* — the analog equivalent of timing a program by
running it. Two mature disciplines stopped doing that decades ago:

| Digital / numerical practice | What it does once | What it then composes by rule |
|---|---|---|
| **Wilkinson / Herbie backward error analysis** | assign a condition number κ to each elementary operation | error of a composite expression = Σ κ_i · (unit roundoff), no re-evaluation |
| **Standard-cell characterization + STA** | characterize each cell once per corner (delay, slew, input capacitance) | path delay = sum of cell delays with the corner's derating, no SPICE on the path |

The mismatch calculus is the same move for analog prime realizations:
**characterize each prime realization once (a tuple of sensitivities), then
compose along the signal-flow graph by rule.** The unit roundoff of the digital
story is replaced by the device-parameter spread of the corner; the condition
number is replaced by a per-source sensitivity; and the one thing analog has
that digital does not — feedback — becomes its own composition rule (R2).

The calculus is **not** a new physical claim. Every rule below is a first-order
sensitivity of a law that the repo's ngspice testbenches already exercise. Its
only claim is *sufficiency*: that the sensitivities carry all the mismatch
information the 200-draw simulations were buying, to within a factor of two.
§11 of `exp_mismatch_calculus.md` is the test of that claim.

---

## 1. The characterization tuple

A prime realization `P` is characterized by

```
char(P) = ( domain,
            s_pc[src]  for src in {Is, n, Vt, K', mirror, gain, offset},
            s_cm[src],
            lever L,
            loop_position ∈ {upstream, at-input, inside, downstream, none},
            residual_class ∈ {static-reparametrizable, data-dependent} )
```

* **domain** ∈ {current-log, current-linear, voltage, charge, time} — fixes
  which device parameters even *have* a sensitivity (a charge-domain P8 has no
  `n`; a time-domain P5 has no `mirror`).
* **s_pc[src] = ∂(relative output error of channel i)/∂(parameter spread of
  src on a per-channel device)**. This is the dangerous column: it becomes
  activation noise.
* **s_cm[src]** — same derivative for a device on a *shared* path. It becomes a
  scale factor, which a trained network absorbs (paper §5.3, Result 4).
* **lever L** — the amplification the domain applies to a *dimensionless*
  parameter spread. In the log domain `L = ln(I/I_s)`; at a saturating stage
  `L = f'(u)/f(u)`; at a threshold `L = k/σ_tot`.
* **loop_position** decides whether R2 applies.
* **residual_class** decides whether R9 applies (can training eat it?).

### 1.1 The tuples this repo has actually measured

`sigma_b ≡ hypot(sigma_Is, sigma_n · L)` is used throughout; it is the sigma of
`d ln I` contributed by **one junction traversal** (R1).

| Realization | domain | lever L | per-channel sources | shared sources | loop pos. | residual class | anchor |
|---|---|---|---|---|---|---|---|
| **P3/P4 junction** (log/exp stage) | current-log | `ln(I/I_s)` = 23.0 at 1 µA / I_s = 1e-16; 19.1 at 2 µA / I_s = 1e-14 | `Is` (s=1), `n` (s=L) | — | — | static | Table 6; `exp_bounded_recursion.md` §3.3 |
| **P2 current mirror** | current-linear | 1 | gain (s=1) if one per channel | gain (s=1) if 1:N shared | — | static | Table 6/7 |
| **P11 VGA / Gilbert** | current-linear | 1 | gain (s=1) | — | at loop input | static | Table 7 rows B |
| **P1 KCL sum** | current-linear | 0 | — | — | — | — | exact by Kirchhoff |
| **P12 translinear ÷√** | current-log | L | — | `Is`,`n` (2 junctions) | — | static | Table 6 |
| **AGC RMS detector** (P11·P1·P2) | current-log | L | — | `Is`,`n` ×(2N+1), mirror | **is the loop** | static, common-mode | Table 7 rows A |
| **P8 capacitor** | charge | 1 | — | — | — | — | `exp_kv_leak_eviction.md` §6 (0.04 %) |
| **P5 race (deterministic)** | time | `ln(I_max/I_S)` = 19.1 | `Is` (s=1), `n` (s=L) → nats of logit | — | — | static | `exp_bounded_recursion.md` §3.3 |
| **P11 triode VCR** (2-terminal) | current-linear | `Δ_max/V_ov,max` | `K'` (gain), `V_t` (**floor**, s = Δ_max/V_ov,max) | — | none (no set-point) | static **until cutoff** | `exp_mamba_vcrc.md` §4.3 |
| **P11 transconductor pair** (3-terminal) | current-log | `1/(2V_T)` | `K'` (tail gain), `V_os` (**state offset**, s = 1/swing) | — | none | static to ~1 %, then tanh-compressive | `exp_mamba_vcrc.md` §4.3 |
| **P6 sigmoid pair** | current-log | `1 − σ(u)` | `V_t` → `δu = V_os/(nV_T)`, `K'` → `δu = ln(1+σ_K)` | — | upstream of any later loop | static | new; §2 R12 |
| **P7 attractor / CAM row** | voltage | `k/σ_tot` | `G` (s = 1/√k rel.) | match-line gain (s = `k/σ_tot`) | inside (attractor) | — | `exp_symbol_margin.md` §4.3, §6.4 |

---

## 2. The composition rules

Notation: `e_i` = relative error of channel `i` at the observation point;
`b_j = ν_j·L_j − ι_j` = the error contributed by junction `j`
(`ν ~ N(0,σ_n)`, `ι ~ N(0,σ_Is)`), entering with **+b** when traversed
current→voltage and **−b** when traversed voltage→current.

---

### R1 — Log-domain lever
**Formula.** A junction traversed once contributes
```
    d ln I  =  ±( ν·L − ι ),      L = ln(I/I_s)
    σ_junction = hypot(σ_Is, σ_n·L)
```
**Domain of validity.** Any device whose I–V law is exponential over the
operating range (diode, BJT, subthreshold MOS); L is evaluated at the *actual*
current of that device, so L varies from device to device inside one stage.
Breaks where the device leaves the exponential region.

**Why it matters.** L ≈ 19–23 at the biases this repo uses, so a 0.3 % spread
in `n` beats a 3 % spread in `I_s` by a factor 2. Every "mismatch-fragile"
verdict in the repo is this one number.

**Anchors.** Table 6 (0.3 % n → ~7 % current error, paper §5.3 Result 2);
`exp_bounded_recursion.md` §3.3 (0.0573 nat vs 0.0300 nat, lever 19.11);
`exp_symbol_margin.md` §1 (the same lever as a match-line score).

---

### R2 — Feedback demotes per-channel to common mode (Bode sensitivity)
**Formula.** For a stage at or downstream of the loop input,
```
    s_pc → 0,      s_cm → s_pc /(1 + L_loop)
```
For a stage **upstream** of the loop input, the tuple is unchanged: full
per-channel exposure.

**Domain of validity.** The loop must pin a *scalar* set-point that the stage's
error can move. It does **not** apply where the quantity of interest *is* the
state (a selective SSM has no set-point — `exp_mamba_vcrc.md` §4.3), and it
does not protect anything upstream of the loop input.

**Anchors.** Table 7 rows A: detector-side mismatch gives *exactly zero*
per-channel distortion at every corner. Result 5: the sum-feedback softmax beats
the open-loop one by only 3.1–3.4× rather than by the full junction count,
because the exponential stage (P3) sits upstream and keeps its exposure.
`exp_symbol_margin.md` §6.4 is the same rule one level up: a Hopfield attractor
loses 6.6 % of capacity at 60 % synaptic mismatch where the open-loop CAM row
already fails at 3 %.

---

### R3 — Per-channel vs shared path; quadrature, no cross terms
**Formula.** With `w_i = x_i²/Σ_j x_j²` the weight with which channel `i`
enters a shared (summed) path:
```
    shared device      → e_i = g  for all i          (pure common mode)
    per-channel device → e_i ⊥ e_j,  Var(Σ c_j b_j) = Σ c_j² σ_j²
    after a scalar normalisation (loop or α-fit):
        per-channel residual  RMS_i = σ_pc · sqrt( 1 − Σ_i w_i² )
        common-mode residue   = second order in σ_pc
```
`E[Σ w_i²] = 2(N−1)/(N²(N+2)) + 1/N`, i.e. 0.30 at N = 8, 0.088 at N = 32,
which is the entire N-dependence of the "0.7 σ_VGA" headline.

**The inversion to watch.** "Shared ⇒ common mode" holds only if the shared
device's error is proportional to the *signal*. Where it is proportional to
something orthogonal to the signal — e.g. a mean-subtraction mirror, whose error
is `−mean(x)·g` along the all-ones vector while the signal `c⁰ = x − mean(x)`
satisfies `⟨1, c⁰⟩ = 0` — a *shared* device lands with full weight in the
**per-channel** residual:
```
    per-channel from the mean path = √2 · |mean(x)| · σ_mirror / RMS(x − mean(x))
```
Measured in ngspice for full LayerNorm: 0.31 % against a predicted 0.24–0.30 %
(`exp_mismatch_calculus.md` §10.3).

**Domain of validity.** Independent mismatch sources; no spatial correlation,
no shared bias network. (`exp_sign_concordance.md` §5.4 flags spatially
correlated polarity errors as the case where the independence assumption fails
and the answer gets *worse*, not better.)

**Anchors.** Table 7 ("the two contributions add without cross terms");
`exp_mismatch_absorption.md` §2.3 (residual 0.7–0.95 σ, not a flat 0.7 σ, with
exactly this N-dependence).

---

### R4 — Threshold / decision
**Formula.**
```
    ε  ≈  Q( m / σ_tot ),           ln ε ≈ −(m/σ_tot)²/2 − ln(z√2π)
    feasibility (no margin rescues it):   σ_tot ≤ d_min / z_ε
    with a k-line dense code:             σ_tot = σ_G·√k,  d_min = 1
    lognormal conductances:               ε_FA ×(1 … 4) at k ≤ 16,
                                          ε_FR ×(1 … 1/8); CLT removes it by k = 256
```
`z_ε = Q⁻¹(ε)` = 3.090 / 4.753 / 5.998 at ε = 10⁻³ / 10⁻⁶ / 10⁻⁹.

**Domain of validity.** Static mismatch and additive noise; a *single*
adversary (R rows add a union-bound factor up to R). Under static mismatch ε is
a **yield**, not a rate: median row 10⁻²¹, fleet mean 10⁻⁵
(`exp_symbol_margin.md` §5.4).

**Anchors.** `exp_symbol_margin.md` K1 (slope −0.5228 over 6.14 decades),
K2 (lognormal prefactor), K3 (m* = 2.879 of the 2.0 available at k = 256,
σ_G = 3 %, ε = 10⁻⁹).

---

### R5 — Gain-accuracy sensitivity of a decision
**Formula.**
```
    d ln ε / dg  =  λ(z)·k/σ_tot,     λ(z) = φ(z)/Q(z)
    g_max (ε within 2×)  =  ln2·σ_tot /(k·λ(z))  ≈  ln2·σ_G /(√k·z_ε)
```
10–14 bits of match-line transimpedance accuracy at k = 64…256.

**Domain of validity.** Systematic (calibratable) gain error, not mismatch.
The rule says a decision stage needs a *gain* spec that no mismatch budget
implies, which is why it was missing from Table 4.

**Anchor.** `exp_symbol_margin.md` §4.3 and §7 (a 0.17 % finite-A₀ gain error
moved the measured false-accept count by 15–30 %).

---

### R6 — Backward paths: signs exact, magnitudes free to a decade
**Formula.** For a trainable stage `γ` behind a **frozen** forward read-out `W`
and a physically separate backward array `B`:
```
    diagonal stage:  converges  ⟺  diag(BᵀW)_i > 0  for EVERY i
    dense stage:     converges  ⟺  Re λ(W Bᵀ) > 0  for EVERY eigenvalue
    tolerated |B| error: one decade;  tolerated backward gain error: 20 %
    tolerated sign errors: ZERO (not a rate — a gate)
```
**Domain of validity.** Frozen forward crossbar. Says nothing about Lillicrap's
setting where `W` itself learns.

**Anchors.** `exp_sign_concordance.md` K1–K4: 0/327 records with ≥1 wrong-sign
channel converged, 69/73 with all signs correct did; S4 converged in exactly the
2/80 draws whose `W Bᵀ` had all-positive eigenvalues (agreement 80/80).

---

### R7 — Same-node rule (conversion accounting)
**Formula.**
```
    if  accumulator_node  ==  selector_drive_node:   conversions = 0
    else:                       conversions = 1 ADC + 1 DAC per channel per step
```
**Anchor.** `exp_bounded_recursion.md` §4.4: if the MCTS Q/N accumulator is not
the node that drives the P5 race, every simulation re-drives `b·d = 256` DAC
channels (12.8 pJ, 256 settles). Same class of hard structural precondition as
R6's sign concordance.

---

### R8 — Runtime rewiring: write latency dominates
**Formula.**
```
    write-bound  ⟺  t_wr  >  t_sel + t_alloc + t_bk
    crossover: 12.0 ns (d=8, parallel backup) … 78.0 ns (d=32, sequential)
```
**Anchor.** `exp_bounded_recursion.md` §4.2, KB2 triggered as designed; the
fabricated IMC-MCTS chip (arXiv:2607.22869) independently keeps the mutable
tree out of the resistive-write path.

---

### R9 — Static reparametrization vs data-dependent residue
**Formula.** Split the mismatch into the part that is a fixed function of the
learned parameters and the part that is not:
```
    e = e_static(θ) + e_data(x)
    e_static is absorbed by training  (X4 result: 0.7–0.95 σ → 0.06–0.16 σ)
    e_data   is not  (device leaves its region: cutoff, clipping, saturation)
```
Operationally: fit the platform's own reparametrization family to the measured
trace; what survives is `e_data`.

**Anchors.** `exp_mismatch_absorption.md` K2/K4; `exp_mamba_vcrc.md` §4.3 (the
pair's ~1000 % collapses to 5–15 % under a 5-constant refit — learnable; the
triode's 8.99 % → 9.86 % does not — cutoff on 5–17 % of tokens).

---

## 3. Rules the brief did not list, found in the same documents

### R10 — Terminal count decides what an offset becomes
**Formula.** For a device used as a controlled conductance,
```
    two-terminal (triode VCR):  V_t offset → CONDUCTANCE error
        g_eff = (1+ε_K)·β·(V_ov − δ_Vt)   ⇒  Δ_eff = (1+ε_K)(Δ − δ_Vt·Δ_max/V_ov,max)
        — a gain-and-floor perturbation, zero current at zero v_d whatever V_t
    three-terminal (transconductor pair): V_os → STATE OFFSET
        the leak relaxes to −V_os, the input pair injects 2V_T·tanh(V_os/2V_T)
        of phantom drive; both measured in full scales of the state swing
```
**Design law.** *Choose the two-terminal element whenever the state is the
thing being multiplied.* A 10 mV offset against a 4 mV swing is 2.5 full
scales, which is the whole of the pair variant's 723–2122 % median error.

**Anchor.** `exp_mamba_vcrc.md` §4.3 ("the structural finding").

---

### R11 — Dynamic range closes two walls at once
**Formula.** For a triode VCR spanning `R = Δ_max/Δ_min`:
```
    state swing ≤ 2·V_ov,max/R        and       V_ov,min = V_ov,max/R
    cutoff when σ_Vt ≳ V_ov,min  ⇒  the residue becomes data-dependent (R9)
```
and, with Mamba's Euler `B̄ = Δ·B` instead of the ZOH the capacitor computes for
free, the write path needs a further `|A|·Δ_max` of range (160× here).

**Anchor.** `exp_mamba_vcrc.md` §4.2.

---

### R12 — Saturating-stage slope lever (generalises R1)
**Formula.** An input-referred offset `δ` at a stage with transfer function `f`
appears at the output *relatively* as
```
    e = δ · f'(u)/f(u)                       (the lever of §1 is f'/f)
    f = exp   ⇒  f'/f = 1/(nV_T)            → R1
    f = σ     ⇒  f'/f = 1 − σ(u)            → largest exactly where the output is small
    f = tanh  ⇒  f'/f = (1−tanh²)/tanh
```
For a logistic stage this means the mismatch is **worst on the channels that
carry least signal**, so it survives any downstream normalisation: a sigmoid is
a per-channel error amplifier in its own left tail. Input-referred offsets
compose additively before the lever is applied:
`δ = V_os/(nV_T) + ln(1+σ_K') + ν·|u|`.

**Status.** Derived here from the tuples of §1; no anchor in the repo when it
was written. **Validated prospectively**: it carries half of the GELU
per-channel error (5.97 % of 8.53 %, `exp_mismatch_calculus.md` §10.2) in a
prediction that ngspice confirmed to 1.00×.

---

### R13 — What training can and cannot reach
**Formula.**
```
    untrained / trained-in-simulation :  σ_pc·sqrt(1 − Σ w_i²)          (R3)
    physics-aware / sign-concordant / perturbative :  the finite-sample
        least-squares floor of the LOSS, which is NOT the floor of the
        per-channel metric (the loss does not divide out the per-sample scalar)
    perturbative cost :  ∝ N  in gradient steps (SPSA variance per coordinate
        is ‖g‖² instead of g_i²)
    identifiability  :  with W also learnable, γ is gauge-unidentifiable
        (γ_i ↔ W_{:,i}/λ_i) and the per-channel metric is meaningless
```
**Anchors.** `exp_mismatch_absorption.md` §3.1–3.4, §4.1 item 2.

---

### R14 — Race: winner latency is field-size-independent, energy is not
**Formula.**
```
    t_(1) = C·V_th/I_max                      independent of N   (5 digits)
    t_(B) = t_(1)·exp(g_1B)                   exponential in the logit gap
    E_race = V_dd·C·V_th·Z_norm·exp(g_1B),    Z_norm = Σ exp(z_i − z_max)
    usable gap floor: g_sel ≳ 6·√2·σ_tot, σ_tot from R1
```
**Anchor.** `exp_bounded_recursion.md` §3.1–3.2 (energy law to 6 significant
figures).

---

### R15 — Measurement floors compose like mismatch
**Formula.** The resolution of the *method* enters the same quadrature as the
device spread:
```
    bisection on a geometric bracket, n halvings of [lo,hi]:
        q = ln(hi/lo)/2ⁿ   uniform in ln G    ⇒   adds q²/12 to Var(ln G)
```
For `settle_gain()` (16 halvings of [1e-3,1e3]) this is q = 2.108e-4, i.e. a
0.005 % floor on any reported common-mode gain shift. It is exactly the floor
visible in Table 7's "0.006 % / 0.006 % / 0.015 %" column, and without it the
σ_VGA = 0.5 % entry looks like a 7× discrepancy.

**Status.** Methodological, not physical; but a calculus that ignores it
mis-scores its own retrodiction.

---

## 4. How to compose (the procedure)

1. **Draw the signal-flow graph** of the realization, one node per physical
   node, one edge per device traversal. Mark each edge forward (I→V) or
   backward (V→I).
2. **Attach the tuple** of §1 to each edge; evaluate `L` at the *actual*
   operating current of that device, not at a nominal.
3. **Walk the graph** from each output back to each source, accumulating
   sensitivity coefficients `c_ij`. Shared devices get the same coefficient in
   every channel; per-channel devices get one coefficient each. Wherever a
   summed node is crossed, the coefficients pick up the energy weights `w_i`
   of R3.
4. **Apply R2** at any loop input: everything from there on is common mode with
   suppression 1/(1+L); everything before it keeps `s_pc`.
5. **Compose** with R3: `Var(e_i) = Σ_j c_ij²σ_j²`. Split by whether `c_ij` is
   channel-dependent (per-channel) or not (common mode).
6. **Apply R12/R4/R5** at any saturating or thresholding stage.
7. **Apply R9/R13** if the stage is trainable: the static part comes off.
8. **Add R15** for whatever numerical procedure reports the number.

Steps 1–3 for the open-loop translinear RMSNorm of the paper give, in full:

```
 e_out,i = (1−w_i)·b_x,i − Σ_{j≠i} w_j·b_x,j        [per channel, R1×R3]
         − b_o,i                                    [per channel, R1]
         + ½ Σ_j w_j·b_sq,j                         [per channel, damped by ½ and w]
         + b_ref − ½·b_ms − b_inv + b_invl          [shared, R3]
         − ½·g_mirror                               [shared]
 Var(e_i) = (1−2w_i+Σw²)σ_b,x² + σ_b,o,i² + ¼Σw_j²σ_b,sq,j²
          + σ_b,ref² + ¼σ_b,ms² + 2σ_b,inv² + ¼σ_mir²
```
The sum of squared coefficients is **5.375** at N = 8. The paper's metric is
`mean_i |out_i − ref_i|` with `RMS(ref) = 1`, so
`error = mean_i( ref_i·√(2/π)·sd(e_i) )`. That is the whole of Table 6 —
three numbers from one line of algebra (`item1_table6()`), agreeing to 5 %.

---

## 5. Worked example — softmax, and why feedback wins by only 3×

The open-loop and the feedback softmax share the exponential stage. Writing
`p_i` for the ideal softmax and `S_i = 1 − 2p_i + Σ_j p_j²`:

```
 open :  Δ_i = (e_i − Σ_j p_j e_j) + b_l,i − b_o,i + b_ref − b_sum
         Var = σ_b,e²·S_i + σ_b,l² + σ_b,o² + σ_b,ref² + σ_b,sum²
 AGC  :  δ_i = (e_i + ε_i) − Σ_j p_j (e_j + ε_j)
         Var = (σ_b,e² + σ_VGA²)·S_i
 E[L1] = √(2/π)·Σ_i p_i·sd_i                        (exact: linearity of E)
```

Two readings fall straight out of the formulas:

* The AGC variance is the *same* `σ_b,e²` multiplied by `S_i`, plus the VGA.
  The exponential stage is untouched — **R2 protects the normalisation path,
  not the primes upstream of it**. That is the paper's Result-5 caveat, derived
  rather than observed.
* The open-loop variance carries **four extra junctions**, two of them shared
  (`b_ref`, `b_sum`) and *not* removed, because an open-loop softmax never
  renormalises. Four extra junctions on top of `σ_b,e²·S_i` with `S_i < 1` is
  what makes the ratio ≈ 3 and not ≈ 10: the shared terms would have been free
  in a circuit that normalises at the end.

Predicted 3.16 / 3.27 / 3.26 against the measured 3.1 / 3.3 / 3.4, and the
absolute L1 values 13.3 % / 4.06 % against the measured 13.6 % / 4.1 %.

---

## 6. Worked example — the VGA residual and its "0.7 σ"

```
 y_i = G·x_i(1+ε_i),   G = Γ/RMS(x(1+ε))          [loop equilibrium]
 α   = ⟨y,y_ref⟩/⟨y_ref,y_ref⟩ = G·RMS(x)·(1 + Σ_j w_j ε_j)
 resid_i = y_i/α − y_ref,i = y_ref,i·(ε_i − Σ_j w_j ε_j)
 RMS_i(resid) = sqrt( Σ_i w_i (ε_i − ε̄_w)² )  →  σ_VGA·sqrt(1 − Σ_i w_i²)
 α − 1 = −½Σ_i w_i ε_i² + ½(Σ_i w_i ε_i)²      [second order ⇒ 0.00x %]
```
`Σ w_i²` is a property of the *input draw*, not of the circuit: 0.30 in
expectation at N = 8, 0.088 at N = 32. Table 7's headline "0.7–0.8 σ_VGA" and
`exp_mismatch_absorption.md`'s cross-check "0.7–0.95 σ" are the same formula at
two different draws. The common-mode column of Table 7 is *below* the
bisection floor of R15 and must be quoted with it.

---

## 7. What the calculus does not do

* **No transients.** Settling times, loop bandwidth and slew are outside it;
  `t_AGC = 4.4 ns` is an input to the cost model, not an output of the algebra.
* **No noise.** Thermal, 1/f, kT/C and supply noise are absent. Every σ here is
  a *static* device-parameter spread. `exp_symbol_margin.md` §5.2 shows that
  0.5 % match-line noise changes a feasibility bound by more than the mismatch
  does; the calculus would have to be extended, not merely re-parameterised.
* **No correlations.** Independent per-device draws, no spatial gradients, no
  shared bias network, no `1/√(WL)` area law.
* **First order in the spread, exact in the map.** Where a sensitivity is not
  small (the sigmoid offset `δu ≈ 0.39`, the pair's 2.5-full-scale offset), the
  code evaluates the *closed-form map* by cheap Monte Carlo instead of
  linearising. That is still no circuit simulation, but it is no longer a
  single derivative.
* **One technology model.** Level-1 MOS, ideal diodes, BJT stand-ins for
  subthreshold pairs, no PDK, no silicon.

---

## 8. Files

| File | Contents |
|---|---|
| `docs/mismatch_calculus.md` | this document: tuples, rules, procedure, worked examples |
| `spice/mismatch_calculus.py` | the algebra as code + the retrodiction of 36 measured numbers + the prospective prediction |
| `docs/exp_mismatch_calculus.md` | pre-registration, retrodiction table, prediction, SPICE arbitration, verdicts |
| `spice/predict_gelu_layernorm_sim.py` | the ngspice arbiter (GELU = P11·P6, LayerNorm = P1·P2 + AGC) |
