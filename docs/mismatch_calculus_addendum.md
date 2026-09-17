# Addendum to the mismatch calculus — R16 to R20

*Extends `docs/mismatch_calculus.md` (R1–R15). Does not replace it; every rule
below is additive except **R19**, which restricts R2 to the static case.
Evidence: `docs/exp_domain_gates.md` §3.3/§4.4 (workload and normalisation),
`docs/exp_calculus_noise.md` (noise and temperature, run 2026-09-16),
`spice/calculus_noise_sim.py`. Written 2026-09-16.*

---

## 0. What the original calculus was missing

Three holes, each found by a measurement rather than by inspection:

| hole | found by | cost of ignoring it |
|---|---|---|
| **the lever is evaluated at a nominal current, not at the realised one** | `exp_domain_gates.md` §3.3 — D1-3 failed its strictest reading | a "technology constant" `V_th* = 155 mV` that is really 176–202 mV and workload-dependent |
| **the σ column mixes two normalisations** | `exp_domain_gates.md` §4.4 | a silent **1.500 bits** between any offset-type and any gain-type entry of the domain table |
| **no dynamic class** | `mismatch_calculus.md` §7, admitted; now measured | a per-channel exposure of **2.0 %** in the AGC bench, 2.9× the mismatch residual the paper names as the only one |

---

## 1. R16 — the lever is *realised*, not nominal

### 1.1 The rule

R1 gives a junction's log lever as `L = ln(I/I_S)`, "evaluated at the actual
current of that device". §4 step 2 of the procedure says the same. What neither
says is what to do when the "actual current" is **a property of the data**, not
of the bias plan. It almost always is: a softmax, a GELU, a normalisation layer
all place their elements over decades of current, and only the *top* element
sits at `I_max`.

```
  R16 (workload-realised lever)

    per element :   L_i = ln(I_i / I_S)                     (exact)
    per stage   :   L_eff = <ln(I/I_S)>_p(x)                (population mean
                                                             over the workload)
    so          :   sigma_stage = hypot( sigma_Is , sigma_n * L_eff )

    ln(I_max/I_S) is an UPPER BOUND on L_eff, attained only when every element
    runs at I_max.  Quoting it as "the" lever over-states the log domain's
    mismatch by  L_max / L_eff  in the sigma_n term.
```

Corollary: **every number in the calculus that carries an `L` is a functional of
the workload distribution.** `L_eff` falls with `N` and with the logit spread,
because a softmax over wider logits pushes more mass into the tail.

### 1.2 Re-prediction of D1's crossing `V_th*`

`domain_axis.md` §5 predicted the crossing where the comparator lever
`σ_Vos/V_th` overtakes the log lever, at `V_th* = σ_Vos/0.0647 = 155 mV`, as a
technology constant. Under R16, with `σ_Vos = 10 mV`, `σ_Is = 3 %`,
`σ_n = 0.3 %`:

```
    V_th*  =  sigma_Vos / sqrt( sigma_Is^2 + <ln(I/I_S)>^2 * sigma_n^2 )
```

| | `L_eff` | `σ_log` predicted | `σ_log` measured (§3.3) | **`V_th*` predicted** | **`V_th*` measured (§3.2)** | ratio |
|---|---|---|---|---|---|---|
| nominal, `ln(I_max/I_S)` | 19.114 | 6.4716 % | — | **154.5 mV** | — | — |
| **N = 8**, realised | 16.14 | **5.6960 %** | 5.731 % | **175.6 mV** | **177.5 mV** | **1.011** |
| **N = 64**, realised | 13.11 | **4.9466 %** | 4.968 % | **202.2 mV** | **200.7 mV** | **0.993** |

**The realised lever closes D1-3.** The nominal lever misses the N = 64 crossing
by 23 % (154.5 vs 200.7 mV) and put it outside the pre-registered 124–186 mV
band; the realised lever hits both crossings to **1.1 % and 0.7 %**. The
failure recorded in `exp_domain_gates.md` §5 was a failure of the *constant*,
not of the formula, and R16 is the repair.

Two things follow that the domain table does not say:

* **`V_th*` is a workload specification.** 155 mV is the limit of an
  infinitely peaked softmax; a real 8-way softmax needs 176 mV and a 64-way one
  202 mV. A compiler that reads 155 mV out of §1.1 and sizes a comparator
  threshold with it will sit on the wrong side of the trade at N ≥ 16.
* **The realised lever is a *draw*, not just a function of N.** The 16.14/13.11
  above are the values of the single fixed input vector that harness used; over
  an ensemble of N(0, 2) logit draws the mean is 16.27 (N = 8) and 14.43
  (N = 64), giving `V_th* = 174.5 / 189.8 mV`. So the honest statement of the
  crossing is a **band, 175–202 mV at N = 8…64**, and the number to put in a
  spec is the one computed from the model's own activation statistics.

### 1.3 Where else R16 moves a published number

| number | as published | under R16 |
|---|---|---|
| `domain_axis.md` §1.1 current-log lever | 6.47 % → 3.0 b | **5.70 % → 3.1 b** at N = 8, **4.95 % → 3.3 b** at N = 64 |
| §2 row P12, open-loop translinear (3 junctions) | 11.2 % → 2.2 b | **9.87 % → 2.3 b** at N = 8 |
| §0.1's calibration `6.47 %·√3 = 11.2 %` vs measured 11.4 % | agreement 2 % | **unchanged** — that pipeline runs all channels near `I_max`, so `L_eff ≈ L_max` there. R16 is a correction for *distributed* workloads, not a global rescale |

The last row is the reason R16 was invisible until D1: the repo's original
RMSNorm calibration happens to be a workload where the bound is tight.

---

## 2. R17 — one normalisation, declared with its workload

### 2.1 The inconsistency

`exp_domain_gates.md` §4.4 measured it: on the D2 operand grid the two metrics
differ by exactly `FS_y/RMS_grid(y) = 2.8284` = **1.500 bits**, identically for
all six arms, and `domain_axis.md` §1.1 reads some levers out of one and some
out of the other:

* **gain-type** levers (error ∝ signal: `δ_Is`, `L·δ_n`, `δ_K′`, `δ_C`) are
  naturally **signal-referred** — metric A, `RMS(err)/RMS(ref)`;
* **offset-type** levers (error independent of the signal: `σ_Vt/V_ov`,
  `σ_Vt/nV_T`, `σ_Vt/V_FS`, `σ_Vos/V_th`, `√(kT/C)/V_FS`) are naturally
  **full-scale-referred** — metric B.

Comparing a 4.48 % triode entry against a 0.100 % charge entry straight out of
that column hides a factor 2.83.

### 2.2 The convention

```
  R17 (single normalisation)

  All sigma_rel in this project are SIGNAL-REFERRED:

        sigma_rel  =  RMS_workload( out - ref ) / RMS_workload( ref )
        ENOB       =  log2( 1 / (2 sigma_rel) )

  An offset-type lever must therefore be multiplied by the workload's
  CREST FACTOR before it may be compared with anything:

        CF = FS_signal / RMS_workload(signal)
        sigma_rel,A = hypot( gain-type parts , CF * offset-type parts )
        ENOB_A      = ENOB_B - log2(CF)

  Declared CF for this project: CF = 2.8284 (the D2 operand grid), i.e.
  exactly 1.500 bits.  A Gaussian activation clipped at 3 sigma gives CF = 3.0
  (1.58 b); a peaked softmax gives far more.  CF is part of the spec, like
  L_eff in R16.

  EXEMPTION: decision stages (P5 argmax, P7 threshold).  A threshold's error
  is referred to the threshold by construction, and R4/R5 -- not ENOB -- price
  it.  Do not apply CF there; quote sigma_Vos/V_th and z_eps.
```

Signal-referred is chosen because it is already the repo's convention
everywhere it matters: `mamba_vcrc_sim.rel_rms`, `tab:mismatch`'s 11.4 %/22.9 %,
`decompose_error()` of Table 7, and every number in
`exp_mismatch_calculus.md`. Metric B survives as a *diagnostic* — it is the
one in which a pure offset lever is a constant — but it is no longer a
currency.

### 2.3 `domain_axis.md` §1.1 restated under R17

| domain | gain-type part | offset-type part | σ_B (as published) | ENOB_B | **σ_A (R17)** | **ENOB_A** | Δ bits |
|---|---|---|---|---|---|---|---|
| **current-log** (`L = 19.11`) | `√(δ_Is²+(Lδ_n)²)` | — | 6.470 % | 2.95 | **6.470 %** | **2.95** | 0.00 |
| current-log, R16 at `⟨L⟩ = 16.14` | — | — | — | — | **5.696 %** | **3.13** | — |
| **subthreshold pair** | — | `σ_Vt/nV_T` | 38.70 % | 0.37 | **109.5 %** | **−1.13** | **−1.50** |
| **triode VCR** | `δ_K′` = 3 % | `σ_Vt/V_ov` = 3.33 % | 4.485 % | 3.48 | **9.894 %** | **2.34** | **−1.14** |
| **Gilbert, strong inv.** | 4.24 % | 4.71 % | 6.342 % | 2.98 | **13.99 %** | **1.84** | **−1.14** |
| **voltage** | — | `σ_Vt/V_FS` | 1.000 % | 5.64 | **2.828 %** | **4.14** | **−1.50** |
| **charge**, σ_C = 0.1 % | `δ_C` | `√(kT/C)/V_FS` | 0.100 % | 8.96 | **0.102 %** | **8.94** | −0.02 |
| **charge**, σ_C = 1 % | `δ_C` | same | 1.000 % | 5.64 | **1.000 %** | **5.64** | 0.00 |
| **time** (race) | `σ_I/I` = 4.48 % | `σ_Vos/V_th` = 2 % | 4.906 % | 3.35 | **7.216 %** | **2.79** | **−0.56** |
| **frequency/phase** | `√(δ_K′²+δ_C²)` | — | 3.020 % | 4.05 | **3.020 %** | **4.05** | 0.00 |
| **current→time comparator** | — | `σ_Vos/V_th` | 2.000 % | 4.64 | **5.657 %** | **3.14** | **−1.50** |

Sanity against the D2 measurements, which are *in metric A*:

| arm | R17 prediction (ENOB_A) | measured (ENOB_A, `exp_domain_gates.md` §4.1) | ratio on σ |
|---|---|---|---|
| triode VCR | 9.894 % → 2.34 b | 8.147 % → 2.62 b | 0.82 |
| charge, σ_C = 0.1 % | 0.102 % → 8.94 b | 0.0941 % → 9.05 b | 0.92 |
| charge, σ_C = 1 % | 1.000 % → 5.64 b | 0.5505 % → 6.50 b | 0.55 |
| subthreshold Gilbert (×3 pairs, §4.4) | 109.5 % → −1.13 b | 335.8 % → −2.75 b | 3.07 |

Three of four land inside 1.25×, against the published table's own admission
that the pair is mis-priced by 4–8× for a *structural* reason (§4.4: one pair
priced, three present) that R17 does not touch.

### 2.4 Which rows of `domain_axis.md` §2 change

Mechanically: **a cell changes iff its lever has an offset-type component.**

| prime | cell | as published | **under R17** | Δ |
|---|---|---|---|---|
| **P1 Σ** | voltage | 5.6 b | **4.1 b** | −1.5 |
| P1 | charge, freq, KCL | 9.0 b / 4.1 b / exact | unchanged | 0 |
| **P2 ×c** | subthr. pair | 0.4 b | **−1.1 b** | −1.5 |
| **P2** | triode | 3.5 b | **2.3 b** | −1.2 |
| **P2** | voltage | 5.6 b | **4.1 b** | −1.5 |
| **P2** | time | 3.3 b | **2.8 b** | −0.6 |
| P2 | current-log, charge, freq | 3.0 / 9.0 (5.6) / 4.1 b | unchanged (log → 3.1 b under R16) | 0 |
| **P3 eˣ** | subthr. pair | 0.4 b | **−1.1 b** | −1.5 |
| **P3** | time (as a magnitude) | 3.3 b | **2.8 b** | −0.6 |
| P3 | current-log; time *as a rank* | 3.0 b; exact | unchanged | 0 |
| **P4 ln x** | time | 3.3 b | **2.8 b** | −0.6 |
| P4 | current-log | 3.0 b | unchanged (3.1 b under R16) | 0 |
| **P5 argmax** | comparator bank / charge comparator | 4.6 b implied | **exempt** — R4/R5, not ENOB | — |
| **P6 σ(x)** | subthr. pair | "0.4 b but learnable" | **−1.1 b, still learnable** | −1.5 |
| **P6** | triode tanh | 3.5 b | **2.3 b** | −1.2 |
| **P6** | voltage | 5.6 b | **4.1 b** | −1.5 |
| **P7 θ(x)** | voltage / time comparator | 4.6 b | **exempt** (R4); if quoted as ENOB, **3.1 b** | −1.5 |
| P7 | charge | 9.0 b | unchanged | 0 |
| P8 ∫ | charge / time / freq | 9.0 b / exact / 4.1 b | unchanged | 0 |
| **P9 d/dt** | voltage | 5.6 b | **4.1 b** | −1.5 |
| **P9** | time | 3.3 b | **2.8 b** | −0.6 |
| P9 | charge (R·C spread) | 4.5 b | unchanged | 0 |
| P10 ξ | — | — | now priced by **R18**, not by a lever | — |
| **P11 a×b** | subthr. pair | 0.4 b | **−1.1 b** (and −2.75 b measured, §4.4) | −1.5 |
| **P11** | triode | 3.5 b | **2.3 b** | −1.2 |
| **P11** | voltage | 5.6 b | **4.1 b** | −1.5 |
| **P11** | charge, two continuous operands | 5.6 b | **2.3 b** (a transconductor re-enters — measured 2.45–4.05 b, §4.3) | −3.3 |
| **P11** | time | 3.3 b | **2.8 b** | −0.6 |
| P11 | current-log, charge with a switched operand | 6.5 %·√k / 9.0 b | unchanged | 0 |
| **P12 1/x** | time | 3.3 b | **2.8 b** | −0.6 |
| P12 | current-log open loop | 2.2 b | unchanged (2.3 b under R16) | 0 |
| P12 | current-log in AGC feedback | 0.000 % per channel | **unchanged for mismatch; see R18 — 2.0 % per channel from noise** | — |

**What does *not* move.** Every ranking in `domain_axis.md` §3.1 survives:
the offset-type domains all lose the same 1.5 bits, so the ordering among them
is untouched, and the two domains the compiler is told to move work *into* —
charge and time — are exactly the two that lose least (0.0 and 0.6 bits).

**What does move.** The *size of the gaps*, and one gap in particular:

* **charge vs triode widens from 5.5 to 6.6 bits** (8.94 vs 2.34), because the
  triode pays the crest factor on its `V_t` term and the charge domain does
  not. The compiler's preference for charge gets stronger, not weaker.
* **The largest single-prime spread of §2 grows.** P11 was quoted as
  0.4 b (pair) → 9.0 b (charge) = 8.6 bits; under R17 it is
  **−1.1 b → 8.9 b = 10.0 bits**.
* **The "8-bit crossover" style comparisons become unusable without a stated
  CF**, which is the point of the rule.

---

## 3. The third lever class: **dynamic**

`domain_axis.md` §1.2 item 2 named two classes and one non-class:

> offset levers are additive and hence **learnable**; gain levers are
> multiplicative and partly **trimmable**; region-exit (cutoff) is neither.

Measurement adds a third that is neither, for a different reason: it is
**re-drawn faster than any calibration or training loop can observe it**.

```
  THE THREE CLASSES

  static-trimmable   gain-type mismatch, drawn once per die
                     -> one constant per channel removes it (R5, R9)
  static-learnable   offset-type mismatch that is a fixed function of the
                     learned parameters -> training absorbs it
                     (R9/R13: 0.7-0.95 sigma -> 0.06-0.16 sigma)
  DYNAMIC            thermal / shot / kT/C noise.  Re-drawn every tau.
                     NOT trimmable, NOT learnable, NOT reducible by area.
                     Sets a floor.  Bought down only with CURRENT, CAPACITANCE
                     or BANDWIDTH -- i.e. with energy and with speed.
```

### R18 — the dynamic lever

```
  R18 (dynamic lever)

  Per edge of the signal-flow graph, alongside the static sigma of R1/R12,
  attach a dynamic sigma -- the relative current (or voltage) error the
  device's own white noise produces in the declared bandwidth B:

     junction, shot noise        sigma_dyn = sqrt( 2 q B / I )
     resistor, thermal           sigma_dyn = sqrt( 4 k T B / R ) / I
     sampled node, kT/C          sigma_dyn = sqrt( k T / C ) / V_signal
     diff. pair, ideal tail      sigma_dyn = sqrt( 2 q B (1 - sigma(u)) / I_a )
                                  -- NOT one more junction; the tail forces
                                     i_a = -i_b, so S_Ia = 2q I_T sigma(1-sigma)

  Composition: EXACTLY R3.  Same +/-1 sensitivities, same quadrature, same
  energy weights w_i at a summed node, same (1 - sum w_i^2) after a scalar
  normalisation.  No separate machinery.

  Two structural differences from R1, both measured:

   (a) NO LEVER L.  Shot noise is already a current fluctuation, so the log
       domain's factor-19 amplification of sigma_n has no dynamic counterpart.
       The log domain's mismatch disadvantage is not a noise disadvantage.
   (b) LOUDEST WHERE THE CURRENT IS SMALLEST -- the same tail channels where
       R12 says the sigmoid offset is worst.  The differential pair is the
       exception and runs the other way: quietest where its output is largest.
```

**Validation** (`exp_calculus_noise.md` §5.1, §5.3). GELU = P11·P6, B = 100 MHz,
N = 8, ngspice `.noise`, no free parameter:

| | per-channel σ range | metric p50 | ratio |
|---|---|---|---|
| ngspice | 0.819 – 2.967 % | **0.6026 %** | — |
| R18 + R3 walk | 0.810 – 2.966 % | **0.6013 %** | **0.998** |

per-channel agreement **1.0001–1.012** on all eight channels over a 26× range
of output current. AGC detector (17 junctions on a shared path): predicted
7.8716·10⁻⁷/√Hz against a measured 7.8717·10⁻⁷/√Hz, **ratio 1.000** — and
**7.744** if the R3 energy weights are dropped, which is the sharpest evidence
in the repo that those weights are load-bearing.

### R19 — feedback, frequency-resolved (restricts R2)

```
  R19 (dynamic feedback)

  R2 as written ("s_pc -> 0, s_cm -> s_pc/(1+L)") is a STATIC rule.  For an
  error with a spectrum, measured on the closed AGC loop:

    shared-path error     -> COMMON MODE at EVERY frequency
                             |H| = 1/2 at DC (it is the loop's reference, so
                             no 1/(1+L) suppression -- for noise as for
                             mismatch), rolling off above f_loop:
                             REJECTED above the loop bandwidth, not demoted.
    per-channel error     -> STAYS PER CHANNEL at every frequency.
                             |H(own channel)| = 1 - w_i  below f_loop
                                              = 1        above f_loop
                             |H(other channel)| = w_i    below f_loop
                                                 = 0     above f_loop

  So: the residual factor of R3 is sqrt(1 - sum w_i^2) BELOW f_loop and
  EXACTLY 1 ABOVE it.  A loop demotes a per-channel error by at most
  1 - w_i ~ 1 - 1/N, and demotes nothing at all above its own bandwidth.

  Bandwidth to use: ENBW = integral |H|^2 df / |H(0)|^2 of the MEASURED
  closed-loop response -- not 1/(2 pi t_settle), which is a large-signal
  number.  On the repo's AGC bench the two differ by 25x
  (56.8 MHz from t_AGC = 4.4 ns vs 1395 MHz measured).
```

Measured, `exp_calculus_noise.md` §5.2, six digits: `H(ch0→y₀)(DC) = 0.909012`
against `1 − w₀ = 0.909012`; `H(ch0→y₁)(DC) = 0.090988` against
`w₀ = 0.090988`; `H(ch0→y₀)(∞) = 1.000000`; `H(det→y₀) ≡ H(det→y₁) ≡
H(det→δG/G)` with `|H(0)| = 0.500000` and `1.4·10⁻⁵` at 100 GHz.

### R20 — the noise floor, and what it trades against

```
  R20 (dynamic floor)

    sigma_dyn ∝ sqrt(B) ∝ 1/sqrt(t_settle)

  A per-channel accuracy spec and a settling-time spec are THE SAME SPEC.
  For a current-mode channel of k junctions at current I:

    sigma_pc,dyn = sqrt( 2 q k B / I ) * sqrt(1 - sum w_i^2 | below f_loop)
                 = sqrt( 2 q k / (4 I t_settle) )

  Buy it down with current (energy), with bandwidth (speed) or with fewer
  junctions -- never with area, never with trimming, never with training.
  The charge-domain twin is sqrt(kT/C)/V_FS: buy it with C (energy) or with
  swing (energy).  This is Kinget's bandwidth-accuracy-power tradeoff
  appearing in the calculus for the first time; the static rules had only the
  Pelgrom half of it.
```

**Consequence for the paper (K1 of `exp_calculus_noise.md`, TRIGGERED).**
The AGC RMSNorm's per-channel error at the loop bandwidth it actually has:

| contribution | size | class |
|---|---|---|
| VGA mismatch, σ_VGA = 1 % (Table 7) | 0.690 % | static-trimmable |
| **VGA shot noise at ENBW_loop = 1395 MHz** | **1.998 %** | **dynamic — neither** |
| VGA shot noise at 100 MHz | 0.563 % | dynamic |
| detector noise (common mode) | 0.735 % | dynamic, common mode |
| kT/C on the 1 pF gain capacitor | 0.009 % | dynamic, common mode |

> The paper's claim that *"the VGA array is the sole per-channel exposure"* is
> **incomplete**, not wrong: the VGA array *is* the sole per-channel exposure,
> but it has **two** of them, and the untrimmable one is 2.9× the trimmable one
> at the bandwidth that buys the advertised 4.4 ns settling. The ranking holds
> only if the per-channel path is band-limited to ≲ 150 MHz.

---

## 4. R21 — temperature gradients are levers, referred to the reference device

Stated separately because a gradient is **static in space but not in time**:
trimmable exactly as far as the thermal map is repeatable, which a
workload-dependent power map is not.

```
  R21 (gradient)

    A gradient enters as a GAIN-type (multiplicative) lever in every domain
    measured, referred to the temperature of the stage's REFERENCE device:

      current-log, translinear:   d ln I_i = L * (T_i - T_ref)/T
                                  -> exactly zero on the channel at T_ref,
                                     antisymmetric around it
      current-log VGA at gain G:  d ln I_i = -ln(G) * (T_i - T_ref)/T
                                  -> a unity-gain log VGA is EXACTLY
                                     temperature-independent
      subthreshold pair:          slope change -dT/T, offset ZERO
                                  (both devices of one pair at one T)
      shared path (detector):     pure common mode (R3, unchanged)

    CAVEAT, and it is the whole of the classification: this holds at CHANNEL
    granularity.  A gradient ACROSS one pair's two devices gives
    dV_os = (EG/q) * dT_pair/T ~ 3.7 mV/K, i.e. offset-type after all.
    Granularity of the thermal model decides the class.
```

Measured (`exp_calculus_noise.md` §5.4), `ΔT = 2 K` across N = 8:

| bench | measured | R21 | ratio |
|---|---|---|---|
| GELU translinear, spread over the array | 12.63 % | `L·ΔT/T = 12.73 %` | **0.99** |
| GELU, relative error under a 4× operand rescale | ×1.0000 | gain-type ⇒ ×1 | exact |
| GELU, absolute error under the same rescale | ×0.2500 | gain-type ⇒ ×0.25 | exact |
| subthreshold pair, slope | −0.6619 % | `−ΔT/T = −0.6663 %` | 0.993 |
| subthreshold pair, input offset | 4.2·10⁻⁷ | 0 | — |
| AGC VGA at `G₀ = 0.717` | 0.000–0.193 % | `−ln(G₀)·ΔT_i/T` | 0.995 |
| AGC detector | −1.548 % common mode, 0 per channel | R3: shared → common | structural |

**The design law.** A 2 K gradient costs the open-loop translinear GELU
**1.41 % per channel** — twice the whole VGA mismatch residual — and costs the
AGC's current-log VGA **0.033 %**, a factor 43 less, for one structural reason:
the translinear chain's log lever multiplies `ΔT/T` by `ln(I/I_S) ≈ 19`, while
a VGA multiplies it by `ln(G) ≈ 0.33`. *Keep the temperature-sensitive
quantity inside a ratio whose logarithm is small.* This is R16's lever
appearing a third time, now with temperature in place of `σ_n` — and it is the
same argument `domain_axis.md` §3.1 item 1 makes for moving P3/P4 into the time
domain.

---

## 5. Updated scope statement (replaces `mismatch_calculus.md` §7 bullets 1–2)

* ~~**No noise.**~~ **White device noise is in**, as R18/R19/R20, validated to
  1.00 on two benches. Still out: **1/f and RTN** (`KF = 0` throughout, and
  flicker is not bandwidth-scalable, so R18 is a *lower* bound in any real
  subthreshold cell), supply and substrate noise, and channel-to-channel
  coupling.
* ~~**No transients.**~~ Partly in: the calculus now takes a **bandwidth** and
  a **loop transfer function** as inputs (R19/R20) and returns a noise floor.
  Still out: slew, large-signal settling, stability margin. `t_AGC = 4.4 ns`
  remains an input, but R20 now says what that input *costs*.
* **No correlations** — unchanged, except that **R21 is the first correlated
  source in the calculus** (a gradient is perfectly correlated across
  channels) and it is handled by referring it to the reference device rather
  than by a covariance matrix.
* **One technology model** — unchanged.

## 6. Rule index

| rule | one line | status |
|---|---|---|
| R16 | the log lever is `⟨ln(I/I_S)⟩` over the workload, not `ln(I_max/I_S)` | validated on D1, 1.01/0.99 |
| R17 | one normalisation (signal-referred) + a declared crest factor; decision stages exempt | resolves the 1.500-bit inconsistency |
| R18 | dynamic lever `√(2qB/I)`, composes exactly like R1 through R3 | validated 0.998 / 1.053 |
| R19 | feedback sorts by path at all frequencies, by spectrum only in how much survives; `√(1−Σw²)` below `f_loop`, 1 above | six-digit AC confirmation |
| R20 | `σ_dyn ∝ 1/√t_settle`: an accuracy spec and a speed spec are one spec | K1 triggered on the paper's own bench |
| R21 | a gradient is a gain lever referred to the stage's reference device; granularity decides the class | 0.99 on four independent readings |
