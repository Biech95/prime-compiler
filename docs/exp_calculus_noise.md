# Experiment: does the mismatch calculus survive thermal noise and a
# temperature gradient?

*Companion to `docs/mismatch_calculus.md` (rules R1–R15),
`docs/mismatch_calculus_addendum.md` (rules R16–R21, written from this run's
verdicts) and `spice/calculus_noise_sim.py` (the arbiter).
Environment: ngspice-46, numpy 2.4, ≤ 4 cores, `OPENBLAS_NUM_THREADS=1`.*

**Status: §0–§4 were written 2026-09-16 BEFORE `spice/calculus_noise_sim.py`
was executed on a single noise or temperature draw. The frozen numbers that
decide K1 and K2 are printed in §3.4 and §3.5. §5 onwards was appended after
the run.**

---

## 0. Pre-registration

### 0.1 Why this experiment exists

`docs/mismatch_calculus.md` §7 states the scope limit in one line:

> **No noise.** Thermal, 1/f, kT/C and supply noise are absent. Every σ here is
> a *static* device-parameter spread. … the calculus would have to be extended,
> not merely re-parameterised.

Everything the calculus has been validated on (36 retrodicted rows, 8
prospective rows, `exp_mismatch_calculus.md`) is **static mismatch**: a number
that is drawn once per die and then stays put. It is therefore

* **trimmable** — one calibration per die removes it (R5, R9), and
* **learnable** — training absorbs the part that is a fixed function of the
  learned parameters (R9/R13: 0.7–0.95 σ → 0.06–0.16 σ).

Thermal noise is neither. It is redrawn every τ, so no trim and no training run
can see it, and it sets a *floor* that no amount of calibration effort buys
down. A temperature gradient is the third thing: static in time but
**systematic in space**, so it violates the independence assumption R3 rests on
("independent per-device draws, no spatial gradients") while being, unlike
noise, in principle trimmable.

If the calculus is to be a compiler back-end and not a mismatch anecdote, it
has to price all three.

### 0.2 Claim under test

**Claim D (dynamic):** the per-channel and common-mode error that white device
noise produces in a composite can be predicted from the same signal-flow walk
the calculus already performs, with one extra per-edge quantity — the junction's
*noise-referred* relative current error `σ_dyn,j = √(2q·B/I_j)` (shot) or
`√(4kT·B/R)/I_j` (thermal) — composed in quadrature exactly like a mismatch σ,
**except** that the feedback rule R2 must be replaced by a frequency-resolved
version.

**Claim T (temperature):** a temperature gradient across the channels of one
stage enters the calculus as a *lever-type* (multiplicative, trimmable) error in
the current-log domain and as an *offset-type* (additive, input-referred) error
in the subthreshold pair, i.e. it is classified by exactly the taxonomy of
`domain_axis.md` §1.2 item 2 and R10, not as a new class.

### 0.3 Pre-registered questions and predictions

| # | Question | Pre-registered prediction |
|---|---|---|
| **(i)** | Does **R2** ("feedback demotes per-channel to common mode") hold for *noise*? | Only for noise components **below the loop bandwidth**. Above `f_loop` the loop cannot act, so nothing is demoted; and detector noise reappears **per-channel** through VGA gain jitter (the scalar gain command multiplies N mismatched VGAs). |
| **(ii)** | Does a 2 K gradient act as a **gain lever** (`V_T ∝ T`, ±0.33 %/K) — hence trimmable — or as an **offset**? | **Gain-type in the log domain, offset-type in the subthreshold pair.** |
| **(iii)** | Can a third lever class "**dynamic**" be defined — error that is neither trimmable nor learnable? | Yes, with the composition rule: adds in quadrature across stages; **not** demoted by feedback above `f_loop`; sets a floor `∝ √(kT/C)` or `√(2qI·B)`. |

### 0.4 Pre-registered kill criteria

| ID | Criterion | Consequence |
|---|---|---|
| **K1** | Noise in the **AGC bench at the loop bandwidth** produces a per-channel error **> 0.69 %** (the measured VGA-mismatch residual at σ_VGA = 1 %, Table 7) | the paper's exposure ranking — *"the VGA array is the sole per-channel exposure"* — is **incomplete**: a second per-channel exposure of comparable size exists that no trim removes. |
| **K2** | The calculus **with the dynamic class** predicts the measured noise-induced per-channel error to within **2×** on **both** benches | the extension **passes**. Otherwise: name the assumption that failed. |

K1 is a statement about the *paper*, K2 about the *calculus*; they are
independent and both can fire.

### 0.5 Declared in advance

* **Corner.** ngspice default `TEMP = TNOM = 27 °C`, i.e. `T₀ = 300.15 K`; the
  brief's "300 K" is realised as this. The 0.05 % difference is irrelevant for a
  gradient study and keeps the two validated benches bit-identical to their
  published runs. `V_T = 0.025864890 V` (ngspice SPICE3-legacy kT/q,
  `spice/README.md`).
* **Physical constants used in the predictions:** `q = 1.602176634e-19 C`,
  `k = 1.380649e-23 J/K`.
* **Noise model.** White only, no flicker (`KF = 0`, ngspice default), so every
  spectrum below must come out flat; *whiteness is itself a check* and is
  reported. Per junction: shot noise `S_I = 2qI`. Per resistor:
  `S_I = 4kT/R`. Behavioural sources (`B`, `E`, `F`) are noiseless in ngspice
  and are declared as such — where the bench uses one to stand in for a real
  device, the corresponding noise is **added explicitly as a physical junction**
  (the VGA, §2.2) or is **absent and declared absent** (the ideal tail current
  source of the GELU pair, the ideal mirrors).
* **Bandwidths, fixed in advance.**
  * **GELU (open loop): `B = 100 MHz`**, as given in the brief. Integration band
    `1 kHz … 100 MHz`; with white noise the lower limit is irrelevant.
  * **AGC (closed loop): `B = ENBW_loop`**, the equivalent noise bandwidth of the
    *measured* closed-loop response, `ENBW = ∫|H(f)|² df / |H(0)|²`. This is a
    property of the circuit, so it is measured, not assumed. **Pre-registered
    point estimate: 318 MHz**, the detector pole `1/(2π·0.5 ns)` of
    `rmsnorm_agc_sim.build_agc_netlist`. Two reference bandwidths are reported
    alongside: `1/(4·t_AGC) = 56.8 MHz` (the ENBW implied by the published
    4.4 ns *large-signal* settling time) and 100 MHz (the GELU band).
* **Temperature gradient.** `T_i = 300 + 2·i/N` K across the N = 8 channels,
  applied as an ngspice per-device instance parameter `temp=` on **every device
  of channel i**. Shared devices (the translinear reference diode, the
  detector's output mirror) sit at the array mean `T₀ + 1 K`. Every gradient
  run is paired with an otherwise identical **uniform-`T₀` control**, and only
  the *difference* is reported, so the 0.15 K offset of §0.5 cancels exactly.
* **Metric, fixed in advance.** `decompose()` of `spice/mismatch_calculus.py`
  — the metric of Table 7 and of `exp_mismatch_calculus.md` §10 — a
  least-squares scalar α removed, residual RMS normalised by `RMS(reference)`,
  reported as p50/p95 in % of signal. This makes every number below directly
  comparable to the 8.518 % (GELU) and 0.690 % (AGC/LayerNorm) mismatch numbers
  already on record.
* **Input draws, fixed in advance.** GELU: `mismatch_calculus.gelu_inputs()`
  (seed 2026) — the identical vector the prospective mismatch test used.
  AGC: `np.random.default_rng(2026).standard_normal(8)*1.5` with the 0.05 floor
  of `agc_mismatch_sweep.run_cell` — the published Table-7 draw.
* **No mismatch is drawn anywhere in this experiment.** Every netlist runs with
  nominal devices. Noise and temperature are studied *alone*, so that the
  comparison against the mismatch numbers is a comparison of two separately
  measured contributions, not a decomposition of one mixed run.

---

## 1. What is new relative to `exp_mismatch_calculus.md`

| | mismatch campaign | this campaign |
|---|---|---|
| ngspice analysis | `.op` × 200 draws | `.noise` (device-exact spectra), `.ac` (loop transfer), `.op` (temperature) |
| randomness | Monte-Carlo over device parameters | **none in ngspice** — noise is a linear-response quantity and is computed exactly; the Monte Carlo is only the cheap numpy propagation of the measured σ's into the p50/p95 metric |
| the unknown | are the sensitivities sufficient? | is the *feedback* rule still right when the error has a spectrum? |

---

## 2. Method

### 2.1 Bench G — GELU = P11 · P6, open loop

Netlist **imported unmodified** from `spice/predict_gelu_layernorm_sim.py`
(`build_gelu_netlist`) with an all-zero mismatch draw, plus two changes needed
to make a noise analysis possible and both declared here:

1. the output current mirror `Fo_i` is terminated into `R_L = 1 GΩ` instead of
   a 0 V sense source, so that the output is a **node voltage** (ngspice
   `.noise` can only observe a voltage). `R_L`'s own current noise is
   `4kT/R_L = 1.66·10⁻²⁹ A²/Hz`, **5 decades below** the smallest junction shot
   noise in the circuit (`2q·0.07 µA = 2.3·10⁻²⁶`), and is reported as a floor;
2. `Iref` carries `AC 1` (ngspice requires an AC-flagged input source for a
   noise analysis; the output noise does not depend on it).

Noise sources present: the four loop diodes per channel (`Dx_i`, `Ds_i`,
`Do_i`) plus the shared `Dref`, and the two pair transistors `Q_ia`, `Q_ib`.
Noise sources *absent by construction and declared*: the tail current source
`It_i`, the mirrors `Fs_i`/`Fo_i`, the loop-closure `Eo_i`.

One `.noise v(nl_i) Iref dec 20 1e3 1e8` run per channel (8 runs).
Read out: `onoise_spectrum` (whiteness check) and `onoise_total`
(ngspice's own band integral). Converted to a relative current error

```
    σ_dyn,i(meas)  =  onoise_total / (R_L · I_out,i)
```

### 2.2 Bench A — AGC RMSNorm, N = 8, closed loop

Three separate ngspice measurements, because the bench is split between a
device-exact detector (`agc_mismatch_sweep.build_detector_netlist`) and a
behavioural loop (`rmsnorm_agc_sim.build_agc_netlist`):

**A-noise-det.** `.noise` on the junction RMS detector at the loop equilibrium
`G₀ = 1/RMS(x)`, nominal devices, output = the 1:N mirror current into
`R_L = 1 GΩ`. Gives the detector's relative output noise
`σ_det = onoise_total/(R_L·I_ms)`, `I_ms = 1 µA`. 2N + 1 = 17 junctions, **all
on the shared path**.

**A-noise-vga.** The bench's VGA is behavioural (`By_i … I={|V(nc_i)|·G}`), so
it carries no noise at all. Giving it physics is unavoidable if the per-channel
exposure is to be priced, so the VGA is realised in the repo's own idiom as a
**two-junction current-log gain cell** — `Dlog_i` fed by the channel current,
an `E`-source adding the gain-control voltage, `Dexp_i` producing the output
current — which is P11 in the current-log domain, the realization
`domain_axis.md` §2 already prices. This is an **extension of the bench** and is
flagged as such in §7. One `.noise` run per channel (8 runs).

**A-ac.** `.ac dec 20 1e5 1e11` on the closed AGC loop
(`build_agc_netlist` parameters: `τ_det = 0.5 ns`, `C_g = 1 pF`, `k = 2e-3`,
`Γ = 1`), with two unit AC injection nodes added:

* `V_ed` — a **relative** perturbation of the detector output
  (`Bmsq … V={(1+V(ed))·mean(y²)}`), the small-signal stand-in for detector
  noise;
* `V_e0` — a **relative** perturbation of channel 0's VGA gain
  (`By_0 … V={V(g)·V(x_0)·(1+V(e0))}`), the stand-in for per-channel VGA noise.

Read out, as functions of frequency:

```
  H_G(f)    = v(g)/v(ed)          detector noise -> common-mode gain jitter
  H_pc(f)   = v(y0)/v(e0)         own-channel response to own VGA noise
  H_x(f)    = v(y1)/v(e0)         cross-channel response  (the demotion term)
  H_yd(f)   = v(y0)/v(ed)         detector noise seen at a channel output
  ENBW      = integral |H|^2 df / |H(0)|^2       (trapezoid on the log grid)
```

`H_pc(0) = 1 − w₀` and `H_pc(∞) = 1` is what R2-for-noise predicts if the loop
demotes per-channel noise the way it demotes per-channel mismatch;
`H_G(0) = −1/2`, `H_G(∞) → 0` is what "shared path → common mode, below the
loop bandwidth only" predicts.

### 2.3 Temperature gradient

**T-log (bench G).** The GELU netlist with `temp=` set per channel; the
sigmoid pair is *frozen* (its drive replaced by the nominal current) in one
variant so that the translinear chain is isolated, and active in the other.
Run at **two input scales**, `s = 1.0` and `s = 0.25`. This is the decisive
test for (ii): if the absolute output error scales `∝ s`, the gradient is a
**gain** error; if it stays constant, it is an **offset**.

**T-pair (bench G).** The differential pair alone, at `T₀` and at `T₀ + 2 K`,
swept over `u ∈ [−4, 4]`. The measured `I_a(u)/I_tail` at the raised temperature
is fitted to `σ((u − δ)/s)`: `δ ≠ 0, s = 1` ⇒ offset-type; `δ = 0, s ≠ 1` ⇒
gain-type.

**T-agc (bench A).** Gradient applied (a) to the detector junctions only,
(b) to the VGA junctions only, (c) to both; loop equilibrium re-solved by the
same 16-step bisection as `agc_mismatch_sweep.settle_gain`; the output
decomposed into common-mode and per-channel by the same metric.

---

## 3. Frozen predictions

### 3.1 The dynamic lever, stated before the run

For a junction traversed once, the mismatch lever R1 has a noise twin:

```
    R1  (static)   d ln I = ±(ν·L − ι),      σ = hypot(σ_Is, σ_n·L)
    R16 (dynamic)  d ln I = ±ξ(t),           σ_dyn = sqrt( 2q·B / I )
```

with the **same ±1 sensitivity and the same quadrature composition** (R3). Two
structural differences are predicted in advance:

* the dynamic lever has **no L**. The log domain's factor-19 amplification of
  `σ_n` has no counterpart in shot noise, because shot noise is already
  expressed as a current fluctuation. The log domain's mismatch disadvantage is
  therefore *not* a noise disadvantage;
* the dynamic lever is **largest where the current is smallest**, i.e. on the
  channels a softmax or a GELU has already squeezed to the tail — the same
  place R12 says the sigmoid offset is worst.

### 3.2 Bench G, frozen

Junction currents at the fixed input draw (µA): `I_x = |x|`,
`I_s = 4·σ(1.702x)`, `I_o = |x|·σ(1.702x)`, `I_ref = 4`.

| channel | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `I_x` / µA | 0.793 | 0.241 | 1.896 | 1.396 | 0.638 | 0.292 | 0.312 | 0.304 |
| `I_s` / µA | 0.824 | 2.404 | 0.153 | 3.660 | 2.991 | 1.513 | 1.481 | 2.506 |
| `I_o` / µA | 0.163 | 0.145 | 0.072 | 1.277 | 0.477 | 0.111 | 0.116 | 0.190 |
| **σ_dyn,i, 4 loop junctions** / % | 1.684 | 1.940 | 2.604 | 0.805 | 1.167 | 2.071 | 2.024 | 1.716 |
| **σ_dyn,i, + naive pair term** / % | 1.796 | 1.974 | 2.980 | 0.858 | 1.212 | 2.122 | 2.077 | 1.753 |

Two variants of the pair term are pre-registered because the differential pair
with an **ideal** tail source constrains `i_a = −i_b`, which may or may not
leave the naive `2qB/I_a`:

* **naive**: the pair counts as one more junction at `I_a = I_s`;
* **correlated**: `S_{I_a} = 2q·I_a·I_b/I_tail·2`, i.e. suppressed by
  `2σ(1−σ)` relative to naive.

**Frozen primary prediction (naive pair term, B = 100 MHz, metric
`decompose`):**

| | p50 | p95 |
|---|---|---|
| **per-channel** | **0.614 %** | 1.084 % |
| **common-mode** | **0.486 %** | — |

(4-junction-only variant: 0.588 % / 1.032 % / 0.452 %.)

Against the **static** mismatch number on record for the same bench and the
same draw: **8.518 % per channel**. Prediction: noise is a **14× smaller**
per-channel contributor than mismatch in the open-loop GELU at 100 MHz.

### 3.3 Bench A, frozen

Channel currents at equilibrium are `|y_i| = |x_i|/RMS(x)` in µA:
0.853 / 0.259 / 2.040 / 1.501 / 0.687 / 0.314 / 0.336 / 0.327.
Two junctions per VGA channel ⇒ `σ_VGA,dyn,i = √(2·2q·B/I_i)`.
`Σ w_i² = 0.3623`, so `√(1 − Σw²) = 0.7986`.

| B | per-channel σ range | **per-channel p50** | p95 | common-mode p50 |
|---|---|---|---|---|
| 56.8 MHz (`1/(4·t_AGC)`) | 0.422–1.186 % | **0.423 %** | 0.695 % | 0.182 % |
| 100 MHz | 0.561–1.574 % | **0.561 %** | 0.924 % | 0.243 % |
| **318 MHz (pre-registered `ENBW_loop`)** | **1.000–2.808 %** | **1.005 %** | **1.651 %** | **0.429 %** |
| 888 MHz (analytic single-pole loop) | 1.670–4.689 % | 1.682 % | 2.745 % | 0.724 % |

Detector side, 17 shared junctions at `I ≈ 1 µA`, `σ_det = √(2qB·Σ_j 1/I_j)`;
the loop converts it to a **common-mode** gain jitter `σ_G/G = ½·σ_det·|H_G|`.
kT/C on the 1 pF gain-control capacitor: `√(kT/C) = 64.4 µV` on a `V_g ≈ 0.72 V`
node ⇒ **0.009 % common-mode**, negligible and reported for completeness.

### 3.4 K1, decided in advance

At the pre-registered `ENBW_loop = 318 MHz` the predicted per-channel noise is
**1.005 %**, which is **above the 0.69 % VGA-mismatch residual**.

> **K1 is predicted to TRIGGER.** If the measured loop ENBW comes out at or
> above ~150 MHz, thermal noise in the VGA path is a per-channel exposure of the
> same size as the VGA mismatch the paper names as the only one — and unlike it,
> untrimmable. If the measured ENBW comes out below ~150 MHz, K1 does not
> trigger and the paper's ranking survives at this corner.

This is written down *before* the AC run, so the threshold is not chosen after
seeing the bandwidth.

### 3.5 K2, decided in advance

K2 compares, on both benches, the **calculus prediction** (§3.2, §3.3, computed
from `√(2qB/I)` and the R3 quadrature walk, no ngspice) against the
**ngspice `.noise` measurement** of the same quantity. The pass band is 0.5×–2×
on the per-channel p50.

Named in advance, the three assumptions that could fail:

1. **the ±1 sensitivity** — that a junction's current noise enters `d ln I`
   with the same coefficient as its mismatch. Fails if the translinear loop
   correlates the noise of its junctions (they share nodes);
2. **the pair term** — either the naive or the correlated form is wrong, which
   would show as a systematic miss concentrated on the low-`I_s` channels
   (channel 2);
3. **bandwidth** — if the measured spectrum is not white, the single number `B`
   is not a sufficient statistic and the extension needs a shaped integral.

### 3.6 Question (ii), frozen

* **Log domain.** `I = I_S(T)·exp(V/(nV_T(T)))`. At fixed drive voltage
  ngspice's default `XTI = 3`, `EG = 1.11 eV` give
  `d ln I/dT|_V = (XTI + EG/kT)/T − ln(I/I_S)/T ≈ (3 + 42.9 − 19.1)/300 =
  0.089 /K`, i.e. **+19 % per 2 K at fixed drive** — enormous. But a translinear
  *loop* takes differences of junction voltages, so what survives is predicted
  to be the *uncancelled* part, entering as a **multiplicative** (gain) error
  on the channel current, hence trimmable. Pre-registered magnitude, from the
  `V_T` term alone: `L·ΔT/T = 19.1·2/300 = 12.7 %` across the array, i.e.
  ~6.4 % per-channel spread — **larger than the whole mismatch budget**.
* **Subthreshold pair.** Within a channel both pair devices sit at the same
  `T_i`, so there is no intra-pair `I_S` imbalance; what changes is `V_T`, i.e.
  the **slope** of the sigmoid. The brief's prediction is "offset-type". The
  test of §2.3 will separate `σ((u−δ)/s)` into `δ` and `s` and is free to
  contradict it.

---

## 4. Wall-clock and compute budget

≤ 4 cores, `OPENBLAS_NUM_THREADS=1`, multiprocessing only from this real `.py`
file (`reference_python_mp_gotcha`). Expected: 8 + 8 + 1 `.noise` runs, 2 `.ac`
runs, ~40 `.op` runs for the temperature arms and 3 × 16 bisection steps for
T-agc — well under a minute of ngspice.

---

<!-- ==================== everything below written after the run ============ -->

## 5. Results

Runner: `spice/calculus_noise_sim.py`, ngspice-46, one core, wall clock **68 s**
(41 `.noise` runs, 2 `.ac` runs, 68 `.op` runs incl. 32 bisection steps).
Raw numbers: `spice/out/calculus_noise.json`.

Bench control, nominal devices: GELU reproduces `x·σ(1.702x)` to
**2.593·10⁻⁶**, i.e. bit-identical to the mismatch campaign's control.
**Every noise spectrum in this campaign is white to 4 decimal places**
(max/min of `onoise_spectrum` over 5–6 decades = 1.0000), so a single
bandwidth `B` is a sufficient statistic and §3.5 assumption 3 is discharged.

### 5.1 Bench G — GELU, per-channel noise at B = 100 MHz

| ch | `I_out` / µA | **ngspice** σ / % | translinear only | calculus, 4 junctions | ratio (TL) | calculus, + pair `(1−σ)` | ratio (full) |
|---|---|---|---|---|---|---|---|
| 0 | 0.163 | 3.046 → **1.773** † | 1.6841 | 1.6839 | **1.0001** | 1.7733 | **1.0001** † |
| 1 | 0.145 | 1.9534 | 1.9397 | 1.9395 | 1.0001 | 1.9532 | 1.0001 |
| 2 | 0.072 | 2.9669 | 2.6044 | 2.6038 | 1.0002 | 2.9664 | 1.0002 |
| 3 | 1.277 | 0.8097 | 0.8051 | 0.8051 | 1.0000 | 0.8097 | 1.0120 ‡ |
| 4 | 0.477 | 1.1780 | 1.1665 | 1.1665 | 1.0000 | 1.1780 | 1.0000 |
| 5 | 0.111 | 2.1031 | 2.0715 | 2.0712 | 1.0002 | 2.1028 | 1.0001 |
| 6 | 0.116 | 2.0579 | 2.0246 | 2.0243 | 1.0001 | 2.0576 | 1.0001 |
| 7 | 0.190 | 1.7304 | 1.7165 | 1.7164 | 1.0001 | 1.7303 | 1.0001 |

† **A bench artefact was found and removed.** In the first run channel 0 came
out at 3.046 % against a predicted 1.773 % (1.72×), while the other seven
channels agreed to 1.0002. A permutation control (`x₀` moved to netlist slot 3,
`x₃` to slot 0) shows the excess **follows the netlist slot, not the signal**:
slot 3 carrying `x₀` gives 1.0001, slot 0 carrying `x₃` gives 1.0120. The
excess is therefore a property of *netlist index 0* in the assembled GELU
(it is absent when the pair is frozen — the translinear column is 1.0001 at
slot 0 — so it lives in the pair→log-diode wiring). The mechanism was not
identified inside this campaign's budget and is carried as an open item (§7).
The table quotes the permuted measurement for channel 0. ‡ residual slot-0
excess in the permuted run, 1.2 %.

**The dynamic lever of §3.1 is confirmed to four significant figures**, on
eight channels spanning a 26× range of output current, with **no free
parameter**: `σ_dyn = √(2q·B·Σ_j 1/I_j)` over the junctions the signal-flow
walk already visits.

The pre-registered **pair term was wrong and the correct one was measured**.
Reading the pair out through a linear load isolates its own current noise:

| u | σ(u) | `I_a` / µA | ngspice (linear load) | `√(2qB(1−σ)/I_a)` | ngspice (log-diode load) | `√(2qB[1+(1−σ)(2−σ)]/I_a)` |
|---|---|---|---|---|---|---|
| −3.228 | 0.0381 | 0.153 | **1.4219 %** | 1.4216 % | 2.4630 % | 2.4630 % |
| −2.000 | 0.1192 | 0.477 | **0.7694 %** | 0.7694 % | 1.3362 % | 1.3362 % |
| −1.350 | 0.2059 | 0.824 | **0.5559 %** | 0.5559 % | 0.9714 % | 0.9714 % |
| −0.497 | 0.3782 | 1.513 | **0.3629 %** | 0.3629 % | 0.6522 % | 0.6522 % |
| 0.000 | 0.5000 | 2.000 | **0.2830 %** | 0.2830 % | 0.5295 % | 0.5295 % |
| +0.409 | 0.6008 | 2.403 | **0.2307 %** | 0.2307 % | 0.4558 % | 0.4558 % |
| +1.000 | 0.7311 | 2.924 | **0.1717 %** | 0.1717 % | 0.3834 % | 0.3834 % |
| +2.000 | 0.8808 | 3.523 | **0.1041 %** | 0.1041 % | 0.3211 % | 0.3211 % |

A differential pair with an ideal tail is **not** one more shot-noise junction.
`i_a = (1−σ)·i_na − σ·i_nb` (the tail forces `i_a = −i_b`), so

```
    S_{I_a} = 2q·I_T·σ(1−σ) = 2q·I_a·(1−σ)      ⇒  σ_dyn,pair = √(2qB(1−σ)/I_a)
```

which ngspice reproduces to four digits at every one of eight operating points.
The **naive** `√(2qB/I_a)` of §3.2 over-counts by `1/√(1−σ)`, up to 2.9× at
σ = 0.88. Note the direction: the pair is *quietest* where its output is
largest, the opposite of R12's mismatch behaviour.

*(The isolated pair driving a log diode obeys `1 + (1−σ)(2−σ)` rather than the
`1 + (1−σ)` that KCL implies — the same interface anomaly as the slot-0 excess,
reproduced here in an 11-line netlist. Inside the full GELU seven of eight
channels obey `1 + (1−σ)`. Open item, §7.)*

**Metric (`decompose()`, 20 000 propagation draws):**

| | ngspice | calculus | ratio |
|---|---|---|---|
| **per-channel p50** | **0.6026 %** | **0.6013 %** | **0.998** |
| per-channel p95 | 1.0469 % | 1.0446 % | 0.998 |
| common-mode p50 | 0.4641 % | 0.4587 % | 0.988 |

Against the **static mismatch** number for the identical bench, draw and metric
(`exp_mismatch_calculus.md` §10.1): **8.518 %**. At 100 MHz thermal noise is
**7.1 % of the mismatch contribution** in the open-loop GELU — it does not
change the verdict on that circuit, and it would take a **200× bandwidth**
(20 GHz) to match the mismatch.

### 5.2 Bench A — what the loop does to a spectrum

Closed-loop AC, `τ_det = 0.5 ns`, `C_g = 1 pF`, `k = 2·10⁻³`; the operating
point lands on `V(g) = 0.717142` against an ideal `G₀ = 0.717142`.

| transfer | `|H|` @ 100 kHz | `|H|` @ 100 GHz | ENBW |
|---|---|---|---|
| detector rel. error → `δG/G` | **0.500000** | 1.41·10⁻⁵ | 1395 MHz |
| detector rel. error → `y₀` | **0.500000** | 1.41·10⁻⁵ | 1395 MHz |
| detector rel. error → `y₁` | **0.500000** | 1.41·10⁻⁵ | 1395 MHz |
| ch-0 VGA rel. error → `y₀` | **0.909012** | **1.000000** | 121 034 MHz |
| ch-0 VGA rel. error → `y₁` | **0.090988** | 2.57·10⁻⁶ | 1395 MHz |

`−3 dB` corner of the detector path **794 MHz**, `ENBW_loop = 1395 MHz`
(the pre-registered estimate of 318 MHz was the *detector* pole alone and is
**4.4× low**; the dominant pole is the loop's own,
`τ = C_g/(4k·√MS(x)) = 0.179 ns`).

Three exact numbers, and each one is a rule:

1. **`H(det→y₀) ≡ H(det→y₁) ≡ H(det→δG/G)` at every frequency.** Detector
   noise is **purely common mode**, with `|H(0)| = ½` — the `½` of
   `G = Γ/√(ms)` — and is **rejected, not demoted, above the loop bandwidth**
   (1.4·10⁻⁵ at 100 GHz). It never becomes per-channel.
2. **`H(ch0→y₀)(0) = 0.909012` against `1 − w₀ = 0.909012`**, and
   **`H(ch0→y₁)(0) = 0.090988` against `w₀ = 0.090988`** — R3's energy weights,
   read off an AC analysis to six digits.
3. **`H(ch0→y₀)(∞) = 1.000000`.** Above the loop bandwidth the loop does
   nothing at all to per-channel noise.

### 5.3 Bench A — noise densities and the per-channel exposure

Detector (17 junctions, all on the shared path):

| | value | ratio |
|---|---|---|
| ngspice `.noise`, relative density | **7.8717·10⁻⁷ /√Hz** | — |
| calculus, R3 energy weights `Σ w_i²(4/I_y,i + 1/I_sq,i) + 1/I_ref` | 7.8716·10⁻⁷ /√Hz | **1.000** |
| calculus, **energy weights forgotten** | 6.0960·10⁻⁶ /√Hz | 7.744 |

The second row is the whole content of R3 in one number: a walk that visits the
17 junctions but forgets the weights at the summed node is wrong by
**7.7×** — larger than every miss in the entire mismatch campaign.

Per-channel (the two-junction current-log VGA):

| B | ngspice σ/ch | calculus σ/ch | **pc p50 ngspice** | pc p50 calculus | ratio | vs the 0.69 % line |
|---|---|---|---|---|---|---|
| **ENBW_loop = 1395 MHz** | 1.99–5.58 % | 2.09–5.88 % | **1.998 %** | 2.103 % | **1.053** | **2.9× above** |
| 100 MHz | 0.56–1.57 % | 0.56–1.57 % | 0.563 % | 0.563 % | 1.000 | below |
| 56.8 MHz (`1/(4 t_AGC)`) | 0.42–1.19 % | 0.42–1.19 % | 0.422 % | 0.424 % | 1.005 | below |

Detector-induced **common-mode** gain jitter at `ENBW_loop`: **0.735 %**
(`½·σ_det·√B`), against the 3.0 % common-mode that detector *mismatch*
produces at the MOS-typical corner. kT/C on the 1 pF gain capacitor:
64.4 µV on a 0.717 V node = **0.009 %**, negligible.

### 5.4 Temperature gradient, 2 K across 8 channels

**Log domain (GELU).** Relative error per channel, `T_i = T₀ + 2i/8`, the
shared reference diode at `T₀ + 1 K`:

```
 translinear chain: +7.429  +5.517  +3.642  +1.803   0.000  −1.769  −3.503  −5.205  %
 full GELU        : +7.429  +5.503  +4.179  +1.752  −0.091  −1.643  −3.343  −5.311  %
```

* the error is **exactly zero on the channel at the reference device's
  temperature** and antisymmetric around it — the gradient enters only as
  `T_i − T_ref`;
* scaling the operand down by 4× leaves the **relative** error unchanged
  (ratio 1.0000) and divides the **absolute** error by 4 (ratio 0.2500) on
  every channel ⇒ **GAIN-type, i.e. multiplicative and trimmable**;
* magnitude: spread 12.63 % over 2 K against the pre-registered
  `L·ΔT/T = 19.11·2/300.15 = 12.73 %` — **ratio 0.99**. The log domain's
  factor-19 lever applies to temperature exactly as it applies to `σ_n`;
* in the Table-7 metric a 2 K gradient costs **1.41 % per channel** and 1.48 %
  common mode — **2× the whole VGA-mismatch residual**, from a gradient a
  designer would not think to budget.

**Subthreshold pair.** Fitting `σ_T₀₊₂ₖ(u)` against `σ_T₀(u)` in logit space
over `u ∈ [−4, 4]`:

```
 logit(T0+2K) = 0.993381 · logit(T0) − 4.1·10⁻⁷
 slope change −0.6619 %   against  −ΔT/T = −0.6663 %   (ratio 0.993)
 input-referred offset    δ = 4.2·10⁻⁷ units of u      (i.e. zero)
```

**The pre-registered prediction (ii) is half right and half falsified.** Gain
type in the log domain: **confirmed**. Offset type in the subthreshold pair:
**refuted** — with both pair devices at the same `T_i` there is no intra-pair
`I_S` imbalance, so a gradient across *channels* changes only `V_T`, i.e. the
**slope** of the sigmoid, to the 7th digit of `−ΔT/T`. The pair would only
produce an offset from a gradient *inside* the pair, which this model
(device-level `temp=`, channel granularity) does not contain.

**AGC bench.** Detector-side gradient: `G` moves 0.717093 → 0.705993, a
**−1.548 % common-mode gain shift and nothing per-channel** — R3's
"shared path → common mode" holds for a gradient exactly as for mismatch.
VGA-side gradient, at the true loop gain `G₀ = 0.71714`
(`V_g = V_T ln G₀ = −8.60 mV`):

```
 measured : 0.0000 0.0277 0.0553 0.0829 0.1105 0.1380 0.1655 0.1929 %
 calculus : 0.0000 0.0277 0.0554 0.0831 0.1108 0.1385 0.1662 0.1939 %   (= −ln G₀ · ΔT_i/T)
```

ratio 0.995 on every channel; common mode 0.066 %, per channel **0.033 %**.
A current-log VGA is temperature-**insensitive in proportion to its own gain**:
at unity gain (`V_g = 0`) the two junctions cancel exactly and a gradient costs
literally nothing. That is a design rule, not a coincidence — and it is the
reason the same 2 K that costs the open-loop GELU 1.41 % costs the AGC 0.03 %.

---

## 6. Verdicts

### 6.1 Question (i) — does R2 hold for noise?

**Partly, and not in the way predicted.** The measurement splits R2 into two
halves that behave differently:

| R2 for static mismatch | R2 for noise, measured |
|---|---|
| `s_pc → 0` for a stage at/after the loop input | **False.** A per-channel noise source is attenuated by `1 − w_i` at DC (0.909 at N = 8) and by **exactly nothing** above `f_loop`. It is never demoted to common mode. |
| shared-path error → common mode | **True at every frequency.** `H(det→y_i)` is identical for all `i` to six digits. |
| suppression `1/(1+L_loop)` | **Not applicable.** `|H(det→δG/G)| = ½` at DC: the detector's error *is* the loop's reference and gets no loop-gain suppression, for noise exactly as for mismatch. |

The brief's prediction that *"above `f_loop` detector noise appears per-channel
through the VGA gain jitter"* is **refuted as a first-order effect**: above
`f_loop` detector noise does not appear at the output at all (1.4·10⁻⁵), and
the scalar gain command reaches every channel identically. The mechanism the
prediction names does exist but is **second order** — `δG·ε_i`, i.e.
0.7 % × 1 % ≈ 5·10⁻⁵ of signal, three decades below the VGA's own noise.

The correct statement, and the one that goes into the calculus as **R19**:
*feedback sorts errors by path (shared → common mode, per-channel → per
channel) at every frequency, and sorts them by spectrum only in how much of
them survives — a shared error is rejected above `f_loop`, a per-channel error
is not.* The per-channel residual factor is `√(1 − Σ w_i²)` below `f_loop` and
**1** above it.

### 6.2 Question (ii) — gain lever or offset?

**Gain lever, in both domains.** Log domain: relative error invariant under a
4× operand rescale (1.0000), absolute error ∝ operand (0.2500), magnitude
`L·ΔT/T` to 1 %. Subthreshold pair: a pure slope change of `−ΔT/T` to 0.7 %,
with an input-referred offset of 4·10⁻⁷ — **zero**. The brief's "offset-type in
the subthreshold pair" is falsified at channel granularity.

Therefore a temperature gradient is, at this fidelity, **trimmable** (a
per-channel gain constant removes it) — but it is *not* static in service: it
moves with the workload's own power dissipation, so it is trimmable only
against a fixed thermal map. It is classified in the addendum as
**static-in-space, dynamic-in-time** (R21), i.e. trimmable exactly as far as
the thermal map is repeatable — and the granularity caveat of R21 says that a
gradient *inside* a pair would be offset-type after all.

### 6.3 Question (iii) — the dynamic class

Defined, and stated as **R18** (the lever), **R19** (feedback, frequency-resolved,
replacing R2 for dynamic errors) and **R20** (the floor, and the
accuracy-vs-settling-time identity) in `docs/mismatch_calculus_addendum.md`;
the gradient becomes **R21**.
Its composition rule survived the only two tests available: quadrature over
junctions (ratio 1.0001 per channel), R3 energy weights at a summed node
(ratio 1.000, and 7.7× wrong without them), and the frequency-resolved
feedback rule (six-digit agreement with `1 − w_i` and with 1).

### 6.4 K1 — **TRIGGERED**

| bandwidth | per-channel noise | the paper's per-channel mismatch (σ_VGA = 1 %) | verdict |
|---|---|---|---|
| **`ENBW_loop` = 1395 MHz (measured)** | **1.998 %** | 0.690 % | **K1 TRIGGERED, 2.9×** |
| 100 MHz | 0.563 % | 0.690 % | would not trigger |
| 56.8 MHz | 0.422 % | 0.690 % | would not trigger |

> **The paper's exposure ranking — "the VGA array is the sole per-channel
> exposure" — is incomplete.** At the loop bandwidth this bench actually has
> (794 MHz −3 dB, ENBW 1395 MHz, the bandwidth that buys the 4.4 ns settling
> the paper advertises), the shot noise of the VGA's own junctions is a
> per-channel error of 2.0 %, **2.9× the VGA mismatch residual and not
> removable by trimming or by training**. The ranking holds if, and only if,
> the per-channel signal path is band-limited to ≲ 130 MHz
> (`0.690 % ⇒ B = 150 MHz`), which the AGC loop as built is not.

This is a statement about the *pairing* of two specifications the paper states
separately — 4.4 ns settling and 0.69 % per-channel error. Speed and
per-channel accuracy trade against each other through
`σ_pc ∝ √B ∝ 1/√t_settle`, and the calculus had no term for that until now.
The pre-registered threshold ("triggers if the measured ENBW ≥ ~150 MHz") was
written before the AC run; the measured ENBW is 9× that.

### 6.5 K2 — **PASSES on both benches**

| bench | quantity | ngspice | calculus | ratio | bar |
|---|---|---|---|---|---|
| **GELU** | per-channel p50 | 0.6026 % | 0.6013 % | **0.998** | 0.5–2 |
| **AGC** | per-channel p50 @ ENBW_loop | 1.998 % | 2.103 % | **1.053** | 0.5–2 |

Per-channel σ's agree to 1.0001–1.012 (GELU) and 1.00–1.06 (AGC). The
extension passes. Two of the three assumptions named in §3.5 in advance did
fail, and both were caught by the measurement rather than by the prediction:

1. **the pair term** (assumption 2) — the pre-registered naive `2qB/I_a`
   over-counts by `1/√(1−σ)`; the correct `2qB(1−σ)/I_a` is derived in §5.1 and
   matches to four digits. On the composite metric the naive form was still
   inside 2× (0.929), so K2 would have passed either way, but the *rule* was
   wrong and is now right;
2. **the analysis code's own R3 walk** for the detector (not one of the three;
   an outright bug) omitted the energy weights and was **7.7×** off. Recorded
   as the campaign's own error, and as the sharpest available demonstration
   that R3's weights are load-bearing;
3. **the ±1 sensitivity** (assumption 1) held to four digits, and
   **whiteness** (assumption 3) held to four decimals.

### 6.6 Summary table

| claim | status |
|---|---|
| Claim D — noise composes on the same signal-flow walk with `σ_dyn = √(2qB/I)` | **confirmed**, 1.00 on both benches |
| Claim T — a gradient is classified by the existing taxonomy | **confirmed for the log domain, and the pair turns out to be gain-type too** |
| R2 unchanged for noise | **refuted** — replaced by R19 |
| Noise is negligible next to mismatch | **true open-loop at 100 MHz (7 %), false in the AGC loop at its own bandwidth (290 %)** |
| VGA is the sole per-channel exposure | **incomplete** (K1) |

---

## 7. Limitations

* **The slot-0 / log-diode-load anomaly is unresolved.** Two independent
  manifestations of one interface effect: netlist index 0 of the assembled
  GELU (1.72× excess, removed by permutation) and the isolated
  pair→log-diode probe (`1+(1−σ)(2−σ)` where KCL gives `1+(1−σ)`). Seven of
  eight channels of the full GELU obey `1+(1−σ)`, and an ideal-current-source
  driven diode shows **zero** branch-current noise (the KCL answer), so the
  effect is confined to the CCCS→diode interface. It may be an ngspice
  artefact or a real path through the `F`-source; it was not chased further.
  Its effect on every headline number is **≤ 1.2 %** after the permutation
  control, and on the pre-permutation metric ≤ 10 %.
* **The VGA of the AGC bench is an addition, not a measurement of the paper's
  circuit.** The published bench's VGA is a behavioural source with no noise at
  all; the two-junction current-log gain cell used here is the realization
  `domain_axis.md` §2 prices for P11 in the current-log domain, but a Gilbert
  cell in strong inversion would have different (larger, thermal-dominated)
  noise. K1 should be read as *"a physically realized VGA of the cheapest kind
  already exceeds the mismatch residual"*, not as a number for one specific
  multiplier.
* **The per-channel signal bandwidth is set equal to the loop bandwidth**, as
  the brief specifies. Physically these are two different bandwidths, and a
  designer who band-limits the channel path below the loop path escapes K1.
  The 100 MHz and 56.8 MHz rows are given precisely so the trade can be read
  off.
* **White noise only.** `KF = 0`: no flicker, no RTN, no popcorn. In a real
  subthreshold cell 1/f would dominate below ~1 MHz and would *not* be
  bandwidth-scalable; the shot-noise floor computed here is therefore a
  **lower** bound on the dynamic class.
* **No supply or substrate noise, no coupling between channels.** Every noise
  source is device-local and independent, which is exactly the assumption R3
  makes and `exp_sign_concordance.md` §5.4 flags as the one that fails first.
* **Temperature enters only through `V_T`, `I_S(T)` and the ngspice
  level-1/Gummel-Poon temperature laws** (`XTI = 3`, `EG = 1.11 eV`,
  `TNOM = 27 °C`). No self-heating, no thermal time constant, no feedback from
  dissipation to the map. The gradient is imposed, not computed, and its
  *shape* (linear in channel index) is a choice.
* **Intra-device gradients are outside the model.** Both halves of a pair sit
  at the same `T_i`, which is what makes the pair come out gain-type; a
  gradient across a pair's two devices would produce
  `ΔV_os = V_T·ln(I_S1/I_S2) ≈ (EG/q)·ΔT_pair/T`, i.e. ~3.7 mV/K, and would
  make the pair offset-type after all. The verdict of §6.2 is therefore
  **granularity-dependent and says so**.
* **One bandwidth definition.** `ENBW = ∫|H|²df/|H(0)|²` on the AC grid;
  ngspice's own `onoise_total` integral on a `dec 20` grid differs from
  `S·√B` by ≤ 5 % when the band edge is not a round decade (visible as the
  1.053 in the 1395 MHz row against 1.000 at 100 MHz).
* **N = 8, one input draw per bench, one corner.** As with the mismatch
  campaign, the `w_i` are a property of the draw (R3), so the `1 − Σw²`
  factors quoted are not universal.

