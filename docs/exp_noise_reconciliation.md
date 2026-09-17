# Reconciling the two per-channel noise numbers of §5.3

Two independent campaigns priced thermal/shot noise as a **per-channel**
exposure of the AGC RMSNorm's VGA array, and disagreed by 2.9×:

| | packet | per-channel noise at `I₀ = 1 µA` | vs the 0.69 % VGA-mismatch residual |
|---|---|---|---|
| **P1** | `docs/exp_phase0_gaps.md` §5, kill **K-C** | **0.682–0.687 %** | 0.99–1.00× — "statistical tie" |
| **P2** | `docs/exp_calculus_noise.md` §5.3, kill **K1** | **1.998 %** | 2.9× — "noise binds" |

Both cannot go into §5.3. This note finds the cause, reproduces both numbers
on **one** bench, and states the single consistent formula.

**Bottom line.** The 2.9× is not a physics disagreement. It is
**2.38× bandwidth convention × 1.43× junction count**, with three small
book-keeping factors (metric statistic, device realization, input draw)
accounting for the remaining 0.86×. The bandwidth half has a mechanism:
the published loop is **underdamped (ζ = 0.299)**, so its settling-envelope
pole (`f_env = 153 MHz`, which P1 used) sits `π/(4ζ²) = 8.8×` below its true
equivalent noise bandwidth (`ENBW = 1395 MHz`, which P2 used).
**P2's convention is the physical one**; P1's band-limit is a settling
number, not a filter that exists anywhere in the circuit.

New bench: `spice/noise_reconciliation.py` (21 `.tran` + 13 `.ac` ngspice
runs, 53 s wall clock on 3 cores,
raw → `spice/out/noise_reconciliation.json`).
Nothing in `prime_compiler_v2.tex`, `paper/`, or any existing `docs/*.md`
was changed.

---

## 1. Method of the cross-check

Everything is re-measured on **one** bench: P1's transient AGC testbench
(`agc_body`-equivalent, `k = 2e-3`, `C_g = 1 pF`, `τ_det = 0.5 ns`, N = 8,
γ = 1, `I₀ = 1 µA`), driven by **P2's input draw** (seed 2026), so the draw
is held fixed while the two conventions are varied one at a time:

* **bandwidth** — P1's single-pole post-filter at `f_env`, vs P2's
  brick-wall integral to `ENBW_loop` (which is what ngspice `.noise`
  computes), vs a brick-wall at P1's *effective* band;
* **junction count** — 1 shot-noise junction per VGA (P1) vs 2 (P2);
* **metric statistic** — RMS over time (P1) vs p50 over 20 000 draws (P2),
  on the identical decomposition `α = ⟨y,y_ref⟩/⟨y_ref,y_ref⟩`,
  `res = y/α − y_ref`.

The same loop is additionally measured three ways in the **frequency**
domain, with P2's unit-relative AC probes (`V_ed`, `V_e0`) added to P1's
netlist.

**Validation that the reproduction is faithful:** the `2 junctions ×
brick @ ENBW_loop × p50` cell of this bench returns **2.10304 %**, against
P2's own published calculus value **2.10304 %** — identical to six digits.
The `1 junction × 1-pole @ f_env × RMS` cell returns **0.6845 %**, against
P1's published **0.682–0.687 %** — dead centre of the published range.

---

## 2. Diagnosis table

Checked in the order requested. "Contribution" is the multiplicative factor
by which the item pushes P2's number above P1's; the five factors multiply
to the published **2.921×**.

| # | item | bench P1 | bench P2 | contribution to the 2.9× |
|---|---|---|---|---|
| 1 | **bias current `I₀` per VGA** | 1 µA (swept 0.1/1/10) | 1 µA (`IUNIT = 1e-6`) | **1.000×** — not the cause |
| 1b | current *per junction* | `I_out = I₀·│y_i│` | `I_in = I_out = I₀·│y_i│` (VGA gain 1) | 1.000× |
| 2 | **noise bandwidth** | `f_env = 1/(2πτ_eq) = 152.8 MHz` from an exponential fit of the noiseless settling envelope; applied as a **1-pole** post-filter, so effective `ENBW = (π/2)·f_env = 240.1 MHz` | `ENBW_loop = 1395.1 MHz`, the `∫│H│²df/│H(0)│²` of the **measured closed-loop AC** response (−3 dB at 794.3 MHz) | **2.384×** (measured; `√(1395.1/240.1) = 2.411` nominal) — **the dominant term** |
| 3a | source PSD | `2qI` on every junction, `4kT/R` declared for resistors | ngspice `.noise` on real diodes (`IS`, `N=1`, `RB=RE=RC=0`, `KF=0`) → shot only | 1.000× |
| 3b | is `4kT/R` double-counted or missing? | No. The only resistor in the transient netlist is `Rdet` (500 Ω), a *modelling* element for τ_det; ngspice resistors are noiseless in `.tran`, and the detector's physical noise is carried by its own `2q·I_det` (= ½·4kT/r_d, the diode-connected identity) | `R_L = 1 GΩ` sense load: `4kT/R_L = 1.66e-29 A²/Hz` vs `2q·1 µA = 3.2e-25` → 5·10⁻⁵ | 1.000× |
| 3c | `NT` | pre-registration says **2 ps**; the code and the results section use **10 ps** (`phase0_gaps_sim.py`, `nt_ps = 10.0`). Both are far above every loop pole (raw source band 250 / 50 GHz) and the in-band PSD is probe-validated at **0.986** of target | not applicable — `.noise` is a frequency-domain analysis, no time-step source | **1.007×** (and a pre-reg/implementation discrepancy worth recording) |
| 3d | referred to input or output? | **output** — `V(nn_i)` added at `y_i = G·x_i + n_i`, PSD `2q·I₀·│y_i│/I₀²` | **output** — `F_o` mirror into `R_L`, normalized by `I_out` | 1.000× |
| 3e | **junctions per VGA** | **1** (VGA output shot noise only; the VGA is behavioural and was given one junction's worth) | **2** (`Dlog` + `Dexp` current-log gain cell; calculus uses `√2·√(2qB/I)` explicitly) | **1.429×** (measured; `√2` nominal) — the second-largest term |
| 4 | **does P2's ENBW include the underdamped peak?** | P1 *measured* the peak (3.36× in power at 460–530 MHz, −3 dB ~780 MHz, +47 % overshoot, 40 crossings in 20 ns) but then band-limited at 153 MHz anyway — its own two measurements of the same loop are inconsistent | Yes: the ENBW integral runs 10⁵–10¹¹ Hz and includes the 3.06× peak at 473 MHz | see §3 — the peak **does not change** the ENBW; the underdamping is nevertheless *the reason* `f_env` is 9× low |
| 5a | **per-channel metric** | identical decomposition; **RMS over the 60–300 ns window** × channels, α re-fitted every timestep | identical decomposition; **p50 over 20 000 Monte-Carlo draws** | **0.902×** (p50/RMS of an 8-channel residual) |
| 5b | averaging window / CM removal | 240 ns after a 60 ns discard; common mode removed per timestep | one draw per realization; common mode removed per draw | included in 5a |
| — | device realization | ideal `2qI` | P2's real two-diode cell measures 0.950× its own ideal-shot calculus (`1.053` ratio, their §5.3) | **0.950×** |
| — | input draw | own draw (`--seed`) | seed 2026 | **1.001×** |

**Product:** `2.384 × 1.429 × 0.902 × 0.950 × 1.001 = 2.921×` = exactly the
published discrepancy `1.998 / 0.684`.

In log terms: bandwidth **81 %**, junction count **33 %**, the three
book-keeping factors **−14 %** of `ln(2.921)`.

### 2.1 The one-bench reproduction

`I₀ = 1 µA`, 4 seeds × 300 ns, P2's draw, per-channel RMS:

| band limit | effective `B` | 1 junction | 2 junctions |
|---|---|---|---|
| 1-pole @ `f_env` (**P1's convention**) | 240.1 MHz | **0.6845 %** | 0.9925 % |
| brick-wall @ `(π/2)·f_env` | 240.1 MHz | 0.6752 % | 0.9654 % |
| brick-wall @ `ENBW_loop` (**P2's convention**) | 1395.1 MHz | 1.6315 % | **2.3316 %** |

Measured ratio P2-cell / P1-cell on one bench: **3.406×**
(nominal `2.411 × 1.414 = 3.409`). The remaining gap to the published
2.921× is the metric statistic (0.902) and P2's device realization (0.950).

---

## 3. Why the two bandwidths differ by 9× — the mechanism

The loop is an integrator (`C_g`) behind a detector pole (`τ_det`), i.e.
second order. Linearizing `dG/dt = (k/C_g)(γ² − msq)`, `msq = LPF_τ(G²·MS(x))`:

```
    ω_n = sqrt( 2 k G0 MS(x) / (C_g tau_det) )      ω_c = 2 k G0 MS(x) / C_g
    zeta = 1 / (2 ω_n tau_det)                      ω_n  = 2 zeta ω_c
```

For the published values (`k = 2e-3`, `C_g = 1 pF`, `τ_det = 0.5 ns`,
`G₀ = 0.7171`, `MS(x) = 1.9444`):

| quantity | analytic | measured on the bench |
|---|---|---|
| `ω_n/2π` | 531.6 MHz | 473 MHz (PSD peak), 828 MHz (ring, large-signal) |
| **`ζ`** | **0.2994** | — |
| envelope pole `ζω_n/2π` (= P1's `f_env`) | 159.2 MHz | **152.8 MHz** |
| `ENBW = ω_n/(8ζ) = ω_c/4` (= P2's `ENBW_loop`) | **1394.4 MHz** | **1395.1 MHz** |
| peaking `1/(4ζ²(1−ζ²))` | 3.06× | **3.06×** |
| `ENBW / f_env = π/(4ζ²)` | 8.76 | 9.13 |

So the two "bandwidths" are the same loop read two ways:

```
    f_env  = zeta * omega_n / 2 pi       <- how fast the ENVELOPE decays
    ENBW   = omega_n / (8 zeta)          <- how much WHITE NOISE gets through
    ratio  = pi / (4 zeta^2)             <- 8.8x at zeta = 0.30
```

**`f_env` is not a filter.** Nothing in the circuit low-passes a channel at
153 MHz. P1's post-filter was a post-processing choice that happens to
coincide with the settling rate. This is the same error `R19` in
`docs/mismatch_calculus_addendum.md` already names — *"Bandwidth to use:
ENBW of the MEASURED closed-loop response — not `1/(2π t_settle)`, which is
a large-signal number"* — committed with a small-signal envelope fit instead
of a large-signal settling time, which is why P1 landed on 153 MHz rather
than R19's 56.8 MHz.

### 3.1 The peak is not the point

The 3.06× resonance does **not** inflate the ENBW: for this topology
`ENBW = ω_n/(8ζ) = ω_c/4` regardless of `ζ`, because `ω_n = 2ζω_c`. Adding a
compensating zero (§5) removes the peak completely and leaves `ENBW`
unchanged at 1387 MHz. Underdamping redistributes the noise in frequency; it
does not add any. What underdamping *does* is decouple `f_env` from `ENBW`
by `π/(4ζ²)` — which is precisely why P1's settling-derived band was 9× low.

### 3.2 The loop does not band-limit the per-channel term at all

Measured on the same netlist with P2's AC probes:

| transfer | `│H│(DC)` | `│H│(∞)` | ENBW |
|---|---|---|---|
| detector rel. error → `δG/G` | 0.500000 | 1.4·10⁻⁵ | **1395 MHz** |
| ch-0 VGA rel. error → `y₀` | 0.909012 (= 1 − w₀) | **1.000003** | **121 034 MHz** |
| ch-0 VGA rel. error → `y₁` | 0.090988 (= w₀) | 2.6·10⁻⁶ | 1395 MHz |

Above the loop bandwidth the loop does **nothing** to per-channel noise
(R19). So `B` in the formula below is *not* a loop property: it is the
**observation bandwidth of whatever consumes `y_i`**. Both packets used a
loop-derived proxy. `ENBW_loop` is the defensible default — a consumer
slower than the loop would throw away the advertised settling spec — but it
is a **declared** quantity and §5.3 must say so.

---

## 4. The unified formula

```
    PER-CHANNEL NOISE ERROR OF THE AGC VGA ARRAY

      sigma_pc  =  C(draw) * sqrt( k_j * 2 q * B / I0 )

      k_j  = shot-noise junctions in the per-channel signal path
             (1 for a single output device, 2 for the current-log
              Dlog/Dexp gain cell the paper's VGA actually is)
      I0   = the current that normalized amplitude 1 represents
      B    = DECLARED observation bandwidth of the per-channel path
             (the loop does not set it -- see 3.2; ENBW_loop is the default)

      C    = sqrt( mean_i[ |y_i| (1 - 2 w_i) ] + sum_j w_j^2 / |y_j| )
             w_i = y_i^2 / sum_j y_j^2 ,  RMS(y) = 1
           = sqrt( 1 - 1/N )  for a flat draw   ( = R3's residual factor )
```

* fitted over 6 `(k_j, B)` cells of the transient bench:
  **`C = 0.780 ± 0.010`** (spread 0.770–0.800);
* closed form for this draw: **`C = 0.7664`** — fit/analytic **1.018**;
* for a flat 8-channel draw `C = √(1 − 1/8) = 0.935`; the 0.766 here is the
  crest factor of the Gaussian draw (noise is loudest on the weak channels,
  `R18(b)`).

Statistic: `C` is fitted to the **RMS**; the p50-over-draws statistic P2
reports is **0.918× `C`**, i.e. `C_p50 = 0.716`.

Scaling, as required: `σ_pc ∝ √(k_j·B/I₀)`.

### 4.1 Where it crosses the 0.69 % mismatch residual

`I₀,cross = C²·k_j·2q·B / σ_mm²`, `B_cross = σ_mm²·I₀ / (C²·k_j·2q)`:

| `k_j` | `B` | `I₀` at which noise = 0.69 % |
|---|---|---|
| 2 (the paper's current-log VGA) | 100 MHz | 0.82 µA |
| **2** | **122 MHz** | **1.00 µA** |
| 2 | 240 MHz | 1.97 µA |
| 2 | 1395 MHz (`ENBW_loop`) | **11.4 µA** |
| 1 (single output device) | 244 MHz | 1.00 µA |
| 1 | 1395 MHz | 5.71 µA |

Equivalently, at `I₀ = 1 µA`: **`B_cross = 122 MHz`** (`k_j = 2`) or
244 MHz (`k_j = 1`). §5 shows that 122 MHz is, to 2 %, the ENBW the
published loop has when it is **critically damped** — the coincidence is the
most useful number in this note.

### 4.2 Both packets' verdicts, restated consistently

At the published operating point (`I₀ = 1 µA`, `k_j = 2`,
`B = ENBW_loop = 1395 MHz`): **`σ_pc = 2.33 %` (this bench), 1.998 % with
P2's measured device** — i.e. **3.3× the mismatch residual**.
P1's "statistical tie at 1 µA" is an artifact of its 240 MHz effective band
and its one-junction VGA; at the same `(k_j, B)` P1's own transient gives
2.33 %. **K-C and K1 are the same kill, and K1 has it right.**

---

## 5. The damping fix and what it costs

There are two independent knobs, and the note's most useful number falls out
of the second one.

```
    zeta * omega_n  =  1 / (2 tau_det)          -- INDEPENDENT OF k
```

The envelope decay rate of this loop is set by the detector pole alone.
Lowering the loop gain `k` therefore does **not** slow the envelope: it only
raises `ζ` (and lowers `ω_c`, hence `ENBW = ω_c/4`). Critical damping is
reached at

```
    k_crit = C_g / (8 G0 MS(x) tau_det) = 1.793e-4      (11.2x below k = 2e-3)
```

Small-signal probe (+10 % input step at 200 ns; P1's 2.5× step is
slew-limited and returns ~5 ns for *every* `k`, which is why the published
campaigns never saw this):

| variant | `ζ` | overshoot | `t_settle` (1 %) | `ENBW_loop` | peaking | `σ_pc` (`k_j`=2) | `σ_pc·√t_settle` |
|---|---|---|---|---|---|---|---|
| **k = 2e-3, R_z = 0 (published)** | **0.30** | **+15.6 %** | **4.13 ns** | 1395 MHz | **3.06×** | **2.33 %** | 4.74 |
| k = 1e-3, R_z = 0 | 0.42 | +6.4 % | 4.58 ns | 697 MHz | 1.70× | 1.65 % | 3.53 |
| k = 5e-4, R_z = 0 | 0.60 | +1.3 % | 4.83 ns | 349 MHz | 1.09× | 1.17 % | 2.56 |
| k = 2.5e-4, R_z = 0 | 0.85 | 0.0 % | 5.09 ns | 174 MHz | 1.00× | 0.82 % | 1.86 |
| **k = k_crit = 1.79e-4, R_z = 0** | **1.00** | **0.0 %** | **5.52 ns** | **125 MHz** | 1.00× | **0.698 %** | **1.64** |
| k = 1e-4, R_z = 0 | 1.34 | 0.0 % | 12.70 ns | 70 MHz | 1.00× | 0.52 % | 1.86 |
| k = 5e-5, R_z = 0 | 1.89 | 0.0 % | 27.76 ns | 35 MHz | 1.00× | 0.37 % | 1.94 |
| k = 2e-3, R_z = 250 Ω | ∞ | +0.3 % | 1.63 ns | 987 MHz | 1.09× | 1.96 % | 2.50 |
| **k = 2e-3, R_z = 500 Ω (= τ_det/C_g)** | ∞ | **0.0 %** | **0.77 ns** | **1387 MHz** | 1.00× | 2.33 % | 2.04 |
| k = 5e-4, R_z = 500 Ω | ∞ | 0.0 % | 3.00 ns | 348 MHz | 1.00× | 1.17 % | 2.02 |
| k = k_crit, R_z = 500 Ω | ∞ | 0.0 % | 8.31 ns | 125 MHz | 1.00× | 0.698 % | 2.01 |

Three statements, and they must not be run together:

1. **Lowering `k` to `k_crit` is nearly free, and it lands exactly on the
   mismatch floor.** `ENBW` falls **11.2×** (1395 → 125 MHz) and `σ_pc` falls
   **3.34×** (2.33 % → **0.698 %**, i.e. the 0.69 % VGA-mismatch residual to
   1 %), while settling degrades only **1.34×** (4.13 → 5.52 ns) and the
   +47 % overshoot and the 3.06× gain-noise peaking disappear entirely. This
   is the whole fix: **`k_crit` is where noise stops being the binding
   per-channel term.** The `B_cross = 122 MHz` of §4.1 and the critically
   damped loop's own `ENBW = 125 MHz` agree to 2 % — that is not a
   coincidence to exploit, it is the design point.
2. **The integrator zero is free but is not the noise fix.**
   `R_z = τ_det/C_g = 500 Ω` cancels the detector pole, makes the loop
   first-order, removes the ringing, and **speeds settling 5.4×
   (4.13 → 0.77 ns)** at **unchanged** `ENBW` and therefore **unchanged
   noise** (`ENBW = ω_n/(8ζ) = ω_c/4` either way — the peaking redistributes
   noise in frequency without adding any). Use it when `k = 2e-3` must stay
   for speed. At any *fixed* `ENBW` the compensated loop is **slower** than
   the uncompensated critically damped one (8.31 ns vs 5.52 ns at 125 MHz),
   because pole-zero cancellation throws away the second pole's contribution
   to the settling.
3. **The `σ·√t_settle` figure of merit has a minimum at `ζ = 1`**: 4.74 at
   the published `ζ = 0.30`, **1.64** at `ζ = 1`, 1.86–1.94 when overdamped,
   ~2.02 for the whole compensated first-order family. The published loop is
   **2.9× off the optimum** of its own topology. `R20`'s
   `σ_dyn ∝ 1/√t_settle` holds exactly only within a fixed damping class —
   which is why the compensated rows are flat at 2.01–2.04 and the
   uncompensated ones are not.

---

## 6. The sentence for §5.3

> Thermal and shot noise is the **third per-channel exposure** of the AGC
> array, alongside VGA gain mismatch and the (architecturally zero) detector
> term. It is carried entirely by the VGA's own junctions — the detector's
> noise, like the detector's mismatch, is demoted to common mode at every
> frequency — and it obeys
> `σ_pc = C·√(k_j·2q·B/I₀)` with `C = 0.780 ± 0.010` for an 8-channel
> Gaussian draw (`C = √(1 − 1/N)` for a flat one), `k_j` the number of
> shot-noise junctions in the channel path (2 for the current-log VGA) and
> `B` the **declared** observation bandwidth of the per-channel path — which
> the loop does not set, since `|H(own channel)| → 1` above the loop
> bandwidth. At `I₀ = 1 µA` noise **equals** the 0.69 % mismatch residual for
> a loop ENBW of **122 MHz**; at the published loop's actual
> `ENBW = 1395 MHz` it is **2.0–2.3 %**, i.e. **3.3× mismatch**, and equality
> would require `I₀ = 11.4 µA`. Noise scales as `√(B/I₀)`: it is neither
> trimmable nor learnable, and is bought down only with current or with
> settling time.

Supporting clause on the loop itself:

> The published loop (`k = 2e-3`, `C_g = 1 pF`, `τ_det = 0.5 ns`) is
> underdamped, `ζ = 0.30`, with 3.06× gain-noise peaking at 473 MHz and
> +47 % overshoot. Because `ζω_n = 1/(2τ_det)` is independent of the loop
> gain, lowering `k` to the critically damped value
> `k_crit = C_g/(8·G₀·MS(x)·τ_det) = 1.8·10⁻⁴` cuts the loop's noise
> bandwidth 11× (1395 → 125 MHz) and the per-channel noise 3.3×
> (2.33 % → 0.70 %, i.e. onto the mismatch floor) for a 1.34× settling cost
> (4.1 → 5.5 ns), and removes the peaking and the overshoot. A series
> resistor `R_z = τ_det/C_g = 500 Ω` in the integrator instead removes the
> ringing at unchanged bandwidth and shortens settling to 0.77 ns, but does
> not reduce noise.

**Consequences for the two existing packets** (stated here, not edited into
them): `exp_phase0_gaps.md` §5.2's "statistical tie at `I₀ = 1 µA`" should be
read as **conditional on a 240 MHz effective band and a one-junction VGA**;
at the paper's own circuit and bandwidth it becomes K1's 3.3×.
`exp_calculus_noise.md` §5.3 and `R19`/`R20` stand as written; `R19`'s
warning against `1/(2π t_settle)` should be extended to cover *any*
settling-derived bandwidth, including a small-signal envelope fit, with
`ENBW/f_env = π/(4ζ²)` as the correction, and `R20`'s
`σ_dyn ∝ 1/√t_settle` should carry the qualifier *at fixed damping*.

---

## 7. Limitations

1. **`B` is declared, not derived.** The strongest result here is negative:
   the loop does not band-limit the per-channel path (`ENBW(ch0→y₀)
   = 121 GHz`, `|H|(∞) = 1.000003`), so no number in this note is the
   bandwidth "of the circuit". `ENBW_loop` is a defensible default, not a
   measurement of the real consumer. A physical VGA has its own pole; both
   benches' VGAs are pole-free (behavioural in P1, `CJE = CJC = TF = 0` in
   P2), so the true `B` is set by devices neither bench models.
2. **`C` is draw-dependent.** 0.7664 for the seed-2026 draw, 0.935 for a flat
   draw. The crossover currents in §4.1 move with the input crest factor by
   the same factor; they are not universal constants.
3. **One draw, one N.** The cross-check was run at N = 8 on a single input
   draw, deliberately (the point was to hold the draw fixed while varying
   conventions). The `N`-dependence of `C` is asserted from the closed form,
   not measured.
4. **The metric factor (p50/RMS) is empirical.** The Monte-Carlo
   p50/analytic-RMS ratio is 0.918 on this draw (20 000 draws); the
   published-to-published factor quoted in §2 is 0.902, because it also
   carries the transient bench's own 1.02× Monte-Carlo excess over the
   analytic RMS. Neither is a distributional identity.
5. **P1's pre-registration/implementation gap on `NT`** (2 ps declared,
   10 ps implemented) is recorded above and shown to be ≤ 0.7 % in amplitude
   by P1's own in-band PSD probe (0.986 of target). It is not a contributor
   to the 2.9×, but it is an un-flagged deviation in a pre-registered run.
6. **P2's 0.950× device factor is taken from P2** (its own measured/calculus
   ratio 1.053 on the two-diode cell) and was not independently re-measured
   here; the transient bench uses ideal `2qI` sources throughout.
7. **The damping sweep is small-signal, and that matters for `k_crit`.**
   All §5 settling numbers are for a +10 % step. Large-signal behaviour
   (P1's 2.5× step) is **slew-limited** by `k/C_g`, so a large-signal step at
   `k_crit` is ~11× slower to slew than at `k = 2e-3` — the 1.34× settling
   cost quoted for `k_crit` is a **small-signal** cost and does not carry
   over to a large input transient. Any design that must track large steps
   keeps `k = 2e-3` and pays the noise, or uses `R_z` plus a separate
   slew path. The §5 rows are internally consistent (all small-signal);
   the 4.13 ns there and P1's 4.03 ns large-signal figure agree only by
   accident.
7b. **`ζ = 1` is the optimum of `σ·√t_settle` for *this* topology only**
   (single integrator behind a single detector pole). The 1.64-vs-2.02
   comparison against the compensated family is a statement about two
   pole configurations, not a general control-theory result, and the
   settling metric (1 % of the step) is one of several defensible ones.
8. **`k_crit` was not re-verified under noise.** The `σ_pc = 0.698 %` for the
   critically damped loop is the fitted formula evaluated at the *measured*
   `ENBW = 125 MHz`, not a noisy transient at `k = k_crit`. The formula is
   validated against noisy transients at 240 and 1395 MHz (§2.1, §3 of the
   script) to 1.7 %, so the extrapolation is short, but it is an
   extrapolation.
9. **No new physics was added to the detector or integrator.** The common-mode
   gain jitter numbers (P1: 7.7 % at 52 µA detector bias; P2: 0.735 %) were
   not reconciled — they differ by detector-bias reading, not by bandwidth
   convention, and they do not affect the per-channel ranking.

---

## 8. Reproducing

```
    python3 spice/noise_reconciliation.py --seeds 4 --tend 300
```

53 s wall clock (21 `.tran` + 13 `.ac` ngspice runs), numpy + scipy.
Raw numbers → `spice/out/noise_reconciliation.json`.
Sections map 1:1 onto the `[1]`–`[5]` blocks of the script's stdout.
