# D1 / D2: the signal-domain axis put under ngspice

*Companion to `docs/domain_axis.md` (§5) and to `spice/domain_gates_sim.py`.
Section 1 below was written into this file **before the script was run**; it is
a verbatim copy of the pre-registration in `docs/domain_axis.md` §5 and has not
been edited since.*

Run: 2026-09-16, ngspice-46 (KLU), numpy 2.4, 6 workers, 200 MC draws per point,
total wall clock **13.3 s** (`spice/out/domain_gates.json`).

**One-line summary.** Both gates survive. D1: the time domain costs
**3.1–4.5 % median L1** against the log domain's **12.5–13.6 %**, a 2.8–4.4× saving,
and the pre-registered crossing lands at 142–178 mV against 155 mV ± 20 % —
except in the strictest reading at N = 64, where it lands at 201 mV and reveals
that the log lever `ln(I_max/I_S)` is *not* a technology constant. D2: the
best-to-worst spread for P11 is **11.8 bits** (σ_C = 0.1 %) / **9.3 bits**
(σ_C = 1 %), far above the 3-bit kill line and above the predicted 8.6 / 5.2.
The third arm, pre-registered against the hypothesis, reaches **2.5–4.1 bits**,
not 9 — §3.2's conversion-cost argument stands.

---

## 1. Pre-registration (verbatim from `docs/domain_axis.md` §5)

> ## 5. Two pre-registerable kill-gates
>
> Both reuse the existing ngspice harness; neither needs a PDK.
>
> ### D1 — exp in log vs exp in time under identical mismatch
>
> Harness: `spice/softmax_agc_sim.py` (log path) + `spice/bounded_recursion_latency.py`
> (race path). Softmax over N = 64, one shared MC draw set (200 draws, 3 % I_S, 0.3 % n,
> 10 mV V_t, V_th = 500 mV): (a) open-loop translinear pipeline; (b) race generator, value
> read as `t_i = t₀e^{−z_i}` by a counter.
>
> **Predictions.** (a) median L1 = **13.6 %** (already measured, paper Result 5).
> (b) `√(0.0647² + 0.020²)` = **6.8 % ± 1 %** as a magnitude; **rank error exactly 0**
> for `g_sel ≥ 0.5` nat.
>
> **Kill.** "The time domain removes the log lever" dies if (b) ≥ 0.75·(a), i.e. (b) >
> **10.2 %**. **Sharper kill — a computable crossover, not a threshold:** the comparator
> lever `σ_Vos/V_th` overtakes the log lever it saved at `V_th* = σ_Vos/0.0647` =
> **155 mV**. Sweep V_th from 100 mV to 1 V; the curves must cross within **155 mV ± 20 %**,
> or the lever model of §1.1 — and this whole document — is wrong.
>
> ### D2 — P11 in Gilbert vs triode vs charge, one corner
>
> Harness: the MC framework of `spice/mamba_vcrc_sim.py`. Multiply two variables,
> 100 draws, 3 % K′, 10 mV V_t, and σ_C at both **0.1 %** and **1 %** (the femtofarad
> measurement above says 1 % is the honest number for small unit caps).
>
> **Predictions.** Subthreshold pair p50 ≈ **40 %** → 0.4 b (repo anchor: 723–2122 %
> state-referred). Triode p50 ≈ **4.5 %** → 3.5 b (anchor: 3.34 % at D = 100). Charge,
> one switched operand: **0.10 %** → 9.0 b, or **1.0 %** → 5.6 b at σ_C = 1 %.
>
> **Kill.** The headline hypothesis predicts a best-to-worst spread of **8.6 bits**
> (σ_C = 0.1 %) or **5.2 bits** (σ_C = 1 %) for one prime at one corner. It dies if the
> measured spread is **< 3 bits**.
>
> **Third arm, pre-registered against my own claim.** A *four-quadrant analog×analog*
> charge multiplier must **not** reach 9 bits: the second continuous operand passes a
> transconductor, re-importing `σ_Vt/V_ov` = 3.3 % → predicted **5.6 b**. If it does reach
> 9 b, "charge is universally best for P11" holds and §3.2's conversion argument is wrong.

### 1.1 Additional pre-registered items from the task brief

These refine, but do not replace, the numbers above. They were recorded here
before the run for the same reason.

* D1 arm (b) is simulated with **σ_Vos = 10 mV at V_th = 200 mV** (the threshold
  of `spice/bounded_recursion_latency.py`, `V_TH = 0.2`). Note that
  `domain_axis.md` §5 derived its 6.8 % prediction at **V_th = 500 mV**
  (`σ_Vos/V_th` = 2.0 %); at 200 mV the same lever model predicts
  `√(0.0647² + 0.050²)` = **8.2 %**, not 6.8 %. Both thresholds are therefore
  reported, and the pre-registered kill (b) > 10.2 % is evaluated at both.
* D1 V_th sweep: **{50, 100, 155, 250, 400, 800} mV** at fixed σ_Vos = 10 mV,
  200 MC draws per point, N = 8 and N = 64. The crossing of the two error
  curves must lie within 155 mV ± 20 %, i.e. **124–186 mV**.
* D1 rank criterion: rank error exactly 0 for logit gaps ≥ 0.5 nat.
* D2: 200 MC draws per arm (the task raises `domain_axis.md`'s 100),
  operands on a grid with |x|, |y| ≤ full-scale/2, capacitor σ_C ∈ {0.1 %, 1 %},
  4 mismatched unit caps in the charge arm.

---

## 2. What was simulated

Fidelity level, in the convention of `spice/README.md`: **device-exact,
wiring-ideal.** Every junction and every channel is a real ngspice device with
its real I–V law and per-device Monte-Carlo mismatch; current copying,
buffering, switches and the comparator are ideal behavioural elements, the
comparator carrying an explicit offset source. `V_T = 0.025864890 V`
(ngspice's SPICE3-legacy kT/q) is used wherever a junction is driven by an
externally computed voltage; junction currents are kept at µA scale.

### 2.1 D1 arm (a) — open-loop translinear softmax (log domain)

The netlist is **imported unmodified** from `spice/softmax_agc_sim.py`
(`build_softmax_netlist`), so this arm is the same cell that produced the
paper's Result 5. Per channel: a DC source at `(BIAS + x_i)·V_T` with
`BIAS = 23` drives the shared exponential stage diode `De_i` into a 0 V sense
source; the open-loop normalisation then adds a per-channel log diode `Dl_i`
fed by the copied channel current, a shared sum diode `Dsum` fed by the KCL sum
of all channel currents, a reference diode `Dref` fed by `I_target` = 1 µA, and
an output diode `Do_i` driven by `V(nl_i) + V(nref) − V(nsum)`. That is
**3N + 2 mismatched junctions**, three of them in each channel's signal path.
`p_i = I(Vo_i)/I_target`. Mismatch: `IS ~ lognormal(0, 3 %)`,
`N ~ normal(1, 0.3 %)`.

The RNG sequence of `softmax_agc_sim.main()` (seed 2026, BJT-grade corner drawn
first) is replayed exactly, so the N = 8 numbers below are the *published*
MOS-typical draws, not merely draws from the same distribution.

### 2.2 D1 arm (b) — time-to-threshold race (time domain)

A DC-operating-point netlist of N race current generators. Per element:

```
.model DN_i D(IS = 1e-14·(1+δ_Is,i), N = 1·(1+δ_n,i))
V_i  a_i 0  {n0·V_T·z_i + V_OFF}      V_OFF = n0·V_T·ln(I_max/I_S) = 0.4943771 V
D_i  a_i m_i DN_i
Vs_i m_i 0  0                          (current sense)
```

with `z = x − max(x)` so that the top logit draws `I_max` = 2 µA, `C = 10 fF`,
`t_1 = C·V_th/I_max = 1.0 ns`, `ln(I_max/I_S) = 19.114`. The arrival time is
`t_i = C·(V_th + V_os,i)/I_i`; the comparator is a behavioural element carrying
an explicit **per-channel offset `V_os,i ~ N(0, 10 mV)`** at `V_th`. The softmax
vector is read as the normalised inverse arrival times,
`p_i = (1/t_i) / Σ_j (1/t_j)`.

**The same junction draws as arm (a)**: element *i* of the race uses exactly
the `(IS, N)` pair of arm (a)'s exponential-stage diode *i*.

The arrival law itself is cross-checked against a full transient netlist
(`B_i 0 c_i I=I(Vs_i)`, `C_i c_i 0 10f`,
`.measure tran t_i WHEN v(c_i)={V_th+V_os,i} RISE=1`): over 160 (N = 8) and
1280 (N = 64) measured arrivals the transient agrees with
`t_i = C(V_th + V_os,i)/I_i` to a **median 5·10⁻⁷ and a maximum 5·10⁻⁶
relative deviation**. GMIN is lowered to 1e-15 S because the tail elements of
an N = 64 softmax over N(0, 2) logits run into the sub-nA decade; the
ideal-device race reproduces the exact softmax to **L1 = 1·10⁻⁶**.

### 2.3 D2 arm (i) — subthreshold Gilbert / differential pair

A junction-exact four-quadrant Gilbert quad (NPN, `BF = 1e6`, `VAF = 1e12`, no
capacitances — the device stands for a subthreshold MOS pair, whose law is the
same exponential). Lower pair carries operand *y*, the two upper pairs carry
operand *x*, tail `I_EE` = 10 µA; outputs sensed by two 0 V sources in the
collectors, `I_out = I(Vk1) − I(Vk2) = I_EE·tanh(x/2V_T)·tanh(y/2V_T)`.
Mismatch: an **input-referred offset per pair**, injected as `IS·exp(±Δ/2V_T)`
(the convention of `mamba_vcrc_sim.mc_draws`, i.e. 10 mV is the *pair*
offset), three independent offsets `Δ_y, Δ_x1, Δ_x2 ~ N(0, 10 mV)`, plus 3 % on
the tail current and 3 % per-device on `IS`. Full scale of both operands is
`n V_T = 25.86 mV` — the denominator of the §1.1 lever `σ_Vt/(nV_T)`.

### 2.4 D2 arm (ii) — triode-region VCR multiplier

Two level-1 NMOS devices in parallel across the differential *x* nodes
(`V_p = V_CM + x/2`, `V_n = V_CM − x/2`, `V_CM` = 0.9 V), gates at
`V_TO + V_CM + V_ov0 ± y/2` with `V_ov0` = 300 mV, `K′` = 100 µA/V², W/L = 1,
currents sensed separately and subtracted:

```
I_A − I_B = β[(V_ov0 + y/2)·x] − β[(V_ov0 − y/2)·x] = β·x·y
```

exactly — the `V_ds²/2` terms cancel in both polarities (the source/drain swap
at `x < 0` is absorbed by the gate-to-source referencing). Measured ideal-device
floor **9.6·10⁻⁵ %**. Mismatch per the repo convention: `K′_A/K′_B = 1 ± δ/2`
with `δ ~ N(0, 3 %)` and `V_TO,A/V_TO,B = V_TO ± Δ/2` with `Δ ~ N(0, 10 mV)`.
Full scale of the gate operand is `V_ov0` = 300 mV — the denominator of the
§1.1 lever `σ_Vt/V_ov`. `V_ds` swing 100 mV, so no device ever leaves triode.

### 2.5 D2 arm (iii) — charge-domain switched-capacitor multiply

Charge-redistribution SC-MAC with **4 mismatched unit caps** (`C_u` = 250 fF,
4 units = 1 pF as in §0.1) sharing onto a summing cap `C_L` = 1 pF. Operand
*y* is the capacitor code `m ∈ {−2,−1,0,1,2}` of 4 units (|y| ≤ full scale/2,
full code = 4 units), realised by driving |m| bottom plates to `sgn(m)·V_x`;
operand *x* is the sampled voltage, `V_FS` = 1 V. Two phases with an ideal
switch (`RON` = 100 Ω, `ROFF` = 1e12 Ω): reset closed while the bottom plates
carry `s_j·V_x`, then reset opens and the bottom plates go to 0. Charge
conservation gives

```
V_top = v_n − (Σ_j s_j C_j)/(Σ_j C_j + C_L) · V_x
```

— the denominator is code-independent because all four units stay connected,
so the multiply is linear in *m*. `kT/C` enters as the sampled reference
`v_n ~ N(0, √(kT/C_tot))` = 45.5 µV rms, injected as a real DC source in series
with the reset switch. All five capacitors carry `σ_C`. Measured ideal-device
floor **3.8·10⁻⁵ %**.

### 2.6 D2 arm (iv) — third arm, charge × transconductor

The same SC front end, but now with **both operands continuous**: all four unit
caps always sample `V_x` (so *y* can no longer be a code), the sampled node is
buffered by ideal unity sources into the differential nodes of the **triode
transconductor of §2.4** (interstage scale 0.2 so that `|V_ds| ≤ 50 mV`), whose
gates carry *y*. This is the circuit `domain_axis.md` §3.2 says must re-import
`σ_Vt/V_ov`. Mismatch: σ_C on the 5 caps *and* 3 % K′ / 10 mV V_t on the
transconductor pair. Measured ideal-device floor **1.4·10⁻⁴ %**.

### 2.7 Metrics

Three, all reported, because they differ by a fixed factor that matters when
comparing a measurement to a lever formula:

| | definition | what it is |
|---|---|---|
| **A** `relRMS` | `RMS_grid(out − ref) / RMS_grid(ref)` | the repo's own convention (`mamba_vcrc_sim.rel_rms`), which produced the 3.34 % triode anchor. **Primary.** |
| **B** `FS-ref` | `RMS_grid(err) / (|G_nom|·RMS_grid(x)·FS_y)` | the error referred to operand-*y* full scale — the quantity the §1.1 `σ_rel` formulas actually compute. Equals the lever exactly when the error is ∝ *x* (true for triode and charge, **false for the Gilbert**, whose offsets enter referred to *x*, to *y* and as a constant). |
| **C** `offs.rem.` | as A with one constant output offset per cell removed | the "learnable bias" of §3.1 item 7. |

`ENOB = log2(1/(2σ_rel))`, the paper's own convention. On this grid A/B is
**exactly** `FS_y/RMS_grid(y)` = 2.8284, i.e. a uniform **1.500 bits** for every
arm (verified: all six arms give 2.8284 to four decimals), so the *spread* —
the actual kill criterion — is identical in A and B. Which of the two
reproduces a given §1.1 lever depends on whether that lever is offset-type or
gain-type; see §4.4.

---

## 3. D1 results

### 3.1 Arm (a) vs arm (b), L1 to the exact softmax

200 paired draws, 3 % I_S, 0.3 % n, σ_Vos = 10 mV.

| | N = 8 | N = 64 |
|---|---|---|
| **(a) open-loop translinear**, p50 / p95 | **13.63 %** / 27.36 % | 12.54 % / 30.37 % |
| (a) after free renormalisation, p50 | 6.56 % | 7.80 % |
| **(b) race @ V_th = 200 mV**, p50 / p95 | **3.78 %** / 7.56 % | **4.45 %** / 9.37 % |
| **(b) race @ V_th = 500 mV**, p50 / p95 | **3.07 %** / 6.69 % | **3.65 %** / 7.59 % |
| kill line 0.75·(a) | 10.22 % | 9.40 % |
| (b)/(a) at 200 mV / 500 mV | 0.277 / 0.225 | 0.355 / 0.291 |
| ideal-device race L1 | 1·10⁻⁶ | 1·10⁻⁶ |

Arm (a) at N = 8 reproduces the published Result 5 exactly: **13.63 % vs the
13.6 % on record.** The harness is therefore known-good before arm (b) is read.

Arm (b) also wins against the *renormalised* arm (a) — 3.78 % vs 6.56 % at
N = 8, 4.45 % vs 7.80 % at N = 64. That control matters because the race output
is normalised by construction while the open-loop translinear output is not:
roughly half of arm (a)'s 13.6 % is a common-mode gain error that a single
downstream constant would remove. Even after handing arm (a) that gift for
free, the time domain is still **1.75× better**.

### 3.2 V_th sweep and the crossing

Median L1 with only the junction lever active (V_os = 0), only the comparator
lever active (ideal diodes), and both. The junction-only column is
V_th-independent by construction — `V_th` cancels in `p_i = I_i/ΣI_j`.

**N = 8**

| V_th / mV | pre-reg | junction only | comparator only | both p50 | both p95 |
|---|---|---|---|---|---|
| 50 | yes | 3.023 % | 8.585 % | 9.972 % | 20.992 % |
| 100 | yes | 3.023 % | 4.324 % | 5.684 % | 10.480 % |
| 155 | yes | 3.023 % | 2.765 % | 4.201 % | 8.112 % |
| 200 | (extra) | 3.023 % | 2.144 % | 3.777 % | 7.556 % |
| 250 | yes | 3.023 % | 1.708 % | 3.476 % | 7.231 % |
| 400 | yes | 3.023 % | 1.062 % | 3.195 % | 6.823 % |
| 500 | (extra) | 3.023 % | 0.849 % | 3.071 % | 6.694 % |
| 800 | yes | 3.023 % | 0.530 % | 3.007 % | 6.554 % |

**N = 64**

| V_th / mV | pre-reg | junction only | comparator only | both p50 | both p95 |
|---|---|---|---|---|---|
| 50 | yes | 3.518 % | 12.087 % | 12.097 % | 25.784 % |
| 100 | yes | 3.518 % | 5.768 % | 6.511 % | 14.002 % |
| 155 | yes | 3.518 % | 3.673 % | 4.948 % | 11.003 % |
| 200 | (extra) | 3.518 % | 2.842 % | 4.445 % | 9.373 % |
| 250 | yes | 3.518 % | 2.279 % | 4.238 % | 8.671 % |
| 400 | yes | 3.518 % | 1.430 % | 3.768 % | 7.832 % |
| 500 | (extra) | 3.518 % | 1.145 % | 3.650 % | 7.586 % |
| 800 | yes | 3.518 % | 0.716 % | 3.637 % | 7.068 % |

The comparator-only column is proportional to 1/V_th to within 1 % across the
whole sweep, so the crossing can be solved rather than eyeballed.

| crossing V_th* | N = 8 | N = 64 | pre-registered |
|---|---|---|---|
| in **median L1** (the literal "error curves") | **141.5 mV** | **162.7 mV** | 155 mV ± 20 % = 124–186 mV |
| in the **effective-logit (nat) metric** | **177.5 mV** | **200.7 mV** | same |

The two rows differ because the two levers do not convert into L1 with the same
efficiency: the junction lever is largest on the highest-current elements,
which are exactly the ones that dominate a peaked probability vector
(L1/lever = 0.53), while the comparator lever is uniform across elements
(L1/lever = 0.42). The nat row removes that metric artefact — both levers are
then additive perturbations of the same quantity, `ln I_i` and
`−ln(1 + V_os,i/V_th)` — and is the reading the lever model of §1.1 is
literally about.

### 3.3 The log lever is not a technology constant

Measured effective-logit spread, against the §1.1 formula:

| | N = 8 | N = 64 |
|---|---|---|
| measured σ_log | 5.731 % | 4.968 % |
| lever with the *realised* mean `⟨ln(I/I_S)⟩` | 5.697 % | 4.946 % |
| realised `⟨ln(I/I_S)⟩` | 16.14 | 13.11 |
| §1.1's nominal lever at `ln(I_max/I_S)` = 19.11 | 6.47 % | 6.47 % |

The formula is right to **0.6 %** once the *realised* bias lever is used. But
`ln(I_max/I_S)` = 19.11 is the value at the **top** element only; every other
element sits at `19.11 + z_i` with `z_i < 0`, and over a softmax on N(0, 2)
logits the population mean falls to 16.1 (N = 8) and 13.1 (N = 64). The log
lever is therefore 5.73 % / 4.97 %, not 6.47 %, and
`V_th* = σ_Vos/σ_log` moves to 178 mV / 201 mV.

### 3.4 Rank error vs logit gap (V_th = 200 mV, both levers active)

Pairwise inversions of the measured arrival order against the true logit order.

| true gap / nat | N = 8: inversions / pairs | rate | N = 64: inversions / pairs | rate |
|---|---|---|---|---|
| [0, 0.1) | 66 / 200 | 33.0 % | 3198 / 11 400 | 28.05 % |
| [0.1, 0.25) | 22 / 200 | 11.0 % | 739 / 15 200 | 4.86 % |
| [0.25, 0.5) | — (no pairs) | — | 15 / 29 400 | 0.051 % |
| **[0.5, 1)** | **0 / 600** | **0** | **0 / 55 600** | **0** |
| **[1, 2)** | **0 / 1600** | **0** | **0 / 101 000** | **0** |
| **[2, ∞)** | **0 / 3000** | **0** | **0 / 190 600** | **0** |
| **all ≥ 0.5** | **0 / 5 200** | **0** | **0 / 347 200** | **0** |

Not a single inversion in 352 400 pairs separated by ≥ 0.5 nat. The last
non-zero bin is [0.25, 0.5) at 5·10⁻⁴, i.e. the threshold sits just below the
pre-registered 0.5 nat, as it should.

One caveat that the pre-registration does not cover: the probability that the
**entire** vector is ordered correctly is 58.5 % (N = 8) and 0 % (N = 64),
because a softmax tail always contains pairs separated by far less than 0.5 nat.
"Rank error 0 for gaps ≥ 0.5 nat" is a statement about *resolvable* pairs, and
that is exactly how §2's P5 row uses it (`top-B set exact at g_sel ≥ 0.5`).
It is not a claim that a race sorts a whole softmax.

---

## 4. D2 results

200 MC draws per arm, 35-point operand grid (7 × 5), |x|, |y| ≤ full scale/2.

### 4.1 Domain × σ_C → error → bits

| arm | σ_C | **A** relRMS p50 (p95) | ENOB_A | **B** FS-ref p50 (rms) | ENOB_B | **C** offs.rem. p50 | ENOB_C | ideal-device floor (A) |
|---|---|---|---|---|---|---|---|---|
| (i) subthreshold Gilbert | — | **335.82 %** (924.6 %) | **−2.75 b** | 118.73 % (169.7 %) | −1.25 b | 107.23 % | −1.10 b | 3.396 % (tanh compression) |
| (ii) triode VCR, V_ov = 300 mV | — | **8.147 %** (24.67 %) | **2.62 b** | 2.880 % (**4.295 %**) | 4.12 b | 8.147 % | 2.62 b | 9.6·10⁻⁵ % |
| (iii) charge SC, 4 unit caps | 0.1 % | **0.0941 %** (0.189 %) | **9.05 b** | 0.0333 % (0.0409 %) | 10.55 b | 0.0561 % | 9.80 b | 3.8·10⁻⁵ % |
| (iii) charge SC, 4 unit caps | 1 % | **0.5505 %** (1.502 %) | **6.50 b** | 0.1946 % (0.2713 %) | 8.00 b | 0.5453 % | 6.52 b | 3.8·10⁻⁵ % |

Against the pre-registration:

| arm | predicted | measured (primary, A) | measured (lever metric, B, rms) | verdict on the *number* |
|---|---|---|---|---|
| subthreshold pair | 40 % → 0.4 b | 335.8 % → −2.75 b | 169.7 % | **lever wrong by 4–8×**, in the direction that makes the domain *worse* (repo anchor for the Mamba pair: 723–2122 % state-referred) |
| triode | 4.5 % → 3.5 b | 8.147 % → 2.62 b | **4.295 %** | **lever confirmed to 4 %** in its own metric |
| charge, σ_C = 0.1 % | 0.10 % → 9.0 b | **0.0941 % → 9.05 b** | 0.0409 % | **hit, to 6 %** |
| charge, σ_C = 1 % | 1.0 % → 5.6 b | 0.5505 % → 6.50 b | 0.2713 % | better than predicted by 0.9 b |

### 4.2 Spread — the kill criterion

| metric | σ_C = 0.1 % | σ_C = 1 % |
|---|---|---|
| **A relRMS (primary)** | pair −2.75 b, triode 2.62 b, charge 9.05 b → **11.80 b** | pair −2.75 b, triode 2.62 b, charge 6.50 b → **9.25 b** |
| B FS-referred | −1.25 / 4.12 / 10.55 b → **11.80 b** | −1.25 / 4.12 / 8.00 b → **9.25 b** |
| C offset removed | −1.10 / 2.62 / 9.80 b → **10.90 b** | −1.10 / 2.62 / 6.52 b → **7.62 b** |
| pre-registered | 8.6 b | 5.2 b |
| kill line | < 3 b | < 3 b |

The smallest spread over all six readings is **7.62 bits**, 2.5× the kill line
and above the pre-registered value in every case.

### 4.3 Third arm (pre-registered against the hypothesis)

A four-quadrant analog × analog charge multiplier, second operand through a
triode transconductor.

| σ_C | relRMS p50 (A) | ENOB_A | FS-ref p50 (rms) | ENOB_B | vs pure charge (A) |
|---|---|---|---|---|---|
| 0.1 % | 8.515 % | **2.55 b** | 3.011 % (4.347 %) | 4.05 b | 0.0941 % → 8.515 %, **90.5× worse, −6.50 b** |
| 1 % | 9.158 % | **2.45 b** | 3.238 % (4.335 %) | 3.95 b | 0.5505 % → 9.158 %, **16.6× worse, −4.05 b** |

**It does not reach 9 bits.** It reaches 2.45–4.05 b against a predicted 5.6 b,
i.e. the conversion is even more expensive than `domain_axis.md` §3.2 claims.
The FS-referred rms (4.35 % / 4.33 %) is indistinguishable from the bare
triode's 4.295 % and is *unchanged* when σ_C goes from 0.1 % to 1 %: the
transconductor completely masks the capacitor mismatch. That is §3.2's
"conversion avoidance, not conversion accuracy" in one measurement — the charge
domain's 9 bits exist only while one operand stays a switch pattern.

### 4.4 A structural result the pre-registration did not anticipate

**§1.1's `σ_rel` column mixes two normalisation conventions.** Metrics A and B
differ by exactly `FS_y/RMS_grid(y)` = 2.8284 — **1.500 bits, identically for
all six arms** — so the choice between them shifts every absolute bit count by
the same amount and leaves the spread untouched. But the §1.1 levers are *not*
all written in the same one:

| lever in §1.1 | type | reproduced by | measured (rms over draws) |
|---|---|---|---|
| triode `√(δ_K′² + (σ_Vt/V_ov)²)` = **4.48 %** | offset (error ∝ x, referred to `V_ov` = full scale of y) | **metric B** | 4.295 % — 4 % low |
| charge `√(δ_C² + kT/C/V_FS²)` = **0.100 % / 1.00 %** | gain (error ∝ x·y) | **metric A** | 0.1158 % / 0.767 % |
| subthreshold pair `σ_Vt/(nV_T)` = **38.7 %** | offset | neither | 480 % (A) / 170 % (B) / 107 % (C) |

An offset-type lever is naturally full-scale-referred (metric B); a gain-type
lever is naturally signal-referred (metric A). Reading both out of one column,
as §1.1 does, is a silent 1.5-bit inconsistency on any grid that runs the
operands down to zero. It does not change any ranking, but a compiler that
compares a triode number to a charge number straight out of that table is
comparing 4.48 % against 0.100 % with a factor 2.83 hidden between them.

**The subthreshold pair is genuinely mis-priced, not mis-normalised.** Both
readings are 3–12× above 38.7 %, because §1.1 prices *one* differential pair
while a four-quadrant multiplier has *three*: the lower pair's offset enters
referred to operand *y*, the two upper pairs' offsets enter referred to *x* and
as a constant. The measured 480 % (rms) / 336 % (p50) sits between §1.1's
40 % prediction and the repo's own Mamba anchor (723–2122 % state-referred),
which is where it should be.

Removing one constant per cell (metric C) buys the Gilbert **1.65 bits**
(335.8 % → 107.2 %) and buys the triode and the charge arm **nothing**
(8.147 % → 8.147 %, 0.5505 % → 0.5453 %). That is a direct confirmation of
§3.1 item 7 / §1.2's "offset levers are additive and hence learnable; gain
levers are multiplicative": the subthreshold cell is the only one of the three
with a trimmable constant, and even after trimming it stays 10 bits behind
charge.

---

## 5. Verdicts

| # | pre-registered kill | measured | verdict |
|---|---|---|---|
| D1-1 | "(b) ≥ 0.75·(a)", i.e. (b) > 10.2 % | (b) = 3.07–4.45 % against (a) = 12.5–13.6 %; worst reading 4.45 % vs a 9.40 % line | **SURVIVES** with a 2.1–3.3× margin |
| D1-2 | (b) = 6.8 % ± 1 % as a magnitude | 3.07 % (N = 8, 500 mV) / 3.65 % (N = 64, 500 mV) median L1; the *lever* is confirmed (§3.3) but L1 converts it at ≈ 0.5 | **prediction over-states by ≈ 2×**; the direction favours the claim. The 6.8 %/8.2 % figures are σ's, not L1's, and `domain_axis.md` conflated the two |
| D1-3 | crossing within 155 mV ± 20 % (124–186 mV) | median-L1 metric: **141.5 mV** (N = 8), **162.7 mV** (N = 64) — both inside. Effective-logit metric: **177.5 mV** (N = 8) inside, **200.7 mV** (N = 64) **outside by 8 %** | **SURVIVES 3 of 4 readings; fails the strictest one.** Cause identified and it is a correction, not a refutation: see below |
| D1-4 | rank error exactly 0 for gaps ≥ 0.5 nat | **0 inversions in 352 400 pairs** | **SURVIVES**, unambiguously |
| D2-1 | spread < 3 bits | 7.62–11.80 bits over six metric × σ_C combinations | **SURVIVES**, 2.5–3.9× the line |
| D2-2 | third arm reaches 9 bits → §3.2 wrong | 2.45–4.05 bits; FS-referred error identical to the bare triode and insensitive to σ_C | **§3.2's conversion-cost argument stands** |

**The one failure and what it costs.** At N = 64 the crossing in the
effective-logit metric lands at 200.7 mV, outside the 124–186 mV band. The
reason is not the lever *formula* — that is accurate to 0.6 % (§3.3) — but the
constant fed into it. `domain_axis.md` §1.1 prices the log domain at
`ln(I_max/I_S)·δ_n` with `ln(I_max/I_S)` = 19.11, which is the amplification
seen by the **largest** current only. Every other element sits lower, so the
population lever is `⟨ln(I/I_S)⟩·δ_n`, and `⟨ln(I/I_S)⟩` falls with N and with
the logit spread: 16.14 at N = 8, 13.11 at N = 64. Consequences:

1. **`V_th*` is workload-dependent, not a technology constant.** It is
   `σ_Vos/√(δ_Is² + ⟨ln(I/I_S)⟩²δ_n²)`: 178 mV for an 8-way softmax, 201 mV
   for a 64-way one, 155 mV only in the limit where every element runs at
   `I_max`. §1.1 and §5 should state the lever as an upper bound.
2. **The direction is unfavourable to the time domain**: a smaller log lever
   means less saved, so a designer must push `V_th` higher than 155 mV to stay
   on the right side of the trade. The 200 mV of the existing race harness is
   already about right; the 500 mV of §0.1 is comfortable.
3. Nothing in §3.1's ranking moves. Even at N = 64 and V_th = 200 mV, where the
   comparator lever is at its most expensive in this sweep, the time domain
   costs 4.45 % against the log domain's 12.54 %.

---

## 6. Limitations

* **No PDK.** Level-1 MOS (no mobility degradation, no body effect —
  `GAMMA = 0`, no `LAMBDA`, no overlap or junction capacitance) and ideal
  bipolar junctions standing in for subthreshold MOS. Absolute currents and
  absolute bit counts are not silicon numbers; the *ratios between domains* at
  one shared corner are what this measures.
* **Static mismatch only.** One draw per cell, held for the whole operand grid.
  No 1/f or RTN noise, no drift, no temperature gradient, no supply noise, no
  V_t/K′ correlation, no gradient (Pelgrom distance) term — all of which a real
  PDK would add and which would hurt the *large* capacitors of the charge arm
  more than the small devices of the others. Area, and therefore Kinget's
  bandwidth-accuracy-power trade, is absent entirely; the charge arm's 9 bits
  are bought with capacitor area that is not priced here.
* **Ideal switches** in the SC arms (`RON` = 100 Ω, `ROFF` = 1e12 Ω, no charge
  injection, no clock feedthrough, no signal-dependent charge dumping). Charge
  injection is the dominant real-world error of an SC multiplier and would
  attack exactly the arm that wins; the 9.05 b should be read as an upper bound.
  `kT/C` *is* included (45.5 µV rms, a real DC source in the netlist).
* **The comparator is a behavioural element with an offset source**, not a
  circuit: no finite gain, no metastability, no delay, no offset drift, and no
  kickback onto the race node. Its only mismatch is the static
  `V_os,i ~ N(0, 10 mV)` the pre-registration names. The race also assumes a
  perfect current copy from diode to integrating capacitor (`B` source), i.e.
  no mirror mismatch beyond the junction's own.
* **Wiring-ideal throughout**, as in the rest of `spice/`: ideal current
  copying, ideal unity buffers (third arm), ideal voltage summation (arm (a)'s
  translinear loop). Real translinear wiring would add error to arm (a), i.e.
  would widen the D1 gap, not narrow it.
* **The operand grid is 35 points and static.** No frequency response, no
  settling, no dynamic range sweep. D2 measures a DC transfer surface; the
  latency and energy columns of `domain_axis.md` §1.2 are untouched here.
* **arm (i) is a four-quadrant Gilbert**, three mismatched pairs. A
  two-quadrant single pair would carry one offset instead of three and would
  land nearer the 38.7 % lever; the arm as built is the honest four-quadrant
  analogue of the other two arms, and choosing the *better* subthreshold
  variant would only shrink the D2 spread by ~1.5 b, leaving the verdict
  unchanged.
* **N = 8 and N = 64 only** for D1, on one logit distribution (N(0, 2)).
  §3.3 shows the crossing depends on that distribution, so the two numbers
  reported there bracket a range rather than fix a constant.
* Arm (a) at N = 8 replays the published RNG sequence and reproduces Result 5
  exactly; every other number here is from fresh draws with fixed seeds
  (`spice/domain_gates_sim.py`, seeds 2026 / 20260916+N / 3–6·10⁶).
