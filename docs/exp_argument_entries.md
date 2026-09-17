# The three argument-only §6 entries, turned into numbers

**Experiment for `docs/missing_primes_mapping.md` §6, rows `D8 CTC loss`,
`A20 Speculative decoding` and `L3 Experience replay`.**

Date: 2026-09-16. Harness: `spice/argument_entries_sim.py`, ngspice-46,
numpy 2.4. Fidelity level: **junction-exact, wiring-ideal** (`spice/README.md`)
— every exponential current comes from a real diode I–V law, every charge
redistribution from real capacitors and switch resistances; mirrors, gates,
comparators and the normalising stage are behavioural with *stated* constants.

Predictor: `docs/mismatch_calculus.md` (rules R1–R15). Arbiter: ngspice.
Domain assignments and levers: `docs/domain_axis.md` §1, §3.

---

## 0. Pre-registration block

*Written and committed to file before `spice/argument_entries_sim.py` existed.
Everything from here to §1 is the pre-registration. Deviations are listed in
§1.1 and nowhere else.*

### 0.0 What is at stake

Three §6 rows are mapped **by argument only** — a mechanism is named, no number
is attached:

| row | now | proposed in §6 | stated mechanism | what is missing |
|---|---|---|---|---|
| D8 | U | B(T, L) | "analog trellis sum-product (Loeliger 2001)"; primes P3·P4·P1·P11 sequential over T | the word *sequential over T* hides an error accumulation; nobody has costed it |
| A20 | U | B(γ, V) | "one-hot compare; rollback ≤ γ" | no code length, no ε, no latency, no energy |
| L3 | G | B(buffer) | "P10 index + decoder + gate" | no fan-in, no read latency, no retention requirement, no storage class |

Each is turned here into **a number plus one circuit check**, with the mismatch
calculus as the predictor and ngspice as the arbiter.

### 0.1 Corner and constants (all sourced or declared)

| symbol | value | source |
|---|---|---|
| `V_T` | 0.025864890 V | `spice/README.md` — ngspice SPICE3-legacy kT/q at 27 °C, **mandatory** for any externally driven junction |
| `σ(I_S)/I_S` | 3 % | `domain_axis.md` §0.1 (MOS-typical corner, = `tab:mismatch`) |
| `σ(n)/n` | 0.3 % | ibid. |
| `σ(V_t)` | 10 mV | ibid. |
| `σ(C)/C` | **0.1 % and 1 %**, both run | ibid. + `domain_axis.md` §1.3 brief correction: 0.1 % is a large-unit-cap number, femtofarad MOM caps are ≈ 1 % |
| `σ(K′)/K′` | 3 % | ibid. |
| `I_S` (diode) | 1e-14 A | `spice/bounded_recursion_latency.py` design constant |
| bias | µA scale (0.1–2 µA) | `spice/README.md` GMIN gotcha |
| `E_ADC`, `t_ADC` | 100 fJ, 1 ns | `prime_compiler_v2.tex` §4.4 |
| `E_DAC`, `t_DAC` | 50 fJ, 0.5 ns | ibid. |
| `t_cmp` | 200 ps | `exp_bounded_recursion.md` §0.2 — declared design constant, not measured |
| `t_1` (race winner) | 1.0000 ns | `exp_bounded_recursion.md` §3.1 — **measured**, N-independent |
| `t_race` per level | 1.2000 ns | ibid. §5.1 — **measured** (`t_1 + t_cmp`) |
| `t_wr` (memristive) | 10 ns … 1 µs, swept | ibid. §0.2 |
| `C_ML`, `V_FS` | 10 fF, 1 V | `exp_symbol_margin.md` §8 |
| `E_line = C·V_FS²` | 10 fJ | ibid. |
| gain-cell τ | 5 ms (Leroux et al. 2025) / **128.6 µs measured** | `exp_kv_leak_eviction.md` §8, `spice/kv_gain_cell.py` |
| memristor write | 10 ns … 1 µs, ~1 pJ/cell | `exp_bounded_recursion.md` §0.2 + task-stated range |
| `V_ov` design point | 300 mV | `domain_axis.md` §0.1 |
| seed | 2026 | house convention |

`σ_b ≡ hypot(σ_Is, σ_n·L)`, `L = ln(I/I_S)` (R1). At `I = 1 µA`, `I_S = 1e-14`:
**L = 18.4207**, **σ_b = 6.288 %**.

### 0.2 E1 — CTC loss: calculus prediction, written first

**Circuit.** One trellis section, `L = 4` states, the general BCJR/HMM form
`α_t(s) = Σ_{s'} α_{t−1}(s')·p_t(s|s')` that Loeliger, Lustenberger,
Helfenstein & Tarköy realise in subthreshold sum-product cells (IEEE T-IT
47(2), 837 (2001), Sec. III, 4-state tail-biting trellis). CTC's own recursion
is the sparser special case `α_t(s) = y_t(l_s)·Σ_{s'∈{s,s−1,s−2}} α_{t−1}(s')`
— 4 multipliers instead of 16 — and is costed alongside.

**Log domain (P3·P4·P11 translinear).** Each product `α·p/I_ref` is a
four-junction translinear loop (input `a`, input `p`, reference, output),
wiring replaced by ideal voltage summation per `spice/README.md`. By R1 each
junction traversal contributes `σ_b`, and by R3 the four are independent:

```
 σ(ln I_product) = 2·σ_b = 12.58 %
```

The KCL sum (P1) is exact. With incoming weights `w_{ss'}` and `Σ_{s'} w² = 0.25`
at uniform weights, R3 gives per output state

```
 Var(e_s) = Σ_{s'} w_{ss'}²·(σ_a² + σ_p² + σ_out²) + σ_ref²
          = 0.25·3σ_b²        [per-channel, branch junctions]
          + σ_b²              [common mode, the shared reference diode]
 σ_pc = 5.45 %,  σ_cm = 6.29 %,  σ_total = 8.32 %
```

CTC-sparse (3-term sum, one emission multiply): **σ ≈ 9.61 %**.

**PRE-REGISTERED PREDICTION E1-L:** per-section log-domain error
**σ_total = 8.3 %** (range 5–10 % over the weight draws), of which
**σ(ln Z) ≈ 6–7 % = 0.06–0.07 nat** is the common-mode part that enters the
log-likelihood.

**Charge/time domain (P8 + switched capacitor).** Probabilities are charges;
`α_{s'}` is sampled on a weight capacitor `C_{ss'} = p_{ss'}·C_u` and the
products are summed by charge redistribution onto the section's output node
(`domain_axis.md` §2, P8 row: "native `Q = ∫I dt`, 9.0 b"; §3.1 item 6). The
sum is exact by charge conservation; the only levers are `σ_C`, `kT/C`, and the
restoring stage. R3 with `Σw² = 0.25` on numerator and denominator:

```
 σ_section ≈ σ_C·√(Σw² + Σw²) = 0.707·σ_C
           = 0.0707 %  at σ_C = 0.1 %      (7.07e-4 nat)
           = 0.707  %  at σ_C = 1 %        (7.07e-3 nat)
 kT/C at C_u = 100 fF: 204 µV on a 0.5 V full scale = 4.1e-4 — subdominant
 kT/C at C_u =  10 fF: 644 µV on a 0.5 V full scale = 1.3e-3 — comparable at σ_C = 0.1 %
```

**PRE-REGISTERED PREDICTION E1-C:** per-section charge-domain error
**0.07 % (σ_C = 0.1 %) / 0.71 % (σ_C = 1 %)**, i.e. **89× / 12× below the log
domain**.

**The restoring stage is declared, not measured.** Both domains need `α`
renormalised every section or it underflows. In the log domain the normaliser is
the repo's own sum-feedback AGC (`tab:agcsweep`: per-channel 0.000 %,
common-mode 3.0–8.1 %, settles 4.4 ns). In the charge domain it is a
switched-capacitor gain stage whose input-referred offset lands additively:
`σ_Vos/V_FS = 10 mV/0.5 V = 2 %`. Because a per-section SPICE model of the
normaliser would be a second experiment, it is carried **analytically** as a
swept term `σ_Vos ∈ {0, 1, 10} mV` and every accumulated number is reported at
all three values. This is declared here so that a favourable choice cannot be
made afterwards.

**Two accumulation models, both pre-registered.**

* **A-indep** — the trellis is *unrolled in space*, T physical copies, fresh
  device draw per section. Independent errors ⇒ `σ(ln P) = √T · σ(ln Z)`.
  This is the model the task names ("state independence assumption").
* **A-reuse** — the trellis is *time-multiplexed* through **one** physical
  section, which is what "bounded T_max" actually buys. The device draw is then
  the **same** at every t, so the shared-junction part of the error accumulates
  **linearly**, `T·σ_cm`, and only the weight-dependent part decorrelates.

Both are run on the same 200 measured draws. Pre-registered predictions:

| model | domain | T = 16 | T = 64 | T = 256 |
|---|---|---|---|---|
| A-indep | log | 0.28 nat | 0.56 nat | **1.12 nat** |
| A-indep | charge, σ_C = 0.1 % | 0.0028 | 0.0057 | 0.0113 |
| A-indep | charge, σ_C = 1 % | 0.028 | 0.057 | 0.113 |
| A-reuse | log | 1.12 | 4.48 | 17.9 |
| A-reuse | charge, σ_C = 0.1 % | 0.011 | 0.045 | 0.181 |
| A-reuse | charge, σ_C = 1 % | 0.113 | 0.453 | 1.81 |

Predicted 0.1-nat crossings (A-indep): log **T ≈ 2**, charge 1 % **T ≈ 200**,
charge 0.1 % **T ≈ 2·10⁴**.

**Effective bits** are reported the repo's way (`domain_axis.md` §0.1),
`ENOB = log2(1/(2σ_rel))`, with `σ_rel = σ(ln P)/|ln P|`; the absolute figure in
nats is reported alongside, because that is the one CTC gradients see.

**PRE-REGISTERED KILL KE1.** *Killed* if the log-domain accumulated error at
T = 256 exceeds **1 nat** (CTC gradients unusable) **AND** the time/charge
domain does not fix it (i.e. also exceeds 1 nat at T = 256, at σ_C = 1 % and
σ_Vos = 10 mV, the pessimistic corner). Report the T at which each domain
crosses **0.1 nat**. Predicted outcome: log domain exceeds 1 nat, charge domain
does not ⇒ **KE1 not killed, but D8 must carry a domain assignment**.

**Measurement (ngspice).** 200 MC draws per domain per σ_C, one trellis section,
`.op` for the log domain, `.tran` charge redistribution (switch `R_on = 1 kΩ`,
so `τ = R_on·C_u`) for the charge domain. Reported per draw:
`e_pc` = RMS over states of the α error after removing the best scalar (R3
residual) and `e_lnZ = ln(Z_circ/Z_ideal)` (the common-mode part). Accumulation
in numpy over the measured draws.

### 0.3 E2 — Speculative decoding: calculus prediction, written first

**Circuit.** Verification compares one draft token against one target token:
a one-hot / coded dot product against the stored token plus a threshold (R4),
`ε = Q(m/σ_tot)`, feasibility `σ_tot ≤ d_min/z_ε` with `d_min = 1` for a dense
code, `σ_tot = σ_G√k` (`exp_symbol_margin.md` §5.2). Rollback is a shift
register of `γ` P8 slots (γ = 5, Leviathan et al. ICML 2023).

**Two encodings, both costed.** A *one-hot* line per vocabulary entry makes the
row `k = V` lines wide (32 768 / 262 144 match-line taps, which is the area
blocker, not the error blocker); a *dense binary code* needs
`k = ⌈log₂V⌉ = 15 / 18` lines. The dense code is the one the kill criterion is
written against.

**Union bound, pre-registered.** A dense code over a full `2^k` alphabet has
exactly `k` Hamming-1 neighbours, so the per-adversary rate must be `ε/k`
(`exp_symbol_margin.md` §10 flags this omission). `z = Q⁻¹(ε/k)`.

**PRE-REGISTERED PREDICTION E2:**

| V | k | σ_G | σ_tot | z (union) | m* required | of 2.0 available | feasible at ε = 10⁻⁶? |
|---|---|---|---|---|---|---|---|
| 32 768 | 15 | 3 % | 0.1162 | 5.274 | **0.613** | 31 % | yes |
| 32 768 | 15 | 1 % | 0.0387 | 5.274 | **0.204** | 10 % | yes |
| 262 144 | 18 | 3 % | 0.1273 | 5.308 | **0.676** | 34 % | yes |
| 262 144 | 18 | 1 % | 0.0424 | 5.308 | **0.225** | 11 % | yes |

`σ_G,max` at ε = 10⁻⁶ (union-bounded): **4.90 % at V = 32k, 4.44 % at V = 256k**.
Gain-accuracy requirement (R5): `g_max ≈ 8.9e-4` at V = 256k, σ_G = 3 %, i.e.
**10.1 bits** of match-line transimpedance accuracy; 11.7 bits at σ_G = 1 %.
Lognormal prefactor (R4 / `exp_symbol_margin.md` §4.2): at k = 15–18 and z ≈ 5.3
budget a factor **≈ 2** on false accepts.

Match-line noise is predicted to be the binding term, not mismatch: at k = 18
the budget is `σ_tot ≤ 0.1884`, mismatch uses 0.1273, leaving **0.772 %**
relative match-line noise — against the 0.5 % that `exp_symbol_margin.md` §5.2
already treats as realistic.

Latency per verify step (all constants from §0.1):
`t_step = t_ML + t_race + t_shift + t_ADC`, with `t_ML` **measured** in this
experiment, `t_race = 1.2 ns`, `t_shift = 1 ns` (switched-capacitor slot) or
`t_wr` (memristive), `t_ADC = 1 ns` for the accepted-prefix length.
Predicted **5–7 ns** with capacitive slots.

Energy per verified token: `γ·E_line + E_ADC + γ·E_shift`, divided by the
expected accepted count `(1−a^{γ+1})/(1−a)` = 2.38 / 3.69 / 4.69 at acceptance
rate a = 0.6 / 0.8 / 0.9. Predicted **~40–70 fJ per verified token**, against a
digital 18-bit XOR-tree comparator at ~2 fJ — i.e. the pre-registered
expectation is that analog comparison is **feasible but loses on energy**, which
is the D-by-position argument of mapping §1/L4 arriving on a second axis.

Rollback register: γ = 5 P8 slots must hold for `t_hold = γ·t_step`. Droop
tolerance is half the remaining margin, `(2 − m*)/(2k)` in relative terms;
required τ and the measured gain-cell τ = 128.6 µs are compared.

**PRE-REGISTERED KILL KE2.** *Triggered* if at V = 256k and σ_G = 3 % the
required code length exceeds **64 bits per token**, or if per-token ε cannot
reach **10⁻⁶**. On trigger: token comparison must stay digital (D-by-position,
like the tokenizer, mapping §6 row R2), and the crossover σ_G is reported
either way.

**Circuit check.** ngspice match line, k = 18, lognormal conductance mismatch at
σ_G = 3 %, finite-gain (A₀ = 10⁴) transimpedance stage, 200 draws instantiated
twice (stored word + Hamming-1 neighbour), paired against the numpy model on the
*same* draws — the `exp_symbol_margin.md` §7 protocol at the speculative-decoding
code width. Extended with a `.tran` to **measure** the match-line settling time
`t_ML`, which §0.1 would otherwise have to assert.

### 0.4 E3 — Experience replay: calculus prediction, written first

**Circuit.** Bounded buffer of B slots, random read: P10 (noise source) → address
decoder → P11 gates on a shared read line → P1 sum.

**Fan-in and tree depth.** Table 4 bounds P1 fan-in at 256
(`missing_primes_mapping.md` §5, Addressing row). A flat B-way read violates it
for every B of interest, so the read is a tree of depth `⌈log₂₅₆ B⌉`:

```
 B = 10⁴  →  depth 2   (256² = 65 536 ≥ 10⁴)
 B = 10⁶  →  depth 3   (256³ = 1.678e7 ≥ 10⁶; 256² = 65 536 < 10⁶)
 address bits: 14 and 20
```

**PRE-REGISTERED PREDICTION E3-A:** depth 2 at B = 10⁴, **depth 3 at B = 10⁶**,
read latency `depth · (t_ML + t_cmp)` with `t_ML` measured in E2.

**Leakage floor.** The P11 gate is realised as the repo's own current-steering
pair (two junctions sharing a tail = the P6 Fermi–Dirac splitter of
`domain_axis.md` §2), select overdrive `ΔV`. An off-gate passes
`exp(−ΔV/(nV_T))` of its slot current, so with 255 off-gates

```
 crosstalk = (Σ_off x_j / x_sel) · exp(−ΔV/(nV_T)) · exp(σ_off²/(2(nV_T)²))
 ΔV for 1 % at equal amplitudes:  nV_T·ln(255/0.01) = 262.4 mV
 lognormal inflation from σ(V_os) = 10 mV: ×1.078
```

**PRE-REGISTERED PREDICTION E3-B:** crosstalk 534 % / 77.3 % / 11.2 % / 1.10 % /
0.234 % at ΔV = 100 / 150 / 200 / 260 / 300 mV. At the repo's own design point
`V_ov = 300 mV` the 1 % bar is met; below ≈ 262 mV it is not. With a slot
dynamic range R = x_max/x_min the requirement rises by `nV_T·ln R` (60 mV per
decade), so R = 10 needs **322 mV**.

**Retention and storage class.** The replay horizon is seconds to minutes. A
gain cell holds `exp(−t/τ)` with τ = 5 ms (Leroux et al.) or the repo's measured
**128.6 µs**. Refresh period for n bits: `t_ref = τ·ln(1/(1−2^{−(n+1)}))`.

```
 τ = 5 ms,     n = 6 b → t_ref = 39.2 µs → 0.255 nW per cell
 τ = 5 ms,     n = 8 b → t_ref =  9.8 µs → 1.02  nW per cell
 τ = 128.6 µs, n = 6 b → t_ref =  1.01 µs → 9.91 nW per cell
 crossover horizon  H* = E_wr·t_ref/(C·V_FS²)  = 3.9 ms (τ=5 ms, n=6, E_wr=1 pJ)
```

**PRE-REGISTERED PREDICTION E3-C:** the compiler must select the **non-volatile
(memristive) class**: the replay horizon exceeds the gain-cell crossover H* by
10²–10⁴, and the refresh power of a B = 10⁶ buffer at 128 cells/slot is tens of
watts against sub-µW of memristive write power at 10³ environment steps/s.

**PRE-REGISTERED KILL KE3.** *Triggered* if the off-gate leakage sum exceeds
**1 %** of the selected signal at 256-way fan-in. On trigger: the P1 fan-in
bound of Table 4 binds the replay read and B = 10⁶ needs a ≥ 3-level tree —
quantified either way, since the fan-in bound produces depth 3 at B = 10⁶
independently of the leakage result.

**Circuit check.** ngspice, 256-way one-hot gated read, 1 % gate (tail) mismatch
plus the MOS-typical corner on the steering junctions, `.op`, 200 MC draws at
the nominal ΔV and a deterministic ΔV sweep.

### 0.5 Sweeps fixed in advance

* E1: `L = 4`; `σ_C ∈ {0.1 %, 1 %}`; `C_u ∈ {10 fF, 100 fF}`;
  `σ_Vos ∈ {0, 1, 10} mV`; `T ∈ {16, 64, 256}`; 200 MC draws per cell;
  both accumulation models; CTC-sparse variant costed analytically.
* E2: `V ∈ {32 768, 262 144}` ⇒ `k ∈ {15, 18}`; `σ_G ∈ {1 %, 3 %}`;
  `ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹}`; `γ = 5`; acceptance rate `a ∈ {0.6, 0.8, 0.9}`;
  slot class ∈ {switched-capacitor, memristive 10 ns … 1 µs}; 200 SPICE draws.
* E3: `B ∈ {10⁴, 10⁶}`; fan-in 256; `ΔV ∈ {100, 150, 200, 260, 300, 400} mV`;
  200 MC draws at ΔV = 300 mV; `n ∈ {6, 8}` bits; τ ∈ {128.6 µs, 5 ms};
  `E_wr ∈ {0.1, 1, 10} pJ`.
* Seed 2026 throughout.

### 0.6 What this experiment cannot settle

Declared in advance:

1. **No search/learning quality.** E1 measures log-likelihood error, not CTC
   training convergence; E2 measures comparison error, not the acceptance rate
   of a real draft model; E3 measures read fidelity, not RL sample efficiency.
2. **No area.** A 262 144-tap one-hot match line and a 10⁶-slot buffer are area
   problems this model does not touch (the same omission as
   `exp_bounded_recursion.md` §0.8).
3. **No thermal noise inside SPICE.** `kT/C` and match-line noise are carried
   analytically, as in `exp_symbol_margin.md` §10.
4. **The normaliser is analytic** (§0.2), swept but not simulated.
5. **One technology model.** Ideal diodes, ideal capacitors, level-1-free
   behavioural gates, no PDK, no memristor compact model, no endurance model.
6. **A-reuse is a model, not a measurement.** No 256-section circuit is
   simulated; the reuse accumulation replays one measured device draw T times.

---

*Everything above was written before `spice/argument_entries_sim.py` existed.
Everything below is the run.*

---

## 1. Method and deviations

`spice/argument_entries_sim.py`, seed 2026, ngspice-46, numpy 2.4, 4 workers,
total wall clock **101.7 s**. Raw numbers in `spice/out/argument_entries.json`,
console log in `spice/out/argument_entries_run.log`.

Reproduce:

```bash
cd spice
OPENBLAS_NUM_THREADS=1 python3 argument_entries_sim.py --part all \
    --mc-runs 200 --workers 4
```

### 1.1 Deviations from the pre-registration

1. **`t_ML` is measured on the passive KCL match line**, not through the
   transimpedance amplifier. A closed-loop settling number would have required a
   declared amplifier GBW, i.e. an asserted constant dressed as a measurement.
   The match line proper — `k` source conductances charging `C_ML` — has no free
   constant, so that is what is measured; the amplifier's *gain* accuracy is
   covered by R5 and its *bandwidth* is declared out of scope.
2. **E1's accumulation uses the closed-form map** of the section (the mode
   `mismatch_calculus.md` §7 licenses), not a T-section netlist. The map is
   validated against ngspice at T = 1 on the same device draws, paired
   (§2.1, §2.2). This was pre-registered in §0.6 item 6 for A-reuse; it also
   applies to A-indep.
3. **E3's per-configuration MC is 200 draws of a 256-slot array** (512 diodes
   per draw), batched 5 draws per netlist. No change to the pre-registered
   sweep.
4. **`|ln P|` for the ENOB conversion is measured on the generator**
   (2.0588 nat per section, §2.4) rather than taken from a real CTC task.
5. Nothing else. All corners, sweeps, kill bars and both accumulation models are
   as pre-registered in §0.

---

## 2. E1 — D8 CTC loss

### 2.1 Per-section error, log domain (ngspice, 200 draws)

One `L = 4` trellis section, 16 four-junction translinear products (a
reference diode, four `α` input diodes, sixteen `p` input diodes, sixteen output
diodes; the loop closed by ideal voltage summation, `spice/README.md` fidelity
level), KCL sum by Kirchhoff.

| quantity | calculus prediction (§0.2) | ngspice, 200 draws | ratio |
|---|---|---|---|
| σ(ln Z) — common mode | 6.29 % | **7.862 %** | 1.25 |
| per-channel RMS residual | 5.45 % | **4.619 %** | 0.85 |
| total RMS | 8.32 % | **9.113 %** | 1.10 |
| paired SPICE vs closed-form map | — | RMS 1.03e-7, max 4.55e-7 | — |

**The calculus lands within 25 % on each component and 10 % on the total**,
which is the accuracy band R1–R3 claim (`mismatch_calculus.md` §0: "to within a
factor of two"). The split moved: the shared reference junction carries slightly
more and the branch junctions slightly less than uniform weights predict,
because the realised trellis weights are not uniform (`Σw²` above 0.25).

*Honest reading of what the SPICE run adds here.* At the wiring-ideal fidelity
level the netlist **is** the closed-form map, so the paired deviation of 1e-7 is
a check that the netlist was built correctly, not an independent physical test.
The arbitration that matters is the first two rows: the **calculus prediction,
written before the run, against the measured junction physics**. That comparison
is not tautological, and it is the one quoted.

### 2.2 Per-section error, charge domain (ngspice `.tran`, 200 draws)

Weight capacitors `C_{ss'} = p_{ss'}·C_u` pre-charged to `V_{s'}`, redistributed
onto the output node through the switch resistance `R_on = 1 kΩ`; charge
conservation does the P1 sum.

| σ_C | C_u | σ(ln Z) SPICE | per-channel RMS | kT/C (relative, analytic) | paired dev |
|---|---|---|---|---|---|
| 0.1 % | 100 fF | **0.0128 %** | 0.0254 % | 0.0407 % | 1.06e-6 |
| 0.1 % | 10 fF | **0.0156 %** | 0.0248 % | 0.1287 % | 1.07e-6 |
| 1 % | 100 fF | **0.1283 %** | 0.2284 % | 0.0407 % | 1.04e-6 |
| 1 % | 10 fF | **0.1281 %** | 0.2539 % | 0.1287 % | 1.07e-6 |

Prediction E1-C was 0.0707 % / 0.707 %; measured 0.0128 % / 0.128 %, i.e. the
calculus **over-predicted by 5.5×**. The reason is in R3 and is a correction to
the prediction, not to the rule: the numerator and denominator of a charge-
redistribution MAC share the *same* capacitors, so their mismatch is correlated
and cancels to first order in the common part — the pre-registration added them
in quadrature as if independent. **Recorded as a calculus correction:** for a
passive ratiometric stage (`out = Σ C_i V_i / Σ C_i`), use
`σ ≈ σ_C·sqrt(Σ_i w_i(1−w_i)²+…)` rather than `σ_C·√(2Σw²)`; the measured
coefficient is **0.128** at `L = 4`, not 0.707.

Against the log domain: **the charge domain is 614× quieter at σ_C = 0.1 % and
61× quieter at σ_C = 1 %** on the log-likelihood channel (7.862 % vs 0.0128 % and
0.1283 %), and 182× / 20× on the per-channel channel. The `kT/C` term overtakes capacitor mismatch below
`C_u ≈ 100 fF` when `σ_C = 0.1 %` — the only place in this experiment where
noise, not mismatch, sets the floor at the device level.

Charge-redistribution settling, measured: **234.8 ps to 99.9 %** at
`C_u = 100 fF` (`R_on·C_tot = 270 ps`), **2.0 ps** at `C_u = 10 fF`.

### 2.3 Accumulation over T (numpy, closed-form map)

40 device draws × 6 input sequences per cell. `bias` = mean error on `ln P`;
`sd(all)` = spread over devices *and* sequences; `sd(seq|dev)` = spread over
sequences at a **fixed** device, which is the number CTC gradients see (a
device-fixed constant offset on `ln P` has zero gradient).

**σ_Vos = 0 (no restoring-stage offset):**

| domain | model | T = 16 | T = 64 | T = 256 |
|---|---|---|---|---|
| log | A-indep, sd(all) | 0.337 | 0.661 | **1.372** |
| log | A-indep, sd(seq\|dev) | 0.287 | 0.571 | 1.207 |
| log | A-reuse, sd(all) | 1.428 | 4.395 | **19.25** |
| log | A-reuse, sd(seq\|dev) | 0.124 | 0.232 | **0.492** |
| charge σ_C = 0.1 % | A-indep, sd(all) | 0.0023 | 0.0046 | **0.0098** |
| charge σ_C = 0.1 % | A-reuse, sd(all) | 0.0033 | 0.0107 | 0.0332 |
| charge σ_C = 1 % | A-indep, sd(all) | 0.0086 | 0.0178 | **0.0341** |
| charge σ_C = 1 % | A-reuse, sd(all) | 0.0187 | 0.0765 | 0.3288 |

(nats). Pre-registered predictions were 0.28 / 0.56 / 1.12 for log A-indep and
17.9 for log A-reuse at T = 256 — **measured 0.337 / 0.661 / 1.372 and 19.25**,
i.e. the √T and linear laws both hold and the pre-registered constants were low
by 20–25 %.

**With the restoring stage's offset (σ_Vos sweep), T = 256:**

| domain | model | σ_Vos = 0 | 1 mV | 10 mV |
|---|---|---|---|---|
| log | A-indep | 1.372 | 1.369 | 1.285 |
| charge σ_C = 0.1 % | A-indep | 0.0098 | 0.0407 | **0.3867** (bias −1.126) |
| charge σ_C = 1 % | A-indep | 0.0341 | 0.0540 | **0.3902** (bias −1.097) |
| charge σ_C = 1 % | A-reuse | 0.3288 | 0.3616 | **0.4772** (bias −1.108) |

**This is the finding the pre-registration did not anticipate:** in the charge
domain the *normaliser's input-referred offset*, not the capacitors, is the
binding term for every T above ≈ 17. At 10 mV the two capacitor corners become
indistinguishable (0.3867 vs 0.3902) — the caps have stopped mattering. In the
log domain the same 10 mV changes nothing (1.372 → 1.285), because a 2 % gain
perturbation is small against the 7.9 % the junctions already contribute.

**The T at which each domain crosses 0.1 nat** (extrapolated from T = 256 with
the law each model obeys: √T for A-indep, linear for A-reuse; on `sd(all)`):

| domain | σ_Vos | A-indep | A-reuse |
|---|---|---|---|
| log | 0 | **T = 1.4** | T = 1.3 |
| log | 10 mV | T = 1.6 | T = 1.4 |
| charge σ_C = 0.1 % | 0 | **T = 26 900** | T = 771 |
| charge σ_C = 0.1 % | 1 mV | T = 1 543 | T = 471 |
| charge σ_C = 0.1 % | 10 mV | **T = 17** | T = 62 |
| charge σ_C = 1 % | 0 | T = 2 205 | T = 78 |
| charge σ_C = 1 % | 10 mV | T = 17 | T = 54 |

The log domain crosses 0.1 nat **before the second section**. That is the whole
of the D8 result in one number.

### 2.4 Effective bits of the final log-likelihood

`|ln P| = 2.0588 nat per section` on this generator (measured over 20 sequences
at each T; a real CTC task runs 1–3 nat/frame, so the ENOB below moves by less
than a bit either way). `ENOB = log2(1/(2·σ_rel))`, `σ_rel = σ(ln P)/|ln P|`,
T = 256, `|ln P| = 527 nat`:

| domain | model | σ_Vos | σ(ln P) | σ_rel | **ENOB** |
|---|---|---|---|---|---|
| log | A-indep | 0 | 1.372 nat | 2.60e-3 | **7.59 b** |
| log | A-reuse | 0 | 19.25 nat | 3.65e-2 | **3.78 b** |
| charge 0.1 % | A-indep | 0 | 0.0098 | 1.85e-5 | **14.72 b** |
| charge 0.1 % | A-reuse | 0 | 0.0332 | 6.30e-5 | 12.95 b |
| charge 1 % | A-indep | 0 | 0.0341 | 6.47e-5 | 12.92 b |
| charge 1 % | A-reuse | 0 | 0.3288 | 6.24e-4 | 9.65 b |
| charge 0.1 % | A-indep | 10 mV | 0.3867 | 7.34e-4 | 9.41 b |
| charge 1 % | A-reuse | 10 mV | 0.4772 | 9.06e-4 | **9.11 b** |

`domain_axis.md` §2 predicts the P8/charge row at 9.0 b and the current-log row
at 3.0 b per stage; a 256-deep accumulation lands at **9.1–14.7 b (charge)**
against **3.8–7.6 b (log)**. The ordering and the ≈ 5-bit gap are exactly the
domain table's, which is the first time that table has been checked on a
*sequential* composite rather than a single stage.

### 2.5 Cost table, T = 256 sections, L = 4

Latency constants: `t_translin ≈ 12 ns` (paper §5.2, the whole open-loop
translinear pipeline — a conservative stand-in for one section, not separately
measured), `t_AGC = 4.4 ns` (paper §5.3 Result 3, measured), charge settle
**measured here**, SC restoring amplifier 1 ns (declared).
Energy: `V_dd·C·nV_T` per junction stage with `V_dd = 0.8 V`
(`domain_axis.md` §1.2), `C_tot·V_FS²` per charge node.

| realisation | products/section | E/section | E, T = 256 | latency/section | latency, T = 256 |
|---|---|---|---|---|---|
| log, dense L², C = 10 fF | 16 × 4 junctions | 13.24 fJ | **3.39 pJ** | ≤ 16.4 ns | ≤ 4.20 µs |
| log, CTC-sparse, C = 10 fF | 4 × 4 junctions | 3.31 fJ | 0.85 pJ | ≤ 16.4 ns | ≤ 4.20 µs |
| log, dense L², C = 1 pF | 16 × 4 junctions | 1.32 pJ | 339 pJ | ≤ 16.4 ns | ≤ 4.20 µs |
| charge, C_u = 10 fF | 16 caps | 27 fJ | **6.91 pJ** | 1.00 ns | **256 ns** |
| charge, C_u = 100 fF | 16 caps | 270 fJ | 69.1 pJ | 1.24 ns | 316 ns |

Reference from the repo's own transition accounting: digitising the `T·L = 1024`
emission probabilities to enter a digital forward–backward costs
**102.4 pJ** of ADC alone (`E_ADC = 100 fJ`), against 3.4–6.9 pJ for the whole
analog trellis. As in `exp_bounded_recursion.md` §2.3, that saving exists **only
if the emission stage is already analog**; if the acoustic model's output is
digital, the `T·L` DACs needed to enter the trellis (51.2 pJ) eat most of it.

*CTC-sparse is 4× cheaper than the dense L² form* and is what CTC actually
needs — the mapping doc's "P3·P4·P1·P11 sequential over T" does not say which,
and the difference is a factor 4 in energy (64 → 16 junction stages) and 12
fewer multipliers (16 → 4).

### 2.6 Verdict — KE1

**KE1 — NOT killed, but the row must carry a domain.**

* The log domain **does** exceed 1 nat at T = 256: **1.372 nat** (A-indep,
  σ_Vos = 0), **19.25 nat** in the realistic time-multiplexed A-reuse model.
  The first half of the kill condition is satisfied.
* The charge domain **does** fix it, at every corner run: the worst charge cell
  in the whole sweep is **0.477 nat** (σ_C = 1 %, σ_Vos = 10 mV, A-reuse) and
  the best is **0.0098 nat**. The second half of the kill condition is not
  satisfied, so KE1 does not fire.

**Three qualifications, all load-bearing.**

1. **The gradient-relevant number is smaller than the headline.** In A-reuse the
   device draw is fixed, so most of the 19.25 nat is a per-chip *constant*, which
   has zero gradient. The spread over inputs at a fixed device is
   **0.492 nat** at T = 256 — still ~5× the 0.1-nat bar, still unusable, but not
   19 nats' worth of unusable. Conversely the constant part is a static
   reparametrization (R9) and is exactly the class of error the repo has measured
   training to absorb (`exp_mismatch_absorption.md`, 0.7–0.95 σ → 0.06–0.16 σ).
   **Not claimed here** — no training run was done — but it is the obvious next
   gate, and it is the difference between "CTC is unmappable in the log domain"
   and "CTC in the log domain needs per-chip calibration".
2. **The charge domain's advantage is spent by the normaliser.** At a 10 mV
   restoring-stage offset the charge domain crosses 0.1 nat at **T = 17**, not
   T = 26 900. The whole 614× device-level advantage of §2.2 survives only if the
   normaliser's offset is trimmed to ≈ 1 mV. That is a spec the mapping doc does
   not name and the domain table does not carry.
3. **A-reuse, not A-indep, is the model that matches "bounded T_max".** Unrolling
   256 trellis sections in space is 256 copies of the hardware; the bound the §6
   row sells is the *time-multiplexed* one, and that is the model in which the
   log domain is worst (19.25 vs 1.37 nat) and in which the charge domain's
   margin is thinnest (0.33 vs 0.034 nat at σ_C = 1 %).

### 2.7 Resulting §6 row text — D8

> | D8 | CTC loss | U | **B(T_max, L_max; domain, σ_Vos)** | analog trellis sum-product (Loeliger et al. 2001). **The bound tuple must name the signal domain.** Log domain (translinear, `domain_axis.md` P3/P4/P11): σ(ln Z) = **7.86 %/section** measured (ngspice, L = 4, MOS-typical corner), so the forward log-likelihood crosses 0.1 nat at **T ≈ 1.4** and reaches **1.37 nat (spatially unrolled) / 19.3 nat (time-multiplexed)** at T = 256 — **7.6 / 3.8 effective bits**, unusable for gradients without per-chip calibration. Charge domain (switched-capacitor redistribution): **0.013–0.128 %/section**, **0.010–0.33 nat** at T = 256, **9.1–14.7 effective bits**. Cost at T = 256, L = 4: **3.4 pJ / 4.2 µs** (log, C = 10 fF) or **6.9 pJ / 256 ns** (charge, C_u = 10 fF), against **102 pJ** of ADC to enter a digital forward–backward — the saving exists only if the emission stage is already analog. CTC's sparse recursion needs **4** multipliers per section, not L² = 16 (4× energy, 12 fewer multipliers). New spec, not in Table 4: the per-section **normaliser offset must be ≲ 1 mV**, or it, not the capacitors, sets the accumulation floor (0.1 nat at T = 17 instead of T = 26 900). |

---

## 3. E2 — A20 speculative decoding

### 3.1 Code length and margin (R4, union-bounded)

Dense binary code, `k = ⌈log₂V⌉`, adversary set = the `k` Hamming-1 neighbours,
so the per-adversary rate is `ε/k` and `z = Q⁻¹(ε/k)`. Separation available
between a stored word and a Hamming-1 neighbour is **2.0 score units at every
k** (`exp_symbol_margin.md` §5.1).

| V | k | σ_G | ε | σ_tot | z (union) | **m\* required** | of the 2.0 available | σ_G,max | gain accuracy (R5) | max ML noise |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 768 | 15 | 3 % | 10⁻³ | 0.1162 | 3.820 | 0.444 | 22.2 % | 6.76 % | 9.6 b | 1.564 % |
| 32 768 | 15 | 3 % | **10⁻⁶** | 0.1162 | 5.274 | **0.613** | 30.6 % | **4.90 %** | **10.0 b** | **0.999 %** |
| 32 768 | 15 | 3 % | 10⁻⁹ | 0.1162 | 6.423 | 0.746 | 37.3 % | 4.02 % | 10.3 b | 0.691 % |
| 32 768 | 15 | 1 % | **10⁻⁶** | 0.0387 | 5.274 | **0.204** | 10.2 % | 4.90 % | 11.6 b | 1.237 % |
| 262 144 | 18 | 3 % | 10⁻³ | 0.1273 | 3.865 | 0.492 | 24.6 % | 6.10 % | 9.7 b | 1.251 % |
| 262 144 | 18 | 3 % | **10⁻⁶** | 0.1273 | 5.308 | **0.676** | 33.8 % | **4.44 %** | **10.1 b** | **0.772 %** |
| 262 144 | 18 | 3 % | 10⁻⁹ | 0.1273 | 6.451 | 0.821 | 41.1 % | 3.65 % | 10.4 b | 0.492 % |
| 262 144 | 18 | 1 % | **10⁻⁶** | 0.0424 | 5.308 | **0.225** | 11.3 % | 4.44 % | 11.7 b | 1.020 % |

**The answer the §6 row was missing: k = 15 bits at V = 32k and k = 18 bits at
V = 256k, with a margin of 0.613 / 0.676 score units of the 2.0 that exist** —
at σ_G = 3 %, ε = 10⁻⁶ per token, union bound included. At σ_G = 1 % the margin
drops to 0.204 / 0.225. Applying the lognormal false-accept prefactor
(`exp_symbol_margin.md` §4.2; between 1.12 and 1.63 at z ≈ 5 for these
`σ_G/√k`, budgeted here at ×2) moves `m*` from 0.676 to **0.692** — a 2 %
change. The prefactor does not bind at this code width.

**One-hot instead of dense.** A one-hot row is `k = V` taps: the *decision* is
easier (only one line is active, `σ_tot = σ_G` independent of V) but the row is
32 768 / 262 144 capacitors wide and the match line's own noise scales with the
tap count. Area, not error, is what rules one-hot out; the dense code is what the
numbers above are for. This distinction is not in the §6 row and needs to be.

### 3.2 Circuit check — ngspice match line, k = 18, σ_G = 3 %, 200 draws

The `exp_symbol_margin.md` §7 protocol at the speculative-decoding code width:
18 lognormal conductances into a finite-gain (A₀ = 10⁴) transimpedance stage,
each draw instantiated twice (stored word + Hamming-1 neighbour), paired against
the numpy model on the *same* draws. Runtime 2.0 s.

| quantity | SPICE | numpy, same draws | ratio |
|---|---|---|---|
| match score σ | 0.134382 | 0.134879 | **0.99631** |
| gain error | **−0.1895 %** | — | predicted `(1+k·R_f/R₀)/A₀ = 0.1900 %` |
| false accepts at m = 1.837 | **10 / 200** | **19 / 200** | 0.53 |

* **σ deviates by 0.37 %** against the 20 % bar the equivalent gate
  (`exp_symbol_margin.md` K4) uses. The symbol-margin noise model is what the
  circuit produces at k = 18 as well as at k = 16.
* **The finite-A₀ gain error is predicted to four digits** (0.1895 % measured vs
  0.1900 % predicted) and it **halves the false-accept count** (19 → 10). At
  k = 16 the same effect was 15–30 % (`exp_symbol_margin.md` §7); at k = 18 and
  a 0.19 % gain error it is 47 %. This is R5 firing exactly as written, and it
  confirms that the 10.1-bit gain-accuracy requirement in §3.1 is the real spec,
  not a theoretical one.
* Sign matters: the gain error here is **negative** (the score reads low), which
  *reduces* false accepts and *increases* false rejects. In speculative decoding
  a false reject costs one wasted draft token; a false accept corrupts the
  output. **Bias the match line low.** That asymmetry is free and is not in the
  mapping doc.

### 3.3 Latency, measured

Match-line settling, `k = 18` source conductances (`R₀ = 100 kΩ` each) charging
`C_ML = 10 fF`, measured in ngspice:

| quantity | measured | RC prediction |
|---|---|---|
| 99 % settle | **242.9 ps** | `τ = R₀/(k+1)·C_ML = 52.6 ps` ⇒ 4.6 τ = 242 ps |
| 99.9 % settle | **364.1 ps** | 6.9 τ = 363 ps |

Per verify step, `t_step = t_ML + t_cmp + t_race + t_shift + t_ADC`:

| slot class | t_step | per verified token (a = 0.8, γ = 5, E[accepted] = 3.69) |
|---|---|---|
| switched-capacitor (1 ns) | **3.764 ns** | **1.020 ns** |
| memristive 10 ns | 12.764 ns | 3.460 ns |
| memristive 100 ns | 102.764 ns | 27.855 ns |
| memristive 1 µs | 1002.764 ns | 271.805 ns |

`t_race = 1.2 ns` is the measured per-level race of `exp_bounded_recursion.md`
§5.1 and is the **largest** term with capacitive slots — the acceptance race,
not the comparison, dominates the verify step. Against a 7 B model's
TPOT of 14.96 ms (`exp_bounded_recursion.md` §0.5), 1.02 ns per verified token
is **1.5×10⁷ times smaller**: verification latency is not where speculative
decoding's time goes, at any slot class in the swept range.

### 3.4 Energy, and why the row still says D-by-position

`E_step = γ·E_line (compare) + E_ADC (accepted-prefix readout) + γ·E_line (slot
shift)`, `E_line = C_ML·V_FS² = 10 fJ` (`exp_symbol_margin.md` §8).

| acceptance rate a | E[accepted] | E per verified token | digital 18-bit XOR tree (~0.1 fJ/gate) | ratio |
|---|---|---|---|---|
| 0.6 | 2.383 | 83.92 fJ | 1.80 fJ | **46.6×** |
| 0.8 | 3.689 | 54.21 fJ | 1.80 fJ | **30.1×** |
| 0.9 | 4.686 | 42.68 fJ | 1.80 fJ | **23.7×** |

This is `exp_symbol_margin.md` §8's conclusion arriving at A20: *the margin is
nearly free, the line drive is not.* An analog token comparison is feasible,
fast and accurate — and it loses to a digital comparator by 24–47× on energy,
because it has to charge a match line to compare two symbols that are already
symbols. **The D-by-position verdict the mapping doc reaches for the BPE
tokenizer (§6 row R2, "mappable but pointless: its input is bytes") applies to
A20's comparator for the same reason and now with a number.** What speculative
decoding actually gets from analog is not the comparison; it is that the draft
and target *logits* never have to be digitised.

### 3.5 Rollback register — γ P8 slots

γ = 5 slots, hold time `t_hold = γ·t_step = 18.82 ns` (capacitive slot). Droop
budget = half the unused margin, `(2 − m*)/(2k) = 3.68 %` at k = 18,
σ_G = 3 %, ε = 10⁻⁶. Required retention **τ ≥ 512 ns**; the repo's measured
gain cell is **τ = 128.6 µs** (`spice/kv_gain_cell.py`, 0.47 % against its
design value) — a **251× margin**. Even at a 1 µs memristive slot shift
(`t_hold = 5.01 µs`) the requirement is τ ≥ 136 µs, which the same cell meets.

**The rollback register is the one part of A20 that is unconditionally free.**
`exp_kv_leak_eviction.md` §8 found the opposite problem for the KV cache —
retention *too long* by 150× — because that buffer is held for thousands of
token-times. A γ = 5 rollback register is held for 19 ns, five orders of
magnitude less, so the same device is 251× *over*-specified. **Retention is a
per-use-site compiler parameter, twice confirmed and in opposite directions.**

### 3.6 Verdict — KE2

**KE2 — NOT triggered.**

* Required code length at V = 256k, σ_G = 3 %, ε = 10⁻⁶: **18 bits**, against a
  64-bit bar. Not exceeded, by 3.6×.
* Per-token ε = 10⁻⁶ is reachable: `m* = 0.676` of the 2.0 score units that
  exist (0.692 with a ×2 lognormal prefactor).

**Crossover σ_G, reported as the gate asks.** Mismatch alone: **σ_G ≤ 4.44 %** at
V = 256k (4.90 % at V = 32k) for ε = 10⁻⁶; 3.65 % / 4.02 % for ε = 10⁻⁹. The
MOS-typical 3 % corner sits comfortably inside.

**But mismatch is not the binding term.** With the remaining budget spent on
match-line noise, `σ_rel ≤ 0.772 %` at V = 256k — against the **0.5 %** that
`exp_symbol_margin.md` §5.2 already treats as realistic. Speculative decoding's
comparator is **noise-limited with 35 % of headroom, not mismatch-limited**,
and a 1 % match line kills it outright. Together with R5's 10.1-bit gain spec,
this is a much tighter design box than "one-hot compare" suggests.

**And the energy verdict stands anyway (§3.4): the comparison should stay
digital.** KE2's kill condition was written about feasibility and does not fire;
the D-by-position conclusion arrives through the energy column instead. That is
worth stating plainly in the row, because it is the opposite of what "U → B"
suggests on its own.

### 3.7 Resulting §6 row text — A20

> | A20 | Speculative decoding | U | **B(γ, V; k, σ_G, σ_ML) — comparator D-by-position** | rollback ≤ γ slots; token comparison = coded dot product + threshold. **k = ⌈log₂V⌉ = 15 (V = 32k) / 18 (V = 256k)** dense bits; at σ_G = 3 %, ε = 10⁻⁶ per token (union bound over the k Hamming-1 neighbours) the required margin is **0.613 / 0.676 of the 2.0 score units available** — feasible, 3.6× inside the 64-bit bar. Binding specs are **not** mismatch (σ_G,max = 4.44 %) but **match-line noise ≤ 0.772 % relative** and **10.1 bits of transimpedance gain accuracy** (R5; a measured 0.19 % gain error halved the false-accept count, ngspice k = 18, 200 draws). Verify step **3.76 ns** (measured match line 364 ps + 200 ps comparator + 1.2 ns acceptance race + 1 ns slot + 1 ns ADC) ⇒ **1.02 ns per verified token** at a = 0.8, γ = 5. Rollback register: γ = 5 P8 slots, `t_hold = 18.8 ns`, requires τ ≥ 512 ns against the measured gain-cell 128.6 µs — **251× margin, unconditionally free**. **Energy says keep it digital:** 42–84 fJ per verified token against ~1.8 fJ for an 18-bit XOR tree (**24–47×**) — like R2 the tokenizer, the comparator is mappable but D-by-position. The analog win in A20 is that the draft/target **logits** need no ADC, not the comparison. |

---

## 4. E3 — L3 experience replay

### 4.1 Circuit check — 256-way one-hot gated read (ngspice)

Each slot is a current-steering pair (two junctions sharing a tail current — the
P6/Fermi–Dirac splitter of `domain_axis.md` §2) with the selected gate at
`V_ref + ΔV` and the 255 others at `V_ref − ΔV`. `Crosstalk` = the summed
current of the 255 off-gates as a fraction of the selected slot's true current.

**Ideal devices, equal slot amplitudes:**

| ΔV | crosstalk, ngspice | law `255·e^{−ΔV/nV_T}` | read error |
|---|---|---|---|
| 100 mV | 522.95 % | 533.89 % | 520.90 % |
| 150 mV | 77.02 % | 77.25 % | 76.72 % |
| 200 mV | 11.177 % | 11.178 % | 11.133 % |
| 260 mV | 1.1016 % | 1.0988 % | 1.0973 % |
| **300 mV** | **0.2364 %** | 0.2340 % | 0.2355 % |
| 400 mV | 0.0060 % | 0.0049 % | 0.0060 % |

The exponential law is confirmed to **0.01 % at ΔV = 200 mV** and to better than
2 % over three decades of crosstalk. `ΔV` for exactly 1 % is
`nV_T·ln(255/0.01) = 262.4 mV`.

**Monte Carlo, 1 % gate (tail) mismatch + 3 % I_S + 0.3 % n + 10 mV offsets,
200 draws per cell:**

| ΔV | slot dynamic range | crosstalk mean | crosstalk p95 | read error | read error sd |
|---|---|---|---|---|---|
| **300 mV** | 1:1 | **0.2553 %** | 0.2666 % | +0.2927 % | 0.9061 % |
| 300 mV | 10:1 | 0.3805 % | **0.9891 %** | +0.3896 % | 1.1093 % |
| 260 mV | 1:1 | 1.1909 % | 1.2335 % | +1.0992 % | 0.9110 % |
| 260 mV | 10:1 | 1.5600 % | 3.7866 % | +1.6398 % | 1.4316 % |
| 200 mV | 1:1 | 12.0823 % | 12.5672 % | +12.0283 % | 0.9964 % |
| 200 mV | 10:1 | 16.1478 % | **42.2272 %** | +16.0228 % | 12.0968 % |

Three readings:

1. **Mismatch inflates the leakage floor by 8 %** (0.2553 % measured vs 0.2364 %
   ideal at 300 mV), which is the lognormal factor `exp(σ_os²/2(nV_T)²) = 1.078`
   predicted in §0.4 — confirmed to 1 %. Mismatch on an off-gate can only make
   leakage worse, never better, because the sum of exponentials is convex.
2. **The read error's *mean* is the crosstalk and its *spread* is the 1 % gate
   mismatch** (sd 0.91 % at every ΔV where crosstalk is controlled). The two
   error sources are cleanly separable, which is what makes the design rule
   below possible: fix ΔV to kill the mean, then the residual is a per-channel
   gain error — static, and therefore R9-absorbable by training.
3. **Slot dynamic range is the hidden parameter.** At 10:1 the *mean* crosstalk
   barely moves (0.26 → 0.38 %) but the **p95 quadruples** (0.27 → 0.99 %) and
   at 200 mV the p95 reaches 42 %. The worst case is a small selected entry
   drowned by 255 large ones, and it costs `nV_T·ln R` = **60 mV of extra
   overdrive per decade** of slot range.

### 4.2 Decoder tree from the P1 fan-in bound

| B | address bits | tree depth `⌈log₂₅₆ B⌉` | capacity 256^depth | read latency `depth·(t_ML + t_cmp)` |
|---|---|---|---|---|
| 10⁴ | 14 | **2** | 65 536 | **1.128 ns** |
| 10⁶ | 20 | **3** | 16 777 216 | **1.692 ns** |

`t_ML = 364 ps` is the match-line settle measured in §3.3; `t_cmp = 200 ps` is
the declared comparator delay. A flat B-way read is not an option at either B:
the Table-4 fan-in bound of 256 forces depth 2 at B = 10⁴ and **depth 3 at
B = 10⁶** — the second half of KE3's statement, and it holds independently of
the leakage result.

**What the tree costs in error.** Each level contributes one 256-way summing
node, so at the 300 mV design point the crosstalk compounds to **0.51 % at
depth 2 and 0.77 % at depth 3**, and the per-channel gate mismatch to
**1.29 % / 1.58 %** (√depth on a 0.91 % per-level sd). At B = 10⁶ the read is
therefore a **0.77 % systematic + 1.6 % random** operation — still under the 1 %
crosstalk bar, but with a factor 1.3 of headroom, not 4.

### 4.3 Retention and the storage class the compiler must select

Replay horizon: seconds to minutes. Refresh period for an n-bit analog cell,
`t_ref = τ·ln(1/(1−2^{−(n+1)}))`; refresh energy `C·V_FS² = 10 fJ` per cell per
refresh; crossover horizon `H* = E_wr·t_ref/(C·V_FS²)` at which one
non-volatile write costs the same as refreshing for that long.

| cell | τ | n | t_ref | refresh power/cell | H* (E_wr = 1 pJ) |
|---|---|---|---|---|---|
| gain cell, repo measured | 128.6 µs | 6 b | 1.009 µs | **9.914 nW** | 0.101 ms |
| gain cell, repo measured | 128.6 µs | 8 b | 0.251 µs | 39.774 nW | 0.025 ms |
| gain cell, Leroux et al. | 5 ms | 6 b | 39.216 µs | **0.255 nW** | 3.922 ms |
| gain cell, Leroux et al. | 5 ms | 8 b | 9.775 µs | 1.023 nW | 0.978 ms |

Buffer level, 6 bits, 128 cells per slot, 10³ environment steps/s:

| B | gain cell 128.6 µs | gain cell 5 ms | memristive, 1 pJ/cell |
|---|---|---|---|
| 10⁴ | 12.7 mW | 326 µW | **0.128 µW** |
| 10⁶ | **1.269 W** | 32.6 mW | **0.128 µW** |

**The storage class is forced, and the number is 10⁴–10⁷×.** A B = 10⁶ replay
buffer on the repo's own measured gain cell burns **1.27 W just standing still**;
on the Leroux cell, 32.6 mW; on a memristive cell written once per environment
step, **0.128 µW**. The memristive figure is independent of B because the write
rate is set by the environment, not by the buffer size — which is the structural
reason the class flips: **refresh power scales with capacity, write power scales
with throughput, and a replay buffer is enormous and slowly written.**

The crossover horizon makes the same point with one number: beyond
**H* = 0.025–3.9 ms** of required retention, non-volatile storage is cheaper than
refreshing. Replay horizons are seconds to minutes, i.e. **10³ to 10⁶ times past
the crossover.** There is no regime in which a volatile replay buffer is the
right choice.

*What this does not say.* Endurance is not modelled. A ring buffer of 10⁶ slots
under 10⁷ environment steps sees ~10 overwrites per slot, which is far inside
any published RRAM endurance figure, but the read-disturb budget of a slot that
is replayed thousands of times is **not** covered here and is the obvious next
question (`exp_bounded_recursion.md` §9 flags read disturb in the same terms).

### 4.4 Verdict — KE3

**KE3 — NOT triggered at the repo's own design point; triggered below it.**

* At `V_ov = ΔV = 300 mV` (`domain_axis.md` §0.1 design point) the off-gate
  leakage sum is **0.2553 %** of the selected signal at 256-way fan-in, with
  200 draws at the MOS-typical corner plus 1 % gate mismatch. The bar is 1 %.
  **Not triggered**, with a factor 3.9 of headroom.
* At **262.4 mV the bar is exactly met**; at 260 mV leakage is 1.19 % and KE3
  **fires**; at 200 mV it is 12.1 % and the read is meaningless.
* With a 10:1 slot dynamic range the p95 at 300 mV is **0.9891 %** — the bar is
  met at the mean and **missed at the 95th percentile**. The requirement grows by
  60 mV per decade of slot range.

**So the honest statement is a condition, not a verdict:** the 256-way one-hot
read meets the 1 % leakage bar **iff the select overdrive is at least
`nV_T·(ln(N_off/0.01) + ln R)` = 262 mV + 60 mV per decade of slot dynamic
range**, which at N = 256 and R = 10 is **322 mV**.

**The fan-in half of KE3 holds regardless.** B = 10⁶ needs a **3-level tree**
(256³ = 1.68e7 ≥ 10⁶, 256² = 65 536 < 10⁶), read latency 1.692 ns, compounded
crosstalk 0.77 % and compounded per-channel spread 1.58 %. B = 10⁴ needs 2
levels, 1.128 ns, 0.51 % / 1.29 %.

### 4.5 Resulting §6 row text — L3

> | L3 | Experience replay | G | **B(B, depth; ΔV_sel, R_slot, retention class)** | P10 index → decoder → P11 gate. **Fan-in:** Table 4's P1 ≤ 256 forces a tree — **depth 2 at B = 10⁴, depth 3 at B = 10⁶** (256² = 65 536 < 10⁶ ≤ 256³); read latency **1.13 / 1.69 ns** (measured match line 364 ps + 200 ps comparator per level). **Leakage floor (ngspice, 256-way, 1 % gate mismatch, 200 draws):** off-gate crosstalk = `255·e^{−ΔV/nV_T}·1.08`, measured **0.255 % at ΔV = 300 mV**, 1.19 % at 260 mV, 12.1 % at 200 mV; the 1 % bar needs **ΔV ≥ 262 mV, +60 mV per decade of slot dynamic range** (322 mV at 10:1, where the p95 otherwise reaches 0.99 %). Compounded over a 3-level tree: **0.77 % systematic + 1.58 % per-channel**. **Storage class is forced, not chosen:** the replay horizon (s–min) is 10³–10⁶× past the refresh/write crossover `H* = 0.025–3.9 ms`, and a B = 10⁶ buffer at 128 cells/slot costs **1.27 W** of refresh on the repo's measured 128.6 µs gain cell, 32.6 mW on a 5 ms cell, and **0.128 µW** on a memristive cell written once per environment step at 10³ steps/s. **The compiler must select non-volatile analog storage; refresh power scales with capacity, write power with throughput, and a replay buffer is large and slowly written.** |

---

## 5. Cross-entry findings

Three results that only appear when the entries are done together:

1. **"Bounded" is not one bound; it is a bound plus a domain plus a spec.** Each
   of the three rows needed a parameter the `B(·)` tuple did not have: D8 the
   signal domain and the normaliser offset, A20 the match-line noise and gain
   accuracy, L3 the select overdrive and the retention class. This is the same
   pattern `exp_bounded_recursion.md` §8 found for MCTS (`t_wr`, not `d`) and
   beam search (`g_floor`, `slot_class`, not `B, L`). **Five for five:
   every §6 bound tuple checked so far has named the wrong parameter.**
2. **Retention is a per-use-site parameter and the errors point in both
   directions.** The KV cache needs a cell **150× leakier** than Leroux's
   (`exp_kv_leak_eviction.md` §8); the rollback register is **251×
   over-specified** by the same cell; the replay buffer needs a class the cell
   cannot reach at all. Maximising retention is wrong at two of the three sites.
3. **Two of the three rows end up D-by-position or domain-switched, and neither
   is a mappability failure.** A20's comparator is feasible and loses on energy;
   D8's log-domain realisation is feasible and loses on accumulated error. The
   mapping doc's §6 headline ("0 unmappable; 28 require a compile-time bound")
   survives — but the useful output of a compiler is *which realisation*, and on
   these three rows the answer is different from the one §6 names in two cases
   out of three.

---

## 6. Limitations

* **Wiring-ideal.** The log-domain translinear loop is closed by ideal voltage
  summation, so §2.1's SPICE run validates the calculus prediction against real
  junction physics but does **not** test loop wiring, mirror errors or the
  Early effect. The paired SPICE-vs-map agreement of 1e-7 is a construction
  check, not a physical one, and is labelled as such in §2.1.
* **The normaliser was never simulated** (declared in §0.2). It turned out to be
  the binding term for the charge domain (§2.3), so the most consequential
  number in E1 rests on a swept parameter rather than a circuit. That is the
  first thing a follow-up should fix.
* **No training.** The A-reuse error is largely a per-chip constant and is
  therefore a candidate for R9 absorption; nothing here measures whether
  training actually removes it.
* **No area.** A 262 144-tap one-hot row, a 10⁶-slot buffer and a T = 256
  unrolled trellis are area problems this model does not touch.
* **No thermal noise inside SPICE.** `kT/C` (E1) and match-line noise (E2) are
  analytic; `exp_symbol_margin.md` §10's warning that a different noise
  parameterisation moves the k-limits applies unchanged to §3.1.
* **E3's off-gate law is a junction law.** Real CMOS pass gates add
  drain-induced barrier lowering, gate leakage and junction leakage, none of
  which are in an ideal-diode model; 262 mV is a floor, not a design value.
* **Endurance and read disturb are absent** from E3's storage-class argument.
* **One trellis size (L = 4), one γ (5), one fan-in (256).** The scaling laws
  are stated analytically but only spot-checked.
* **Nothing here is silicon.** No PDK, no memristor compact model, no layout.
