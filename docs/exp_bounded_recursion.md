# X3 bounded recursion: latency and energy of the B(·) reclassifications

**Experiment for `missing_primes_mapping.md` §3 (X3) and §6 rows `L5 MCTS` →
B(M nodes, d) and `P5* Beam search` → B(B, L).**

Date: 2026-09-15. Harness: `spice/bounded_recursion_latency.py`, ngspice-46,
numpy 2.4. Fidelity level: **junction-exact, wiring-ideal** (`spice/README.md`)
— every exponential current comes from a real diode I–V law; current mirrors,
comparators and gating are ideal behavioural elements with *stated* delays.

---

## 0. Pre-registration block

*Written and committed to file before the first ngspice run. Everything below
this line up to §1 is the pre-registration; deviations are listed in §1.1.*

### 0.1 Claim under test

§6 of the mapping doc reclassifies two `U` (fundamentally unmappable) rows as
bounded:

| row | now | proposed | stated mechanism |
|---|---|---|---|
| P5\* Beam search | U | B(B, L) | race ranking + gates |
| L5 MCTS | U | B(M nodes, d) | memristive adjacency + P5 + sequential backup; *"Latency is sequential, not impossible."* |

"Latency is sequential, not impossible" is an existence argument, not a number.
This experiment asks whether the sequential latency, costed with the paper's own
transition constants, is **competitive** or merely **finite**. A reclassification
that yields a mappable-but-1000×-slower circuit is a bookkeeping change, not an
engineering claim; the mapping doc itself (§7) concedes that area and rewrite
latency "may be prohibitive". This is the test of exactly that concession.

### 0.2 Cost constants fixed before running (all sourced)

| symbol | value | source |
|---|---|---|
| `E_ADC`, `t_ADC` | 100 fJ, 1 ns | `prime_compiler_v2.tex` §4.4 Phase 4, transition cost list |
| `E_DAC`, `t_DAC` | 50 fJ, 0.5 ns | ibid. |
| `t_AGC` | 4.4 ns cold start, 5.3 ns after a 2.5× input step | ibid. §5.3 Result 3 (ngspice, 1 pF integrator) |
| `t_AGC,softmax` | 3.2 ns cold, 2.1 ns after a logit switch | ibid. §5.3 Result 5 |
| `t_translin` | ~12 ns open-loop translinear pipeline propagation | ibid. §5.2 |
| `t_wr` | memristor/RRAM write, 10 ns … 1 µs (swept, not assumed) | task-stated range; bracketed by the 768 ns RRAM-crossbar *read* of Molom-Ochir et al. (see 0.5) |
| `V_T` | 0.025864890 V | `spice/README.md` (ngspice SPICE3-legacy kT/q at 27 °C) — mandatory for any externally driven junction |
| `t_cmp` | **200 ps** WTA/comparator detection delay, declared here, modelled as a first-order lag in the netlist | design constant, stated not measured |

Race-cell design constants (declared here so the SPICE numbers are falsifiable):
`C_int = 10 fF`, `V_th = 0.2 V`, `I_max = 2 µA` (µA scale as `spice/README.md`
demands, to keep ngspice `GMIN = 1e-12 S` out of the signal path), diode
`IS = 1e-14 A`, `N = 1` nominal. Hence the fastest arrival
`t_1 = C·V_th/I_max = 1.0 ns`.

### 0.3 Model, Part A — beam search B(B, V, L)

Per decoding step, B beams, vocabulary V, `N = B·V` candidates:

1. **Score** — B parallel crossbar score vectors of length V (P1·P2), then the
   exponential stage (P3). Analog end to end.
2. **Select** — one race over `N = B·V` exponential elements; the *ordering* is
   the arrival order (multi-computation Op. 3, `multi_computation_v2.tex` §3.3:
   a single race yields the complete rank ordering, readout O(N) — here only the
   first B arrivals are read, so readout is O(B)).
3. **Copy** — the B surviving hypothesis states are moved into the next step's
   slots (P11 gate on a bounded slot memory).

Pre-registered arrival law (to be confirmed or refuted by SPICE, not assumed):
with `I_i = I_S exp(z_i)` charging `C` to `V_th`,

    t_i = C·V_th / I_i  =  t_1 · exp(z_max − z_i)

so the **B-th arrival is exponential in the logit gap**:

    t_(B) = t_1 · exp(g_1B),   g_1B = z_(1) − z_(B)

and the per-step analog latency is

    t_step = t_DAC + t_score + t_(B) + t_cmp + t_readout + t_copy

with `t_readout = B·t_ADC` (serial) or `t_ADC` (B parallel TDC channels), and
`t_copy ∈ {~1 ns switched-capacitor slot, t_wr memristive slot}`.

**Where ADC/DAC are unavoidable (stated in advance):** the argmax *index* is
digital by nature — a beam id is a symbol, not a voltage. One ADC-equivalent
(index latch/TDC read, costed at `E_ADC`) per selected beam per step is
irreducible: **B ADC per step, not B·V**. Conversely, if the logits arrive from a
*digital* matmul, `B·V` DACs per step are needed to drive the race, and the race
loses. Pre-registered prediction: the race is only worth it when the score stage
is already analog; this is the beam-search analogue of the
`exp_kv_leak_eviction.md` §8 finding ("the score is free, the selection costs one
P5 race").

Pre-registered energy law: total race charge = `V_dd · t_(B) · Σ_i I_i`, i.e.

    E_race = V_dd · C · V_th · Z_norm · exp(g_1B),   Z_norm = Σ_i exp(z_i − z_max)

— **independent of N except through the partition function** `Z_norm`. To be
checked in SPICE by integrating the sense-source currents.

### 0.4 Model, Part B — MCTS B(M, d, b, K)

Per simulation, branching factor b, depth d, node pool M:

* **Selection**: d sequential races over ≤ b children each; level k starts when
  level k−1's winner is detected. `t_sel = d·(t_race,level + t_cmp)`.
* **Expansion**: one adjacency-row write (`t_wr`) + free-slot allocation by race
  over free flags (one race over M flags, t_1-scale).
* **Rollout**: out of scope (policy network, already mappable).
* **Backup**: two variants, both analysed —
  (a) **sequential**: d P1·P8 accumulator updates, `t_bk,seq = d·(t_pulse + t_cmp)`;
  (b) **parallel one-hot**: the selected path is held as a one-hot vector on the
  slot memory, so a single current pulse updates all d ancestors at once,
  `t_bk,par = t_pulse`. Variant (b) is the interesting one: it converts an
  O(d) recursion into O(1), which is exactly the "pay in space" row of §3.
* Per move: `K = 800` simulations (AlphaZero, Silver et al., *Science* 362, 1140
  (2018)) — the simulations are strictly sequential (each reads statistics the
  previous one wrote).

### 0.5 External anchors (fetched before running)

* **Molom-Ochir, Morris, He, Gajjar, Pedretti, Li, Chen, Ignowski, Natarajan,
  "Multi-primitive in-memory computing for Monte Carlo tree search",
  arXiv:2607.22869 (24 Jul 2026)** — 22 nm, fabricated RRAM-array parameters,
  500 MHz digital clock. Reported: **≈ 2,400 iterations/s sustained at 9×9 Go**
  (18,471 it/s at 2×2, 510 it/s at 19×19) ⇒ **≈ 417 µs per simulation**;
  RRAM crossbar read latency **768 ns** (256 ns wordline charging + 512 ns ADC);
  ~60 mW sustained during a 5,000-iteration-per-move search; 96× energy
  efficiency over a CPU, 65×–2,059× over an H100; **3.4× latency advantage over
  the batched H100 baseline**. The authors state explicitly that the 2,400 it/s
  "is set by MCTS's strict sequential dependency between iterations …, not by
  single-iteration pipeline latency".
* **CPU reference**: single-core 9×9 Go MCTS on the order of 10–50 µs per
  simulation (Pachi/Michi class engines; Pachi's own documentation reports
  playouts scaling with threads and 10,000-rollout configurations as the
  standard strength setting). Used as the *stricter* of the two anchors.
* **Digital beam-search reference**: per-output-token latency of a 7 B model on
  an A100 80 GB under vLLM measured at **TPOT = 14.96 ms** (DatabaseMart vLLM
  A100 80GB benchmark, DeepSeek-R1-Distill-Qwen-7B); commonly quoted range
  20–30 ms. The *beam-specific* per-step overhead (a top-k over B·V = 128 k
  fp16 logits plus the beam-state gather) is memory-bound: 128 k × 2 B = 256 kB
  at ~2 TB/s ≈ 0.13 µs of traffic plus CUDA kernel-launch overhead of a few µs.
  **Pre-registered digital reference: 1 µs per step** — deliberately the most
  GPU-favourable defensible number, so that the kill bar is hard to clear.

### 0.6 PRE-REGISTERED KILL CRITERIA

**KA1** — If the modelled analog per-step latency at (B = 4, V = 32 768,
L = 64) exceeds the digital reference (**1 µs** per step, §0.5) by more than
**10×**, i.e. > 10 µs, then "bounded-mappable with sequential latency" is a
hollow reclassification for beam search and the §6 row `P5*` must carry the
latency figure, not just the bound tuple.

**KA2** — If the top-B *set* is wrong in **> 10 %** of Monte-Carlo draws at a
**selection-boundary logit gap of 1 nat** (`g_sel = z_(B) − z_(B+1) = 1`) under
MOS-typical mismatch (**3 % σ on I_S, 0.3 % σ on n**), the race ranking is not a
usable P5 for beam selection and §6's "race ranking" mechanism is refuted.
*Definition fixed in advance*: the gap that matters is the one straddling the
selection boundary (B-th vs (B+1)-th), not the top-1 gap.

**KB1** — If the modelled analog per-simulation latency exceeds **10×** the
Molom-Ochir anchor (417 µs ⇒ bar at 4.17 ms) **or** 10× the stricter CPU
reference (10 µs ⇒ bar at 100 µs), the B(M, d) reclassification is hollow for
MCTS. Both bars are reported; the **CPU bar (100 µs) is the binding one**.

**KB2** — If the expansion-step memristor write is **> 50 %** of per-simulation
time at *any* write speed in the published 10 ns … 1 µs range, then the bound is
**write latency, not recursion depth**, and the §6 `L5` row must be restated in
those terms. (This criterion is designed to trigger; the question it settles is
*where* in the range the crossover sits.)

### 0.7 Sweeps fixed in advance

* Part A SPICE: `N = B·V ∈ {64, 256, 1024}`, `B = 4`; top-gap sweep
  `g_1B ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 4.0}` nats; MC 200 draws per
  (N, `g_sel`) with `g_sel ∈ {0.1, 0.25, 0.5, 1.0, 2.0}`.
* Part B SPICE: `d ∈ {8, 16, 32}`, `b = 8`; nominal + 200 MC draws per d.
* Analytic cost model evaluated over `t_wr ∈ {10, 30, 100, 300, 1000} ns`.
* Seed 2026 throughout (house convention).

### 0.8 What this experiment cannot settle

Declared in advance: this is a latency/energy model with one circuit check per
algorithm at the wiring-ideal level. It says nothing about (a) area — a 128 k
race array with 10 fF per cell is 1.28 nF of capacitance and is almost certainly
the real blocker for beam search; (b) whether the analog scores are accurate
enough for *search quality* (only for *ranking correctness*); (c) routing,
clocking or the M×M adjacency array's own read disturb.

---

## 1. Method and deviations

`spice/bounded_recursion_latency.py`, seed 2026, ngspice-46, 8 workers,
total wall clock **362 s**. Raw numbers in `spice/out/bounded_recursion.json`,
console log in `spice/out/bounded_recursion_run.log`.

**Race cell (Parts A and B share it).** Each candidate is a diode driven by
`V_i = n·V_T·z_i + V_off`, `V_off = n·V_T·ln(I_max/I_S) = 0.4943 V`, so the
junction law gives `I_i = I_S·exp(V_i/(n·V_T)) = I_max·exp(z_i)`. An ideal
mirror (`B`-source reading the sense current) charges `C_int = 10 fF`; the
element "fires" when its node reaches `V_th = 200 mV`. `I_max = 2 µA` keeps the
tail elements at nA–µA, i.e. out of ngspice's `GMIN = 1e-12 S` leakage floor
(`spice/README.md`); `V_T = 0.025864890` as required for an externally driven
junction.

**Part B chain.** Level *k*'s eight children are gated by
`u(v(en_k) − 0.5)`; the level's WTA output is `u(max_j v(c_kj) − V_th)` passed
through an R–C lag with `τ = t_cmp/ln2 = 288.5 ps`, so the 50 % crossing of
`en_{k+1}` lands exactly `t_cmp = 200 ps` after the winner crosses `V_th`.
That lag **is** the stated comparator delay; it is a declared design constant,
not a measured one.

### 1.1 Deviations from the pre-registration

1. **A5 added.** `N = B·V = 131 072` is out of ngspice budget, so the
   real-vocabulary set-correctness point is computed in numpy using the
   arrival law `t_i = C·V_th/I_i` that A1 verifies in SPICE to 0.002 %, with
   the same mismatch model. It is an extrapolation of a SPICE-verified law, not
   a SPICE result, and is labelled as such.
2. **A2 tail currents.** Elements that do not fire inside `tstop` get their
   current from the analytic `I_max·exp(z_i)` rather than from `C·V_th/t_i`.
   This affects only the sub-fJ tail of the energy sum; the SPICE-vs-law
   agreement in §3.2 is to four digits, which the substitution cannot manufacture
   (the fired elements dominate the sum).
3. Nothing else. All sweeps, kill bars, constants and the `g_sel` definition are
   as pre-registered in §0.

---

## 2. Cost model A — beam search

### 2.1 Per-step cost, stage by stage

`N = B·V`. Every constant is from §0.2.

| # | stage | prime | latency | energy | domain |
|---|---|---|---|---|---|
| 1 | beam state → crossbar drive | — | `t_DAC = 0.5 ns` (all lines in parallel) | `B·d_model·E_DAC` **if the slot memory is digital**; 0 if analog | DAC, *avoidable* |
| 2 | B score vectors, V each | P1·P2 | folded into 3 | — | analog |
| 3 | exponential / sum-feedback stage | P3 (+AGC) | `t_softmax = 3.2 ns` cold, 2.1 ns after a logit switch | — | analog |
| 4 | race to the B-th arrival | P5 | `t_(B) = t_1·exp(g_1B)`, `t_1 = 1.0000 ns` **measured** | `V_dd·C·V_th·Z_norm·exp(g_1B)` **measured** | analog |
| 5 | WTA detection | — | `t_cmp = 0.2 ns` | — | analog |
| 6 | index readout, B first arrivals | — | `B·t_ADC = 4 ns` serial, or `t_ADC = 1 ns` with B TDC channels | `B·E_ADC = 400 fJ` | **ADC, unavoidable** |
| 7 | copy B hypothesis states into next slots | P11 | `1 ns` switched-cap, or `t_wr = 10 ns…1 µs` memristive | — | analog |

**Where ADC/DAC are unavoidable.** Stage 6 only. A beam id is a symbol, not a
voltage, so exactly **B ADC-equivalents per step** are irreducible — *not* `B·V`.
That is the whole point of the race: it replaces `N = 131 072` conversions by
`B = 4`. Stage 1 is avoidable iff the slot memory is analog; if the LM head is
digital, `N` DACs per step are needed to *enter* the race and the accounting
reverses (§2.3).

### 2.2 Per-step latency, B = 4, V = 32 768, L = 64 (N = 131 072)

| `g_1B` | `t_(B)` / ns | `t_step`, sw-cap slot + serial ADC | + parallel TDC | + 10 ns memristive slot | + 1 µs memristive slot |
|---|---|---|---|---|---|
| 0.25 | 1.284 | **10.18 ns** | 7.18 ns | 19.18 ns | 1009.18 ns |
| 0.5 | 1.649 | **10.55 ns** | 7.55 ns | 19.55 ns | 1009.55 ns |
| 1.0 | 2.718 | **11.62 ns** | 8.62 ns | 20.62 ns | 1010.62 ns |
| 2.0 | 7.389 | **16.29 ns** | 13.29 ns | 25.29 ns | 1015.29 ns |

Over `L = 64` steps: **0.65–1.04 µs** of beam-search-specific latency with
capacitive slots; 64.6–65.0 µs if each hypothesis copy is a 1 µs memristive
write.

### 2.3 Per-step energy, and the direction the accounting runs

`E_race = V_dd·C_int·V_th·Z_norm·exp(g_1B)` (derived in §0.3, verified in §3.2).
`Z_norm = Σ exp(z_i − z_max) = 1/p_max` is the partition function over all `B·V`
candidates — **the race energy does not scale with N**.

| `Z_norm` | `g_1B` | `E_race` | B ADC (stage 6) | N ADC if every logit were digitised | N DAC if the logits arrive digital |
|---|---|---|---|---|---|
| 10 | 0.25 | 20.5 fJ | 400 fJ | 13.107 nJ | 6.554 nJ |
| 50 | 1.0 | 217.5 fJ | 400 fJ | 13.107 nJ | 6.554 nJ |
| 200 | 1.0 | 869.9 fJ | 400 fJ | 13.107 nJ | 6.554 nJ |
| 1000 | 1.0 | 4.35 pJ | 400 fJ | 13.107 nJ | 6.554 nJ |

* **Inside an analog logit path**: race + B ADC costs 0.4–4.7 pJ against 13.1 nJ
  for the digitise-everything baseline — **2 760× to 31 167×**.
* **Outside one** (digital LM head): you must spend `N·E_DAC = 6.554 nJ` to get
  into the race in order to avoid `N·E_ADC = 13.107 nJ`. Net saving **2.0×**,
  bought with a 131 072-cell race array (1.311 nF of integration capacitance).
  **Not worth it.**

This is the beam-search form of the `exp_kv_leak_eviction.md` §8 result. There
the finding was *"the score is free, the selection costs one P5 race"*; here it
is the mirror image: **the selection is nearly free, but only if the score is
already analog.** The race is a *fusion* optimisation, not a standalone one.

---

## 3. Circuit check A — ngspice race, N ∈ {64, 256, 1024}

### 3.1 Arrival law: 1st vs B-th arrival (B = 4), ideal devices

| N | `g_1B` | `t_(1)` / ns | `t_(B)` / ns | `t_(B) − t_(1)` / ns | `t_(B)/t_(1)` | `exp(g_1B)` | dev |
|---|---|---|---|---|---|---|---|
| 64 | 0.10 | 1.0000 | 1.1052 | 0.1052 | 1.1052 | 1.1052 | +0.002 % |
| 64 | 0.25 | 1.0000 | 1.2840 | 0.2840 | 1.2840 | 1.2840 | +0.002 % |
| 64 | 0.50 | 1.0000 | 1.6487 | 0.6487 | 1.6487 | 1.6487 | +0.002 % |
| 64 | 1.00 | 1.0000 | 2.7183 | 1.7183 | 2.7183 | 2.7183 | +0.002 % |
| 64 | 2.00 | 1.0000 | 7.3892 | 6.3892 | 7.3890 | 7.3891 | +0.002 % |
| 64 | 4.00 | 1.0000 | 54.5984 | 53.5984 | 54.5973 | 54.5982 | +0.000 % |
| 256 | 0.10 … 4.00 | 1.0000 | identical to N = 64 to 5 digits | | | | ≤ 0.002 % |
| 1024 | 0.10 … 4.00 | 1.0000 | identical to N = 64 to 5 digits | | | | ≤ 0.002 % |

Two results, both load-bearing:

1. **`t_(1)` is exactly N-independent** (1.0000 ns at N = 64, 256 and 1024).
   The race's *winner* latency is set by one current, not by the field size.
   This is the property that makes free-slot allocation over an M-entry pool
   (Part B) an O(1) operation.
2. **`t_(B) = t_(1)·exp(g_1B)` to 0.002 %.** The top-B readout latency is
   **exponential in the logit gap**, not constant. A 4-nat gap costs 54.6 ns for
   the 4th arrival where a 0.1-nat gap costs 1.1 ns. The race is fast *because*
   the candidates are close, which is precisely the regime beam search operates
   in — but the compiler cannot assume it (§8.1).

### 3.2 Energy law

| N | `g_1B` | `Z_norm` | `E_race` SPICE / fJ | `E_race` law / fJ | N·E_ADC / fJ | ratio |
|---|---|---|---|---|---|---|
| 64 | 0.25 | 16.42 | 33.741 | 33.741 | 6 400 | 190× |
| 64 | 1.00 | 9.02 | 39.220 | 39.219 | 6 400 | 163× |
| 64 | 2.00 | 4.32 | 51.050 | 51.050 | 6 400 | 125× |
| 256 | 0.25 | 64.80 | 133.135 | 133.134 | 25 600 | 192× |
| 256 | 1.00 | 30.17 | 131.219 | 131.218 | 25 600 | 195× |
| 256 | 2.00 | 11.68 | 138.136 | 138.135 | 25 600 | 185× |
| 1024 | 0.25 | 244.34 | 501.975 | 501.974 | 102 400 | 204× |
| 1024 | 1.00 | 116.60 | 507.122 | 507.119 | 102 400 | 202× |
| 1024 | 2.00 | 43.24 | 511.143 | 511.140 | 102 400 | 200× |

`E_race = V_dd·C·V_th·Z_norm·exp(g_1B)` holds to **six significant figures**.
Note the 1024-row block: `Z_norm` falls 5.7× from `g = 0.25` to `g = 2.0` while
`exp(g)` rises 5.6×, and the measured energy stays at ~505 fJ. The race spends
its energy on the partition function, and the partition function is exactly the
quantity the multi-computation paper reads off the *winner time*
(`multi_computation_v2.tex` Op. 2). *Observation, not a validated claim*: the
race's supply charge is a sixth single-race observable, `Q/(V_dd·C·V_th) =
Z_norm·e^{g_1B}`.

### 3.3 Monte-Carlo mismatch: is the top-B **set** right?

3 % σ on `I_S`, 0.3 % σ on `n`, 200 draws per cell, B = 4,
`g_sel = z_(B) − z_(B+1)`:

| N | 0.10 | 0.25 | 0.50 | **1.00** | 2.00 |
|---|---|---|---|---|---|
| 64 | 0.340 | 0.970 | 1.000 | **1.000** | 1.000 |
| 256 | 0.130 | 0.915 | 1.000 | **1.000** | 1.000 |
| 1024 | 0.020 | 0.830 | 1.000 | **1.000** | 1.000 |
| 131 072 (A5, extrapolated) | 0.000 | 0.203 | 1.000 | **1.000** | 1.000 |

**Error budget (A4), which explains the table.** The mismatch enters the
*log* domain:

    ln(I_i/I_nom) ≈ ln(1 + δ_Is) + (z_i + ln(I_max/I_S))·(1/n_i − 1)

* 3 % `I_S` mismatch → **0.0300 nat** of effective-logit error.
* 0.3 % `n` mismatch → **0.0573 nat**, because it is multiplied by the bias
  offset `ln(I_max/I_S) = 19.11`. **The 0.3 % ideality mismatch hurts twice as
  much as the 3 % saturation-current mismatch**, and it gets worse the harder
  the diodes are biased. This is the design constraint the race imposes and it
  is not in the mapping doc.
* Total σ = **0.0647 nat** per element, **0.0915 nat** pairwise.

A 1-nat gap is 10.9 pairwise σ; a 0.25-nat gap is 2.7 σ, and with 131 072
competitors piled up below the boundary, 2.7 σ is not enough (0.203). The
usable floor is `g_sel ≳ 0.5 nat` at real vocabulary size.

---

## 4. Cost model B — MCTS

### 4.1 Per-simulation cost

Branching `b = 8`, depth `d`, pool `M`; per-level race latency **measured**
at 1.2000 ns (§5.1).

| phase | prime | latency | note |
|---|---|---|---|
| Selection (UCB argmax down d levels) | P5 × d, sequential | `d · 1.2000 ns` | 9.60 / 19.20 / 38.40 ns for d = 8/16/32 |
| Free-slot allocation | P5 over M free flags | `t_1 + t_cmp = 1.2 ns` | **independent of M** (§3.1 result 1) |
| Expansion: one adjacency-row write | X2 mechanism 2 | `t_wr = 10 ns … 1 µs` | the swept unknown |
| Backup (a) sequential | P1·P8 × d | `d · 1.2 ns` | |
| Backup (b) parallel one-hot | P1·P8 × 1 | `1.2 ns` | path held as a one-hot on the slot memory |
| Rollout | — | excluded by scope | |

### 4.2 Per-simulation latency and the write fraction

`t_sim` in ns, and the expansion write's share of it:

| d | backup | `t_wr` = 10 ns | 30 ns | 100 ns | 300 ns | 1 µs |
|---|---|---|---|---|---|---|
| 8 | sequential | 30.4 (33 %) | 50.4 (**60 %**) | 120.4 (83 %) | 320.4 (94 %) | 1020.4 (98 %) |
| 8 | parallel | 22.0 (**46 %**) | 42.0 (**71 %**) | 112.0 (89 %) | 312.0 (96 %) | 1012.0 (99 %) |
| 16 | sequential | 49.6 (20 %) | 69.6 (43 %) | 139.6 (**72 %**) | 339.6 (88 %) | 1039.6 (96 %) |
| 16 | parallel | 31.6 (32 %) | 51.6 (**58 %**) | 121.6 (82 %) | 321.6 (93 %) | 1021.6 (98 %) |
| 32 | sequential | 88.0 (11 %) | 108.0 (28 %) | 178.0 (**56 %**) | 378.0 (79 %) | 1078.0 (93 %) |
| 32 | parallel | 50.8 (20 %) | 70.8 (42 %) | 140.8 (**71 %**) | 340.8 (88 %) | 1040.8 (96 %) |

The 50 % crossover is at `t_wr = t_sel + t_alloc + t_bk`, i.e.

| d | parallel backup | sequential backup |
|---|---|---|
| 8 | **12.0 ns** | 20.4 ns |
| 16 | **21.6 ns** | 39.6 ns |
| 32 | **40.8 ns** | 78.0 ns |

**Everything to the right of 12–78 ns is write-bound, not recursion-bound.**

Per move at K = 800 (AlphaZero): **17.6 µs** (d = 8, parallel backup, 10 ns
write) to **862 µs** (d = 32, sequential backup, 1 µs write).

### 4.3 The parallel-backup trick is worth less than it looks

One-hot parallel backup removes `(d−1)·1.2 ns`: 8.4 ns at d = 8, 37.2 ns at
d = 32. It converts an O(d) recursion into O(1) — exactly the "pay in space"
row of mapping §3 — but it only matters when `t_wr ≲ 30 ns`. Above that the
write swamps the saving. **The interesting architectural idea in the mapping
doc's MCTS row is the one that the dominant cost makes irrelevant.**

### 4.4 A requirement the mapping doc does not state

Each level's race needs the children's UCB values *as currents*. If the
Q/N accumulators live anywhere else, every simulation must re-drive `b·d = 256`
DAC channels (256 × 50 fJ = 12.8 pJ and 256 settle times). The design only works
if **the P8 accumulator node and the P5 race driver are the same physical node**
— the capacitor that integrates the backup charge is the gate that sets the
selection current. Backup then costs one current pulse and selection costs no
conversion at all. This is a hard structural constraint, of the same kind as the
reciprocity requirement the doc flags for the X4 gradient family (§7 of the
mapping doc).

---

## 5. Circuit check B — ngspice d-level sequential race chain (b = 8)

### 5.1 Nominal

| d | `t_sel` SPICE / ns | per level / ns | model `d·(t_1 + t_cmp)` / ns | dev |
|---|---|---|---|---|
| 8 | 9.5998 | 1.20000 | 9.6000 | −0.002 % |
| 16 | 19.1997 | 1.19998 | 19.2000 | −0.002 % |
| 32 | 38.3995 | 1.19998 | 38.4000 | −0.001 % |

The cascade is exactly additive: each level contributes one race (1.0 ns) plus
one stated comparator delay (0.2 ns), with **no accumulation of settling error**
across 32 levels. Nothing in the chain degrades with depth — the "sequential,
not impossible" claim of mapping §3 is correct *at this level of the model*.

### 5.2 Under mismatch (3 % `I_S`, 0.3 % `n`, 200 draws)

| d | mean / ns | sd / ns | CV | p5 | p95 |
|---|---|---|---|---|---|
| 8 | 9.6046 | 0.1758 | 1.83 % | 9.318 | 9.914 |
| 16 | 19.2379 | 0.2408 | 1.25 % | 18.884 | 19.601 |
| 32 | 38.4541 | 0.3619 | 0.94 % | 37.870 | 39.015 |

The spread **falls** with depth (1.83 % → 0.94 %, close to the 1/√d that
independent per-level errors predict): a deep chain averages its own mismatch.
Mean latency is biased high by only +0.05 % to +0.14 % — the winner of a race is
biased towards the *positively* mismatched element, so the winning current is
slightly above nominal and the bias is, if anything, favourable.

*Caveat stated plainly*: mismatch also changes **which** child wins when two UCB
values are within ~0.09 nat. In MCTS that is a tie-break perturbation on an
algorithm that is already stochastic by construction, so it is benign for search
quality in a way the beam-search top-B set is not; this experiment does not
measure search quality (§0.8).

---

## 6. External anchors

| anchor | value | our number | ratio |
|---|---|---|---|
| Molom-Ochir et al., arXiv:2607.22869 — 9×9 Go, 22 nm, fabricated RRAM params | ≈ 2 400 iterations/s ⇒ **417 µs / simulation** | 22–1078 ns / simulation | **387× – 19 000× faster** |
| same, RRAM crossbar read | 768 ns (256 ns wordline + 512 ns ADC) | our whole d = 32 selection chain: 38.4 ns | 20× |
| same, energy | ~60 mW sustained, 96× over CPU, 65–2059× over H100 | not modelled here | — |
| CPU 9×9 Go MCTS (Pachi/Michi class) | ~10–50 µs / simulation | 22–1078 ns | **9× – 2 270× faster** |
| 7 B LLM decode, A100 + vLLM | TPOT **14.96 ms** / token | — | — |
| beam-specific GPU step overhead (top-k over 131 072 fp16 + gather) | ≈ 1 µs (pre-registered, GPU-favourable) | 10.2–16.3 ns (cap slots) | **61× – 98× faster** |

**Honest scope warning on the MCTS comparison.** The 387×–19 000× is *not*
like-for-like. Molom-Ochir's 417 µs is a full sustained iteration including the
rollout (their 768 ns RRAM crossbar read, repeated down a playout) and all
tree-statistics traffic; ours is selection + allocation + expansion + backup
with the rollout excluded by scope (§0.4) and the UCB arithmetic assumed to live
on the accumulator node (§4.4). The defensible statement is narrower and still
useful: **the phases the mapping doc calls "sequential recursion" cost tens of
nanoseconds, i.e. they are not where an MCTS accelerator's time goes.** Their own
paper says the same thing from the other side — the 2 400 it/s "is set by MCTS's
strict sequential dependency between iterations …, not by single-iteration
pipeline latency".

**Corroboration of KB2 from the built chip.** Molom-Ochir map expansion to
*combinational logic* and backpropagation to *SRAM*, using RRAM only for the
read-mostly rollout crossbar. A fabricated multi-primitive MCTS engine
therefore deliberately keeps the mutable tree out of the resistive-write path.
That is exactly the conclusion §4.2 reaches from the cost model.

---

## 7. Verdicts against the pre-registered kill criteria

**KA1 — NOT killed, with room to spare.** Per-step analog latency at
(B = 4, V = 32 768, L = 64) is **10.18–16.29 ns** with capacitive hypothesis
slots and serial B-ADC readout — **61× to 98× below** the pre-registered
GPU-favourable digital reference of 1 µs, and 600× below the 10 µs kill bar.
Even the worst swept configuration (1 µs memristive slot write) lands at
1.015 µs, still an order of magnitude under the bar. Over L = 64 steps the
beam-search-specific latency is 0.65–1.04 µs, against a 7 B model's own
14.96 ms *per token*. For beam search, "bounded-mappable with sequential
latency" is **not** a hollow reclassification: the sequential latency is
negligible against the forward pass it sits inside.

**KA2 — NOT killed.** At `g_sel = 1 nat` the top-4 set is correct in
**200/200 draws at N = 64, 256 and 1024** (and in 400/400 numpy draws at
N = 131 072). The bar was > 10 % failures; observed 0 %. The race ranking is a
usable P5 for beam selection. The criterion was, in hindsight, generous: the
binding number is the *floor*, `g_sel ≈ 0.5 nat` at real vocabulary size
(0.203 correct at 0.25 nat, 0.000 at 0.1 nat), and the dominant error source is
the 0.3 % ideality mismatch amplified by the 19.1-nat bias offset, not the 3 %
saturation-current mismatch.

*A caveat that the multi-computation paper flags and that does not apply here.*
`multi_computation_v2.tex` §3.3 reports that per-trial exact top-K set match is
**low** for an sMTJ race, because a thermally-activated race is stochastic and
the ranking is correct only *in distribution*. The race simulated here is the
**deterministic** subthreshold-diode race of the prime compiler's softmax stage:
arrival times are `C·V_th/I_i` exactly, so the ordering is exact up to device
mismatch. **The two P5 realisations are not interchangeable for beam search.**
Op. 3's "beam search without sorting" holds for the deterministic race; for the
sMTJ race it requires repeated races. The mapping doc's `P5*` row should say
which race it means.

**KB1 — NOT killed, at any point of the swept write range.** Analog
per-simulation latency is **22.0–1078.0 ns** (d = 8…32, `t_wr` = 10 ns…1 µs,
both backup variants). Against the Molom-Ochir anchor (417 µs, bar 4.17 ms) it
is 387×–19 000× under; against the stricter CPU anchor (10 µs, bar 100 µs) it is
9×–450× under. Per move at K = 800: 17.6–862 µs. The B(M, d) reclassification
is not hollow on latency grounds — **but see the scope warning in §6, which is
where the honest uncertainty in this verdict lives.**

**KB2 — TRIGGERED, as designed.** The expansion write exceeds 50 % of
per-simulation time for every `t_wr` above **12.0 ns** (d = 8, parallel backup)
to **78.0 ns** (d = 32, sequential backup). At a 100 ns write — a conservative
but entirely ordinary RRAM programming time — the write is 56–89 % of the
simulation; at 1 µs it is 93–99 %. Only the fastest end of the published
10 ns–1 µs range leaves the recursion in charge, and only for deep trees.

**Restatement required by KB2, in the mapping doc's terms.** The §6 row

> `L5 | MCTS | U | B(M nodes, d) | memristive adjacency + P5 + sequential backup`

should become

> `L5 | MCTS | U | B(M nodes, d, t_wr) | memristive adjacency + P5 + sequential backup; the binding parameter is the adjacency-row write time t_wr, not the depth d — selection+backup cost d·1.2 ns (SPICE, b=8, d≤32) while one row write costs 10 ns–1 µs. Above t_wr ≈ 12–78 ns the mapping is write-bound. Cross-check: the built IMC-MCTS chip (arXiv:2607.22869) puts expansion in logic and backup in SRAM, keeping the mutable tree out of the resistive-write path.`

and mapping §3's sentence *"Latency is sequential, not impossible"* should read
*"Latency is sequential and small; the cost is the write, not the recursion."*

---

## 8. What the compiler must know at compile time

### 8.1 Beam search — the bound tuple `B(B, L)` is incomplete

`B(B, L)` names the two static parameters and omits the two that actually govern
the circuit. The compiler needs **`B(B, V, L; g_floor, g_ceiling, slot_class)`**.
`V` enters through array area (131 072 cells × 10 fF = **1.311 nF** of
integration capacitance) and through the `Z_norm`-scaled race energy, not
through latency — `t_(1)` is N-independent (§3.1). The two new entries are
data-dependent and therefore must be *budgeted*, not computed: `g_ceiling` is
the largest top-1-to-top-B logit gap the design will wait for, and it fixes the
race time-out `t_max = t_1·exp(g_ceiling)` (4 nats already costs 54.6 ns), with a
declared fallback for the steps where fewer than B elements have fired;
`g_floor` is the smallest selection-boundary gap the design will trust, and it
must satisfy `g_floor ≳ 6·σ_tot·√2`, where σ_tot = 0.0647 nat is set by the 0.3 %
ideality mismatch times the bias offset `ln(I_max/I_S)` — at real vocabulary size
that is `g_floor ≈ 0.5 nat`, below which the compiler must fall back to a digital
tie-break over the near-boundary candidates. `slot_class` (switched-capacitor vs
memristive) is the single largest latency lever, worth 100×. Finally the compiler
must verify a *placement* precondition, not a bound: the logit stage must be
analog, or the `N` DACs needed to enter the race cancel the `N` ADCs it saves
(§2.3).

### 8.2 MCTS — the bound tuple `B(M, d)` names the wrong parameter

The compiler needs **`B(M, d, b; t_wr, backup_mode)`**. `M` is an area bound
(the M×M adjacency array) and contributes **nothing to latency**: free-slot
allocation is one race over M flags and the winner's arrival time is
M-independent to 5 digits (§3.1). `d` and `b` set the selection and
sequential-backup chains at a measured 1.2000 ns per level, mismatch-tight
(CV 0.94 % at d = 32) and non-accumulating — i.e. the "unbounded recursion"
that put MCTS in the `U` column costs 38 ns at depth 32. What the compiler must
actually pin down is `t_wr`, the adjacency-row write time of the chosen device,
because it crosses 50 % of the simulation budget at 12–78 ns and reaches 99 %
at 1 µs; and `backup_mode`, since the one-hot parallel backup that collapses the
O(d) chain to O(1) is only worth choosing when `t_wr ≲ 30 ns`. Two structural
preconditions come with the tuple and are not bounds at all: the P8 accumulator
node must *be* the P5 race driver (§4.4), or 256 DAC channels reappear per
simulation; and the tree must be mutable at the write speed the search demands,
which is why the one fabricated multi-primitive MCTS engine keeps expansion in
logic and backup in SRAM.

---

## 9. Limitations

* Wiring-ideal: mirrors, gating, the WTA threshold and the 200 ps comparator
  are behavioural. A real WTA over 131 072 lines has its own fan-in problem that
  this model does not touch, and 200 ps is asserted, not designed.
* Area is out of scope and is the most likely real blocker for beam search:
  1.311 nF of integration capacitance and 131 072 diodes for one decode step.
* The MCTS model excludes the rollout and assumes UCB values are already
  currents on the accumulator nodes; §6 states what that does to the comparison.
* The Monte Carlo covers device mismatch only — no thermal noise, no supply
  noise, no crossbar IR drop, no read disturb on the adjacency array.
* Search quality is not measured for either algorithm, only ranking correctness
  (A) and latency spread (B).
* A5 (N = 131 072) is a numpy extrapolation of a law verified in SPICE at
  N ≤ 1024, not a SPICE result.

## 10. Reproducing

```bash
cd spice
OPENBLAS_NUM_THREADS=1 python3 bounded_recursion_latency.py --part all --mc-runs 200 --workers 8
# ~6 min wall clock, 8 cores; writes out/bounded_recursion.json
```
