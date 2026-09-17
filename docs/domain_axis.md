# The signal-domain axis: a second compiler choice below analog-vs-digital

*Draft 2026-09-16. Companion to `prime_compiler_v2.1.tex` and `docs/missing_primes_mapping.md`.
Status: analytic cost model calibrated against three measured points in this repo.
No new simulation was run for this document. Two pre-registerable gates in §5.*

---

## 0. Claim

The paper's compiler decides **analog vs digital per prime**. Below that sits an axis
it does not have: the **signal domain** — the physical quantity carrying the value.
Eight are available in CMOS: current-log (translinear/subthreshold), current-linear
(triode, Gilbert), voltage, charge (switched-capacitor, capacitor integration), time
(time-to-threshold, pulse width, race), frequency/phase, and mixed (current→time).

Hypothesis: **each domain has one dominant mismatch lever, the levers differ by more
than an order of magnitude at the same device corner, and domain choice per prime is
therefore a larger cost lever than the analog/digital choice.**

Three measured points in this repo already say so, at one corner:

| Repo result | Domain | Measured |
|---|---|---|
| Translinear RMSNorm pipeline (`tab:mismatch`) | current-log | 11.4 % / 22.9 % → **2.1 eff. bits** |
| Race ranking, N = 131 072 (`exp_bounded_recursion.md` §3.3) | time | **top-B set exact** (p = 1.000) at `g_sel ≥ 0.5` nat |
| Mamba cell (`exp_mamba_vcrc.md` §4.3) | triode vs subthreshold pair | **3.3–18.8 % vs 723–2122 %** at the *same* 3 % K′ / 10 mV V_t |

The last row is a ~200× spread between two domains for one prime at one corner — no
analog/digital decision in the paper moves a number that far.

### 0.1 Method and units

**Corner (MOS-typical, as in `tab:mismatch`):** σ(K′)/K′ = 3 %, σ(V_t) = 10 mV,
σ(I_S)/I_S = 3 %, σ(n)/n = 0.3 %, σ(C)/C = 0.1 %. T = 300 K, V_T = 25.85 mV, n ≈ 1 so
nV_T ≈ 26 mV; bias lever `ln(I_max/I_S) = 19.11` (measured, `exp_bounded_recursion.md`
§3.3). Design point: V_ov = 300 mV, V_FS = 1 V, comparator V_th = 500 mV, C = 1 pF.

**Effective bits:** `ENOB = log2( 1 / (2·σ_rel) )`, σ_rel = RMS relative signal error
from the lever — the paper's own convention, reproducing `tab:mismatch` exactly
(4.5 % → 4.5 b; 22.9 % → 2.1 b; 34.4 % → 1.5 b).

**Calibration.** The log-domain lever predicts 6.47 % per junction stage; a three-stage
RMSNorm pipeline gives 6.47 %·√3 = 11.2 % against the **measured 11.4 %** — accurate to
~2 % on the one cell with ground truth. Everything else here is the same arithmetic
applied elsewhere, and is *predictive, not validated*.

---

## 1. The eight domains

### 1.1 Lever formulas

| Domain | Governing law | Mismatch lever σ_rel | Value at corner |
|---|---|---|---|
| **Current-log** (translinear, subthreshold) | `V = nV_T ln(I/I_S)` | `√( δ_Is² + (ln(I/I_S)·δ_n)² )` | `√(0.03² + (19.11·0.003)²)` = **6.47 %** |
| **Subthreshold transconductor** (diff. pair, Gilbert in weak inv.) | `I = I_0 e^{V_gs/nV_T}` | `σ_Vt / (nV_T)` | `10/25.85` = **38.7 %** |
| **Current-linear / triode** (two-terminal VCR) | `G = K′(W/L)·V_ov` | `√( δ_K′² + (σ_Vt/V_ov)² )` | `√(0.03² + (10/300)²)` = **4.48 %** |
| **Gilbert, strong inversion** | square law, two matched pairs | `√2 · triode lever` | **6.34 %** |
| **Voltage** (open-loop) | `V` carries the value | `σ_Vt / V_FS` (additive offset) | `10 mV / 1 V` = **1.00 %** |
| **Charge** (switched-cap, cap. integration) | `Q = CV`, charge sharing | `√( δ_C² + kT/C /V_FS² )` | `√(0.001² + (64 µV/1 V)²)` = **0.100 %** |
| **Time** (time-to-threshold, pulse width, race) | `t = C·V_th / I`, so `t ∝ 1/I` | `√( (σ_I/I)² + (σ_Vos/V_th)² )`, **unity gain in the current error** | `√(0.0448² + 0.020²)` = **4.91 %** |
| **Frequency / phase** | `f = 1/(2Nτ)`; `φ = 2π∫f dt` | `√( δ_K′² + δ_C² )`, *static, does not average out* | **3.02 %** |
| **Mixed: current→time** | comparator fires at `V_th` | `σ_Vos / V_th` | `10 mV / 500 mV` = **2.00 %** |

Two structural facts the table encodes and the paper does not:

1. **The log domain is the only one with a gain > 1 on a device-parameter spread.**
   A 0.3 % ideality spread becomes 5.73 % of signal because it multiplies the bias
   offset `ln(I_max/I_S) = 19.11` — *twice the damage of the 3 % saturation-current
   spread* (measured, `exp_bounded_recursion.md` §3.3). All other domains map spread
   to error with gain ≤ 1.
2. **Offset levers (voltage, subthreshold pair, comparator) are additive and hence
   learnable; gain levers (triode, charge, log) are multiplicative and partly
   trimmable; region-exit (cutoff) is neither.** The repo measured all three: the
   pair's ~1000 % collapses to 5–15 % under a six-constant refit, while the triode's
   8.99 % barely moves to 9.86 % because 5–17 % of tokens hit cutoff
   (`exp_mamba_vcrc.md` §4.3). This distinction, not the raw error, is what the
   compiler needs.

### 1.2 Dynamic range, energy, latency

| Domain | Dynamic range | Energy / op | Energy scaling | Settle / latency |
|---|---|---|---|---|
| Current-log / subthreshold | **≥ 5 decades** (anchor: pair variant passed K1 over Δ∈[0.001,1], `exp_mamba_vcrc.md`) | `V_dd·C·nV_T` ≈ 26 fJ @1 pF | **flat in ENOB** — bits cannot be bought with energy | `τ = C·nV_T/I`; pipeline ≈ 12 ns (paper §7) |
| Triode | ~10²–10³ before cutoff (design law `swing ≤ 2V_ov,max·Δ_min/Δ_max`, measured) | `V_dd·C·V_ov` ≈ 300 fJ | `E ∝ 2^ENOB` (V_ov buys bits linearly) | `C·V_ov/I` |
| Voltage | `V_FS/σ_Vt` = 100 (6.6 b) uncalibrated | `C·V_FS²` = 1 pJ | `E ∝ 4^ENOB` | GBW-limited |
| Charge | `V_FS/√(kT/C)` = 1.5×10⁴ (12.9 b) @1 pF | `C·V_FS²` = 1 pJ @1 pF; **10 fJ @10 fF (anchor, `exp_symbol_margin.md` §8)** | `E ∝ 4^ENOB` (Pelgrom area law → Kinget's fixed BW-accuracy-power tradeoff) | RC of the switch, sub-ns |
| Time | `t_max/t_min` ≈ 10⁵ (ns→100 µs); **`t_(B)/t_(1) = e^g` verified to 0.002 % over g = 0.1–4** | `E = V_dd·C·V_th·Z_norm·e^{g}`, **6 significant figures** (anchor); 34–511 fJ for N = 64–1024, i.e. ~0.5 fJ/element, **190–204× below N ADCs** | `E ∝ e^{margin}` — you pay only for the margin you need | **`t_(1)` exactly N-independent = 1.0000 ns** (anchor); `t_(B) = t_(1)e^g` (1.1 ns at g=0.1, 54.6 ns at g=4) |
| Frequency/phase | Q-limited (Hajimiri–Lee) | `N_cycles·C·V_dd²` | **`E ∝ 2^ENOB` *and* latency ∝ 2^ENOB** — worst of both | `N_cycles/f` |

Bolded Time/Charge entries are measurements in this repo (`exp_bounded_recursion.md`
§3.1–§3.2, `exp_symbol_margin.md` §8); the rest is lever arithmetic.

**Energy note (`exp_symbol_margin.md` §8).** In the charge domain the *margin* is
nearly free — one restoration to ε = 10⁻⁹ at k = 256, σ_G = 1 % costs 141 zJ against
an 86 zJ Landauer-class bound, **factor 1.6** — but the *line drive* is `C·V_FS²` =
10 fJ, **1.17×10⁵ × the bound**. So any energy argument about a domain must attack `C`
and `V_FS`, never the noise margin: in every voltage-like domain accuracy is cheap,
swing is expensive.

### 1.3 Verified anchors, one per domain

| Domain | Reference | Status |
|---|---|---|
| all MOS mismatch | Pelgrom, Duinmaijer, Welbers, *Matching properties of MOS transistors*, IEEE JSSC **24**(5):1433–1439, Oct 1989 | verified |
| σ–area–power | Kinget, *Device mismatch and tradeoffs in the design of analog circuits*, IEEE JSSC **40**(6):1212–1224, Jun 2005 — "a fixed bandwidth-accuracy-power tradeoff … set by technology constants" | verified |
| current-log | Gilbert, *Translinear circuits: a proposed classification*, Electronics Letters **11**(1):14–16, 1975, DOI 10.1049/el:19750011 | verified |
| subthreshold | Vittoz & Fellrath, *CMOS analog integrated circuits based on weak inversion operation*, IEEE JSSC **12**(3), Jun 1977 | verified. **Brief correction:** the canonical paper is Vittoz & Fellrath 1977, not "Vittoz 1985" |
| charge | McCreary & Gray, *All-MOS charge redistribution A/D conversion techniques, I*, IEEE JSSC **10**(6):371–379, Dec 1975 | verified |
| charge (CIM) | Valavi, Ramadge, Nestler, Verma, IEEE JSSC **54**(6):1789–1799, Jun 2019 | title/venue verified; its matching and SNR numbers **UNVERIFIED** (paywalled) |
| charge vs current | Chen et al., *CAP-RAM* (arXiv:2107.02388): "the matching of transistors is more difficult thus limiting the linearity of the CiM computing in current domain" | verified (quoted from the arXiv text) |
| capacitor matching | *Matching Properties of Femtofarad and Sub-Femtofarad MOM Capacitors*: σ = **0.8 % and 1.2 %**, area scaling per Pelgrom | partially verified (abstract only). **Brief correction:** 0.1 % is a *large-unit-cap* number; at femtofarad size σ ≈ 1 % |
| time | Sayal, Fathima, Nibhanupudi, Kulkarni, ISSCC 2019 paper 14.4 / IEEE JSSC **55**(1):60–75, 2020: **12.08 TOPS/W** (peak 13.46), works to 375 mV at >90 % MNIST | verified |
| time | Miyashita et al., time-domain NN, "48.5 TSOp/s/W"; journal version IEEE JSSC **52**(10):2679–2689, 2017 | JSSC venue verified; the brief's **ISSCC 2014** attribution is **UNVERIFIED** |
| frequency/phase | Hajimiri & Lee, *A general theory of phase noise in electrical oscillators*, IEEE JSSC **33**(2):179–194, 1998 | verified |
| analog-vs-digital | Sarpeshkar, Neural Computation **10**(7):1601–1638, 1998, DOI 10.1162/089976698300017052 | verified. The "~8-bit crossover" often attributed to it is **UNVERIFIED here** |

---

## 2. Prime × domain

Feasible = the domain can carry the prime without a helper stage. ENOB from the §1.1
lever at the stated corner. **Bold = anchored to a measurement in this repo.**

| Prime | current-log | subthr. pair | triode | voltage | charge | time | freq. |
|---|---|---|---|---|---|---|---|
| **P1 Σ** | KCL **exact, O(0)** (fan-in ≤ 256) | — | KCL, same | needs a gain stage: 5.6 b | charge share: 9.0 b, but 1/N attenuation | ✗ (times do not sum on a wire; serial chain only) | phase sums: 4.1 b |
| **P2 ×c** | 6.5 % → 3.0 b | 38.7 % → 0.4 b | 4.5 % → **3.5 b** | 1.0 % → 5.6 b | **0.10 % → 9.0 b** (σ_C = 1 %: 5.6 b) | delay ratio: 4.9 % → 3.3 b | 3.0 % → 4.1 b |
| **P3 eˣ** | native, 6.5 % → 3.0 b | native (Boltzmann) but 0.4 b | ✗ | ✗ | ✗ | **native in the time axis**: `t = t₀e^{−z}`, **verified 0.002 %** ideal; 3.3 b as magnitude, **exact as a rank** | ✗ |
| **P4 ln x** | native, **lever ×19.11** → 3.0 b | — | ✗ | ✗ | ✗ | **`ln I = −ln t` read off the time axis, no device lever at all** → 3.3 b | ✗ |
| **P5 argmax** | WTA, needs gain | WTA | — | comparator bank | comparator on charge | **race: top-B set exact (p = 1.000) at g ≥ 0.5 nat, N = 131 072**; 190–204× below N ADCs | ✗ |
| **P6 σ(x)** | — | **native Fermi–Dirac**; 10 mV = 0.39 units of input shift → *learnable bias*, keeps its bits downstream | tanh via triode: 3.5 b | 5.6 b | ✗ | ✗ | ✗ |
| **P7 θ(x)** | — | — | — | comparator: **4.6 b** (σ_Vos/V_th) | 9.0 b | comparator: 4.6 b | ✗ |
| **P8 ∫** | ✗ | ✗ | ✗ | ✗ | **native `Q = ∫I dt`, 9.0 b**; gain-cell τ 128.0 µs design vs **128.6 µs measured (0.47 %)** | counter: exact | phase = ∫f: 4.1 b |
| **P9 d/dt** | ✗ | ✗ | ✗ | RC high-pass: 5.6 b | RC: 4.5 b (R·C spread) | edge spacing: 3.3 b | ✗ |
| **P10 ξ** | Johnson/RTN | Johnson/RTN | Johnson/RTN | kT/C | kT/C | **jitter is the native noise** | phase noise, Q-limited |
| **P11 a×b** | translinear, 6.5 %·√k | **723–2122 % → 0.4 b** | **3.3–18.8 % → 3.5 b** | 5.6 b | **9.0 b but only if one operand is a switch pattern**; two continuous operands → 5.6 b (a transconductor re-enters) | pulse-AND × current = charge: 3.3 b | ✗ |
| **P12 1/x** | translinear loop, 3 stages → **2.2 b** (the paper's fragile case); **in AGC feedback: per-channel 0.000 %, common-mode 3.0–8.1 %, settles 4.4 ns** (`tab:agcsweep`) | ✗ | ✗ | ✗ | ✗ | **`t ∝ 1/I` *is* the inversion — zero extra devices, lever unamplified → 3.3 b** | ✗ |

Largest cell-to-cell spread for one prime: **P11, 0.4 b (subthreshold pair) → 9.0 b
(charge) = 8.6 bits**; its measured half (pair vs triode, ~200× in error) is already
in the repo.

---

## 3. Compiler consequence

### 3.1 Primes that should move

1. **P3 and P4 out of current-log into time.** In the log domain a 0.3 % ideality
   spread is multiplied by `ln(I/I_S) = 19.11`; in the time domain the exponential
   *is the time axis* (`t = t₀e^{−z}`, verified to 0.002 %) and no device parameter
   multiplies it. This removes the only >1 gain in the table. Cost: one comparator.
2. **P12 into feedback where a set-point exists, into time where it does not, never
   open-loop translinear.** Open-loop: 3 cascaded junctions, 2.2 b. Feedback: 0.000 %
   per-channel (measured). Time: free, because `t ∝ 1/I` is the device law. The
   condition is the one the Mamba gate found — an SSM has no target to pin, so the AGC
   trick does not transfer (`exp_mamba_vcrc.md` §4.3); those cases go to time.
3. **P11 into charge (one switched operand) or triode (two continuous), never into a
   subthreshold pair.** Anchor: 3.3–18.8 % vs 723–2122 %. Mechanism from the repo: a
   triode VCR is *two-terminal*, so V_t mismatch is a conductance error and never an
   offset; a transconductor is *three-terminal* and its input-referred offset is 2.5
   full scales on a 4 mV state swing. **Rule: pick the two-terminal element whenever
   the state is what is being multiplied.**
4. **P1 stays KCL.** Charge sharing is equally exact but attenuates by 1/N and costs
   `C·V_FS²`. What binds is fan-in (≤ 256), not precision.
5. **P5 into time** — already there: exact ranking at 131 072 competitors, `t_(1)`
   N-independent, 190–204× below the ADC alternative.
6. **P8 into charge** — the only domain where integration is a device law.
7. **P6 stays subthreshold** despite 0.4 bits, because its lever is an *input-referred
   shift*, i.e. a learnable bias; the repo measured static per-channel error absorbed
   by physics-aware or sign-concordant training (0.7–0.95 σ → 0.06–0.16 σ,
   `exp_mismatch_absorption.md`). **Score levers by type, not by size.**

### 3.2 Conversion costs

| Conversion | Circuit | Lever | Energy |
|---|---|---|---|
| current → time | **comparator** | **`σ_Vos/V_th` = 10/500 mV = 2.0 %** — static, per-channel, trimmable, *order-preserving* (cannot reorder currents whose ratio exceeds `e^{σ_Vos/V_th}`) | ~0.5 fJ/element (34–511 fJ over N = 64–1024) |
| current → charge | integrate on C | exact (`Q = ∫I dt`); costs a time window | `C·V_FS²` |
| time → current | current DAC / charge pump | re-imports the source lever | — |
| charge/voltage → current | **transconductor** | **re-imports `σ_Vt/nV_T` = 38.7 %** (subthreshold) or `σ_Vt/V_ov` = 3.3 % (triode) | — |
| log → linear | exponential device | 6.5 % | 26 fJ |

**The asymmetry is the compiler rule.** Conversions *into* time and charge are cheap
(a comparator, a capacitor) and order-preserving; conversions *back out* into
subthreshold current are the most expensive move in the table. A schedule should push
work downstream into time/charge and not return. The repo's MCTS analysis hits the
same constraint from the other side: the P8 accumulator node and the P5 race driver
must be *the same physical node*, or every simulation re-drives 256 DAC channels at
12.8 pJ (`exp_bounded_recursion.md` §4.4). **Conversion avoidance, not conversion
accuracy, is what domain assignment actually optimises.**

---

## 4. Prior art

| Work | What it does | Does it select the signal domain? |
|---|---|---|
| Sarpeshkar 1998 (Neural Computation 10(7):1601–1638) | analog vs digital as a resource question; efficient computation is *hybrid*, distributed over many wires at an optimal SNR per wire | No — the axis one level **above** this one |
| Hasler, ICRC 2016; Hasler & Black, *Physical Computing: Unifying Real Number Computation…* (JLPEA 2021) | unifies analog, neuromorphic, optical and quantum as *real-valued* computation; real-valued Turing model, analog abstraction, FPAA | Taxonomy axis is **substrate + real-valuedness**. Whether the full texts carry a domain axis: **UNVERIFIED** (abstracts only) |
| Legno (Achour & Rinard, ASPLOS 2020) | first compiler targeting a *physical* reconfigurable analog device (HCDCv2); noise-/quantization-/variation-aware; 0.50–5.92 ms, 0.28–5.67 µJ | No — the device fixes the domain |
| Ark (Wang, Cowan, Rührmair, Achour, ASPLOS 2024, arXiv:2309.08774) | language for *describing* an analog compute paradigm + validator + compiler; case studies: transmission-line PUF, cellular nonlinear network, oscillator-based computing | No — the **human picks the paradigm**; Ark codifies its design space |
| Shem (Wang & Achour, arXiv:2411.03557, 2024) | hardware-aware optimisation by differentiating through time-domain simulation (neural-ODE style), with noise, mismatch, discrete behaviour | No — optimises **parameters within** a given design |
| **Freye, Lou, Lanius, Gemmeke, *Merits of Time-Domain Computing for VMM — A Quantitative Comparison*, arXiv:2403.18367 (RWTH Aachen, 2024)** | **closest existing work**: time-domain vs charge-domain analog vs digital for VMM. Digital dominates at error ≤ 0.5 LSB; TD wins at small/medium vectors under approximation; analog wins at large vectors as ADC cost amortises; TD not area-competitive | **No** — a design-space **comparison for one fixed workload**, per macro, not a synthesis decision |
| CAP-RAM (arXiv:2107.02388), Valavi et al. JSSC 2019 | charge-domain CIM beats current-domain on linearity under variation, because capacitor matching beats transistor matching | Domain choice is a **design argument**, made once per chip |

**What exists:** every pairwise comparison in §1 exists somewhere, and charge-vs-current
and time-vs-charge are well made (CAP-RAM, Freye et al.). The analog-vs-digital axis is
Sarpeshkar's, the real-valued-substrate axis Hasler's, analog compilation Achour's.

**What is new:** I found **no published framework treating the signal domain as a
per-operation compile-time choice with a mismatch-lever cost model.** Legno, Ark and
Shem fix the domain before they begin; Freye et al. compare domains but do not select
among them; Sarpeshkar and Hasler sit one axis up. What is available here is the
*axis as a compiler input* plus the lever table that prices it — not any individual
domain comparison.

---

## 5. Two pre-registerable kill-gates

Both reuse the existing ngspice harness; neither needs a PDK.

### D1 — exp in log vs exp in time under identical mismatch

Harness: `spice/softmax_agc_sim.py` (log path) + `spice/bounded_recursion_latency.py`
(race path). Softmax over N = 64, one shared MC draw set (200 draws, 3 % I_S, 0.3 % n,
10 mV V_t, V_th = 500 mV): (a) open-loop translinear pipeline; (b) race generator, value
read as `t_i = t₀e^{−z_i}` by a counter.

**Predictions.** (a) median L1 = **13.6 %** (already measured, paper Result 5).
(b) `√(0.0647² + 0.020²)` = **6.8 % ± 1 %** as a magnitude; **rank error exactly 0**
for `g_sel ≥ 0.5` nat.

**Kill.** "The time domain removes the log lever" dies if (b) ≥ 0.75·(a), i.e. (b) >
**10.2 %**. **Sharper kill — a computable crossover, not a threshold:** the comparator
lever `σ_Vos/V_th` overtakes the log lever it saved at `V_th* = σ_Vos/0.0647` =
**155 mV**. Sweep V_th from 100 mV to 1 V; the curves must cross within **155 mV ± 20 %**,
or the lever model of §1.1 — and this whole document — is wrong.

### D2 — P11 in Gilbert vs triode vs charge, one corner

Harness: the MC framework of `spice/mamba_vcrc_sim.py`. Multiply two variables,
100 draws, 3 % K′, 10 mV V_t, and σ_C at both **0.1 %** and **1 %** (the femtofarad
measurement above says 1 % is the honest number for small unit caps).

**Predictions.** Subthreshold pair p50 ≈ **40 %** → 0.4 b (repo anchor: 723–2122 %
state-referred). Triode p50 ≈ **4.5 %** → 3.5 b (anchor: 3.34 % at D = 100). Charge,
one switched operand: **0.10 %** → 9.0 b, or **1.0 %** → 5.6 b at σ_C = 1 %.

**Kill.** The headline hypothesis predicts a best-to-worst spread of **8.6 bits**
(σ_C = 0.1 %) or **5.2 bits** (σ_C = 1 %) for one prime at one corner. It dies if the
measured spread is **< 3 bits**.

**Third arm, pre-registered against my own claim.** A *four-quadrant analog×analog*
charge multiplier must **not** reach 9 bits: the second continuous operand passes a
transconductor, re-importing `σ_Vt/V_ov` = 3.3 % → predicted **5.6 b**. If it does reach
9 b, "charge is universally best for P11" holds and §3.2's conversion argument is wrong.

---

## 6. What this does not establish

- **Nothing here is simulated.** §2 is lever arithmetic calibrated on one cell
  (`tab:mismatch`: 11.2 % predicted vs 11.4 % measured). Three cells carry repo
  anchors; the other ~70 are predictions.
- **Area is absent.** Kinget's tradeoff says the charge domain's 9 bits are bought
  with capacitor area, and the femtofarad data (σ ≈ 1 %) says the brief's 0.1 % corner
  is a large-cap number. Read the charge column as *9.0 b at large unit caps, 5.6 b at
  femtofarad unit caps*.
- **The frequency/phase row has no anchor in this repo**; its lever (static mismatch
  does not average out, so phase error grows linearly with integration time) is
  asserted, not measured.
- **"Largest single cost lever" is comparative and untested**, supported by one
  measured 200× spread (pair vs triode) and one measured exactness result (race
  ranking). Gate D2 is the test.
- Temperature gradients, supply noise, and V_t/K′ correlation in a real PDK are ignored.
