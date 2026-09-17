# Kill-gate 1: Mamba's selective SSM as a voltage-controlled-R/C cell

*Experiment run 2026-09-15 against `docs/missing_primes_mapping.md` §2 (X2, claim
"data-dependent gain is P11, not dynamic topology") and §8 kill-gate 1.
Code: `spice/mamba_vcrc_sim.py`. Raw numbers:
`spice/out/mamba_vcrc_results.json`. Figure: `spice/out/mamba_vcrc.png`.
Tools: ngspice-46, numpy 2.4, scipy 1.17. Total compute ≈ 2 min.*

---

## 0. Claim under test

> Mamba's selective SSM (Gu & Dao, arXiv:2312.00752, Alg. 2) is **not** dynamic
> topology (X2). Per state channel it is a variable × variable multiplication
> (P11) acting on an integrator (P8): `A_n` is a *fixed* learned negative
> constant, and the input-dependent step `Δ(x) = softplus(w·x + b)` is a
> **time-warp** of a single RC integrator whose leak conductance and input
> transconductance are both set by the current token.

If that factorization is physics and not analogy, then a capacitor with two
voltage-controlled conductances, clocked one token per `T_tok`, must reproduce
the discrete recurrence to useful precision. This document pre-registers the
criteria, derives the mapping, builds the circuit in ngspice without a PDK, and
reports the verdicts.

---

## 1. Pre-registration (fixed before any circuit was simulated)

**K1 — fidelity.** If with *ideal* devices the circuit cannot track the discrete
recurrence to ≤ 1 % relative RMS (≈ 7 effective bits) over the Δ range
0.001–1, the P11-realization claim is KILLED at this fidelity. If it fails, the
report must say whether the failure is triode nonlinearity (physics) or
ZOH-vs-Euler discretization mismatch (a modelling choice, not physics).

**K2 — mismatch.** If under ±3 % K′ and ±10 mV V_t mismatch on the
transconductor devices the **median** error exceeds 8 % (≈ 3.6 bits), the
realization is "mismatch-fragile" in the same sense as the open-loop
translinear pipeline of the paper (`spice/rmsnorm_agc_sim.py`, Table 6). This
is *not* a kill of the factorization, only of the naive circuit; the report must
then say whether a feedback form exists that makes the mismatch common-mode.

**K3 — selectivity.** If the selective-copy toy fails qualitatively — spikes not
retained while Δ is small, or not forgotten while Δ is large — the claim
"selectivity = data-dependent gain" is KILLED.

Metrics fixed in advance: relative RMS error of `h` sampled at token boundaries
against the numpy reference, `rel = ‖h_circ − h_ref‖₂ / ‖h_ref‖₂`, and
`effective bits = −log₂(rel)`. Monte Carlo: 100 draws, report p50 and p95.

Sweep fixed in advance: Δ spanning 0.001–0.1, 0.001–1, 0.001–10 (units of
1/T_tok); three channels with the S4D-real init `A = −1, −4, −16`; `B_t = 1`;
three input sequences (a) 256 Gaussian tokens, (b) selective-copy toy,
(c) step + reset.

---

## 2. Derivation

### 2.1 The circuit's continuous-time law

Take a capacitor `C` holding the state voltage `h`, a **voltage-controlled leak
conductance** `g_leak(x)` from the state node to the state's reference, and a
**voltage-controlled input transconductance** `g_in(x)` driving current
`g_in(x)·x` into the node. KCL at the state node:

    C · dh/dt = − g_leak(x) · h + g_in(x) · x

    ⇒  h' = − (g_leak(x)/C) · h + (g_in(x)/C) · x                      (1)

This is Mamba's continuous form `dh/dt = A·h + B(x)·x` with
`A_eff = −g_leak/C` and `B_eff = g_in/C`. Note that both coefficients are
*products of two live quantities* — a conductance set by the current token
multiplying a state (and an input) set by the data. That is P11, a
variable × variable multiplication, on P8, an integrator. No wire moves.

### 2.2 Clocking: one token per T_tok

Hold `x_t`, `g_leak(x_t)`, `g_in(x_t)` constant over the slot
`[(t−1)T_tok, t·T_tok)`. Equation (1) is then linear with constant
coefficients and integrates exactly:

    h(t·T_tok) = e^{−g_leak(x_t)·T_tok/C} · h((t−1)T_tok)
                 + (g_in(x_t)/g_leak(x_t)) · (1 − e^{−g_leak(x_t)·T_tok/C}) · x_t   (2)

### 2.3 The mapping

Mamba's ZOH discretization of `dh/dt = A h + B x` with step `Δ_t` is

    h_t = e^{Δ_t A} h_{t−1} + ((e^{Δ_t A} − 1)/A) · B_t · x_t          (3-ZOH)

and the *simplified Euler* form actually implemented (Alg. 2, and the reference
CUDA kernel: `deltaA = exp(delta*A)`, `deltaB = delta*B`) is

    h_t = e^{Δ_t A} h_{t−1} + Δ_t · B_t · x_t                          (3-Euler)

Comparing (2) with (3-ZOH), term by term, gives the mapping asked for:

| Mamba quantity | circuit quantity | explicit form |
|---|---|---|
| `Δ_t · A`  | `− g_leak(x_t) · T_tok / C` | `g_leak(x_t) = \|A\| · C · Δ_t / T_tok` |
| `Δ_t · B_t` | `g_in(x_t) · T_tok / C` | `g_in(x_t) = C · Δ_t · B_t / T_tok` |

The decay factor matches **exactly and unconditionally**:
`e^{−g_leak T_tok/C} = e^{Δ_t A}`. With `g_in/g_leak = B_t/|A|` the input term
of (2) becomes `(B_t/|A|)(1 − e^{Δ_t A}) x_t = ((e^{Δ_t A}−1)/A) B_t x_t`, i.e.
the circuit is a **ZOH-exact** realization: the physical RC integrator computes
the exact matrix exponential of its own generator, for free, in the analog
domain. The "discretization" that costs a digital implementation a choice is
simply what the capacitor does.

**The Euler correction.** To reproduce (3-Euler) instead, only the *write* path
needs rescaling — the decay is already identical. Requiring
`(g_in/g_leak)(1 − e^{Δ_t A}) = Δ_t B_t` gives

    g_in(x_t) = (C · Δ_t · B_t / T_tok) · κ_t ,
    κ_t = |A|Δ_t / (1 − e^{−|A|Δ_t}) ,   κ → 1 as Δ|A| → 0            (4)

`κ_t` is a scalar computable in the same digital path that already computes
`softplus`. So **the ZOH-vs-Euler discrepancy is not a physical limit at all**;
it is one multiplicative calibration constant per token in the D/A map. Both
mappings are implemented and both are reported below.

Physical scales for `C = 1 pF`, `T_tok = 10 ns`: `g_leak = |A|·Δ·100 µS`, i.e.
50 nS at `|A|Δ = 5·10⁻⁴` up to 16 mS at `|A|Δ = 160`. Five decades of
conductance — this is the number that decides the experiment.

### 2.4 The differential cell (exact linearity in level-1 triode)

A single NMOS in triode between two nodes `h⁺`, `h⁻` carries

    I = β[(V_g − V_t)(h⁺ − h⁻) − (h⁺² − h⁻²)/2]
      = β[(V_g − V_t) − (h⁺+h⁻)/2] · (h⁺ − h⁻)

so if the **common mode `(h⁺+h⁻)/2` is fixed**, the quadratic term vanishes
identically and the device is a *perfectly linear* voltage-controlled resistor
with `g = β(V_g − V_t − V_cm)` (Czarnul/Tsividis MOS-C linearization). In the
cell below the common mode is preserved structurally: every current is injected
differentially (`+i` at `h⁺`, `−i` at `h⁻`) and the leak current flows *between*
the two nodes, so `d(h⁺+h⁻)/dt = 0` exactly. A DC probe confirms it: with
`V_cm = 0.9 V`, `V_g = 2.0 V`, the measured `I(v_d)` is linear to a relative
error of `2·10⁻⁸`–`2·10⁻⁷` over `v_d = ±50 mV`.

With `v_d = h⁺ − h⁻` and `C` on each half-node, `C·dv_d/dt = 2i − 2g·v_d`, so
the single-ended conductances of §2.3 are realized by devices carrying
`g_leak/2` and `g_in/2`.

---

## 3. Circuit and netlist

Two variants, both at the repo's established fidelity level — **device-exact,
wiring-ideal**: the transconductor devices are real ngspice devices with real
I–V laws and real Monte-Carlo mismatch; current copying and common-mode
bookkeeping are ideal `F`-sources.

### Variant A — MOS triode voltage-controlled resistors (the brief's primary)

```
* per cell k (9 cells = 3 Delta-ranges x 3 channels, all independent)
Chp{k} hp{k} 0 1p
Chn{k} hn{k} 0 1p
Ml{k}  hp{k} gl{k} hn{k} 0 ML{k}  W=.. L=..     ; leak VCR, g_leak/2 = beta*Vov
Mi{k}  xp    gi{k} xs{k} 0 MI{k}  W=.. L=..     ; input transconductor (P11)
Vs{k}  xs{k} xn DC 0                            ; sense
Fp{k}  0     hp{k} Vs{k} 1.0                    ; ideal differential copy
Fn{k}  hn{k} 0     Vs{k} 1.0
Vgl{k} gl{k} 0 PWL(...)                         ; D/A boundary: Vt+Vcm+Vov(Delta_t)
Vgi{k} gi{k} 0 PWL(...)                         ; D/A boundary: Vt+Vcm+Vov(Delta_t*kappa_t*B_t)
.model ML{k} NMOS(LEVEL=1 VTO=0.7 KP=100u LAMBDA=0 GAMMA=0 ...)
```

`Mi{k}` is the P11 element in the strict sense: its current is
`β(V_gi − V_t − V_cm) · (x⁺ − x⁻)` — gate overdrive (carrying `Δ_t·B_t`) times
drain–source voltage (carrying `x_t`), two live variables multiplied by one
device. `Ml{k}` is the same multiplication with the *state* as the second
factor.

Sizing rule: per cell, `β` is chosen so that the largest conductance the
sequence demands sits at the maximum overdrive `V_ov,max = 1.0 V`
(`W/L = β/K′`, ranging from `W/L = 0.05` to `W/L = 80`). Common mode
`V_cm = 0.9 V`, `V_TO = 0.7 V`, `K′ = 100 µA/V²`, `LAMBDA = GAMMA = 0`
(no channel-length modulation in triode, no body effect with bulk at 0 and
`GAMMA = 0`).

### Variant B — junction-exact differential pair with a current-mode tail

Added because variant A runs out of dynamic range (§5.2), which the brief
anticipated. A differential pair with tail current `I_tail` gives
`ΔI = I_tail·tanh(v_d/2V_T)`, i.e. `g = I_tail/(2 V_T)`: the conductance is set
by a **current**, and a current-mode DAC spans five decades without changing
operating region. Required tails, from §2.3:

    I_tail,leak = 2 V_T C |A| Δ_t / T_tok        (5.2 nA … 830 µA)
    I_tail,in   = 2 V_T C Δ_t B_t κ_t / T_tok · (s_h/s_x)

with `V_T = 0.025864890 V`, ngspice's SPICE3-legacy thermal voltage
(`spice/README.md` gotcha — the tail currents are computed against that value,
not CODATA). Bipolar devices (`IS = 1e-16`, `BF = 1e6`, no parasitics) stand in
for a subthreshold MOS pair, which is what a real implementation would use:
ngspice's level-1 MOS model has no weak-inversion region, and the repo's
existing testbenches already use junctions to carry exponential device physics.
`BF = 1e6` encodes the assumption of a gate-current-free (MOS) input pair; see
limitations.

### Signal scaling and the D/A boundary

Everything digital in Mamba stays digital: `Δ_t = softplus(w·x_t + b)`, the
affine scale that makes `Δ` span the swept range, `B_t`, and the Euler
correction `κ_t` are computed in numpy and enter the circuit **only** as
PWL gate voltages (variant A) or PWL tail currents (variant B). That is the
D/A boundary of this experiment, and it is where a real implementation would
put a per-channel DAC. The token grid is enforced with PWL breakpoints exactly
at `k·T_tok` (transitions 1 ps later), so token-boundary samples are solution
points, not interpolation artefacts; ngspice's `set interp` then returns exactly
one row per token.

State swing is scaled to a 4 mV peak differential; input swing likewise 4 mV.
Solver: `reltol=1e-6 abstol=1e-16 vntol=1e-10 gmin=1e-14`, Gear-2,
`tmax = T_tok/50`.

---

## 4. Results

### 4.0 How far apart the two discretizations are (numpy only)

Relative RMS distance between the Euler and the exact-ZOH reference, Gaussian
input:

| Δ range | A = −1 | A = −4 | A = −16 |
|---|---|---|---|
| 0.001–0.1 | 3.29 % | 13.73 % | 40.57 % |
| 0.001–1   | 29.54 % | 65.35 % | 88.04 % |
| 0.001–10  | 82.70 % | 94.64 % | 98.64 % |

So the choice of discretization is, at large `Δ|A|`, a bigger effect than any
device imperfection studied below. It is *not* physics; §2.3 removes it with one
scalar per token. All tables below use the matched pair
(Euler mapping ↔ Euler reference, ZOH mapping ↔ ZOH reference).

### 4.1 K1 — ideal devices

Relative RMS error, **worst over the three input sequences**, Euler mapping vs.
the Mamba recurrence:

| Δ range | A | variant A (triode) | eff. bits | variant B (diff-pair) | eff. bits |
|---|---|---|---|---|---|
| 0.001–0.1 | −1  | 0.0036 % | 14.8 | 0.201 % | 9.0 |
| 0.001–0.1 | −4  | 0.0035 % | 14.8 | 0.206 % | 8.9 |
| 0.001–0.1 | −16 | 0.0082 % | 13.6 | 0.197 % | 9.0 |
| 0.001–1   | −1  | 0.383 %  | 8.0  | 0.211 % | 8.9 |
| 0.001–1   | −4  | 0.524 %  | 7.6  | 0.195 % | 9.0 |
| 0.001–1   | −16 | **1.066 %** | 6.6 | 0.108 % | 9.9 |
| 0.001–10  | −1  | **12.11 %** | 3.0 | 0.168 % | 9.2 |
| 0.001–10  | −4  | **32.25 %** | 1.6 | 0.053 % | 10.9 |
| 0.001–10  | −16 | **40.87 %** | 1.3 | 0.053 % | 10.9 |

Per input (triode / diff-pair, %, Euler mapping):

| input | 0.001–0.1 | 0.001–1 | 0.001–10 |
|---|---|---|---|
| (a) Gaussian, 256 tok | 0.0015–0.0082 / 0.069–0.134 | 0.004–0.015 / 0.045–0.089 | 0.008–0.021 / 0.049–0.053 |
| (b) selective copy    | 0.0029–0.0036 / 0.197–0.206 | 0.383–1.066 / 0.108–0.211 | 12.1–40.9 / 0.017–0.168 |
| (c) step + reset      | 0.0004–0.0011 / 0.041–0.144 | 0.0006–0.0071 / 0.015–0.057 | 0.005–0.455 / 0.006–0.018 |

The ZOH mapping against the ZOH reference gives numbers identical to 3 digits
(`spice/out/mamba_vcrc_results.json`, key `ideal`), confirming that the Euler
correction κ is exact and costs no fidelity.

**Reading.** The diff-pair variant meets K1 everywhere, worst case 0.21 %
(8.9 bits). The triode variant is the more accurate of the two where it works
(0.0036 %, ~15 bits) but degrades with the Δ dynamic range, and only on input
(b) — the one sequence containing a *long* run of tokens at Δ_min. §5.2
identifies the mechanism.

### 4.2 K1 mechanism — it is device-region exit, not nonlinearity or ZOH/Euler

Analytic overdrive floors of the triode variant (Euler mapping) and the measured
effect of shrinking the state swing until no device ever leaves triode
(input (b)):

| Δ range | A | V_ov,min leak | V_ov,min input | tokens in triode | swing | rel. RMS at 4 mV → at min swing |
|---|---|---|---|---|---|---|
| 0.001–0.1 | −1  | 10 mV | 9.52 mV | 100 % | 4 mV | 0.0036 % → 0.0036 % |
| 0.001–0.1 | −16 | 10 mV | 5.03 mV | 100 % | 4 mV | 0.0029 % → 0.0029 % |
| 0.001–1   | −1  | 1 mV  | 0.632 mV | 97.3 % | 0.1008 mV | 0.383 % → 0.034 % |
| 0.001–1   | −16 | 1 mV  | 0.063 mV | 73.0 % | 0.1008 mV | 1.066 % → 0.018 % |
| 0.001–10  | −1  | 0.1 mV | 0.0100 mV | 78.1 % | 1.008 nV | 12.11 % → 1.45 % |
| 0.001–10  | −16 | 0.1 mV | 0.00063 mV | 60.2 % | 1.008 nV | 40.87 % → 1.00 % |

Shrinking the swing recovers 12–13 effective bits at 0.001–1 and cuts the error
by a factor 12–41 at 0.001–10 (the ~1 % residual left there is the solver floor:
`vntol = 0.1 nV` against a 1.008 nV swing). So the triode failure is **not**
triode *nonlinearity* (which §2.4 shows is identically zero in level-1 with a
pinned common mode), and **not** ZOH-vs-Euler (which the κ map removes): it is
the leak/input device sliding out of triode into saturation during the
low-Δ hold, where the conductance law changes from `β·V_ov·v_d` to
`(β/2)V_ov²` and the hold leak becomes 1–2 orders of magnitude too small.

This is a scaling law, not an accident. For a triode VCR,
`g = β·V_ov`, so `g_max/g_min = V_ov,max/V_ov,min`, while the triode condition
demands `|v_d| ≤ 2 V_ov` at *every* token. With `R = Δ_max/Δ_min`:

    state swing ≤ 2 · V_ov,max / R        and        V_ov,min = V_ov,max / R

With `V_ov,max ≲ 1 V`: `R = 10²` → 20 mV swing and 10 mV control margin;
`R = 10³` → 2 mV and 1 mV; `R = 10⁴` → 0.2 mV and 0.1 mV. Two independent walls
close at once: signal swing and control margin both collapse as `1/R`, while the
mismatch that fights the control margin (`σ_Vt = 10 mV`) does not.

A second, sharper observation falls out of the same table: with the **ZOH**
mapping `V_ov,min = V_ov,max/R` on *both* devices (10 / 1 / 0.1 mV), but with the
**Euler** mapping the input device's floor drops by a further factor
`κ_max = |A|Δ_max` (to 0.00063 mV at `A = −16`, Δ_max = 10). Insisting on
Mamba's Euler `B̄ = Δ·B` instead of the ZOH `B̄` the capacitor computes for free
therefore **multiplies the required write-path dynamic range by up to
|A|·Δ_max** — here 160×. The simplification that saves a digital kernel one
`expm1` costs the analog kernel two decades of headroom.

### 4.3 K2 — Monte-Carlo mismatch (100 draws, σ(K′/tail) = 3 %, σ(V_t/V_os) = 10 mV)

| variant | Δ range | A | ideal | p50 | p95 | residual after static refit (p50) |
|---|---|---|---|---|---|---|
| triode | 0.001–0.1 | −1  | 0.003 % | 3.34 % | 7.37 % | 1.04 % |
| triode | 0.001–0.1 | −4  | 0.002 % | 5.10 % | 10.24 % | 2.88 % |
| triode | 0.001–0.1 | −16 | 0.007 % | 7.43 % | 15.02 % | 8.35 % |
| triode | 0.001–1   | −1  | 0.004 % | 6.03 % | 14.20 % | 5.69 % |
| triode | 0.001–1   | −4  | 0.013 % | **8.99 %** | 18.90 % | 9.86 % |
| triode | 0.001–1   | −16 | 0.013 % | **13.73 %** | 32.60 % | 14.72 % |
| triode | 0.001–10  | −1  | 0.011 % | **12.78 %** | 25.72 % | 14.76 % |
| triode | 0.001–10  | −4  | 0.017 % | **16.70 %** | 39.93 % | 18.73 % |
| triode | 0.001–10  | −16 | 0.015 % | **18.79 %** | 51.05 % | 21.18 % |
| pair | 0.001–0.1 | −1  | 0.172 % | **2122 %** | 6220 % | 14.5 % |
| pair | 0.001–0.1 | −4  | 0.117 % | **945 %** | 3051 % | 4.8 % |
| pair | 0.001–0.1 | −16 | 0.076 % | **723 %** | 1801 % | 13.1 % |
| pair | 0.001–1   | −1  | 0.088 % | **710 %** | 1671 % | 8.6 % |
| pair | 0.001–1   | −4  | 0.061 % | **860 %** | 2311 % | 32.8 % |
| pair | 0.001–1   | −16 | 0.063 % | **1022 %** | 2895 % | 43.6 % |
| pair | 0.001–10  | −1  | 0.059 % | **1102 %** | 2760 % | 42.5 % |
| pair | 0.001–10  | −4  | 0.065 % | **996 %** | 3191 % | 61.7 % |
| pair | 0.001–10  | −16 | 0.063 % | **959 %** | 2935 % | 102.9 % |

Same Monte Carlo with σ(V_t/V_os) reduced to 1 mV (50 draws):

| variant | 0.001–0.1 | 0.001–1 | 0.001–10 |
|---|---|---|---|
| triode p50 | 2.04–2.39 % | 3.24–3.54 % | 3.57–4.24 % |
| triode p95 | 5.57–6.87 % | 6.29–7.96 % | 8.42–9.81 % |
| pair p50 | 84.7–184 % | 81.8–97.1 % | 113–127 % |

**The structural finding.** A triode VCR is a *two-terminal* element: it carries
zero current at zero differential voltage whatever its threshold is. Its V_t
mismatch therefore appears as a **conductance error**, never as an offset — and
the conductance error is `Δ_eff = (1+ε)(Δ + σ_Vt·Δ_max/V_ov,max)`, i.e. a
gain-and-floor perturbation of Δ. A transconductor pair is a *three-terminal*
element with an input-referred offset `V_os`: its leak relaxes the state to
`−V_os` instead of to 0, and its input pair injects a drive
`∝ I_tail·tanh(V_os/2V_T)` that is present even when `x_t = 0`. On a 4 mV state
swing a 10 mV offset is 2.5 full scales, hence the 700–2100 % medians. This is
the price paid for the five decades of control range that made variant B pass
K1, and it is the design fact this experiment produces: **choose the two-terminal
element whenever the state is the thing being multiplied.**

**Is the mismatch absorbable by training?** The last column fits, per channel and
per draw, the six-constant family

    h_t = e^{−|A|(aΔ_t + p)} h_{t−1} + b(Δ_tκ_t + q)B_t(x_t + x₀) + (1−ā)c

where `a` is the learned `A_n`, `p` a Δ-floor (`dt_min`), `b` the scale of the
`B` projection, `x₀` the bias of the depthwise Conv1d in front of the SSM, and
`c` a constant state offset (exactly absorbable: the SSM is linear, so
`h → h + c` is an output bias). Only `q` sits outside the standard
parametrization. Result: for the *pair* variant the residual collapses from
~1000 % to 5–15 % in the two narrower Δ ranges — its mismatch is very largely a
**static reparametrization**, not noise, and a hardware-aware training pass
would eat it. For the *triode* variant the refit barely helps (8.99 % → 9.86 %,
18.79 % → 21.18 %): once `σ_Vt = 10 mV` exceeds `V_ov,min` (1 mV, 0.1 mV) the
device is pushed into **cutoff** for a fraction of the tokens
(`P(cutoff) = 5–17 %`, table in §4.2's JSON key `triode_feasibility`), and
cutoff is a hard nonlinearity that no reparametrization of a linear recurrence
can represent.

**Does a feedback form make it common-mode?** No — and the reason is structural,
not a missing idea. The AGC results of the paper work because there is a
*target* to pin (the output RMS), so the loop drives the gain error into a
common-mode shift. A selective SSM has no target: `h` **is** the output, its
value is the useful signal, and any loop that pinned it would destroy the
memory. What replaces feedback here is the refit above: the mismatch of the
pair variant is static and therefore learnable, which routes this circuit to
kill-gate 3 (physical-gradient absorption of mismatch) rather than to a feedback
topology. The triode variant's cutoff is the part that is neither common-mode
nor learnable, and it is fixed at the design level (keep `V_ov,min ≫ σ_Vt`,
i.e. keep the Δ dynamic range per channel ≤ ~10², or move to current control).

### 4.4 K3 — selective-copy toy

Δ range 0.001–1, Euler mapping, 256 tokens: 6 sparse ±1 spikes in the first
quarter, a ~110-token hold window at `Δ = Δ_min`, then a 5-token flush burst
(marker set, `x = 0`, `Δ = Δ_max`).

| variant | A | write amplitude ref/circ | retention over hold ref/circ | post-flush residual ref/circ |
|---|---|---|---|---|
| triode | −1  | 0.02924 / 0.02942 | 0.8940 / 0.8999 | 6.72e−3 / 6.76e−3 |
| triode | −4  | 0.06234 / 0.06245 | 0.6389 / 0.6400 | 2.04e−9 / 3.96e−7 |
| triode | −16 | 0.09369 / 0.09376 | 0.1666 / 0.1646 | 1.72e−35 / 9.40e−7 |
| pair | −1  | 0.02924 / 0.02916 | 0.8940 / 0.8917 | 6.72e−3 / 6.70e−3 |
| pair | −4  | 0.06234 / 0.06220 | 0.6389 / 0.6377 | 2.04e−9 / 1.97e−9 |
| pair | −16 | 0.09369 / 0.09356 | 0.1666 / 0.1665 | 1.72e−35 / 1.61e−11 |

Retention agrees with the reference to 0.1–1.2 % relative; erasure is complete
in every case. The residuals of 4·10⁻⁷ / 9·10⁻⁷ in the triode rows are the
circuit's leakage/solver floor (GMIN = 1e−14 S, `vntol` = 0.1 nV), 5–6 orders
below the held signal — qualitatively "forgotten" by any standard. The
channel-dependent memory the toy is supposed to expose is reproduced: the
`A = −1` channel keeps 89 % of the spike across the hold, `A = −16` keeps 17 %.
The figure also shows one mismatched triode draw still writing, holding and
flushing, with the amplitude wrong — selectivity survives mismatch even where
precision does not.

### 4.5 Figure

`spice/out/mamba_vcrc.png` — left column: `h(t)` for the selective-copy input,
Δ ∈ [10⁻³, 1], all three channels, showing the numpy reference, both circuit
variants, and one mismatched triode draw. Right column: normalised error per
token on a log axis with the 1 % K1 threshold marked.

---

## 5. Verdicts

**K1 — NOT KILLED.** Over the pre-registered Δ range 0.001–1 (six cells: three
channels × the two Δ-ranges contained in it, each on three inputs) the
diff-pair realization reproduces the discrete selective-SSM recurrence to
0.015–0.211 % (8.9–12.7 effective bits), i.e. it passes on 6 of 6 cells; the
triode realization reaches 0.0004–0.524 % on 5 of those 6 cells (up to 18
effective bits) and misses the 1 % threshold on the sixth (1.066 %, A = −16,
selective-copy input). The claim in §2 of the mapping
document is therefore supported at 7–15 bits: **Mamba's selective scan is an
RC integrator with two data-set conductances, not a topology change.** The one
threshold miss is attributable (§4.2) to the device leaving triode, is removed
by shrinking the state swing (1.066 % → 0.018 %), and is a property of the
*chosen device*, not of the factorization. Explicitly, per the pre-registration:
the failure mode is **neither** triode nonlinearity (identically zero in
level-1 with a pinned common mode, verified to 2·10⁻⁷) **nor** ZOH-vs-Euler
(a per-token scalar κ in the D/A map removes it exactly). It is device-region
exit under a large Δ dynamic range.

Bonus finding beyond the gate: the capacitor performs the ZOH discretization
exactly and for free, so the ZOH mapping is the *cheap* one physically; Mamba's
Euler simplification is what costs the circuit up to `|A|·Δ_max` (here 160×) of
extra write-path dynamic range.

**K2 — TRIGGERED for both naive circuits; "mismatch-fragile" confirmed.** Under
the pre-registered ±3 % K′ / ±10 mV V_t the median error exceeds 8 % in 5 of 9
triode cells (8.99–18.79 %) and in 9 of 9 diff-pair cells (723–2122 %). The
naive open-loop VCRC is therefore mismatch-fragile in exactly the sense the
paper reports for the open-loop translinear pipeline. Two qualifications, both
measured: (i) at σ(V_t) = 1 mV the triode variant passes everywhere
(p50 2.0–4.2 %, p95 5.6–9.8 %) — this circuit needs either large devices or
trimming, not a different principle; (ii) the diff-pair variant's mismatch is
overwhelmingly a *static reparametrization* (residual after a 6-constant refit:
5–15 % in the two narrower Δ ranges, from ~1000 %), so it is learnable rather
than random, which hands the problem to kill-gate 3. No feedback form pins it,
because a selective SSM has no set-point to pin (§4.3). **This is a kill of the
naive circuit, not of the factorization** — as pre-registered.

**K3 — NOT KILLED.** Spikes are retained while Δ is small (retention 0.894 /
0.639 / 0.167 for A = −1/−4/−16, matched by the circuit to 0.1–1.2 %) and
erased while Δ is large (residual ≤ 10⁻⁶ relative, against a held signal of
order 10⁻¹). "Selectivity = data-dependent gain" is realized by a voltage on a
gate; no addressing, no routing, no wire moves.

### Consequence for `docs/missing_primes_mapping.md`

§2's proposed reclassification of **Mamba K2 from X2 to M, factorization
P8·P2·P1·P11**, survives its kill-gate. The addition this experiment makes to
the table entry is the *cost column* the mapping document asks for elsewhere:
the bound is the **Δ dynamic range per channel**, and it is paid in state swing
and control margin (`swing ≤ 2 V_ov,max·Δ_min/Δ_max` for voltage control), or
in DAC decades (for current control). §8's gate-1 kill condition
("needs > 8 b linearity to match") is not met: the ideal-device fidelity is
8.9–15.8 bits with linearity that is *exact* by construction in the triode cell.

---

## 6. Limitations

1. **PWL control boundary.** `Δ_t`, `κ_t`, `B_t` and the softplus are computed
   in numpy and delivered as ideal PWL gate voltages / tail currents. The DAC
   itself — its resolution, its own mismatch, its settling within `T_tok` — is
   not modelled. Five decades of Δ means a ≥ 17-bit-equivalent or
   segmented/logarithmic DAC per channel; that converter is the unpriced part
   of this design and could easily dominate the area and energy budget.
2. **Wiring-ideal.** Current copies, the differential injection and the
   common-mode bookkeeping are `F`-sources. Real mirrors add their own
   mismatch, finite output conductance and bandwidth. For variant B, `BF = 1e6`
   asserts a gate-current-free (subthreshold MOS) input pair; a real bipolar
   pair would droop the state node through base current.
3. **No PDK, level-1 MOS.** The triode result is *exact* in level-1 partly
   because level-1 is a simple model: no mobility degradation, no velocity
   saturation, no short-channel `V_t` roll-off, no channel-length modulation.
   The linearity number (2·10⁻⁷) is a property of the model, not of silicon.
   The dynamic-range scaling law of §4.2, by contrast, follows from the triode
   condition itself and survives any model.
4. **Ideal capacitor, no noise.** No `kT/C` noise (for `C = 1 pF`,
   `√(kT/C) ≈ 64 µV` — which, against the 4 mV swing used here, is already
   ~36 dB SNR and against the nanovolt swings of §4.2 is fatal), no leakage
   beyond GMIN, no dielectric absorption. A noise analysis would tighten the
   dynamic-range bound further in the same direction.
5. **Scalar channel, `B_t = 1`.** Three channels of a single state dimension
   with a fixed `B`; the code supports an input-dependent `B_t` but the reported
   runs do not use it. The `C_t` readout projection and the gated MLP branch of
   a Mamba block are out of scope — they are P1/P2/P11 and are covered by the
   existing RMSNorm/softmax testbenches.
6. **Mismatch model.** Independent Gaussian `K′` and `V_t` per device, no
   spatial correlation, no `1/√(WL)` area dependence, and the same absolute
   offset applied to both variants — which is fair on input-referred grounds but
   pessimistic for a bipolar pair (10 mV is a realistic subthreshold-MOS
   `σ_Vos`, not a bipolar one).
7. **Single seed per configuration** for the input sequences (seed 2026); the
   Monte Carlo is 100 draws, the σ = 1 mV sensitivity run 50.

---

## 7. Reproduce

```bash
cd spice
python3 mamba_vcrc_sim.py --tokens 256 --mc-runs 100 --mc-tokens 128 --seed 2026
# ~2 min, writes out/mamba_vcrc_results.json and out/mamba_vcrc.png
python3 mamba_vcrc_sim.py --quick        # 6 s smoke test
```
