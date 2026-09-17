# SPICE validation testbenches

Circuit-level validation for Section 5.3 of the paper (`prime_compiler_v2.pdf`).
Fidelity level: **junction-exact, wiring-ideal** — all log-domain arithmetic is
done by real diode I–V characteristics (the exponential junction law, i.e. the
physics of P3/P4/P12); translinear loop wiring is replaced by ideal voltage
summation. Monte-Carlo mismatch on saturation current, ideality factor, and
mirror ratios. See the paper for what this level does and does not cover.

## Requirements

- `ngspice` in `PATH` (developed against ngspice-46; no PDK needed)
- Python 3.10+, `numpy`; `matplotlib` optional (plots)

Each script generates its netlists into `spice/out/` (gitignored), runs
ngspice in batch mode, and prints a summary. Everything is regenerable;
total runtime for all three scripts is roughly 15–20 minutes.

## Scripts → paper results

| Script | Paper artifact | What it does |
|---|---|---|
| `rmsnorm_agc_sim.py` | Table 6 (mismatch corners), Result 1–3, left panel of Fig. 1 | Open-loop translinear RMSNorm at three technology corners (200 MC runs each) + continuous-time AGC transient (1 pF integrator, settling times) |
| `agc_mismatch_sweep.py` | Table 7 (error decomposition), Result 4, N-scaling | Junction-based RMS detector, loop equilibrium via bisection; splits output error into common-mode gain shift vs. per-channel residual, separately for detector-side and VGA-side mismatch; N = 8/32/128 |
| `calculus_noise_sim.py` | `docs/exp_calculus_noise.md`, rules R18-R21 of `docs/mismatch_calculus_addendum.md` | Thermal/shot noise and a 2 K temperature gradient on the GELU (P11.P6) and AGC RMSNorm benches: `.noise` spectra per channel, `.ac` closed-loop transfer functions (what feedback does to a *spectrum*), paired `.op` gradient runs. No Monte Carlo, no mismatch draw; ~70 s on one core |
| `softmax_agc_sim.py` | Result 5 | Paired comparison: open-loop translinear softmax vs. sum-feedback AGC, both sharing identical mismatched exponential stages; plus AGC transient |

Reproduce everything:

```bash
python3 rmsnorm_agc_sim.py --mc-runs 200
python3 agc_mismatch_sweep.py --mc-runs 100
python3 softmax_agc_sim.py --mc-runs 200
```

Seeds are fixed (`--seed 2026`), so the numbers match the paper exactly.

## Gotcha: ngspice's thermal voltage

ngspice uses the SPICE3-legacy physical constants, so its effective kT/q at
the default temperature (27 °C) is **0.025864890 V** — not the CODATA value.
Any netlist that *drives* a junction with an externally computed voltage
(e.g. the softmax exponential stage, `V = x·n·VT`) must use this value, or
every logit is silently scaled by ~1.0001 and the "ideal device" error floor
rises from ~1e-6 to ~1e-4. Diode-to-diode stages are immune (VT cancels in
ratios). The value was measured with a one-diode probe:
drive 0.5 V, read I, solve `VT = 0.5 / ln(I/Is + 1)`.

Also keep junction currents at µA scale (the testbenches bias accordingly);
at nA scale ngspice's default GMIN leakage (1e-12 S) becomes a visible error
term.

## Version 3 testbenches: paper section / table / figure → script → evidence record


| Paper artifact | Script | Evidence record |
|---|---|---|
| §3.1 Table "Reclassification of the 28 non-M entries" | no script: the reclassification is a mapping argument, stated in full in the paper | the seven rows that *were* tested at circuit level are the §7 rows below, plus §6.1 for backpropagation |
| §4.3 signal-domain axis, Table `tab:domains` | `spice/domain_gates_sim.py` | `docs/domain_axis.md` (model), `docs/exp_domain_gates.md` (gates D1/D2) |
| §4.4 mismatch calculus, Table `tab:rules`, Fig. `fig_v3_calculus.png` | `spice/mismatch_calculus.py`, `spice/predict_gelu_layernorm_sim.py` | `docs/mismatch_calculus.md` (R1–R15), `docs/mismatch_calculus_addendum.md` (R16–R21), `docs/exp_mismatch_calculus.md` |
| §5.2 open-loop translinear pipeline; §5.3 Results 1–3, Table `tab:mismatch`, Fig. `fig_rmsnorm_spice.png` | `spice/rmsnorm_agc_sim.py` | `docs/exp_phase0_gaps.md` §1 (anchor reproduction) |
| §5.3 Result 4, Table `tab:agcsweep`, N-scaling | `spice/agc_mismatch_sweep.py` | `docs/exp_phase0_gaps.md` §1 |
| §5.3 Result 5 (softmax, sum-feedback vs open-loop) | `spice/softmax_agc_sim.py` | `docs/exp_phase0_gaps.md` §1 |
| §5.4 the four validation gaps, Table `tab:gaps` | `spice/phase0_gaps_sim.py` | `docs/exp_phase0_gaps.md` |
| §5.4.4 thermal/shot noise as a third per-channel exposure, Fig. `fig_v3_noise.png` | `spice/noise_reconciliation.py`, `spice/calculus_noise_sim.py` | `docs/exp_noise_reconciliation.md`, `docs/exp_calculus_noise.md` |
| §5.5 full LayerNorm extension (measured per-channel cost) | `spice/predict_gelu_layernorm_sim.py` | `docs/exp_mismatch_calculus.md` §10 |
| §6.1 five gradient sources, Table `tab:trainconds`, Fig. `fig_v3_training.png` | `spice/train_mismatch_absorption.py` | `docs/exp_mismatch_absorption.md` |
| §6.2 sign concordance, Fig. `fig_v3_signflip.png` | `spice/train_sign_concordance.py` | `docs/exp_sign_concordance.md` |
| §6.3 cost of needing no backward path (perturbative scaling in N) | `spice/train_gelu_absorption.py` (Part B) | `docs/exp_gelu_training.md` §5 |
| §6.4 where the mechanism stops working (GELU offset) | `spice/train_gelu_absorption.py` (Part A) | `docs/exp_gelu_training.md` §§3–4 |
| §7.1 Mamba as a voltage-controlled R/C cell, Fig. `fig_v3_mamba.png` | `spice/mamba_vcrc_sim.py` | `docs/exp_mamba_vcrc.md` |
| §7.2 symbol error rate vs noise margin (the exact-alphabet residual) | `spice/symbol_margin_sim.py` | `docs/exp_symbol_margin.md` |
| §7.3 bounded recursion: MCTS and beam search latency/energy | `spice/bounded_recursion_latency.py` | `docs/exp_bounded_recursion.md` |
| §7.4 the race as a top-k selector | `spice/race_topk_sim.py` | `docs/exp_race_topk.md` |
| §7.5 KV-cache eviction by leak alone | `spice/kv_leak_eviction.py`, `spice/kv_gain_cell.py` | `docs/exp_kv_leak_eviction.md` |
| §7.6 the three entries mapped by argument only (CTC loss, speculative decoding, experience replay) | `spice/argument_entries_sim.py` | `docs/exp_argument_entries.md` |
| all five `fig_v3_*.png` figures | `spice/make_v3_figures.py` | reads only `spice/out/*.json` |

`spice/out/` is not shipped in this archive: it is regenerated by the scripts
and runs to several hundred megabytes of netlists. Every number the paper
quotes is also written out in the corresponding `docs/exp_*.md`.

Some `docs/exp_*.md` files cross-reference internal working notes that are not
part of this archive; where they do, the experiment record itself is
self-contained — it carries its own pre-registration, its own numbers and its
own verdict.

