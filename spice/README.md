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
