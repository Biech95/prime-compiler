# Changelog — Computational Primes, version 3

Previous published version: v2.1 (Zenodo DOI 10.5281/zenodo.21179525, 3 July 2026).
Concept DOI (all versions): 10.5281/zenodo.19432503.

Version 3 is a substantial revision. It adds four new result groups, corrects
five published numbers or claims, and corrects the headline algorithm count.

## 1. Corrections to previously published material

- **Algorithm count.** v2.1 stated "107 algorithms, 79 M / 22 G / 6 U". The
  supplementary factorization table it summarized contains **114** rows
  (A1–A20, B1–B12, C1–C7, D1–D8, E1–E4, F1–F6, G1–G7, H1–H5, I1–I3, J1–J5,
  K1–K5, L1–L5, M1–M4, N1–N4, O1–O5, P1–P5, Q1–Q5, R1–R4; no duplicate
  identifier) with **86 M / 22 G / 6 U**. The summary line understated the M
  column and the total by seven. The G and U columns were correct, so the 28
  non-M entries that v3 re-derives are unchanged. All counts in the paper, the
  supplement and the conclusion now use 114.
- **Equilibrium gain error.** v2's "below 1e-4" for the AGC loop is an
  ideal-integrator figure. With a realistic 60 dB integrator DC gain the loop
  closure is 7.2e-4 — still below 1e-3, but seven times above the published
  figure.
- **VGA-only common-mode residual.** The "0.006 %" entries of v2 are the
  quantization floor of the 16-step bisection used to close the loop, not
  physics. With a converged solver the values are 0.0024 % (σ_VGA = 1 %) and
  0.0007 % (0.5 %) — second order in the mismatch, as theory requires, and a
  factor 2.5–9 stronger than published.
- **Normalization priority claim withdrawn.** v2 stated that no prior work had
  implemented this fully in analog. A full-circuit memristor transformer with an
  analog Layer Normalization module and no ADC/DAC exists (Yang, Wang & Zeng,
  IEEE Trans. Circuits Syst. I 69(4):1395–1407, 2022). It is now cited and
  distinguished: the contribution claimed in v3 is the identity between RMSNorm
  and an automatic-gain-control loop — normalization obtained as the equilibrium
  of a feedback constraint rather than computed forward — together with the
  mismatch analysis that identity makes possible.
- **Mamba re-factorized.** A data-dependent gain on a state is
  variable×variable multiplication, not a topology change; the entry moves from
  GPU-mappable to fully mappable, with a circuit-level check.
- **Bibliography.** Fourteen references inherited from v2.1 carried a wrong
  author, title, venue, year, volume or page range. All fourteen are corrected
  and each correction is recorded, with its verification source, in
  `docs/errata_v2.1.md`.

## 2. New in version 3

- **The four "missing primes" re-read.** They are not operations. They are the
  four components a finite circuit lacks relative to a Turing machine: an exact
  alphabet, head movement, an unbounded tape, and program-as-data. The line they
  mark is the circuit/software line, not the analog/digital line, and every
  hardware synthesis flow meets it in the same place. Consequently **0 of 114**
  algorithms are fundamentally unmappable once a compile-time bound is supplied;
  28 entries change status, carrying an explicit bound, a corrected
  factorization, or a digital-by-position verdict (revised counts: 93 M / 20 B /
  1 D / 0 U).
- **Feedback as a design principle extends to training.** A gradient measured on
  the mismatched forward path lets the normalization affine absorb per-channel
  mismatch (0.7–0.95 σ down to 0.06–0.16 σ, the analytic least-squares floor).
  The backward path needs *sign concordance* with the forward crossbar but not
  reciprocity: magnitudes may be wrong by a decade and the backward gain by
  20 %, while the tolerable number of wrong-sign channels is zero, not a small
  percentage. Feedback alignment is unavailable for a *frozen* forward crossbar,
  with the obstruction measured as an eigenvalue condition.
- **A signal-domain axis.** Below analog-vs-digital sits the choice of signal
  domain (current-log, charge, time, KCL current-linear), whose mismatch levers
  differ by up to 10 effective bits for one prime at one corner. Conversion
  avoidance, not conversion accuracy, is what domain assignment optimizes.
- **A mismatch calculus.** 21 composition rules over a per-prime sensitivity
  tuple retrodict 36 measured numbers with none outside a factor 2, and predict
  two composites never simulated before to 1.00× (GELU) and 1.05× (LayerNorm).
- **The four open validation gaps of v2 are closed.** Four-quadrant signalling
  preserves the architectural zero but only with class-AB rails; N = 4096 holds
  at 1.09e-7 ideal-device error provided the summing node stays at virtual
  ground; loop non-idealities leave closure below 1e-3 at 60 dB integrator gain;
  and thermal/shot noise is a third per-channel exposure, equal to the mismatch
  residual at I0 = 1 µA and B = 122 MHz.
- **Seven reclassified entries tested at circuit level** (Mamba, KV-cache
  eviction, MCTS, beam search, CTC loss, speculative decoding, experience
  replay), plus a symbol-margin test of the exact-alphabet residual. Five for five, every bound tuple that was checked named
  the wrong parameter — MCTS is bounded by write latency, not recursion depth;
  beam search by the selection-boundary logit gap; and so on. The bound tuples
  for untested rows are labelled as first drafts.
- **Digital-by-position** is introduced as a status: the operation is mappable
  and the mapping is pointless, because it sits at a boundary where the input is
  already symbolic.

## 3. Presentation

- Estimates are now separated from measurements. The ~2 ns current-mirror
  settling and ~7 ns total of the LayerNorm extension, and the ~12 ns
  propagation delay of the open-loop pipeline, are explicitly marked as design
  estimates with no testbench in this paper.
- The perturbative-gradient cost is now reported per arm: 4.5–210× the forward
  evaluations up to N = 128 at a fixed step size (an arm that fits N^1.39 and
  fails to converge at N = 512), or 2.5–116× across N = 8…512 once the step size
  is annealed with the array size (the arm that recovers the N^0.97 law).
- A new supplement, `supplementary_factorizations_v3.tex`, carries the recounted
  summary statistics and both readings (v2 status and v3 status) of the table.
- The companion preprints are now cited by DOI rather than as undated preprints.

## 4. Record housekeeping

The v2/v2.1 preprint is publicly indexed under two version DOIs of the same
concept record — 10.5281/zenodo.21138358 and 10.5281/zenodo.21179525 (concept
DOI 10.5281/zenodo.19432503). Version 3 supersedes 10.5281/zenodo.21179525.

## Late corrections in this version

* Section 10 no longer claims priority for the partitioning framework or
  the per-prime mismatch budget; both sentences now state what is new
  relative to the established analog design-automation practice of
  budgeting and sensitivity analysis.
* Section 2.2 adds the general-purpose-analog-computer anchor (Pouly,
  Bournez and Graca 2013) so that the circuit/software line is not read as
  a claim about analog computation in general.
* Section 5.3 cites the concurrent time-domain analog softmax circuit
  (Singh et al. 2026); Section 10 cites Akrout et al. 2019 on learning
  without weight transport.
* The reclassification table caption now points to both sections that
  hold circuit-level tests (Sections 6 and 7).
