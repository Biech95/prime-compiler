# Prime Compiler — Research Roadmap

*Created 2026-07-03. Companion to `prime_compiler_v2.1.tex`.*

## Organizing principle

The mapping is done and it was the cheap part. The 107 algorithm→prime
factorizations are interpretive and fast to produce; they carry no weight on
their own. **All weight lives in the physics validation.** This roadmap is a
deliberate pivot away from *more mapping* toward *things that survive a test
against physics, silicon, or an external collaborator.*

Every phase has a **kill-gate up front** — run the cheap test before building,
not after publishing.

**Hard rule:** no more domain-mapping papers ("primes for domain X"). That is
the Math-Primes failure mode: volume, not weight. Extend in the *physics*
(substrate, training), never in the *analogy*.

---

## Phase 0 — Finish the one result you already have  (now, ~2 weeks)

Depth over breadth. The RMSNorm/AGC result is ~80% there; the paper's own
"What this validation does not cover" lists the gaps. Turn a framework paper
into *the* definitive analog-normalization result.

- **Do:** extend SPICE to signed / four-quadrant signalling (currently
  magnitude-only currents); detector dynamic-range study at N=4096; add
  thermal noise; drop the loop-wiring idealization one notch.
- **Kill-gate:** if the common-mode-benign story breaks at N=4096 or under
  thermal noise, that is the single most important thing to learn and it
  changes everything downstream. Test first.
- **Why first:** it is the only already-validated result. Depth here is the
  credibility currency with the scene (the Jumper lesson: one deep result
  beats five broad framework papers).

## Phase 1 — Generalize the feedback principle into a taxonomy  (parallel, cheap)

Extend the *one* validated principle — feedback demotes device mismatch to a
common-mode gain error — across every P12-containing operation. Reuses the
existing SPICE harness, so low cost.

- **Deliverable:** table of operation × {feedback-realizable? mismatch payoff
  in effective bits}. Candidates: Adam `1/sqrt(v)`, attention scaling,
  whitening/decorrelation, L2 embedding norm, BatchNorm.
- **Kill-test per op:** does feedback actually help, or is the exposure
  *upstream* (as with softmax's exp stage)? The boundary is already known from
  the softmax result — systematize it.
- **Output:** natural v3 paper, falsifiable, zero new mapping.

## Phase 2 — The strategic bet: pick ONE  (gated)

### 2a Photonics — highest scene leverage
Same "operation → physical primitive → domain assignment" logic on photonic
accelerators (MZI meshes do matmul natively; the nonlinearity is the
bottleneck — same partitioning problem, underserved).
- **Gate before investing:** 3 days, read 2–3 photonic-accelerator papers,
  confirm the analog/digital partitioning problem there is (a) real and
  (b) unsolved. If yes → biggest door to Normal / Extropic / Jülich-type people.

### 2b Equilibrium Propagation bridge — most original
Connect the feedback principle to analog *training* (the X4 blocker).
EqProp (Scellier/Bengio) is the ML-theory version of "training via feedback
equilibrium."
- **Gate:** 1-week theory sketch — is EqProp's feedback equilibrium expressible
  in the prime algebra? Yes/No decides.

**Choose by:** reach (2a) vs. originality (2b). Not both.

## Phase 3 — Collaborator / real-silicon path  (the trajectory-changer)

The paper says "silicon remains the final arbiter." Every direction above hits
a ceiling without someone who can fab or has measured devices. This is an
outreach task, not a paper task.

- **Contact artifact:** prime_compiler + the sharpened RMSNorm result (Phase 0).
  *Not* the Unified-Primes series.
- **Target:** one analog-CIM or photonics group, or a thermodynamic-computing
  startup (Normal, Extropic). The engineer who is fluent in ML + hardware +
  software is their stated bottleneck — that profile.
- **This is the gate between "interesting preprints" and "work that ships."**

---

## Thread

- Phase 0 + 1 = safe depth (reuse, falsifiable).
- Phase 2 = the one bet, with a gate in front.
- Phase 3 = the multiplier that makes everything else land.
