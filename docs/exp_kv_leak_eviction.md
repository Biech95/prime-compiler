# X2 experiment: content-adaptive KV eviction by leak alone

*Status: pre-registered 2026-09-15, run 2026-09-15. Companion to
`docs/missing_primes_mapping.md` §2 (X2) and §10 (X2 row).
Code: `spice/kv_leak_eviction.py`, `spice/kv_gain_cell.py`.
This document carries no weight beyond the kill-gates recorded below.*

---

## 0. Pre-registration block

**Written and committed to file before any simulation was run.** Everything below
section 0 was filled in afterwards; nothing in section 0 was edited after the
first run.

### Claim under test

`docs/missing_primes_mapping.md` §10, X2 row:

> **Content-adaptive KV eviction for free**: leaky analog storage (gain cell
> τ ≈ 5 ms, Leroux et al. 2025 Nat. Comput. Sci.) + read-triggered refresh
> proportional to attention weight = heavy-hitter eviction (H2O, Zhang et al.
> NeurIPS 2023) by physics.

Mechanism asserted: each cached key/value lives in a leaky cell; every attention
read refreshes the cell's charge in proportion to that token's attention weight
a_{t,i}; cells decaying below a threshold are evicted. Retention equilibrates
between reinforcement and leak — the Physarum/memristor law dD/dt = f(Q) − D.
**No argmin, no scoring logic.**

### Model (fixed before running)

Discrete-time, one step per token, per slot i:

    q_i[t+1] = q_i[t] · exp(−1/τ) + κ · a_{t,i}        (τ in token-times)
    slot i is "live"  ⟺  q_i > θ

θ ≡ 1 without loss of generality (the dynamics are scale-free in (q, θ, κ, q0)
jointly; only κ/θ and q0/θ matter). A newly written token is initialised to
q0 = κ, i.e. one unit of refresh. Per-step order: decay → write token t → read
(attention over live slots, renormalised) → refresh live slots by κ·a_{t,i}.

Cache of K slots, sequence length T, K/T ∈ {0.1, 0.2, 0.4}.
Write rule, two variants:
* **argmin** — free slot if one exists (q ≤ θ), else overwrite the slot with the
  smallest q. This one comparison is a P5 race; it is the only comparison allowed.
* **drop** — free slot if one exists, else the new token is simply not cached.
  No comparison at all.

### Baselines

1. **H2O** (Zhang et al. 2023): K slots split 50/50 into a recent window of K/2
   and the top-K/2 tokens by accumulated attention.
2. **FIFO** sliding window of the last K tokens (Leroux et al.'s actual policy).
3. **Full** cache (oracle, no eviction).

### Metrics

* **m1** Jaccard overlap |live ∩ live_H2O| / |live ∪ live_H2O| per step,
  averaged over steps t ≥ K (before that every policy is trivially identical).
* **m2** attention-output error ‖o_t^cache − o_t^full‖ / ‖o_t^full‖ with the real
  value vectors, averaged over t ≥ K, heads, layers, sequences. Masked softmax
  over a live subset S is exactly the renormalisation a_i / Σ_{j∈S} a_j, so m2 is
  computed exactly from the attention matrix and V.
* **m3** perplexity on ~50 sequences × 512 tokens, per policy, with the cache
  policy actually driving the model forward pass (recursive: the policy's state
  is fed by the *masked* attention it produced, not by oracle attention).

### Sweep

τ ∈ {8, 32, 128, 512} token-times; κ ∈ {0.25, 0.5, 1, 2, 4, 8, 16};
K/T ∈ {0.1, 0.2, 0.4}; write variant ∈ {argmin, drop}.
Report the best τ per K/T and whether one τ serves all three budgets.

### PRE-REGISTERED KILL CRITERIA

* **K1 (kill).** If at K/T = 0.2 the best leak policy's m2 is not better than
  FIFO's by at least 20 % relative, the "content-adaptive for free" claim is
  **KILLED**: leak gives nothing beyond a sliding window.
* **K2 (downgrade).** If the best leak policy's m2 is worse than H2O's by more
  than 30 % relative, the claim is **downgraded to "partial"**: leak captures
  some but not all of the heavy-hitter benefit.
* **K3 (design constraint, not a kill).** If no single τ within a factor 4 works
  across the three K/T budgets (τ must be retuned per budget), record it as a
  design constraint.
* **K4 (strengthening).** If the drop variant is within 10 % of the argmin
  variant, the P5 race is unnecessary; reporting this strengthens the claim.

### Trace source decision rule

If `torch` + `transformers` import and a small causal LM is cached locally, use
real attention traces and real perplexity. Otherwise fall back to synthetic
traces (Zipf sinks + exponential locality + uniform noise) and state explicitly
that synthetic results show only the mechanism, never a real-model benefit.

---

*(Sections 1 onward were written after the runs.)*

## 1. Method and trace source

**Trace source: REAL.** `torch` 2.11+rocm7.11, `transformers` 5.5.0 and a local
GPT-2 small (124 M, 12 layers × 12 heads) were available in
`~/.cache/huggingface`, together with `wikitext-2-raw-v1`. All numbers below
come from real attention matrices and real value vectors; m3 is a real
perplexity. The synthetic Zipf/locality generator is implemented
(`synthetic_traces()`, `--synthetic`) but was **not used** for any reported
number.

**Gauge anchor.** The experiment needs control over the per-step attention, so
GPT-2's forward pass is re-implemented from the state dict rather than called
through `transformers`. `kv_leak_eviction.py --check` compares the two:
maximum relative logit deviation **1.6e-6** (GPU) / **6.8e-7** (CPU). The
hand-rolled path is the same model.

**Encoding.** K physical slots are represented as a per-token charge vector
q[0..T−1] with the invariant |{i : q_i > θ}| ≤ K. A token occupies at most one
slot, so this is an exact re-encoding and it makes the whole sweep vectorisable
over (config × sequence × head). Masked softmax over a live subset S equals the
renormalisation a_i / Σ_{j∈S} a_j, so m2 is computed exactly, not approximated.

**Refresh signal.** A cell is refreshed by the attention weight the cache
*actually applied* (renormalised over the live set), not by the full-attention
weight — that is the current that would physically flow through the cell.

**Two stages.**
* Stage A (`--stage a`, 69 s): oracle traces. Policy state is driven by the
  weights the cache applied, but the underlying scores come from a
  full-attention forward pass. 8 sequences × 512 tokens × 12 layers × 12 heads
  = 1152 (sequence, head) traces per layer. Gives m1, m2 over the full sweep.
* Stage B (`--stage b`, 199 s): closed loop. The policy drives the actual
  forward pass, so evicted hidden states propagate into all later keys and
  values. 50 sequences × 512 tokens. Gives m3.

### Deviations from the pre-registration

Three, all made after the first run and all recorded here rather than by
editing section 0:

1. **κ grid extended** from the pre-registered {0.25 … 16} to {0.25 … 64}. In a
   single-layer timing run the optimum sat on the upper boundary at two of the
   three budgets, so the grid was not bracketing the optimum; it was widened
   before the first full 12-layer run. Restricting the final 12-layer grid back
   to κ ≤ 16 post hoc gives best m2 = 0.1928 / 0.1222 / 0.0729 (vs 0.1880 /
   0.1222 / 0.0688), so K1 and K2 are unaffected. **K3 is affected**: with
   κ ≤ 16 no single (τ, κ) is within 7 % at all three budgets (the best
   candidate, τ=32/κ=16, is +54 % at K/T = 0.4), so the "one configuration
   serves every budget" result in §7 depends on the extension. The extension is
   also what exposed the κ-saturation described in §4. Flagged as a deviation
   that moves the evidence in favour of the claim.
2. **FIFO+sink4 baseline added.** Plain FIFO turned out to be catastrophically
   bad on GPT-2 (m2 > 1.0 in deep layers) because a sliding window discards the
   token-0 attention sink — in the head measured in §6 that sink absorbs 38 %
   of the attention mass on average. Passing K1 against a baseline that is
   broken for a known, unrelated reason would be worthless, so **FIFO+sink4**
   (StreamingLLM-style: 4 pinned initial tokens + a window of K−4) was added.
   Every verdict below is reported against both; this one is the honest one.
3. **H2O's heavy/recent split was tuned** over {0.25, 0.5, 0.75} instead of
   being fixed at the pre-registered 50/50, so that K2 compares the leak
   against the *best* H2O rather than a handicapped one. This makes K2 harder
   to pass, not easier.

Deviations 2 and 3 both move the comparison against the claim under test.
Deviation 1 moves it in favour, and is flagged as such.

## 2. Results — m2, attention-output error (stage A)

Mean over 12 layers × 12 heads × 8 sequences × steps t ≥ K. "argmin" = the one
P5 race; "drop" = no comparison at all. H2O's heavy/recent split was itself
tuned over {0.25, 0.5, 0.75} so that K2 compares against the best H2O; the
column below takes the best split **per layer**, which is a per-layer oracle
and therefore flatters H2O slightly. With a single global split the best H2O is
0.2014 / 0.1262 / 0.0625 (at 0.25 / 0.5 / 0.5) instead of 0.1981 / 0.1230 /
0.0618 — a difference too small to move any verdict.

| K/T | FIFO | FIFO+sink4 | H2O | leak/argmin (best) | leak/drop (best) |
|---|---|---|---|---|---|
| 0.1 | 1.3691 | 0.2279 | 0.1981 | **0.1880** (τ=8, κ=64) | 0.2212 (τ=32, κ=2) |
| 0.2 | 1.2288 | 0.1709 | 0.1230 | **0.1222** (τ=128, κ=4) | 0.1578 (τ=32, κ=8) |
| 0.4 | 1.0931 | 0.1212 | 0.0618 | 0.0688 (τ=32, κ=64) | 0.1027 (τ=128, κ=2) |

Relative to the baselines (positive = leak better):

| K/T | argmin vs FIFO | argmin vs FIFO+sink4 | argmin vs H2O | drop vs FIFO+sink4 | drop vs H2O |
|---|---|---|---|---|---|
| 0.1 | +86.3 % | +17.5 % | +5.1 % | +3.0 % | −11.6 % |
| 0.2 | +90.1 % | +28.5 % | +0.6 % | +7.7 % | −28.3 % |
| 0.4 | +93.7 % | +43.2 % | −11.3 % | +15.2 % | −66.1 % |

## 3. Results — m1, Jaccard overlap with H2O's live set

| K/T | leak/argmin (best-m2 config) | leak/drop | live fraction (argmin) |
|---|---|---|---|
| 0.1 | 0.632 | 0.581 | 0.908 |
| 0.2 | 0.602 | 0.554 | 1.000 |
| 0.4 | 0.609 | 0.600 | 0.879 |

The leak does **not** reproduce H2O's set: only ~0.60 Jaccard, i.e. roughly a
quarter of each policy's retained tokens is not in the other's — yet the
attention-output error comes out slightly *better* at K/T = 0.1 and 0.2 and
11 % worse at 0.4. Leak finds a *different*, comparably good retained set, so
"leak = H2O by physics" is the wrong description; "leak is a comparably good
eviction rule of the same family" is the right one. Note also the live
fractions: at the best configurations the cache is 88–100 % full, i.e. the
threshold frees almost nothing on its own (see §4 and §8).

## 4. τ–κ sweep (m2, argmin variant, all 12 layers)

τ is in token-times; θ ≡ 1, so κ is really κ/θ, i.e. the inverse threshold.

**K/T = 0.2** (FIFO 1.2288 · FIFO+sink4 0.1709 · H2O 0.1230)

| τ \ κ | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 3.553 | 3.553 | 0.676 | 0.404 | 0.332 | 0.281 | 0.234 | 0.194 | 0.157 |
| 32 | 3.553 | 3.553 | 0.457 | 0.230 | 0.170 | 0.137 | 0.131 | 0.131 | 0.131 |
| 128 | 3.553 | 3.553 | 0.247 | 0.122 | **0.122** | 0.122 | 0.122 | 0.122 | 0.122 |
| 512 | 3.553 | 3.553 | 0.371 | 0.356 | 0.356 | 0.356 | 0.356 | 0.356 | 0.356 |

**K/T = 0.4** (H2O 0.0618)

| τ \ κ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 8 | 0.684 | 0.418 | 0.346 | 0.297 | 0.254 | 0.213 | 0.170 |
| 32 | 0.468 | 0.249 | 0.193 | 0.144 | 0.106 | 0.079 | **0.069** |
| 128 | 0.263 | 0.097 | 0.073 | 0.073 | 0.073 | 0.073 | 0.073 |
| 512 | 0.179 | 0.133 | 0.133 | 0.133 | 0.133 | 0.133 | 0.133 |

Two structural facts fall out of the grid:

* κ ≤ θ is degenerate (m2 ≈ 3.5 everywhere): a cell written with q0 = κ < θ is
  dead before the next read, so the cache collapses to self-attention. The
  model needs κ > θ, which in circuit terms is "one write must clear the
  eviction threshold".
* For the **argmin** variant m2 saturates as κ grows. That is the informative
  part: large κ/θ means the threshold almost never fires, the leak evicts
  nothing, and the P5 race does all the selecting. In that regime the physics
  supplies the *score* (an exponentially-discounted accumulated attention) and
  the comparison supplies the *selection*. For the **drop** variant the grid is
  non-monotone in κ with an interior optimum — as it must be, because there the
  threshold is the only eviction mechanism there is.

## 5. Results — m3, closed-loop perplexity (stage B)

GPT-2 small, 50 sequences × 512 tokens of wikitext-2, cache policy driving the
actual forward pass. Full cache: **ppl = 37.298**.

| policy | K/T = 0.1 | K/T = 0.2 | K/T = 0.4 |
|---|---|---|---|
| full cache | 37.30 | 37.30 | 37.30 |
| FIFO | 5845.9 | 2041.7 | 331.1 |
| FIFO+sink4 | 78.09 | 58.26 | 45.20 |
| H2O | **51.95** | **41.86** | **38.15** |
| leak/argmin (per-budget τ, κ) | 60.58 | 44.36 | 39.27 |
| leak/argmin (universal τ=32, κ=64) | 62.70 | 46.62 | 39.27 |
| leak/drop (no comparison) | 65.24 | 46.18 | 39.93 |

In excess perplexity over the full cache, H2O keeps a consistent lead that m2
did not show: at K/T = 0.2, H2O costs +4.57 ppl and leak/argmin +7.06 ppl, i.e.
leak's excess is 54 % larger even though its m2 was 0.6 % *better*. The
oracle-trace metric flatters the leak; the closed loop is the honest one. The
ordering leak ≫ FIFO+sink4 ≫ FIFO holds in both metrics.

## 6. SPICE: is the discrete model physically realisable?

`spice/kv_gain_cell.py`, ngspice-46, one RC storage node per cached token.

    C      = 10 fF          R_leak = 1.28e10 Ω     τ = R·C = 128 µs
    1 token = 1 µs  ⇒  τ = 128 token-times (the stage-A optimum at K/T = 0.2)
    θ = 100 mV,  κ = 400 mV per unit attention weight  (κ/θ = 4)
    refresh = 50 ns current pulse, amplitude I = κ·C·a_{t,i} / 50 ns

Three cells are driven with three real attention columns from the same GPT-2
trace (layer 5, head 3, 400 tokens of wikitext-2):

| cell | token | mean a_{t,i} after write | peak | final | outcome |
|---|---|---|---|---|---|
| attention sink | 0 | 3.76e-1 | 18883 mV | 18252 mV | never evicted (≥ 400 token-times) |
| heavy hitter (strongest non-sink) | 1 | 6.76e-3 | 1249 mV | 76 mV | evicted after 365 token-times |
| one-off | 67 | 3.29e-4 | 420 mV | 34 mV | evicted after 191 token-times |

* **Effective τ: designed 128.00 µs, measured 128.57 µs (+0.45 %)**, from a
  least-squares exponential fit to the one-off cell's free decay.
* Analytic check: a never-refreshed cell crosses θ after τ·ln(κ/θ) = 177.4 µs =
  177 token-times; the one-off cell, which still receives a trickle of
  attention, survives 191.
* The node voltage tracks the discrete difference equation
  q ← q·e^(−1/τ) + κ·a to **0.039 % of peak** on all three columns.

So the discrete model is the circuit, to four digits. Two caveats that the
circuit itself exposes:

* The sink cell equilibrates at **18.9 V** (= κ·ā·τ = 0.4 × 0.376 × 128). A
  linear RC node has unbounded dynamic range; a real gain cell saturates near
  ~1 V. Saturation is probably harmless for eviction (it compresses the top of
  the ranking, where the ordering does not matter — everything up there is
  retained), but it is not what was simulated, and the argmin variant *does*
  read the analog value, so a saturating cell would degrade the race.
* This section demonstrates realisability only. It says nothing about whether
  the policy is any good; that is what sections 2–5 measure.

## 7. Verdicts against the pre-registered kill criteria

**K1 — NOT killed.** At K/T = 0.2 the best leak policy's m2 is **90.1 %** better
than FIFO's (0.1222 vs 1.2288), far past the 20 % bar. Against the stronger
post-hoc FIFO+sink4 baseline it is still **28.5 %** better, so the verdict does
not depend on FIFO's attention-sink pathology. The claim survives K1.
*Caveat:* the comparison-free **drop** variant clears the bar against
pre-registered FIFO (+87.2 %) but only by +7.7 % against FIFO+sink4 — it would
**fail** K1 had the criterion been written against StreamingLLM rather than
plain FIFO.

**K2 — NOT downgraded (for the argmin variant).** Best leak m2 vs best H2O m2:
−5.1 % (K/T=0.1), −0.6 % (0.2), +11.3 % (0.4), where positive = worse. The
worst case is 11.3 %, well inside the 30 % bar. The **drop** variant is +11.6 %
/ +28.3 % / +66.1 % worse and therefore **triggers K2 at K/T = 0.4**: without
the comparison, the claim is "partial" at the largest budget.
On m3 (closed loop), H2O leads at every budget and by more than m2 suggested —
recorded in §5, but K2 is defined on m2 and on m2 it does not fire.

**K3 — split, and the wording is the reason.** K3 was written ambiguously and
its two readings disagree, so both are reported.

*Reading (a), literal ("the argmin-best τ per budget must span ≤ factor 4").*
The best τ is 8 / 128 / 32 at K/T = 0.1 / 0.2 / 0.4 — a span of **factor 16**.
**K3 fires**: recorded as a design constraint.

*Reading (b), practical ("does one τ work everywhere?").* Yes, comfortably. The
m2 surface is flat in τ near its optimum, so the argmax being noisy costs
almost nothing: **τ = 32, κ/θ = 64** is within **+2.6 % / +7.2 % / +0.0 %** of
the per-budget best at all three budgets, and κ need not be retuned either.
Under this reading K3 does not fire.

Reading (b) is the one that matters for a compiler — the span of argmaxes on a
flat surface is not a design constraint, the width of the usable window is.
That window is **τ ∈ [32, 128] token-times at T = 512**, a factor 4: τ = 512
(≈ the whole sequence) fails everywhere (+93 % to +191 % over the per-budget
best) and τ = 8 fails at large budgets (+146 % at K/T = 0.4).

*Both readings depend on deviation 1 in §1.* On the pre-registered κ ≤ 16 grid
the numbers reverse: reading (a) gives best τ = 32 / 128 / 128, a span of
factor 4, so K3 would **not** fire; reading (b) gives no single (τ, κ) within
7 % at all three budgets (the best candidate, τ = 32 / κ = 16, is +45 % at
K/T = 0.4), so K3 **would** fire. The κ extension flipped both. The claim that
is robust to all four combinations is only the narrow one: **τ must be of order
tens of tokens, not hundreds and not thousands.**

**K4 — NOT satisfied on the pre-registered primary metric; satisfied on m3.**
The drop variant's m2 is **17.7 % / 29.1 % / 49.2 %** worse than argmin's at the
three budgets — all beyond the 10 % bar, so on m2 the P5 race is **not**
dispensable. On closed-loop perplexity the same comparison gives **+7.7 % /
+4.1 % / +1.7 %**, i.e. *inside* the 10 % bar at every budget. K4 did not name
its metric; both numbers are reported and the split is the finding, not a
choice to be made after the fact. The defensible summary: dropping the
comparison costs a large fraction of the attention-output fidelity but only a
few percent of perplexity.

## 8. What the claim actually buys

The mapping doc's X2 row says "content-adaptive KV eviction for free … no
argmin, no scoring logic". The experiment splits that into two claims that
fare differently:

1. **The scoring is free.** Confirmed. A leaky cell refreshed by its own read
   current *is* an exponentially-discounted accumulated-attention score. No
   counter, no adder, no comparator is needed to maintain it, and using it
   matches a tuned H2O within 11 % on attention-output error while retaining a
   set that overlaps H2O's by only ~0.6 Jaccard.
2. **The selection is not free.** The best configurations sit where the leak
   threshold almost never fires (live fraction 0.88–1.00) and one comparison —
   the P5 race over the analog charges — does the evicting. Removing it
   entirely still works and still beats a sliding window, but costs 18–49 % in
   m2 (2–8 % in perplexity). "No argmin" is therefore too strong; the honest
   version of the X2 row is *"the score comes from the physics, the selection
   still costs one P5 race."* This is a weaker claim than the doc makes, but it
   is a bounded one, and B(N slots) already budgets a race.

A third, quantitative consequence for the compiler, which was not anticipated:
**Leroux's retention is ~150× too long.** τ ≈ 5 ms at one token per µs is 5000
token-times; the usable window measured here is τ ≈ 32–128 token-times at
T = 512. A 5 ms cell driven at µs token rate never evicts anything and the
policy degenerates to pure argmin. Either the cell must be made deliberately
leakier (τ ≈ 32–128 µs, i.e. the opposite of the usual retention-engineering
direction), or the token rate must drop by two orders of magnitude. Retention
becomes a *compiler-set parameter tied to the sequence length*, not a device
constant to be maximised.

## 9. Limitations

* One model, one size, one dataset: GPT-2 small (124 M) on wikitext-2. Nothing
  here shows the τ window or the leak/H2O gap transfers to a modern model with
  GQA, RoPE and a 100k+ context, where the attention statistics differ.
* T = 512 only. τ is reported in token-times at a single sequence length, so
  whether the usable window is absolute (τ ≈ 32–128 tokens) or relative
  (τ/T ≈ 1/16 to 1/4) is **not** determined by this experiment. The distinction
  matters for the design constraint in §8 and needs a T sweep to settle.
* Stage A is an oracle-trace evaluation and, as §5 shows, it flatters the leak
  relative to the closed loop. Only m3 is a deployment-relevant number.
* m3 is teacher-forced perplexity, not generation quality; H2O's published
  benefits are largely on long-generation tasks that are not measured here.
* The SPICE cell is a linear RC with ideal current pulses: no transistor, no
  read disturb, no write latency, no device-to-device mismatch, no saturation
  (see the 18.9 V node in §6), and no ADC on the argmin race. Fidelity level is
  below the "junction-exact, wiring-ideal" standard of the other testbenches in
  `spice/` — it is a realisability demonstration, not a circuit result.
* The eviction race is assumed to be exact. A P5 race over analog charges that
  differ by less than the device noise floor will pick the wrong cell; the
  effect of that on m2 is not measured.
* H2O was tuned only over its heavy/recent split; the leak was tuned over a
  4 × 9 grid. The leak therefore had the larger hyperparameter budget, which
  biases §2 in its favour.
* κ ≤ θ was left in the sweep and is degenerate by construction (§4); it
  contributes nothing except a sanity check.

## 10. Reproducing

    cd spice
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 kv_leak_eviction.py --check    # gauge anchor
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 kv_leak_eviction.py --stage a  #  69 s
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 kv_leak_eviction.py --stage b  # 199 s
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python3 kv_gain_cell.py                #  ~60 s

Seeds fixed at 2026. Total compute for everything reported here: **~6 minutes**
(AMD Radeon 8060S / gfx1151, torch 2.11+rocm7.11; the venv at
`~/venv-gfx1151` is required — system torch does not support gfx1151).
Stage A also runs on CPU with `--device cpu`. Outputs land in `spice/out/`:
`kv_leak_stage_a.json` (full per-layer sweep), `kv_leak_stage_a_summary.json`,
`kv_leak_best.json`, `kv_leak_stage_b.json`, `kv_gain_cell.json`,
`kv_gain_cell.npz`. Synthetic-trace fallback: `--synthetic`.
