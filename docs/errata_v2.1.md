# Errata for prime_compiler v2.1 (surfaced 2026-09-15 by the X1–X4 literature gate)

Not yet applied to `prime_compiler_v2.tex` (published as v2.1 on Zenodo). Each
item gives the exact location and the replacement text. Items 1–3 are pure
corrections; items 4–5 change a classification and therefore the headline
counts (79/22/6), so they belong to a v2.2 together with the missing-primes
analysis, not to a silent fix.

## 1. Ref. [18] — wrong authors and superseded title (`prime_compiler_v2.tex` line 579)
Old:
    \bibitem{raceit2023} H.~Errahmouni~Barkam \textit{et al.}, ``RACE-IT: Reconfigurable analog CAM-crossbar engine for in-memory transformer acceleration,'' \textit{arXiv:2312.06532}, 2023.
New:
    \bibitem{raceit2023} L.~Zhao, A.~Natarajan, L.~Buonanno, A.~Gajjar, R.~M.~Roth, S.~Serebryakov, J.~Moon, J.~Ignowski, and G.~Pedretti, ``RACE-IT: A reconfigurable analog computing engine for in-memory transformer acceleration,'' \textit{arXiv:2312.06532}, v3, 2025.
Verified against arXiv abstract page (v1 Nov 2023, v3 Aug 2025).

## 2. Ref. [19] — wrong year (line 581)
Old:
    ... \textit{Nature Communications}, 2025.
New:
    X.~Zhang, M.~Hu, S.~Lu, S.~Kim, E.~Y.-J.~Lee, Y.~Liu, and W.~D.~Lu, ``Compute-in-memory implementation of state space models for event sequence processing,'' \textit{Nature Communications}, vol.~17, 1513, 2026.
Published 9 Jan 2026 (DOI 10.1038/s41467-025-68227-w; the DOI slug carries the submission year).

## 3. §2.2 X3 — unsourced figure (line 139)
Old:
    ... Physical hardware is finite. GPU stack limited to ${\sim}1024$ frames. Blocks: MCTS.
New (either cite the CUDA per-thread stack documentation, or drop the number):
    ... Physical hardware is finite; on GPUs the per-thread stack is a fixed, small, configurable allocation. Blocks: MCTS.

## 4. Table 3 row 28b and supplement K2 — Mamba mis-factorized (line 189; supplement line 259)
Gu & Dao (arXiv:2312.00752, Alg. 2): A is a learned parameter; only Δ, B, C are
input-dependent. A data-dependent gain on a state is variable×variable
multiplication (P11), not a topology change. Circuit-level check in
`docs/exp_mamba_vcrc.md`: a capacitor with two voltage-controlled conductances
tracks the selective-scan recurrence to 7–15 bits with ideal devices.
Old:
    28b & Mamba (selective SSM) & P8$\cdot$P2$\cdot$P1$\cdot$P11 + X2 & \textbf{G} & Input-dep.\ weight reprogramming \\
New:
    28b & Mamba (selective SSM) & P8$\cdot$P2$\cdot$P1$\cdot$P11 & \textbf{M} & $\Delta(x)$, $B(x)$, $C(x)$ are data-dependent gains (P11) on a fixed $A$; cost: $\Delta$ dynamic range per channel \\
Consequence: counts become 80 M / 21 G / 6 U in abstract (line 26), §3 summary (line 210), conclusion, and supplement key findings (line ~419) — unless v2.2 adopts the bounded-mappable status, in which case the whole count line is rewritten.

## 5. `src/prime_compiler/factorizations.py` — `sort` contradicts ref. [7]
`sort` was ({P5}, {X2}, G, "Iterated WTA needs dynamic routing"). Ref. [7]
(multi-computation, Op. 3) states that one race yields the complete ranking;
Yu et al., Nat. Electron. 2025 (doi:10.1038/s41928-025-01405-2) demonstrate
sort-in-memory on memristors. Applied in code on 2026-09-15 (see git diff);
not in any published table, so no paper text changes.

## Not errata, but v2.2 material
- Table 2 columns "Analog / GPU" should read "Circuit / Software" (X3 is the
  HLS obstruction; Vitis HLS UG1399: "Recursive functions cannot be synthesized").
- Status B (bounded-mappable) replacing G/U for 28 entries; see
  `docs/missing_primes_mapping.md` §6.
- Feedback-as-design-principle (§7.2) extends to training: physics-aware or
  sign-concordant gradients absorb VGA mismatch (`docs/exp_mismatch_absorption.md`).

## 6. Bibliography: 14 references of v2.1 carried wrong fields (corrected in v3)

A citation audit of every bibitem against Crossref found fourteen entries that
exist in `prime_compiler_v2.tex` (published as v2.1) with at least one wrong
field. All fourteen are corrected in `prime_compiler_v3.tex`. None is a
fabricated reference; every one resolves to a real record. Each row below was
re-verified against the Crossref REST API (`api.crossref.org`) before being
written here; the DOI given in the last column is the record that was matched.

| # | Key | Wrong field in v2.1 | Corrected value in v3 | Verified against |
|---|---|---|---|---|
| 6.1 | `denram2024` | first author `T.~Dalgaty`; title truncated to `...with RRAM`; no volume | `S.~D'Agostino et al.`; `DenRAM: Neuromorphic dendritic architecture with RRAM for efficient temporal processing with delays`; `vol.~15` | Crossref 10.1038/s41467-024-47764-w (1st author Simone D'Agostino, Nat. Commun. 15, 2024) |
| 6.2 | `rram_swish2022` | authors `H.~Alahmadi et al.`; short title; venue `IEEE ICECS`; no pages | `A.~Fatima and A.~Pethe`; `Implementation of RRAM based Swish activation function and its derivative on 28nm FD-SOI`; `Int. Electrical Engineering Congress (iEECON)`; `pp.~1--4` | Crossref 10.1109/ieecon53204.2022.9741701 (Afroz Fatima, 2 authors, iEECON 2022, pp. 1-4) |
| 6.3 | `reram_aln2022` | author `Y.-C.~Hou et al.`; title `CNN accelerator`; journal `IEEE Trans. Electron Devices`; year 2022; no volume/pages | `S.-G.~Gi, H.~Lee, J.~Jang, and B.-G.~Lee`; `...convolutional neural network accelerator...`; `IEEE Trans. Industrial Electronics`; `vol.~70, pp.~6442--6451`; year **2023** | Crossref 10.1109/tie.2022.3190876 (Sang-Gyun Gi, 4 authors, IEEE TIE 70:6442-6451, June 2023) |
| 6.4 | `cimmlc2024` | first author `Y.~Zhao`; no pages | `S.~Qu et al.`; `pp.~185--200` | Crossref 10.1145/3620665.3640359 (Songyun Qu, ASPLOS 2024, pp. 185-200) |
| 6.5 | `cinm2024` | first author `S.~Khan`; title without `(Cinnamon)`; no pages | `A.~A.~Khan et al.`; `CINM (Cinnamon): ...`; `pp.~31--46` | Crossref 10.1145/3622781.3674189 (Asif Ali Khan, ASPLOS 2024, pp. 31-46) |
| 6.6 | `cmswitch2025` | first author `Z.~Chen`; no pages | `S.~Zhao et al.`; `pp.~63--78` | Crossref 10.1145/3676641.3716248 (Shixin Zhao, ASPLOS 2025, pp. 63-78) |
| 6.7 | `htvm2023` | first author `N.~Delm` | `J.~Van Delm et al.` | Crossref 10.1109/dac56929.2023.10247664 (Josse Van Delm, DAC 2023) |
| 6.8 | `hybrid_partition2024` | first author `A.~Varadarajan`; venue given only as `IEEE`; year 2024 | `F.~Kre\ss{} et al.`; `IEEE Int. Symp. on Quality Electronic Design (ISQED)`; year **2025** | Crossref 10.1109/isqed65160.2025.11014471 (Fabian Kress, ISQED 2025) |
| 6.9 | `aihwkit2023` | first author `M.~J.~Rasch` (Rasch is last author); title truncated; no volume | `M.~Le Gallo et al.`; `...kit for neural network training and inference`; `vol.~1` | Crossref 10.1063/5.0168089 (Manuel Le Gallo, APL Machine Learning 1, 2023) |
| 6.10 | `navcim2024` | first author `S.~Kim`; title `analog CIM`; no pages | `J.~Park, B.~Kim, and H.~Sung`; `...analog computing-in-memory architectures`; `pp.~168--182` | Crossref 10.1145/3656019.3676946 (Juseong Park, 3 authors, PACT 2024, pp. 168-182) |
| 6.11 | `cryptoristor2024` | first author `H.~Kim`; title `...for true random number generation` | `S.-I.~Kim et al.`; `Cryptographic transistor for true random number generator with low power consumption` | Crossref 10.1126/sciadv.adk6042 (Seung-Il Kim, Sci. Adv. 10, 2024) |
| 6.12 | `mram_sigmoid2022` | first author `M.~Amin`; title `analog sigmoid`; no pages | `M.~H.~Amin et al.`; `...analog sigmoid function...`; `pp.~319--323` | Crossref 10.1145/3526241.3530376 (Md Hasibul Amin, GLSVLSI 2022, pp. 319-323) |
| 6.13 | `ibm_supernetwork2026` | title truncated (`...mapping to mixed-precision hardware...`); no volume | `Supernetwork-based efficient mapping of deep learning applications to mixed-precision hardware using model adaptation`; `vol.~17` | Crossref 10.1038/s41467-026-71071-1 (Hadjer Benmeziane, Nat. Commun. 17, 2026) |
| 6.14 | `legno2020` | authors `S.~Achour et al.` (there are only two); title in the pre-publication word order; no pages | `S.~Achour and M.~Rinard`; `Noise-aware dynamical system compilation for analog devices with Legno`; `pp.~149--166` | Crossref 10.1145/3373376.3378449 (Sara Achour, 2 authors, ASPLOS 2020, pp. 149-166) |

Five further bibliography corrections from the same audit (`ark2024`, `shem2024`,
`yu2025sort`, `vittoz1977`, `leblanc1922`) are **not** errata for v2.1: those
keys do not exist in `prime_compiler_v2.tex` and appear for the first time in v3.
