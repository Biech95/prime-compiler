"""Multi-computation fusion detection — Phase 3 of the prime compiler."""

from __future__ import annotations
from dataclasses import dataclass
from prime_compiler.primes import Prime as P


@dataclass
class FusionOpportunity:
    """A detected multi-computation fusion."""
    pattern_name: str
    fusion_class: str
    multiplier: str
    node_names: list
    primes_involved: set

    def __str__(self):
        names = ", ".join(self.node_names[:3])
        if len(self.node_names) > 3:
            names += f", ... ({len(self.node_names)} nodes)"
        return f"{self.pattern_name} [{self.fusion_class}] {self.multiplier} — {names}"


# Fusion patterns: (required_primes, class_name, count, description, priority)
#
# `count` is the number of operations obtained per physical event, as in the
# fusion table of the paper (v3.1) and the multi-computation companion paper
# (v3.1, doi:10.5281/zenodo.23218448).  It is an operation count, not a
# throughput or precision multiplier: from a single race only the winner
# reaches a usable precision.  "not re-established" marks patterns whose count
# the companion paper does not carry.
# `priority` fixes the greedy matching order (higher first); it reproduces the
# order of version 3, so the set of detected fusions is unchanged.
FUSION_PATTERNS = [
    ({P.P3, P.P5},         "Race",          "4 ops/event (5 with two-array coincidence)",
     "Softmax via race: winner + Z estimate + log-ratios + entropy estimate", 4),
    ({P.P3, P.P5, P.P4},   "Race-Loss",     "not re-established",
     "Softmax + CrossEntropy fused into single race", 5),
    ({P.P1, P.P2, P.P11, P.P12}, "Normalization", "N outputs + 1 static channel",
     "RMSNorm/LayerNorm via AGC equilibrium", 3),
    ({P.P10, P.P7},         "Threshold",     "not re-established",
     "Ising/Boltzmann: decision + confidence", 1),
    ({P.P1, P.P2, P.P8},   "Equilibrium",   "4 independent observables",
     "Resistor network: solution + R_eff + spectrum + currents", 2),
]


def find_fusions(nodes: list) -> list[FusionOpportunity]:
    """Scan node list for multi-computation fusion opportunities.

    Uses greedy non-overlapping matching: once nodes are consumed by a
    fusion, they cannot participate in another fusion of the same class.
    Patterns are matched in a fixed priority order (see FUSION_PATTERNS).
    """
    physics_nodes = [n for n in nodes if n.domain == "physics" and n.primes]
    if not physics_nodes:
        return []

    # Fixed priority order (greedily pick fusions in this order)
    sorted_patterns = sorted(FUSION_PATTERNS, key=lambda p: p[4], reverse=True)

    fusions = []
    consumed = set()  # indices of physics_nodes already in a fusion

    for pattern_primes, class_name, multiplier, description, _prio in sorted_patterns:
        i = 0
        while i < len(physics_nodes):
            if i in consumed:
                i += 1
                continue

            window_primes = set()
            window_indices = []
            window_names = []
            found = False

            for j in range(i, min(i + 5, len(physics_nodes))):
                if j in consumed:
                    continue
                window_primes |= physics_nodes[j].primes
                window_indices.append(j)
                window_names.append(physics_nodes[j].name)

                if pattern_primes.issubset(window_primes):
                    fusions.append(FusionOpportunity(
                        pattern_name=description,
                        fusion_class=class_name,
                        multiplier=multiplier,
                        node_names=list(window_names),
                        primes_involved=pattern_primes,
                    ))
                    consumed.update(window_indices)
                    found = True
                    break

            i = (max(window_indices) + 1) if found else (i + 1)

    return fusions
