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


# Fusion patterns: (required_primes, class_name, multiplier, description)
FUSION_PATTERNS = [
    ({P.P3, P.P5},         "Race",          "5×",    "Softmax via race: 5 observables from 1 event"),
    ({P.P3, P.P5, P.P4},   "Race-Loss",     "6×",    "Softmax + CrossEntropy fused into single race"),
    ({P.P1, P.P2, P.P11, P.P12}, "Normalization", "4×", "RMSNorm/LayerNorm via AGC equilibrium"),
    ({P.P10, P.P7},         "Threshold",     "2-3×",  "Ising/Boltzmann: decision + confidence"),
    ({P.P1, P.P2, P.P8},   "Equilibrium",   "4-12×", "Resistor network: solution + R_eff + currents + power"),
]


def find_fusions(nodes: list) -> list[FusionOpportunity]:
    """Scan node list for multi-computation fusion opportunities.

    Uses greedy non-overlapping matching: once nodes are consumed by a
    fusion, they cannot participate in another fusion of the same class.
    Higher-multiplier patterns are matched first.
    """
    physics_nodes = [n for n in nodes if n.domain == "physics" and n.primes]
    if not physics_nodes:
        return []

    # Sort patterns by multiplier descending (greedily pick best fusions first)
    sorted_patterns = sorted(FUSION_PATTERNS, key=lambda p: p[2], reverse=True)

    fusions = []
    consumed = set()  # indices of physics_nodes already in a fusion

    for pattern_primes, class_name, multiplier, description in sorted_patterns:
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
