"""Prime Compiler — 4-phase compilation pipeline."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

import torch
import torch.fx

from prime_compiler.primes import Prime, MissingPrime, Status, GPU_AVAILABLE
from prime_compiler.factorizations import FACTORIZATIONS, get_factorization, classify
from prime_compiler.chips import ChipProfile, get_chip, CHIPS
from prime_compiler.fusion import find_fusions, FusionOpportunity


@dataclass
class NodeAnalysis:
    """Analysis result for a single graph node."""
    name: str
    op_type: str
    target: str
    primes: set = field(default_factory=set)
    missing: set = field(default_factory=set)
    status: Status = Status.M
    domain: str = "undecided"  # "physics" or "digital"
    note: str = ""
    matched_key: str = ""


@dataclass
class CompilationReport:
    """Full compilation report."""
    model_name: str
    chip_name: str
    nodes: list = field(default_factory=list)
    fusions: list = field(default_factory=list)
    transitions: int = 0
    optimal_transitions: int = 0

    @property
    def total_nodes(self) -> int:
        return len([n for n in self.nodes if n.primes or n.missing])

    @property
    def analog_nodes(self) -> int:
        return len([n for n in self.nodes if n.domain == "physics"])

    @property
    def digital_nodes(self) -> int:
        return len([n for n in self.nodes if n.domain == "digital"])

    @property
    def routing_nodes(self) -> int:
        return len([n for n in self.nodes if not n.primes and not n.missing])

    @property
    def status_counts(self) -> dict:
        c = Counter(n.status for n in self.nodes if n.primes or n.missing)
        return {s.name: c.get(s, 0) for s in Status}

    @property
    def prime_histogram(self) -> dict:
        c = Counter()
        for n in self.nodes:
            c.update(n.primes)
        return {p.value: count for p, count in c.most_common()}

    @property
    def summary(self) -> str:
        sc = self.status_counts
        total = self.total_nodes
        analog = self.analog_nodes
        digital = self.digital_nodes
        routing = self.routing_nodes

        lines = [
            f"═══ Prime Compiler Report: {self.model_name} ═══",
            f"Target chip: {self.chip_name}",
            f"",
            f"Nodes: {total} compute + {routing} routing = {len(self.nodes)} total",
            f"  Analog (physics): {analog} ({analog/max(total,1)*100:.0f}%)",
            f"  Digital:          {digital} ({digital/max(total,1)*100:.0f}%)",
            f"",
            f"Mappability:",
            f"  M (fully mappable):  {sc['M']}",
            f"  G (GPU-mappable):    {sc['G']}",
            f"  U (unmappable):      {sc['U']}",
            f"",
            f"Multi-computation fusions: {len(self.fusions)}",
        ]
        for f in self.fusions:
            lines.append(f"  {f}")

        lines.extend([
            f"",
            f"Domain transitions (ADC/DAC): {self.transitions}",
            f"  Optimal (after island merging): {self.optimal_transitions}",
            f"",
            f"Prime histogram:",
        ])
        for sym, count in self.prime_histogram.items():
            lines.append(f"  {sym:>6}: {count}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "chip": self.chip_name,
            "total_nodes": self.total_nodes,
            "analog_nodes": self.analog_nodes,
            "digital_nodes": self.digital_nodes,
            "routing_nodes": self.routing_nodes,
            "status_counts": self.status_counts,
            "prime_histogram": self.prime_histogram,
            "fusions": [str(f) for f in self.fusions],
            "transitions": self.transitions,
            "optimal_transitions": self.optimal_transitions,
            "nodes": [
                {
                    "name": n.name,
                    "op": n.target,
                    "primes": [p.value for p in n.primes],
                    "missing": [m.value for m in n.missing],
                    "status": n.status.name,
                    "domain": n.domain,
                    "note": n.note,
                }
                for n in self.nodes
                if n.primes or n.missing
            ],
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class PrimeCompiler:
    """The 4-phase prime compilation pipeline."""

    def __init__(self, chip: str | ChipProfile = "full_analog"):
        if isinstance(chip, str):
            self.chip = get_chip(chip)
        else:
            self.chip = chip

    def compile(self, model: torch.nn.Module, model_name: str = "",
                example_input: Optional[torch.Tensor] = None) -> CompilationReport:
        """Run the full 4-phase compilation pipeline."""
        if not model_name:
            model_name = model.__class__.__name__

        # Trace model with torch.fx
        if example_input is None:
            example_input = torch.randn(1, 64)

        try:
            tracer = torch.fx.Tracer()
            graph = tracer.trace(model)
            gm = torch.fx.GraphModule(model, graph)
        except Exception as e:
            # Fallback: analyze module structure without tracing
            return self._compile_from_modules(model, model_name)

        # Phase 1: Annotate
        nodes = self._phase1_annotate(gm)

        # Phase 2: Domain assignment
        self._phase2_assign(nodes)

        # Phase 3: Multi-computation fusion
        fusions = self._phase3_fuse(nodes)

        # Phase 4: Transition minimization
        transitions, optimal = self._phase4_minimize(nodes)

        return CompilationReport(
            model_name=model_name,
            chip_name=self.chip.name,
            nodes=nodes,
            fusions=fusions,
            transitions=transitions,
            optimal_transitions=optimal,
        )

    def _phase1_annotate(self, gm: torch.fx.GraphModule) -> list[NodeAnalysis]:
        """Phase 1: Annotate each node with its prime factorization."""
        nodes = []
        for node in gm.graph.nodes:
            analysis = NodeAnalysis(
                name=node.name,
                op_type=node.op,
                target=str(node.target) if node.target else node.op,
            )

            if node.op == "call_function":
                target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                analysis.target = target_name
                key, fact = get_factorization(target_name)
                if fact:
                    analysis.primes = fact[0]
                    analysis.missing = fact[1]
                    analysis.status = fact[2]
                    analysis.note = fact[3]
                    analysis.matched_key = key

            elif node.op == "call_module":
                module = _get_module(gm, node.target)
                if module is not None:
                    mod_type = type(module).__name__.lower()
                    key, fact = get_factorization(mod_type)
                    if fact:
                        analysis.primes = fact[0]
                        analysis.missing = fact[1]
                        analysis.status = fact[2]
                        analysis.note = fact[3]
                        analysis.matched_key = key
                    analysis.target = type(module).__name__

            elif node.op == "call_method":
                key, fact = get_factorization(str(node.target))
                if fact:
                    analysis.primes = fact[0]
                    analysis.missing = fact[1]
                    analysis.status = fact[2]
                    analysis.note = fact[3]
                    analysis.matched_key = key

            nodes.append(analysis)
        return nodes

    def _phase2_assign(self, nodes: list[NodeAnalysis]):
        """Phase 2: Assign each node to physics or digital domain."""
        for node in nodes:
            if not node.primes and not node.missing:
                node.domain = "routing"
                continue

            if node.missing:
                # Has missing primes → must be digital (or GPU)
                node.domain = "digital"
                continue

            # Check if all primes are available on the chip
            if node.primes.issubset(self.chip.analog_primes):
                node.domain = "physics"
            else:
                node.domain = "digital"

    def _phase3_fuse(self, nodes: list[NodeAnalysis]) -> list[FusionOpportunity]:
        """Phase 3: Find multi-computation fusion opportunities."""
        return find_fusions(nodes)

    def _phase4_minimize(self, nodes: list[NodeAnalysis]) -> tuple[int, int]:
        """Phase 4: Count and minimize domain transitions."""
        compute_nodes = [n for n in nodes if n.domain in ("physics", "digital")]
        if not compute_nodes:
            return 0, 0

        # Count raw transitions
        transitions = 0
        for i in range(1, len(compute_nodes)):
            if compute_nodes[i].domain != compute_nodes[i-1].domain:
                transitions += 1

        # Greedy island merging: flip small islands to majority neighbor
        domains = [n.domain for n in compute_nodes]
        optimal_domains = list(domains)

        # Merge islands of size 1 into neighbors
        for i in range(1, len(optimal_domains) - 1):
            prev_d = optimal_domains[i-1]
            next_d = optimal_domains[i+1]
            if prev_d == next_d and optimal_domains[i] != prev_d:
                optimal_domains[i] = prev_d

        optimal = sum(1 for i in range(1, len(optimal_domains))
                      if optimal_domains[i] != optimal_domains[i-1])

        return transitions, optimal

    def _compile_from_modules(self, model: torch.nn.Module, model_name: str) -> CompilationReport:
        """Fallback: analyze from module tree when tracing fails."""
        nodes = []
        for name, module in model.named_modules():
            if name == "":
                continue
            mod_type = type(module).__name__.lower()
            key, fact = get_factorization(mod_type)

            analysis = NodeAnalysis(
                name=name,
                op_type="module",
                target=type(module).__name__,
            )
            if fact:
                analysis.primes = fact[0]
                analysis.missing = fact[1]
                analysis.status = fact[2]
                analysis.note = fact[3]
                analysis.matched_key = key

            nodes.append(analysis)

        self._phase2_assign(nodes)
        fusions = self._phase3_fuse(nodes)
        transitions, optimal = self._phase4_minimize(nodes)

        return CompilationReport(
            model_name=model_name,
            chip_name=self.chip.name,
            nodes=nodes,
            fusions=fusions,
            transitions=transitions,
            optimal_transitions=optimal,
        )


def _get_module(gm: torch.fx.GraphModule, target: str):
    """Safely get a submodule by dotted path."""
    parts = target.split(".")
    mod = gm
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return None
    return mod


def compile(model: torch.nn.Module | str, chip: str = "full_analog",
            model_name: str = "", example_input: Optional[torch.Tensor] = None) -> CompilationReport:
    """Convenience function: compile a model with the prime compiler.

    Args:
        model: PyTorch model or model name string
        chip: Target chip profile name
        model_name: Display name for the report
        example_input: Example input tensor for tracing

    Returns:
        CompilationReport with full analysis
    """
    if isinstance(model, str):
        model_name = model_name or model
        raise ValueError(f"Pass a torch.nn.Module, not a string. Got: '{model}'")

    compiler = PrimeCompiler(chip=chip)
    return compiler.compile(model, model_name=model_name, example_input=example_input)
