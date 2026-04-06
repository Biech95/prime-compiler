"""CLI entry point: prime-compile."""

import argparse
import sys
import torch
from prime_compiler.compiler import PrimeCompiler
from prime_compiler.chips import CHIPS
from prime_compiler.factorizations import FACTORIZATIONS
from prime_compiler.primes import Status


def main():
    parser = argparse.ArgumentParser(
        prog="prime-compile",
        description="Computational Primes: Analyze neural networks for analog/digital partitioning",
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze ──
    p_analyze = sub.add_parser("analyze", help="Analyze a PyTorch model")
    p_analyze.add_argument("model", help="Model class (e.g., 'torchvision.models.resnet18')")
    p_analyze.add_argument("--chip", default="full_analog", help=f"Target chip: {', '.join(CHIPS.keys())}")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")

    # ── chips ──
    p_chips = sub.add_parser("chips", help="List available chip profiles")

    # ── primes ──
    p_primes = sub.add_parser("primes", help="List all computational primes")

    # ── stats ──
    p_stats = sub.add_parser("stats", help="Show factorization database statistics")

    args = parser.parse_args()

    if args.command == "chips":
        cmd_chips()
    elif args.command == "primes":
        cmd_primes()
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


def cmd_chips():
    print("Available chip profiles:\n")
    for name, chip in CHIPS.items():
        primes = ", ".join(p.value for p in sorted(chip.analog_primes, key=lambda x: x.name))
        print(f"  {name:<20} {chip.name}")
        print(f"  {'':20} Analog primes: {primes}")
        print(f"  {'':20} ADC: {chip.adc_cost_fj} fJ, DAC: {chip.dac_cost_fj} fJ")
        print(f"  {'':20} {chip.description}")
        print()


def cmd_primes():
    from prime_compiler.primes import PRIME_DB, MissingPrime
    print("The 12 Physical Primes:\n")
    for p, info in PRIME_DB.items():
        print(f"  {p.name:>3} {info.symbol:>6}  {info.name:<20} {info.physics}")
    print("\nThe 4 Missing Primes:\n")
    for x in MissingPrime:
        gpu = "✓" if x in {MissingPrime.X2, MissingPrime.X4} else "✕"
        print(f"  {x.name:>3} {x.value:>6}  GPU: {gpu}")


def cmd_stats():
    total = len(FACTORIZATIONS)
    m = sum(1 for _, (_, _, s, _) in FACTORIZATIONS.items() if s == Status.M)
    g = sum(1 for _, (_, _, s, _) in FACTORIZATIONS.items() if s == Status.G)
    u = sum(1 for _, (_, _, s, _) in FACTORIZATIONS.items() if s == Status.U)

    print(f"Factorization database: {total} operations\n")
    print(f"  M (fully mappable):  {m:>4} ({m/total*100:.0f}%)")
    print(f"  G (GPU-mappable):    {g:>4} ({g/total*100:.0f}%)")
    print(f"  U (unmappable):      {u:>4} ({u/total*100:.0f}%)")

    # Prime frequency
    from collections import Counter
    from prime_compiler.primes import Prime
    c = Counter()
    for _, (primes, _, _, _) in FACTORIZATIONS.items():
        c.update(primes)
    print(f"\nMost used primes:")
    for p, count in c.most_common():
        print(f"  {p.name:>3} {p.value:>6}: {count}")


def cmd_analyze(args):
    # Try to import the model
    parts = args.model.rsplit(".", 1)
    if len(parts) == 2:
        import importlib
        mod = importlib.import_module(parts[0])
        model_fn = getattr(mod, parts[1])
        model = model_fn()
    else:
        print(f"Error: Specify model as 'module.class', e.g., 'torchvision.models.resnet18'")
        sys.exit(1)

    compiler = PrimeCompiler(chip=args.chip)
    report = compiler.compile(model, model_name=args.model)

    if args.json:
        print(report.to_json())
    else:
        print(report.summary)


if __name__ == "__main__":
    main()
