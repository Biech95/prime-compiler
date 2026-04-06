"""Computational Primes — the 12 irreducible operations + 4 missing."""

from enum import Enum, auto
from dataclasses import dataclass


class Prime(Enum):
    """The 12 physical primes with analog hardware implementations."""
    P1 = "Σ"        # Accumulation — KCL
    P2 = "×c"       # Scaling — Ohm's law
    P3 = "eˣ"       # Exponential — Boltzmann / subthreshold FET
    P4 = "ln"       # Logarithm — diode voltage
    P5 = "argmax"   # Selection — race / WTA
    P6 = "σ"        # Sigmoid — Fermi-Dirac
    P7 = "θ"        # Threshold — comparator
    P8 = "∫"        # Integration — capacitor
    P9 = "d/dt"     # Differentiation — RC high-pass
    P10 = "ξ"       # Stochasticity — thermal noise
    P11 = "×"       # Variable multiplication — Gilbert cell
    P12 = "1/x"     # Inversion — translinear loop


class MissingPrime(Enum):
    """The 4 missing primes — no analog implementation."""
    X1 = "SYM"      # Symbol manipulation
    X2 = "DYN"      # Dynamic topology (available on GPU)
    X3 = "REC"      # Unbounded recursion
    X4 = "TAPE"     # Symbolic tracing (available on GPU)


class Status(Enum):
    """Mappability status of an algorithm."""
    M = auto()   # Fully mappable to analog
    G = auto()   # GPU-mappable (X2/X4 available)
    U = auto()   # Unmappable (needs X1 or X3)


# GPU availability of missing primes
GPU_AVAILABLE = {MissingPrime.X2, MissingPrime.X4}


@dataclass
class PrimeInfo:
    """Metadata for a physical prime."""
    prime: Prime
    name: str
    symbol: str
    physics: str
    hardware: str
    physics_if: str
    digital_if: str

PRIME_DB = {
    Prime.P1:  PrimeInfo(Prime.P1,  "Accumulation",    "Σ",      "KCL",                    "Wire junction",        "fan-in < 256",            "fan-in > 1000"),
    Prime.P2:  PrimeInfo(Prime.P2,  "Scaling",         "×c",     "Ohm's Law",              "Memristor/resistor",   "variability < tolerance", "precision > 8b"),
    Prime.P3:  PrimeInfo(Prime.P3,  "Exponential",     "eˣ",     "Boltzmann",              "sMTJ/subthreshold",    "range < 10 decades",      "extreme range"),
    Prime.P4:  PrimeInfo(Prime.P4,  "Logarithm",       "ln",     "Diode V=kT/q·ln(I/Is)",  "Diode/subthreshold",   "η < 1.2",                 "ideality factor"),
    Prime.P5:  PrimeInfo(Prime.P5,  "Selection",       "argmax", "Race to threshold",       "sMTJ array / WTA",     "always physics",          "—"),
    Prime.P6:  PrimeInfo(Prime.P6,  "Sigmoid",         "σ",      "Fermi-Dirac",            "Subthreshold FET pair","kT/q slope matches",      "exact slope needed"),
    Prime.P7:  PrimeInfo(Prime.P7,  "Threshold",       "θ",      "Phase transition",       "Schmitt trigger",      "always physics",          "—"),
    Prime.P8:  PrimeInfo(Prime.P8,  "Integration",     "∫",      "Charge Q=∫Idt",          "Capacitor/memristor",  "hold > period",           "leakage issue"),
    Prime.P9:  PrimeInfo(Prime.P9,  "Differentiation", "d/dt",   "V=LdI/dt",               "RC high-pass",         "bandwidth OK",            "HF noise"),
    Prime.P10: PrimeInfo(Prime.P10, "Stochasticity",   "ξ",      "Thermal fluctuation",    "Johnson-Nyquist/RTN",  "always physics",          "—"),
    Prime.P11: PrimeInfo(Prime.P11, "Multiplication",  "×",      "Transistor square law",  "Gilbert cell",         "THD < tolerance",         "linearity > 1%"),
    Prime.P12: PrimeInfo(Prime.P12, "Inversion",       "1/x",    "Translinear loop",       "BJT/MOSFET circuit",   "translinear range OK",    "range > 40dB"),
}
