"""Chip profiles — hardware specifications for domain assignment."""

from dataclasses import dataclass, field
from prime_compiler.primes import Prime as P


@dataclass
class ChipProfile:
    """Hardware profile for a target analog/digital chip."""
    name: str
    description: str
    # Which primes are available in analog on this chip
    analog_primes: set = field(default_factory=set)
    # ADC cost in femtojoules
    adc_cost_fj: float = 100.0
    # DAC cost in femtojoules
    dac_cost_fj: float = 50.0
    # ADC latency in nanoseconds
    adc_latency_ns: float = 1.0
    # DAC latency in nanoseconds
    dac_latency_ns: float = 0.5
    # Noise budget per prime (0-1, fraction of signal)
    noise_budget: float = 0.01
    # Max fan-in for P1 (accumulation)
    max_fan_in: int = 256
    # Precision bits available in analog
    analog_bits: int = 6


# ── Built-in chip profiles ──

CHIPS = {
    "ibm_pcm": ChipProfile(
        name="IBM 14nm PCM",
        description="Phase-change memory crossbar, 35M devices, ALBERT demo (Nature Comms 2025)",
        analog_primes={P.P1, P.P2},  # Only MatMul
        adc_cost_fj=200, dac_cost_fj=100,
        analog_bits=4, max_fan_in=512,
    ),
    "tsmc_cim": ChipProfile(
        name="TSMC Mixed CIM",
        description="Memristor+SRAM heterogeneous CIM (Nature 2025), 40.91 TFLOPS/W",
        analog_primes={P.P1, P.P2, P.P8},
        adc_cost_fj=150, dac_cost_fj=80,
        analog_bits=8, max_fan_in=256,
    ),
    "juelich_gaincell": ChipProfile(
        name="Jülich Gain-Cell",
        description="Analog attention for LLMs (Nature Comp Sci 2025), 70000x energy reduction",
        analog_primes={P.P1, P.P2, P.P7},  # QK^T + HardSigmoid
        adc_cost_fj=100, dac_cost_fj=50,
        analog_bits=6, max_fan_in=1024,
    ),
    "extropic_xtr0": ChipProfile(
        name="Extropic XTR-0",
        description="Thermodynamic computing board, Boltzmann machines for diffusion (arXiv 2025)",
        analog_primes={P.P1, P.P2, P.P3, P.P7, P.P10},
        adc_cost_fj=50, dac_cost_fj=30,
        analog_bits=1, max_fan_in=64,
    ),
    "normal_cn101": ChipProfile(
        name="Normal Computing CN101",
        description="RLC-based sampling ASIC (Nature Comms 2025), Gaussian sampling + matrix inversion",
        analog_primes={P.P1, P.P2, P.P8, P.P10},
        adc_cost_fj=80, dac_cost_fj=40,
        analog_bits=8, max_fan_in=128,
    ),
    "full_analog": ChipProfile(
        name="Full Analog (theoretical)",
        description="Hypothetical chip with all 12 primes in analog — theoretical upper bound",
        analog_primes={P.P1, P.P2, P.P3, P.P4, P.P5, P.P6, P.P7, P.P8, P.P9, P.P10, P.P11, P.P12},
        adc_cost_fj=50, dac_cost_fj=25,
        analog_bits=8, max_fan_in=256,
    ),
    "smtj_race": ChipProfile(
        name="sMTJ Race Array",
        description="Stochastic barrier attention chip — Softmax + Multi-Computation (Bieg 2025/2026)",
        analog_primes={P.P1, P.P2, P.P3, P.P5, P.P7, P.P10},
        adc_cost_fj=30, dac_cost_fj=20,
        analog_bits=1, max_fan_in=1024,  # Race scales well
    ),
}


def get_chip(name: str) -> ChipProfile:
    if name not in CHIPS:
        available = ", ".join(CHIPS.keys())
        raise ValueError(f"Unknown chip '{name}'. Available: {available}")
    return CHIPS[name]
