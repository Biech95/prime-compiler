"""Tests for the prime compiler."""

import torch
import torch.nn as nn
from prime_compiler import compile, PrimeCompiler
from prime_compiler.primes import Prime as P, Status
from prime_compiler.factorizations import get_factorization, FACTORIZATIONS
from prime_compiler.chips import get_chip


# ── Test factorization database ──

def test_factorization_lookup():
    key, fact = get_factorization("relu")
    assert key == "relu"
    assert P.P7 in fact[0]
    assert fact[2] == Status.M

def test_factorization_softmax():
    key, fact = get_factorization("softmax")
    assert P.P3 in fact[0]
    assert P.P1 in fact[0]
    assert fact[2] == Status.M

def test_factorization_layer_norm():
    key, fact = get_factorization("layer_norm")
    assert P.P12 in fact[0]  # Needs inversion
    assert fact[2] == Status.M

def test_factorization_ctc_unmappable():
    key, fact = get_factorization("ctc_loss")
    assert fact[2] == Status.U

def test_factorization_batch_norm_gpu():
    key, fact = get_factorization("batch_norm")
    assert fact[2] == Status.G

def test_all_factorizations_have_status():
    for key, (primes, missing, status, note) in FACTORIZATIONS.items():
        assert isinstance(status, Status), f"{key} has no status"


# ── Test chip profiles ──

def test_chip_ibm():
    chip = get_chip("ibm_pcm")
    assert P.P1 in chip.analog_primes
    assert P.P2 in chip.analog_primes
    assert P.P3 not in chip.analog_primes

def test_chip_full_analog():
    chip = get_chip("full_analog")
    assert len(chip.analog_primes) == 12


# ── Test compilation on simple models ──

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_compile_mlp():
    model = SimpleMLP()
    report = compile(model, chip="full_analog", model_name="SimpleMLP")
    assert report.model_name == "SimpleMLP"
    assert report.total_nodes > 0
    assert report.analog_nodes >= 0

def test_compile_mlp_ibm():
    model = SimpleMLP()
    report = compile(model, chip="ibm_pcm")
    # IBM only has P1,P2 — ReLU needs P7,P11 → digital
    assert report.digital_nodes > 0


class TransformerBlock(nn.Module):
    def __init__(self, d=128, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x

def test_compile_transformer():
    model = TransformerBlock()
    report = compile(model, chip="full_analog", model_name="TransformerBlock",
                     example_input=torch.randn(1, 16, 128))
    assert report.total_nodes > 0
    print(report.summary)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        return self.fc(x.flatten(1))

def test_compile_convnet():
    model = ConvNet()
    report = compile(model, chip="tsmc_cim", model_name="ConvNet",
                     example_input=torch.randn(1, 3, 32, 32))
    assert report.total_nodes > 0


# ── Test JSON output ──

def test_json_output():
    model = SimpleMLP()
    report = compile(model)
    j = report.to_json()
    import json
    d = json.loads(j)
    assert "model" in d
    assert "nodes" in d


# ── Test CLI stats ──

def test_database_coverage():
    """At least 80 ops in the database."""
    assert len(FACTORIZATIONS) >= 80


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
