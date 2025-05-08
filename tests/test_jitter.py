import sys, os
import numpy as np
import pytest
import VuVoPy.features.utils.jitter as mod
from VuVoPy.features.utils.jitter import jitterPPQ
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_f0_and_vswav(monkeypatch):
    # 1) Save originals
    original_f0 = mod.f0
    original_from_wav = mod.vs.from_wav

    # 2) Fake FundamentalFrequency class
    class FakeF0:
        def __init__(self, sample, plim, hop_size, dlog2p, dERBs, sTHR):
            pass
        def get_f0(self):
            return FakeF0.f0_array

    # 3) Patch f0 and vs.from_wav (to avoid I/O)
    monkeypatch.setattr(mod, "f0", FakeF0)
    monkeypatch.setattr(mod.vs, "from_wav", lambda path: None)

    yield FakeF0

    # 4) Restore originals
    monkeypatch.setattr(mod, "f0", original_f0)
    monkeypatch.setattr(mod.vs, "from_wav", original_from_wav)


def test_jitterppq_too_few_points(tmp_path, patch_f0_and_vswav):
    # Provide fewer voiced points than n_points ⇒ returns 0.0
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([0.0, 100.0, 0.0])  # only one >0, but default n_points=3 ⇒ len<3
    result = jitterPPQ("ignored.wav", n_points=3)
    assert result == 0.0


def test_jitterppq_constant_freqs(tmp_path, patch_f0_and_vswav):
    # Three identical voiced samples ⇒ zero jitter
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([0.0, 200.0, 200.0, 200.0, 0.0])
    # After filtering >0 ⇒ [200,200,200], k=1 ⇒ deviations all zero ⇒ mean=0
    result = jitterPPQ("ignored.wav", n_points=3)
    assert pytest.approx(result, abs=1e-12) == 0.0


def test_jitterppq_nontrivial(tmp_path, patch_f0_and_vswav):
    # Use frequencies [100,50,100] ⇒ periods [0.01,0.02,0.01], k=1 ⇒ single deviation=0.5
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([0.0, 100.0, 50.0, 100.0, 0.0])
    # filtering leaves [100,50,100]
    expected = 0.5
    result = jitterPPQ("ignored.wav", n_points=3)
    assert pytest.approx(result, rel=1e-9) == expected


def test_jitterppq_custom_n_points(patch_f0_and_vswav):
    """Custom n_points logic produces expected deviation"""
    FakeF0 = patch_f0_and_vswav
    # five voiced frequencies
    FakeF0.f0_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    n_points = 5
    # Compute expected deviation dynamically
    T = 1.0 / FakeF0.f0_array
    k = n_points // 2
    T_bar = np.mean(T)
    expected = abs(T[k] - T_bar) / T_bar
    result = jitterPPQ("ignored.wav", n_points=n_points)
    assert pytest.approx(result, rel=1e-9) == expected