import numpy as np
import pytest
from VuVoPy.features.utils.relf0sd import relF0SD
import VuVoPy.features.utils.relf0sd as mod
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_f0_and_vswav(monkeypatch):
    """Patch out FundamentalFrequency (mod.f0) and vs.from_wav so we control get_f0()."""
    # Save originals
    original_f0 = mod.f0
    original_vs_from_wav = mod.vs.from_wav

    # Fake F0 extractor
    class FakeF0:
        def __init__(self, sample, plim, hop_size, dlog2p, dERBs, sTHR):
            pass
        def get_f0(self):
            return FakeF0.f0_array

    # Apply patches
    monkeypatch.setattr(mod, "f0", FakeF0)
    monkeypatch.setattr(mod.vs, "from_wav", lambda path: None)
    yield FakeF0

    # Restore originals
    monkeypatch.setattr(mod, "f0", original_f0)
    monkeypatch.setattr(mod.vs, "from_wav", original_vs_from_wav)


def test_relf0sd_basic(patch_f0_and_vswav):
    """Mean/std for [100,200,300] → mean=200, std≈81.6497 → ratio≈2.4494897."""
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([100.0, 200.0, 300.0])
    result = relF0SD("ignored.wav")
    expected = np.mean(FakeF0.f0_array) / np.std(FakeF0.f0_array)
    assert pytest.approx(result, rel=1e-9) == expected


def test_relf0sd_constant(patch_f0_and_vswav):
    """If f0 is constant, std=0 ⇒ ratio → +inf."""
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([150.0, 150.0, 150.0])
    result = relF0SD("ignored.wav")
    assert np.isnan(result)


def test_relf0sd_empty(patch_f0_and_vswav):
    """Empty F0 array ⇒ mean/std = nan/nan ⇒ result nan."""
    FakeF0 = patch_f0_and_vswav
    FakeF0.f0_array = np.array([])
    result = relF0SD("ignored.wav")
    assert np.isnan(result)
