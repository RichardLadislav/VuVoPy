import numpy as np
import pytest
from VuVoPy.features.utils.relf1f2sd import relF1SD, relF2SD
import VuVoPy.features.utils.relf1f2sd as mod


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    # Save originals
    original_ff = mod.ff
    original_vs = mod.vs.from_wav
    original_pp = mod.pp.from_voice_sample
    original_sg = mod.sg.from_voice_sample

    # Fake FormantFrequencies class
    class FakeFF:
        def __init__(self):
            pass

        @classmethod
        def from_voice_sample(cls, seg):
            return cls()

        def get_formants_preem(self):
            # returns a (N,3) array assigned per-test to FakeFF.array
            return FakeFF.array

    # Patch in our fakes
    monkeypatch.setattr(mod, "ff", FakeFF)
    monkeypatch.setattr(mod.vs, "from_wav", lambda path: None)
    monkeypatch.setattr(mod.pp, "from_voice_sample", lambda vsamp: None)
    monkeypatch.setattr(mod.sg, "from_voice_sample", lambda ppobj, winlen, wintype, winover: None)

    yield FakeFF

    # Restore originals
    monkeypatch.setattr(mod, "ff", original_ff)
    monkeypatch.setattr(mod.vs, "from_wav", original_vs)
    monkeypatch.setattr(mod.pp, "from_voice_sample", original_pp)
    monkeypatch.setattr(mod.sg, "from_voice_sample", original_sg)


def test_relF1SD_constant(patch_pipeline):
    """Constant F1 values ⇒ zero std ⇒ +inf."""
    FakeFF = patch_pipeline
    FakeFF.array = np.array([[100.0, 200.0, 300.0],
                              [100.0, 200.0, 300.0]])
    result = relF1SD("ignored.wav")
    assert np.isnan(result)


def test_relF2SD_constant(patch_pipeline):
    """Constant F2 values ⇒ zero std ⇒ +inf."""
    FakeFF = patch_pipeline
    FakeFF.array = np.array([[100.0, 200.0, 300.0],
                              [100.0, 200.0, 300.0]])
    result = relF2SD("ignored.wav")
    assert np.isnan(result)


def test_relF1SD_nontrivial(patch_pipeline):
    """Nontrivial F1 sequence yields correct mean/std ratio."""
    FakeFF = patch_pipeline
    # F1 values = [100, 200, 300]
    FakeFF.array = np.array([[100.0,   0.0,  0.0],
                              [200.0,   0.0,  0.0],
                              [300.0,   0.0,  0.0]])
    arr = FakeFF.array[:, 0]
    expected = np.mean(arr) / np.std(arr)
    result = relF1SD("ignored.wav")
    assert pytest.approx(result, rel=1e-9) == expected


def test_relF2SD_nontrivial(patch_pipeline):
    """Nontrivial F2 sequence yields correct mean/std ratio."""
    FakeFF = patch_pipeline
    # F2 values = [400, 500, 600]
    FakeFF.array = np.array([[  0.0, 400.0,   0.0],
                              [  0.0, 500.0,   0.0],
                              [  0.0, 600.0,   0.0]])
    arr = FakeFF.array[:, 1]
    expected = np.mean(arr) / np.std(arr)
    result = relF2SD("ignored.wav")
    assert pytest.approx(result, rel=1e-9) == expected


def test_relF_empty(patch_pipeline):
    """Empty formant array ⇒ mean/std = nan."""
    FakeFF = patch_pipeline
    FakeFF.array = np.zeros((0, 3))
    r1 = relF1SD("ignored.wav")
    r2 = relF2SD("ignored.wav")
    assert np.isnan(r1)
    assert np.isnan(r2)
